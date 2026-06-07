from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.models.schemas import CandidateEvaluation, ExtractedFeatures
from app.services.langchain_structured_chat import OpenAICompatibleStructuredChat
from app.services.llm_cache import LLMCache

_DEFAULT_TIMEOUT = object()


class ArchitectureRecommendationPayload(BaseModel):
    candidates: list[CandidateEvaluation] = Field(default_factory=list)
    composition_recommendation: dict[str, Any] = Field(default_factory=dict)
    review_notes: list[str] = Field(default_factory=list)


class LLMClient:
    """OpenAI-compatible LLM adapter. DeepSeek, Qwen compatible gateways can use it."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = (os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS") or os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "12"))
        self.chat_url = self._build_chat_url(self.base_url)
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or self.api_key
        self.embedding_base_url = (os.getenv("EMBEDDING_BASE_URL") or "").rstrip("/")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "")
        self.embedding_url = self._build_embedding_url(self.embedding_base_url) if self.embedding_base_url else ""
        self.last_error: str | None = None
        cache_enabled = os.getenv("LLM_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        resolved_cache_path = cache_path or Path(os.getenv("LLM_CACHE_PATH", "data/llm_cache.json"))
        self.cache = LLMCache(resolved_cache_path) if cache_enabled else None
        set_llm_cache(self.cache)

    async def generate_report(
        self,
        requirement: str,
        features: ExtractedFeatures,
        candidates: list[CandidateEvaluation],
    ) -> str | None:
        if not self.api_key:
            return None

        prompt = self._build_prompt(requirement, features, candidates)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是软件体系结构评审专家，输出简洁、可追溯、适合课程作业展示的中文报告。"},
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "stream": False,
        }
        try:
            return await self._chat(payload)
        except Exception:
            return None

    async def recommend_architectures(
        self,
        requirement: str,
        features: ExtractedFeatures,
        styles: list[dict[str, Any]],
        top_k: int,
        case_references: list[dict[str, Any]] | None = None,
    ) -> tuple[list[CandidateEvaluation], dict[str, Any]] | None:
        if not self.api_key:
            self.last_error = "DeepSeek API Key 未配置，无法进行架构匹配。"
            return None

        candidate_count = max(3, min(top_k, 6))
        style_payload = [
            {
                "id": style.get("id"),
                "name": style.get("name"),
                "category": style.get("category"),
                "description": style.get("description"),
                "suitable_for": style.get("suitable_for", []),
                "quality_scores": style.get("quality_scores", {}),
            }
            for style in styles
        ]
        case_references = case_references or []
        case_payload = [
            {
                "title": item["title"],
                "requirement": item["requirement"],
                "expected_styles": item["expected_styles"],
                "recommended_styles": item["recommended_styles"],
                "notes": item["notes"],
                "similarity": item["similarity"],
            }
            for item in case_references
        ]
        prompt = (
            "你是架构匹配 Agent。请完全基于用户需求、结构化特征和候选架构知识库，推荐候选体系结构风格。\n"
            "只返回 JSON，不要 Markdown，不要解释。\n\n"
            "输出 JSON Schema：\n"
            "{\n"
            "  \"candidates\": [\n"
            "    {\n"
            "      \"style_id\": \"必须来自候选架构 id\",\n"
            "      \"name\": \"架构中文名\",\n"
            "      \"score\": 0到100,\n"
            "      \"raw_score\": 0到100,\n"
            "      \"recommendation_role\": \"核心推荐|备选方案|专项补充|不推荐\",\n"
            "      \"confidence\": \"高|中高|中|中低|低\",\n"
            "      \"matched_reasons\": [\"理由\"],\n"
            "      \"risks\": [\"风险\"],\n"
            "      \"deductions\": [\"扣分原因\"],\n"
            "      \"quality_scores\": {\"scalability\":0到1,\"performance\":0到1,\"reliability\":0到1,\"modifiability\":0到1,\"complexity\":0到1,\"realtime\":0到1}\n"
            "    }\n"
            "  ],\n"
            "  \"composition_recommendation\": {\n"
            "    \"composition_needed\": true或false,\n"
            "    \"primary_style\": \"主架构\",\n"
            "    \"supporting_styles\": [{\"style_id\":\"id\",\"style\":\"名称\",\"role\":\"职责\",\"apply_to\":[\"组件或能力\"],\"reason\":\"原因\",\"score\":0到100}],\n"
            "    \"reason\": \"组合或不组合原因\",\n"
            "    \"triggers\": [\"触发依据\"],\n"
            "    \"overengineering_warnings\": [\"过度设计提醒\"]\n"
            "  },\n"
            "  \"review_notes\": [\"候选排序复核意见\"]\n"
            "}\n\n"
            "约束：\n"
            f"1. candidates 必须返回 {candidate_count} 个。\n"
            "2. style_id 必须来自候选架构知识库，不能编造。\n"
            "3. 简单低并发 CRUD/审批/库存类系统不要过度推荐微服务或事件驱动。\n"
            "4. 高并发、强实时、最终一致性、独立部署等需求可以推荐微服务、事件驱动或 CQRS。\n"
            "5. score 必须拉开合理差距，不要所有架构都给 100。\n\n"
            "相似历史案例只作为参考材料：命中案例可以帮助解释相近场景，但不得覆盖当前需求特征和候选架构知识库。\n\n"
            f"用户需求：{requirement}\n"
            f"结构化特征：{features.model_dump_json()}\n"
            f"候选架构知识库：{json.dumps(style_payload, ensure_ascii=False)}\n"
            f"相似历史案例：{json.dumps(case_payload, ensure_ascii=False)}\n"
        )
        try:
            structured = await self._ainvoke_structured(
                system_message="你是软件体系结构风格匹配 Agent，只输出可解析 JSON。",
                user_prompt=prompt,
                schema=ArchitectureRecommendationPayload,
                temperature=0.1,
                max_tokens=3200,
            )
            recommendation = (
                structured
                if isinstance(structured, ArchitectureRecommendationPayload)
                else ArchitectureRecommendationPayload.model_validate(structured)
            )
            data = recommendation.model_dump()
            return self._sanitize_architecture_recommendation(data, styles, candidate_count)
        except Exception as exc:
            self.last_error = f"DeepSeek 架构匹配 JSON 校验失败：{exc}"
            return None

    async def stream_report(
        self,
        requirement: str,
        features: ExtractedFeatures,
        candidates: list[CandidateEvaluation],
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            return

        prompt = self._build_prompt(requirement, features, candidates)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是软件体系结构评审专家，输出简洁、可追溯、适合课程作业展示的中文报告。"},
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "stream": True,
        }
        async for chunk in self._stream_chat(payload):
            yield chunk

    async def generate_topology(
        self,
        requirement: str,
        features: ExtractedFeatures,
        winner: CandidateEvaluation,
    ) -> str | None:
        if not self.api_key:
            return None

        prompt = (
            "请根据本次软件需求和最终推荐架构，生成一张定制化 Mermaid 架构拓扑图。\n"
            "只返回 Mermaid 源码，不要 Markdown 代码块，不要解释。\n"
            "要求：\n"
            "1. 使用 flowchart LR 或 flowchart TD。\n"
            "2. 节点必须体现本需求中的业务模块、数据存储、消息/事件通道、外部客户端或第三方服务。\n"
            "3. 节点文字使用中文，控制在 2-8 个字。\n"
            "4. 不要使用括号、引号、emoji 或特殊符号，避免 Mermaid 渲染失败。\n\n"
            f"需求：{requirement}\n"
            f"抽取特征：{features.model_dump_json()}\n"
            f"最终推荐：{winner.model_dump()}\n"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是软件架构图生成 Agent，只输出合法 Mermaid flowchart 源码。"},
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "stream": False,
            "max_tokens": 1000,
        }
        content = await self._chat(payload)
        if not content:
            return None
        return self._sanitize_mermaid(content)

    async def extract_capabilities(
        self,
        requirement: str,
        features: ExtractedFeatures,
    ) -> list[str]:
        if not self.api_key:
            return []

        prompt = (
            "请从软件需求中提取架构拓扑所需的业务能力模块，只返回 JSON，不要 Markdown。\n"
            "格式：{\"capabilities\":[\"能力1\",\"能力2\"]}\n"
            "能力名称使用 2-6 个中文字符，优先返回业务能力，不要返回抽象质量属性。\n"
            "例如社交媒体应包含 Feed、关系、内容、互动、评论、私信、推荐、审核。\n\n"
            f"需求：{requirement}\n"
            f"特征：{features.model_dump_json()}\n"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是架构能力识别 Agent，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.1,
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": 800,
        }
        content = await self._chat(payload)
        if not content:
            return []
        try:
            data = self._extract_json(content)
            capabilities = data.get("capabilities", [])
            return [str(item).strip() for item in capabilities if str(item).strip()]
        except Exception:
            return []

    async def extract_features(
        self,
        requirement: str,
    ) -> ExtractedFeatures | None:
        if not self.api_key:
            self.last_error = "DeepSeek API Key 未配置，无法进行需求解析。"
            return None

        features = await self._request_feature_extraction(requirement, strict=False)
        if features:
            return features

        return await self._request_feature_extraction(requirement, strict=True)

    async def _request_feature_extraction(self, requirement: str, strict: bool = False) -> ExtractedFeatures | None:
        strict_rules = (
            "严格补充要求：\n"
            "1. 必须从用户需求中按业务动作拆出 4 到 8 个业务能力，例如录入、查询、选择、下单、审核、核验、扣费、统计、提醒等。\n"
            "2. topology_expectations.must_have_components 必须覆盖每个业务能力对应的服务组件和数据存储。\n"
            "3. topology_expectations.must_have_relations 必须体现主要业务链路，例如录入->库存、下单->订单、归还->核验、扣费->记录。\n"
            "4. 不能只返回订单服务、库存服务这类过少组件；对租借、领用、预约、审批、归还、扣费、统计等普通业务系统也要完整拆分。\n\n"
            if strict
            else ""
        )
        prompt = (
            "请作为主需求解析器，从软件需求中抽取体系结构推荐所需结构化特征，并只返回 JSON，不要 Markdown。\n"
            "重要：下面 input.requirement 字段就是原始需求，必须逐字读取并围绕它抽取领域、关键词、业务能力和质量属性。\n"
            "如果 input.requirement 非空，不允许把需求判定为空，也不允许返回空的 business_capabilities。\n"
            "JSON 字段必须为：domain, keywords, business_capabilities, architecture_drivers, topology_expectations, quality_attributes, constraints, data_flow, ambiguity_notes。\n"
            "business_capabilities 必须是具体业务能力，不要写“业务处理”“数据处理”等泛化词；应覆盖用户、商家、管理员等角色的关键业务动作。\n"
            "architecture_drivers 表示影响架构选型的驱动因素，例如高并发、弹性伸缩、最终一致性、灰度发布。\n"
            "topology_expectations 必须包含 must_have_components, must_have_relations, quality_infrastructure 三个数组；must_have_components 要包含业务服务和对应数据存储。\n"
            "quality_attributes 必须包含 concurrency, realtime, reliability, scalability, data_intensity, ai_reasoning，值为 0 到 1。\n"
            "data_flow 只能是 event_stream, pipeline, transactional, request_response 之一。\n"
            "constraints 至少包含 scale_mentions, deployment, requires_high_availability, requires_future_extension。\n\n"
            f"{strict_rules}"
            "示例 JSON：\n"
            "{\"domain\":\"电商交易\",\"keywords\":[\"秒杀\",\"支付\",\"库存\"],\"business_capabilities\":[\"商品浏览\",\"购物车\",\"订单管理\",\"支付结算\",\"库存一致性\",\"秒杀活动\",\"物流跟踪\"],\"architecture_drivers\":[\"高并发\",\"弹性伸缩\",\"最终一致性\",\"灰度发布\"],\"topology_expectations\":{\"must_have_components\":[\"订单服务\",\"支付服务\",\"库存服务\",\"秒杀服务\",\"消息队列\"],\"must_have_relations\":[\"订单服务->支付服务\",\"订单服务->库存服务\",\"秒杀服务->消息队列\"],\"quality_infrastructure\":[\"负载均衡\",\"缓存集群\",\"监控服务\"]},\"quality_attributes\":{\"concurrency\":0.95,\"realtime\":0.4,\"reliability\":0.85,\"scalability\":0.9,\"data_intensity\":0.65,\"ai_reasoning\":0},\"constraints\":{\"scale_mentions\":[\"每秒数万笔订单\"],\"deployment\":[\"微服务\",\"独立部署\",\"灰度发布\"],\"requires_high_availability\":true,\"requires_future_extension\":true},\"data_flow\":\"transactional\",\"ambiguity_notes\":[]}\n\n"
            "待解析输入：\n"
            f"{json.dumps({'requirement': requirement}, ensure_ascii=False)}\n"
        )
        try:
            structured = await self._ainvoke_structured(
                system_message="你是软件架构需求分析 Agent，擅长把模糊中文需求转为结构化架构特征。",
                user_prompt=prompt,
                schema=ExtractedFeatures,
                temperature=0.1,
                max_tokens=1800 if strict else 1200,
            )
            features = structured if isinstance(structured, ExtractedFeatures) else ExtractedFeatures.model_validate(structured)
            features = self._strengthen_topology_expectations(features)
            if not self._features_consistent_with_requirement(requirement, features):
                self.last_error = "DeepSeek 需求解析结果与输入不一致：输入需求非空，但模型返回了空需求或未提取到有效业务特征。"
                return None
            return features
        except Exception as exc:
            self.last_error = f"DeepSeek 需求解析 JSON 校验失败：{exc}"
            return None

    async def review_candidates(
        self,
        requirement: str,
        features: ExtractedFeatures,
        candidates: list[CandidateEvaluation],
    ) -> list[str]:
        if not self.api_key:
            return []

        prompt = (
            "请作为架构评审 Agent，复核候选架构排序是否合理。只返回 3 条以内中文短句，指出补充理由或风险，不要改分。\n\n"
            f"需求：{requirement}\n"
            f"特征：{features.model_dump_json()}\n"
            f"候选：{[item.model_dump() for item in candidates]}\n"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是谨慎的软件体系结构评审专家，输出简洁可追溯的复核意见。"},
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "stream": False,
        }
        content = await self._chat(payload)
        if not content:
            return []
        return [line.strip("- 1234567890.、") for line in content.splitlines() if line.strip()][:3]

    async def propose_topology_knowledge_patch(
        self,
        requirement: str,
        features: ExtractedFeatures,
        coverage: dict[str, Any],
        graph_knowledge: dict[str, Any],
        request_timeout: float | None | object = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any] | None:
        if not self.api_key:
            return None

        prompt = (
            "你是软件架构知识图谱补全 Agent。当前系统要生成架构拓扑图，但 Neo4j 知识覆盖率不足。\n"
            "请根据用户需求、已抽取特征、已有图谱知识和多维覆盖率缺口，补全领域能力、架构组件、数据存储和依赖关系。\n"
            "只返回 JSON，不要 Markdown，不要解释。\n\n"
            "JSON 格式必须为：\n"
            "{\n"
            "  \"scenario_id\": \"领域场景英文或拼音标识\",\n"
            "  \"scenario_name\": \"领域场景中文名\",\n"
            "  \"capabilities\": [\n"
            "    {\"name\":\"能力名\", \"components\":[\"组件名\"], \"stores\":[\"存储名\"], "
            "\"edges\":[{\"source\":\"源组件\", \"target\":\"目标组件或存储\", \"label\":\"关系\", \"kind\":\"sync|event\"}]}\n"
            "  ],\n"
            "  \"components\": [\"组件名\"],\n"
            "  \"stores\": [\"存储名\"],\n"
            "  \"edges\": [{\"source\":\"源组件\", \"target\":\"目标组件\", \"label\":\"关系\", \"kind\":\"sync|event\"}],\n"
            "  \"reason\": \"一句话说明补全依据\"\n"
            "}\n\n"
            "约束：\n"
            "1. 组件名称使用中文，2 到 8 个字。\n"
            "2. 只补充和本需求直接相关的组件，避免泛化过度。\n"
            "3. scenario_id 必须代表当前需求场景，不要复用无关场景；无法归入已有领域时生成新的稳定标识。\n"
            "4. 每个 capability 必须按业务能力独立成组，组内给出该能力自己的组件、存储和关键边。\n"
            "5. 高并发/秒杀场景必须考虑缓存、消息队列或事件总线。\n"
            "6. 最终一致性场景必须考虑事件关系或异步消息关系。\n"
            "7. 支付/订单/库存/物流等业务能力需要体现服务和数据存储。\n\n"
            "强约束：\n"
            "1. coverage.missing_capabilities 中的每一个业务能力都必须在 capabilities 数组中逐项出现，"
            "不要合并成“消息处理”“业务处理”“数据处理”等泛化能力名。\n\n"
            "2. coverage.missing_components 中的组件应尽量出现在 components、capabilities.components 或 stores 中。\n"
            "3. coverage.missing_relations 中使用“源->目标”格式的关系，应尽量在 edges 中用完全相同的 source 和 target 补齐。\n"
            "4. coverage.missing_quality_infrastructure 中的高并发、高可用、可扩展基础设施应显式补齐。\n\n"
            f"用户需求：{requirement}\n"
            f"需求特征：{features.model_dump_json()}\n"
            f"覆盖率评估：{json.dumps(coverage, ensure_ascii=False)}\n"
            f"已有图谱知识：{json.dumps(graph_knowledge, ensure_ascii=False)}\n"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是谨慎的软件架构知识图谱补全 Agent，只输出可解析 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.1,
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": 1400,
        }
        content = await self._chat(payload, request_timeout=request_timeout)
        if not content:
            return None
        try:
            data = self._extract_json(content)
            return self._sanitize_topology_patch(data)
        except Exception:
            return None

    async def review_topology_coverage_gap(
        self,
        requirement: str,
        features: ExtractedFeatures,
        graph_knowledge: dict[str, Any],
        request_timeout: float | None | object = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any] | None:
        if not self.api_key:
            return None

        prompt = (
            "你是架构拓扑完整性复核 Agent。请对照原始需求和当前拓扑知识，找出当前架构图漏掉的业务能力、组件、数据存储和关系。\n"
            "只返回 JSON，不要 Markdown，不要解释。\n\n"
            "判断原则：\n"
            "1. 必须覆盖原始需求中的每个关键业务动作，例如录入、选择日期、下单、归还、核验、扣费、统计、提醒、审核等。\n"
            "2. 如果当前 graph_knowledge 已覆盖某能力，不要重复补。\n"
            "3. 只补与当前需求直接相关的内容，不要泛化成无关平台能力。\n\n"
            "JSON 格式与知识补丁一致：\n"
            "{\n"
            "  \"scenario_id\":\"场景标识\",\n"
            "  \"scenario_name\":\"场景名称\",\n"
            "  \"capabilities\":[{\"name\":\"能力名\",\"components\":[\"组件\"],\"stores\":[\"存储\"],\"edges\":[{\"source\":\"源\",\"target\":\"目标\",\"label\":\"关系\",\"kind\":\"sync|event\"}]}],\n"
            "  \"components\":[\"组件\"],\n"
            "  \"stores\":[\"存储\"],\n"
            "  \"edges\":[{\"source\":\"源\",\"target\":\"目标\",\"label\":\"关系\",\"kind\":\"sync|event\"}],\n"
            "  \"reason\":\"一句话说明\"\n"
            "}\n"
            "如果没有缺口，返回空数组字段。\n\n"
            f"原始需求：{requirement}\n"
            f"结构化特征：{features.model_dump_json()}\n"
            f"当前拓扑知识：{json.dumps(graph_knowledge, ensure_ascii=False)}\n"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是谨慎的软件架构拓扑复核 Agent，只输出可解析 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.1,
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": 1400,
        }
        content = await self._chat(payload, request_timeout=request_timeout)
        if not content:
            return None
        try:
            return self._sanitize_topology_patch(self._extract_json(content))
        except Exception:
            return None

    async def embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings through an OpenAI-compatible embedding endpoint.

        DeepSeek chat endpoints do not guarantee embedding support, so the
        embedding service is configured independently. If it is not configured,
        callers should avoid permanent semantic merge decisions instead of
        falling back to string rules.
        """
        if not texts:
            return []
        if not (self.embedding_api_key and self.embedding_url and self.embedding_model):
            self.last_error = "Embedding 服务未配置，无法进行语义近义节点召回。"
            return None

        headers = {"Authorization": f"Bearer {self.embedding_api_key}", "Content-Type": "application/json"}
        # Qwen text-embedding-v4 supports at most 10 input strings per request.
        batches = [texts[index : index + 10] for index in range(0, len(texts), 10)]
        vectors: list[list[float]] = []
        self.last_error = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for batch in batches:
                payload = {"model": self.embedding_model, "input": batch}
                cache_payload = self._cache_payload(self.embedding_url, payload)
                if self.cache:
                    cached_vectors = self.cache.get("embedding", cache_payload)
                    if cached_vectors is not None:
                        if not self._valid_embedding_vectors(cached_vectors, len(batch)):
                            raise ValueError("Embedding cache entry has invalid vector shape.")
                        vectors.extend(cached_vectors)
                        continue
                try:
                    response = await client.post(self.embedding_url, headers=headers, json=payload)
                    if response.status_code >= 400:
                        self.last_error = f"Embedding 服务调用失败：HTTP {response.status_code}，{response.text[:500]}"
                        return None
                    data = response.json()
                    rows = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
                    batch_vectors = [item.get("embedding") for item in rows]
                    if len(batch_vectors) != len(batch) or any(not isinstance(vector, list) for vector in batch_vectors):
                        self.last_error = "Embedding 服务返回结构不符合 OpenAI-compatible 格式。"
                        return None
                except Exception as exc:
                    self.last_error = str(exc)
                    return None
                if self.cache:
                    self.cache.set("embedding", cache_payload, batch_vectors)
                vectors.extend(batch_vectors)
        return vectors

    async def adjudicate_semantic_merge(
        self,
        requirement: str,
        candidate_node: dict[str, Any],
        top_matches: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.api_key:
            return None

        prompt = (
            "你是软件架构知识图谱节点消歧 Agent。请判断 LLM 新生成的节点是否应与 Neo4j 中已有节点合并。\n"
            "只返回 JSON，不要 Markdown，不要解释。\n\n"
            "判断原则：\n"
            "1. 只比较同一种节点类型，不能跨类型合并。\n"
            "2. 名称不同但业务语义、职责、上下游关系一致时可以 merge。\n"
            "3. 语义接近但无法确认是否同一职责时返回 temporary，表示仅本次临时使用，不写入 Neo4j。\n"
            "4. 明确是不同能力、组件或存储时返回 create。\n\n"
            "JSON 格式：\n"
            "{\"decision\":\"merge|create|temporary\",\"canonical\":\"已有节点名或新节点名\",\"confidence\":0.0,\"reason\":\"一句话理由\"}\n\n"
            f"用户需求：{requirement}\n"
            f"新节点：{json.dumps(candidate_node, ensure_ascii=False)}\n"
            f"Neo4j Top-K 候选：{json.dumps(top_matches, ensure_ascii=False)}\n"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是谨慎的软件架构知识图谱节点消歧 Agent，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.05,
            "stream": False,
            "response_format": {"type": "json_object"},
            "max_tokens": 500,
        }
        content = await self._chat(payload)
        if not content:
            return None
        try:
            data = self._extract_json(content)
            decision = str(data.get("decision", "")).strip().lower()
            if decision not in {"merge", "create", "temporary"}:
                return None
            confidence = float(data.get("confidence", 0))
            return {
                "decision": decision,
                "canonical": str(data.get("canonical", "")).strip(),
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(data.get("reason", "")).strip()[:160],
            }
        except Exception:
            return None

    @staticmethod
    def _build_prompt(requirement: str, features: ExtractedFeatures, candidates: list[CandidateEvaluation]) -> str:
        return (
            "请基于以下需求分析和候选架构，只生成“适配理由补充”部分的 Markdown 内容。\n"
            "严格要求：\n"
            "1. 不要输出标题，不要输出完整报告。\n"
            "2. 不要输出对比表格，表格由系统生成。\n"
            "3. 输出 3 到 5 条要点，每条使用 '- ' 开头。\n"
            "4. 每条要点必须包含加粗关键词，例如 **高并发**。\n"
            "5. 聚焦需求特征、候选差异、最终推荐可信度、主要风险和落地关注点。\n\n"
            f"原始需求：{requirement}\n"
            f"抽取特征：{features.model_dump_json()}\n"
            f"候选架构：{[item.model_dump() for item in candidates]}\n"
        )

    @staticmethod
    def _sanitize_topology_patch(data: dict[str, Any]) -> dict[str, Any]:
        def clean_list(items) -> list[str]:
            if not isinstance(items, list):
                return []
            return [str(item).strip() for item in items if str(item).strip()][:16]

        flattened_edges = []
        capabilities = []
        for item in data.get("capabilities", []):
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    capability_edges = []
                    for edge in item.get("edges", []):
                        if not isinstance(edge, dict):
                            continue
                        source = str(edge.get("source", "")).strip()
                        target = str(edge.get("target", "")).strip()
                        if not source or not target:
                            continue
                        kind = str(edge.get("kind", "sync")).strip()
                        clean_edge = {
                            "source": source,
                            "target": target,
                            "label": str(edge.get("label", "依赖")).strip() or "依赖",
                            "kind": "event" if kind == "event" else "sync",
                            "capability": name,
                        }
                        capability_edges.append(clean_edge)
                        flattened_edges.append(clean_edge)
                    capabilities.append(
                        {
                            "name": name,
                            "components": clean_list(item.get("components", [])),
                            "stores": clean_list(item.get("stores", [])),
                            "edges": capability_edges[:12],
                        }
                    )
            elif str(item).strip():
                capabilities.append({"name": str(item).strip(), "components": [], "stores": []})

        edges = []
        for item in list(data.get("edges", [])) + flattened_edges:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            if not source or not target:
                continue
            kind = str(item.get("kind", "sync")).strip()
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "label": str(item.get("label", "依赖")).strip() or "依赖",
                    "kind": "event" if kind == "event" else "sync",
                    "capability": str(item.get("capability", "")).strip(),
                }
            )

        return {
            "scenario_id": str(data.get("scenario_id", "")).strip(),
            "scenario_name": str(data.get("scenario_name", "")).strip(),
            "capabilities": capabilities[:12],
            "components": clean_list(data.get("components", [])),
            "stores": clean_list(data.get("stores", [])),
            "edges": edges[:24],
            "reason": str(data.get("reason", "")).strip()[:160],
        }

    @staticmethod
    def _sanitize_architecture_recommendation(
        data: dict[str, Any],
        styles: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[CandidateEvaluation], dict[str, Any]]:
        style_map = {str(style.get("id", "")).strip(): style for style in styles}
        candidates: list[CandidateEvaluation] = []
        limit = max(3, min(top_k, 12))
        seen: set[str] = set()
        for item in data.get("candidates", []):
            if not isinstance(item, dict):
                continue
            style_id = str(item.get("style_id", "")).strip()
            if style_id not in style_map or style_id in seen:
                continue
            seen.add(style_id)
            style = style_map[style_id]
            quality_scores = item.get("quality_scores")
            if not isinstance(quality_scores, dict):
                quality_scores = style.get("quality_scores", {})
            score = LLMClient._clamp_score(item.get("score", 0))
            raw_score = LLMClient._clamp_score(item.get("raw_score", score))
            candidates.append(
                CandidateEvaluation(
                    style_id=style_id,
                    name=str(item.get("name") or style.get("name") or style_id),
                    score=score,
                    raw_score=raw_score,
                    recommendation_role=LLMClient._clean_choice(
                        item.get("recommendation_role"),
                        {"核心推荐", "备选方案", "专项补充", "不推荐", "核心推荐/组合候选", "组合备选"},
                        "备选方案" if candidates else "核心推荐",
                    ),
                    confidence=LLMClient._clean_choice(
                        item.get("confidence"),
                        {"高", "中高", "中", "中低", "低"},
                        "中",
                    ),
                    matched_reasons=LLMClient._clean_text_list(item.get("matched_reasons", []), 5),
                    risks=LLMClient._clean_text_list(item.get("risks", []), 4),
                    deductions=LLMClient._clean_text_list(item.get("deductions", []), 4),
                    quality_scores={
                        key: max(0.0, min(1.0, float(quality_scores.get(key, 0))))
                        for key in ["scalability", "performance", "reliability", "modifiability", "complexity", "realtime"]
                    },
                )
            )
            if len(candidates) >= limit:
                break
        if len(candidates) < 3:
            raise ValueError("候选架构少于 3 个或 style_id 不在知识库中")
        candidates.sort(key=lambda item: item.score, reverse=True)
        # The first candidate is normalized as the final recommendation consumed by the UI.
        candidates[0].recommendation_role = "核心推荐"
        composition = data.get("composition_recommendation", {})
        if not isinstance(composition, dict):
            composition = {}
        composition = {
            "composition_needed": bool(composition.get("composition_needed", False)),
            "primary_style": str(composition.get("primary_style") or candidates[0].name),
            "supporting_styles": [
                item for item in composition.get("supporting_styles", [])
                if isinstance(item, dict)
            ][:3],
            "reason": str(composition.get("reason", "")).strip(),
            "triggers": LLMClient._clean_text_list(composition.get("triggers", []), 6),
            "overengineering_warnings": LLMClient._clean_text_list(composition.get("overengineering_warnings", []), 6),
        }
        review_notes = LLMClient._clean_text_list(data.get("review_notes", []), 5)
        return candidates, {**composition, "review_notes": review_notes}

    @staticmethod
    def _clamp_score(value: Any) -> float:
        return round(max(0.0, min(100.0, float(value))), 1)

    @staticmethod
    def _clean_choice(value: Any, allowed: set[str], default: str) -> str:
        clean = str(value or "").strip()
        return clean if clean in allowed else default

    @staticmethod
    def _clean_text_list(items: Any, limit: int) -> list[str]:
        if not isinstance(items, list):
            return []
        return [str(item).strip() for item in items if str(item).strip()][:limit]

    @staticmethod
    def _features_consistent_with_requirement(requirement: str, features: ExtractedFeatures) -> bool:
        if not requirement.strip():
            return True
        empty_notes = any("原始需求为空" in str(note) for note in features.ambiguity_notes)
        empty_result = (
            features.domain.strip() in {"", "未知"}
            and not features.keywords
            and not features.business_capabilities
            and not features.architecture_drivers
        )
        return not (empty_notes or empty_result)

    @staticmethod
    def _strengthen_topology_expectations(features: ExtractedFeatures) -> ExtractedFeatures:
        expectations = features.topology_expectations or {}
        components = [
            str(item).strip()
            for item in expectations.get("must_have_components", [])
            if str(item).strip()
        ]
        relations = [
            str(item).strip()
            for item in expectations.get("must_have_relations", [])
            if str(item).strip()
        ]
        quality_infrastructure = [
            str(item).strip()
            for item in expectations.get("quality_infrastructure", [])
            if str(item).strip()
        ]
        capabilities = [
            str(item).strip()
            for item in features.business_capabilities
            if str(item).strip()
        ]

        for capability in capabilities:
            compact = re.sub(r"(管理|处理|记录|创建|选择|核验|统计|提醒|审核|录入|扣费|归还|下单)$", "", capability).strip()
            base = compact or capability
            service_name = capability if capability.endswith("服务") else f"{base}服务"
            store_name = capability if capability.endswith(("库", "数据库", "缓存", "索引")) else f"{base}库"
            if not any(base in item or capability in item for item in components):
                components.append(service_name)
                components.append(store_name)

        service_components = [
            item for item in components
            if item.endswith("服务") and item not in {"API网关", "监控服务", "审计服务"}
        ]
        for left, right in zip(service_components, service_components[1:]):
            relation = f"{left}->{right}"
            if relation not in relations:
                relations.append(relation)

        normalized = {
            "must_have_components": list(dict.fromkeys(components))[:24],
            "must_have_relations": list(dict.fromkeys(relations))[:24],
            "quality_infrastructure": list(dict.fromkeys(quality_infrastructure))[:12],
        }
        return features.model_copy(update={"topology_expectations": normalized})

    async def _ainvoke_structured(
        self,
        *,
        system_message: str,
        user_prompt: str,
        schema: type[BaseModel],
        temperature: float,
        max_tokens: int,
        request_timeout: float | None | object = _DEFAULT_TIMEOUT,
    ) -> BaseModel:
        # LangChain owns prompt composition; the adapter still calls the same OpenAI-style endpoint.
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_message}"),
                ("user", "{user_prompt}"),
            ]
        )
        structured_model = OpenAICompatibleStructuredChat(
            self,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        ).with_structured_output(schema, method="json_mode")
        chain = prompt | structured_model
        return await chain.ainvoke({"system_message": system_message, "user_prompt": user_prompt})

    async def _chat(
        self,
        payload: dict[str, Any],
        request_timeout: float | None | object = _DEFAULT_TIMEOUT,
    ) -> str | None:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = self.timeout if request_timeout is _DEFAULT_TIMEOUT else request_timeout
        cache_payload = self._cache_payload(self.chat_url, payload)
        if self.cache:
            cached_content = self.cache.get("chat", cache_payload)
            if cached_content is not None:
                if not isinstance(cached_content, str):
                    raise ValueError("Chat cache entry must be a string.")
                self.last_error = None
                return cached_content
        try:
            self.last_error = None
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.chat_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            self.last_error = self._exception_message(exc)
            return None
        if self.cache:
            self.cache.set("chat", cache_payload, content)
        return content

    async def _stream_chat(self, payload: dict[str, Any]) -> AsyncGenerator[str, None]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            self.last_error = None
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.chat_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            break
                        try:
                            payload = json.loads(data)
                            delta = payload["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except Exception:
                            continue
        except Exception as exc:
            self.last_error = self._exception_message(exc)

    async def ping(self) -> dict[str, Any]:
        if not self.api_key:
            return {"configured": False, "ok": False, "model": self.model, "base_url": self.base_url}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a health check endpoint. Reply with ok."},
                {"role": "user", "content": "ping"},
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 8,
            "stream": False,
        }
        content = await self._chat(payload)
        return {
            "configured": True,
            "ok": bool(content),
            "model": self.model,
            "base_url": self.base_url,
            "chat_url": self.chat_url,
            "error": None if content else self.last_error,
        }

    @staticmethod
    def _cache_payload(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"url": url, "payload": payload}

    @staticmethod
    def _exception_message(exc: Exception) -> str:
        detail = str(exc).strip()
        if detail:
            return detail
        return exc.__class__.__name__

    @staticmethod
    def _valid_embedding_vectors(value: Any, expected_count: int) -> bool:
        if not isinstance(value, list) or len(value) != expected_count:
            return False
        for vector in value:
            if not isinstance(vector, list):
                return False
            if any(isinstance(item, bool) or not isinstance(item, int | float) for item in vector):
                return False
        return True

    @staticmethod
    def _build_chat_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            return base_url
        if path.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/chat/completions"

    @staticmethod
    def _build_embedding_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/embeddings"):
            return base_url
        if path.endswith("/v1"):
            return f"{base_url}/embeddings"
        return f"{base_url}/embeddings"

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
        if fenced:
            content = fenced.group(1)
        else:
            start = content.find("{")
            end = content.rfind("}")
            content = content[start : end + 1] if start >= 0 and end >= start else content
        return json.loads(content)

    @staticmethod
    def _sanitize_mermaid(content: str) -> str:
        fenced = re.search(r"```(?:mermaid)?\s*(.*?)\s*```", content, flags=re.S)
        if fenced:
            content = fenced.group(1)
        lines = [line.rstrip() for line in content.strip().splitlines() if line.strip()]
        if not lines:
            return ""
        if not lines[0].startswith("flowchart"):
            lines.insert(0, "flowchart LR")
        return "\n".join(lines)
