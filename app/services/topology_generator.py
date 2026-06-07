from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.models.schemas import CandidateEvaluation, ExtractedFeatures


@dataclass(frozen=True)
class TopologyNode:
    id: str
    name: str
    layer: str


@dataclass(frozen=True)
class TopologyEdge:
    source: str
    target: str
    label: str = ""
    kind: str = "sync"


class TopologyGenerator:
    """Topology renderer driven by LLM features, Neo4j knowledge, and schema checks."""

    CANONICAL_COMPONENT_IDS = {
        "客户端": "client",
        "CDN": "cdn",
        "负载均衡": "lb",
        "API网关": "gateway",
        "直播网关": "live_gateway",
        "设备网关": "device_gateway",
        "视频网关": "video_gateway",
        "用户服务": "user",
        "用户库": "user_db",
        "课程服务": "course",
        "课程库": "course_db",
        "直播服务": "live",
        "回放服务": "replay",
        "回放库": "replay_db",
        "作业服务": "homework",
        "作业库": "homework_db",
        "考试服务": "exam",
        "考试库": "exam_db",
        "关系服务": "relation",
        "关系图谱": "graph_db",
        "内容服务": "content",
        "内容库": "content_db",
        "搜索索引": "search",
        "互动服务": "interaction",
        "互动库": "interaction_db",
        "评论服务": "comment",
        "私信服务": "message",
        "消息库": "message_db",
        "Feed服务": "feed",
        "Feed缓存": "feed_cache",
        "推荐服务": "recommend",
        "特征库": "feature_store",
        "审核服务": "moderation",
        "通知服务": "notify",
        "消息服务": "message_service",
        "状态服务": "status_service",
        "状态缓存": "status_cache",
        "信令服务": "signaling_service",
        "商品服务": "product",
        "商品库": "product_db",
        "购物车服务": "cart",
        "购物车缓存": "cart_cache",
        "订单服务": "order",
        "订单库": "order_db",
        "支付服务": "payment",
        "支付库": "payment_db",
        "库存服务": "inventory",
        "库存库": "inventory_db",
        "促销服务": "promotion",
        "秒杀服务": "flash_sale",
        "退款服务": "refund",
        "物流服务": "logistics",
        "物流库": "logistics_db",
        "搜索服务": "search_service",
        "风控服务": "risk_control",
        "配置中心": "config_center",
        "缓存集群": "cache",
        "消息队列": "event_bus",
        "事件总线": "event_bus",
        "媒体服务": "media",
        "转码服务": "transcode",
        "对象存储": "object_store",
        "AI服务": "ai",
        "服务注册": "service_registry",
        "监控服务": "monitoring",
        "审计服务": "audit",
        "实验人员服务": "lab_user",
        "试剂申领服务": "reagent_request",
        "申领审核服务": "approval",
        "试剂库存服务": "reagent_inventory",
        "出库服务": "stock_out",
        "采购提醒服务": "purchase_alert",
        "试剂库": "reagent_db",
        "申领记录库": "request_db",
        "出库记录库": "stock_out_db",
    }

    SINGLETON_COMPONENTS = {
        "客户端", "CDN", "负载均衡", "API网关", "直播网关", "设备网关", "视频网关",
        "事件总线", "消息队列", "任务队列", "数据管道", "缓存集群", "对象存储",
        "搜索索引", "特征库", "模型库", "服务注册", "监控服务", "审计服务", "配置中心",
    }

    SOCIAL_KEYWORDS = ["社交", "发帖", "点赞", "评论", "私信", "内容推荐", "关注", "粉丝"]
    EDUCATION_KEYWORDS = ["在线教育", "上课", "课程", "直播", "录播", "回放", "课后互动", "作业", "考试", "课堂"]
    MEDIA_KEYWORDS = ["视频", "图片", "短视频", "转码", "直播", "录播"]
    COMMERCE_KEYWORDS = ["电商", "商品", "购物车", "订单", "支付", "退款", "促销", "下单", "秒杀", "物流", "灰度", "交易"]
    LAB_REAGENT_KEYWORDS = ["实验室", "试剂", "申领", "领用", "出库", "采购"]
    IOT_KEYWORDS = ["设备", "传感器", "采集", "告警", "远程控制"]
    AI_KEYWORDS = ["AI", "智能", "推荐算法", "生成", "大模型"]

    def generate(
        self,
        requirement: str,
        features: ExtractedFeatures,
        winner: CandidateEvaluation,
        extra_capabilities: list[str] | None = None,
        graph_knowledge: dict | None = None,
        composition_recommendation: dict | None = None,
    ) -> tuple[str, list[str]]:
        graph_knowledge = graph_knowledge or {}
        composition_recommendation = composition_recommendation or {}
        # Coverage decides whether Neo4j leads generation or only enriches LLM expectations.
        graph_primary = self._graph_coverage_sufficient(requirement, features, graph_knowledge, extra_capabilities or [])
        capabilities = self._topology_capabilities(requirement, features, graph_knowledge, extra_capabilities or [], graph_primary)
        capabilities.extend(extra_capabilities or [])
        capabilities = list(dict.fromkeys(capabilities))
        nodes, edges, notes = self._build_graph(capabilities, features, winner, graph_knowledge, composition_recommendation, graph_primary)
        return self._render_mermaid(nodes, edges), notes

    def generate_views(
        self,
        requirement: str,
        features: ExtractedFeatures,
        winner: CandidateEvaluation,
        extra_capabilities: list[str] | None = None,
        graph_knowledge: dict | None = None,
        composition_recommendation: dict | None = None,
    ) -> tuple[dict[str, str], list[str]]:
        graph_knowledge = graph_knowledge or {}
        composition_recommendation = composition_recommendation or {}
        graph_primary = self._graph_coverage_sufficient(requirement, features, graph_knowledge, extra_capabilities or [])
        capabilities = self._topology_capabilities(requirement, features, graph_knowledge, extra_capabilities or [], graph_primary)
        capabilities.extend(extra_capabilities or [])
        capabilities = list(dict.fromkeys(capabilities))
        nodes, edges, notes = self._build_graph(capabilities, features, winner, graph_knowledge, composition_recommendation, graph_primary)
        views = self._build_view_diagrams(nodes, edges)
        notes.append("拓扑可视化：已生成总览、完整、业务链路、数据流和支撑设施多视图")
        return views, notes

    def generate_graph_views(
        self,
        requirement: str,
        features: ExtractedFeatures,
        winner: CandidateEvaluation,
        extra_capabilities: list[str] | None = None,
        graph_knowledge: dict | None = None,
        composition_recommendation: dict | None = None,
    ) -> tuple[dict[str, str], dict[str, dict], list[str]]:
        graph_knowledge = graph_knowledge or {}
        composition_recommendation = composition_recommendation or {}
        graph_primary = self._graph_coverage_sufficient(requirement, features, graph_knowledge, extra_capabilities or [])
        capabilities = self._topology_capabilities(requirement, features, graph_knowledge, extra_capabilities or [], graph_primary)
        capabilities.extend(extra_capabilities or [])
        capabilities = list(dict.fromkeys(capabilities))
        nodes, edges, notes = self._build_graph(capabilities, features, winner, graph_knowledge, composition_recommendation, graph_primary)
        diagrams = self._build_view_diagrams(nodes, edges)
        graphs = self._build_structured_view_graphs(nodes, edges)
        notes.append("拓扑可视化：已生成结构化 JSON 拓扑，前端可直接渲染可拖拽节点和连线")
        return diagrams, graphs, notes

    def assess_coverage(
        self,
        requirement: str,
        features: ExtractedFeatures,
        graph_knowledge: dict,
        extra_capabilities: list[str] | None = None,
    ) -> dict:
        expected = self.extract_business_capabilities(requirement, features, extra_capabilities or [])
        if not expected:
            return {
                "score": 1.0,
                "dimensions": {},
                "expected_capabilities": [],
                "covered_capabilities": [],
                "missing_capabilities": [],
                "expected_components": [],
                "covered_components": [],
                "missing_components": [],
                "expected_relations": [],
                "covered_relations": [],
                "missing_relations": [],
                "expected_quality_infrastructure": [],
                "covered_quality_infrastructure": [],
                "missing_quality_infrastructure": [],
            }

        available_names = set(graph_knowledge.get("components", [])) | set(graph_knowledge.get("stores", []))
        available_names |= set(graph_knowledge.get("capabilities", []))
        covered = [name for name in expected if name in available_names]
        missing = [name for name in expected if name not in available_names]
        expected_components = self._expected_components(expected, features)
        covered_components = [name for name in expected_components if name in available_names]
        missing_components = [name for name in expected_components if name not in available_names]
        expected_relations = self._expected_relations(expected_components, features)
        available_relations = {
            f"{edge.get('source')}->{edge.get('target')}"
            for edge in graph_knowledge.get("edges", [])
            if isinstance(edge, dict) and edge.get("source") and edge.get("target")
        }
        covered_relations = [relation for relation in expected_relations if relation in available_relations]
        missing_relations = [relation for relation in expected_relations if relation not in available_relations]
        expected_quality_infra = self._expected_quality_infrastructure(features)
        covered_quality_infra = [name for name in expected_quality_infra if name in available_names]
        missing_quality_infra = [name for name in expected_quality_infra if name not in available_names]

        # Business capabilities receive the highest weight because they anchor domain correctness.
        dimensions = {
            "business_capability": self._dimension_score(expected, covered, missing),
            "component": self._dimension_score(expected_components, covered_components, missing_components),
            "relation": self._dimension_score(expected_relations, covered_relations, missing_relations),
            "quality_infrastructure": self._dimension_score(expected_quality_infra, covered_quality_infra, missing_quality_infra),
            "architecture_responsibility": {
                "score": 1.0,
                "expected": [],
                "covered": [],
                "missing": [],
            },
        }
        score = round(
            dimensions["business_capability"]["score"] * 0.35
            + dimensions["component"]["score"] * 0.25
            + dimensions["relation"]["score"] * 0.20
            + dimensions["quality_infrastructure"]["score"] * 0.15
            + dimensions["architecture_responsibility"]["score"] * 0.05,
            2,
        )
        return {
            "score": score,
            "dimensions": dimensions,
            "expected_capabilities": expected,
            "covered_capabilities": covered,
            "missing_capabilities": missing,
            "expected_components": expected_components,
            "covered_components": covered_components,
            "missing_components": missing_components,
            "expected_relations": expected_relations,
            "covered_relations": covered_relations,
            "missing_relations": missing_relations,
            "expected_quality_infrastructure": expected_quality_infra,
            "covered_quality_infrastructure": covered_quality_infra,
            "missing_quality_infrastructure": missing_quality_infra,
        }

    def merge_knowledge_patch(self, graph_knowledge: dict, patch: dict) -> dict:
        merged = {
            "components": list(graph_knowledge.get("components", [])),
            "stores": list(graph_knowledge.get("stores", [])),
            "edges": list(graph_knowledge.get("edges", [])),
            "scenarios": list(graph_knowledge.get("scenarios", [])),
            "capabilities": list(graph_knowledge.get("capabilities", [])),
        }
        for capability in patch.get("capabilities", []):
            name = str(capability.get("name", "")).strip() if isinstance(capability, dict) else str(capability).strip()
            if name:
                merged["capabilities"].append(name)
            if isinstance(capability, dict):
                merged["components"].extend(str(item).strip() for item in capability.get("components", []) if str(item).strip())
                merged["stores"].extend(str(item).strip() for item in capability.get("stores", []) if str(item).strip())
        merged["components"].extend(str(item).strip() for item in patch.get("components", []) if str(item).strip())
        merged["stores"].extend(str(item).strip() for item in patch.get("stores", []) if str(item).strip())
        for edge in patch.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if source and target:
                merged["edges"].append(
                    {
                        "source": source,
                        "target": target,
                        "label": str(edge.get("label", "依赖")).strip() or "依赖",
                        "kind": str(edge.get("kind", "sync")).strip() or "sync",
                    }
                )
        for key in ["components", "stores", "scenarios", "capabilities"]:
            merged[key] = list(dict.fromkeys(merged[key]))
        seen_edges = set()
        deduped_edges = []
        for edge in merged["edges"]:
            key = (edge.get("source"), edge.get("target"), edge.get("label"), edge.get("kind"))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            deduped_edges.append(edge)
        merged["edges"] = deduped_edges
        return merged

    def extract_capabilities(self, requirement: str, features: ExtractedFeatures) -> list[str]:
        return self.extract_business_capabilities(requirement, features, [])

    def extract_business_capabilities(
        self,
        requirement: str,
        features: ExtractedFeatures,
        extra_capabilities: list[str],
    ) -> list[str]:
        llm_caps = [item for item in features.business_capabilities if item and not self._is_generic_capability(item)]
        extra_caps = [item for item in extra_capabilities if item and not self._is_generic_capability(item)]
        merged = list(dict.fromkeys(llm_caps + extra_caps))
        return merged

    @staticmethod
    def _is_generic_capability(name: str) -> bool:
        generic = {"业务处理", "数据处理", "消息处理", "系统管理", "服务处理", "通用能力"}
        return name.strip() in generic

    @staticmethod
    def _keyword_business_capabilities(text: str) -> list[str]:
        rules = {
            "商品浏览": ["商品浏览", "商品", "浏览", "搜索"],
            "购物车": ["购物车"],
            "订单管理": ["订单", "下单"],
            "支付结算": ["支付", "结算"],
            "库存管理": ["库存"],
            "库存一致性": ["最终一致性", "一致性"],
            "促销活动": ["促销"],
            "秒杀活动": ["秒杀"],
            "物流跟踪": ["物流", "轨迹"],
            "退款售后": ["退款", "售后"],
            "风控审计": ["风控", "审计", "安全"],
            "灰度发布": ["灰度", "独立部署", "发布"],
            "通知提醒": ["通知", "提醒"],
            "商品管理": ["商品管理"],
            "直播教学": ["直播"],
            "录播回放": ["录播", "回放"],
            "课堂互动": ["互动"],
            "作业考试": ["作业", "考试"],
            "内容发布": ["发帖", "内容发布"],
            "互动行为": ["点赞", "评论", "互动"],
            "私信通信": ["私信"],
            "关注关系": ["关注", "关系"],
            "信息流": ["信息流", "Feed", "推荐"],
            "内容推荐": ["推荐算法", "内容推荐", "推荐"],
            "内容审核": ["审核"],
        }
        result: list[str] = []
        for capability, keywords in rules.items():
            if any(keyword in text for keyword in keywords):
                result.append(capability)
        return result

    @classmethod
    def _components_for_capabilities(cls, capabilities: list[str]) -> list[str]:
        spec = cls._domain_topology_spec()
        capability_map = spec.get("capabilities", {})
        expected_components: list[str] = []
        for capability in capabilities:
            item = capability_map.get(capability)
            if not item:
                continue
            expected_components.extend(item.get("components", []))
            expected_components.extend(item.get("stores", []))
        return list(dict.fromkeys(expected_components))

    @classmethod
    def _expected_components(cls, capabilities: list[str], features: ExtractedFeatures) -> list[str]:
        expected_components = cls._components_for_capabilities(capabilities)
        expectations = features.topology_expectations or {}
        for name in expectations.get("must_have_components", []):
            if str(name).strip():
                expected_components.append(str(name).strip())
        return list(dict.fromkeys(expected_components))

    @classmethod
    def _expected_relations(cls, expected_components: list[str], features: ExtractedFeatures) -> list[str]:
        expected_set = set(expected_components)
        relations: list[str] = []
        expectations = features.topology_expectations or {}
        for relation in expectations.get("must_have_relations", []):
            if "->" in str(relation):
                relations.append(str(relation).strip())

        spec = cls._domain_topology_spec()
        for source, target, _label in spec.get("dependencies", []):
            if source.startswith("*") or target.startswith("*"):
                continue
            if source in expected_set and target in expected_set:
                relations.append(f"{source}->{target}")
        return list(dict.fromkeys(relations))

    @classmethod
    def _expected_quality_infrastructure(cls, features: ExtractedFeatures) -> list[str]:
        spec = cls._domain_topology_spec()
        expected: list[str] = []
        for quality, score in features.quality_attributes.items():
            if score >= 0.65:
                expected.extend(spec.get("quality_components", {}).get(quality, []))
        expectations = features.topology_expectations or {}
        expected.extend(str(item).strip() for item in expectations.get("quality_infrastructure", []) if str(item).strip())
        return list(dict.fromkeys(expected))

    @staticmethod
    def _dimension_score(expected: list[str], covered: list[str], missing: list[str]) -> dict:
        if not expected:
            score = 1.0
        else:
            score = round(len(covered) / len(expected), 2)
        return {
            "score": score,
            "expected": expected,
            "covered": covered,
            "missing": missing,
        }

    @classmethod
    def _scenario_capabilities(cls, text: str, domain: str) -> list[str]:
        spec = cls._domain_topology_spec()
        result: list[str] = []
        if domain:
            for scenario in spec.get("scenarios", []):
                scenario_text = " ".join([scenario.get("id", ""), scenario.get("name", ""), " ".join(scenario.get("keywords", []))])
                if domain == scenario.get("name") or domain in scenario_text:
                    return list(dict.fromkeys(scenario.get("capabilities", [])))

        scored_scenarios: list[tuple[int, list[str]]] = []
        for scenario in spec.get("scenarios", []):
            hits = sum(1 for keyword in scenario.get("keywords", []) if keyword in text)
            if hits >= 2:
                scored_scenarios.append((hits, scenario.get("capabilities", [])))
        if scored_scenarios:
            scored_scenarios.sort(key=lambda item: item[0], reverse=True)
            result.extend(scored_scenarios[0][1])
        return list(dict.fromkeys(result))

    @staticmethod
    @lru_cache(maxsize=1)
    def _domain_topology_spec() -> dict:
        path = Path(__file__).resolve().parents[2] / "data" / "domain_topology.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_graph(
        self,
        capabilities: list[str],
        features: ExtractedFeatures,
        winner: CandidateEvaluation,
        graph_knowledge: dict,
        composition_recommendation: dict,
        graph_primary: bool,
    ) -> tuple[list[TopologyNode], list[TopologyEdge], list[str]]:
        nodes: dict[str, TopologyNode] = {}
        edges: list[TopologyEdge] = []
        notes: list[str] = []

        def add(node_id: str, name: str, layer: str) -> None:
            canonical_id = self._canonical_node_id(node_id, name)
            if self._is_singleton(name) and canonical_id in nodes:
                notes.append(f"拓扑规则去重：{name} 作为全局单例合并展示")
                return
            nodes[canonical_id] = TopologyNode(canonical_id, name, layer)

        def link(source: str, target: str, label: str = "", kind: str = "sync") -> None:
            canonical_source = self._canonical_node_id(source)
            canonical_target = self._canonical_node_id(target)
            if canonical_source in nodes and canonical_target in nodes:
                edges.append(TopologyEdge(canonical_source, canonical_target, label, kind))

        # Build order matters: base entry points, LLM expectations, graph knowledge, then validators.
        self._ensure_base_infrastructure(features, add, notes, graph_primary)
        self._add_llm_capability_nodes(capabilities, add)
        self._add_llm_expected_nodes(features, add)

        relevant_graph_knowledge = self._relevant_graph_knowledge(
            graph_knowledge,
            features,
            graph_primary,
        )
        if self._has_graph_knowledge(relevant_graph_knowledge):
            self._apply_graph_knowledge(relevant_graph_knowledge, nodes, edges, add, link, notes)
        elif self._has_graph_knowledge(graph_knowledge):
            notes.append("拓扑图谱增强：Neo4j 命中内容与本次 DeepSeek 拓扑期望相关性不足，已跳过泛化节点")

        if graph_primary:
            notes.append("拓扑生成策略：Neo4j coverage 达标，作为主知识源增强 DeepSeek 拓扑期望")
            self._ensure_quality_infrastructure(features, add, notes)
        else:
            notes.append("拓扑生成策略：Neo4j coverage 不足，仅作为增强；DeepSeek 拓扑期望作为本次图最低保障")

        self._connect_common(nodes, edges, link)
        self._connect_llm_expected_topology(features, nodes, link, notes)
        if graph_primary:
            self._connect_graph_services(nodes, edges, link, notes)
        self._validate_and_repair_topology(capabilities, features, nodes, add, link, notes, graph_primary)
        nodes, edges = self._dedupe_graph(nodes, edges, notes)
        nodes, edges = self._prune_irrelevant_topology(capabilities, features, graph_knowledge, nodes, edges, notes, graph_primary)
        if composition_recommendation.get("composition_needed"):
            self._apply_composition_architecture_topology(nodes, edges, winner, composition_recommendation, notes)
            nodes, edges = self._dedupe_graph(nodes, edges, notes)

        return list(nodes.values()), edges, notes

    def _apply_composition_architecture_topology(
        self,
        nodes: dict[str, TopologyNode],
        edges: list[TopologyEdge],
        winner: CandidateEvaluation,
        composition: dict,
        notes: list[str],
    ) -> None:
        if not composition.get("composition_needed"):
            return

        style_specs = [
            {
                "style_id": winner.style_id,
                "style": composition.get("primary_style") or winner.name,
                "role": self._primary_style_role(winner.style_id),
                "apply_to": self._primary_apply_targets(winner.style_id, nodes),
                "node_id": "style_primary",
            }
        ]
        for index, item in enumerate(composition.get("supporting_styles", []), start=1):
            style_name = str(item.get("style", "辅助架构模式")).strip() or "辅助架构模式"
            style_specs.append(
                {
                    "style_id": self._normalize_style_id(item.get("style_id", ""), style_name),
                    "style": style_name,
                    "role": item.get("role", "局部能力增强"),
                    "apply_to": item.get("apply_to", []),
                    "node_id": f"style_support_{index}",
                }
            )

        added = 0
        for spec in style_specs:
            targets = self._resolve_responsibility_targets(spec["style_id"], spec["apply_to"], nodes)
            if not targets:
                continue
            added += self._apply_style_structure(spec["style_id"], spec["style"], targets, nodes, edges)

        if added:
            notes.append("组合架构建图：已按推荐组合架构重组微服务、事件流、CQRS 等架构子图")

    def _apply_style_structure(
        self,
        style_id: str,
        style_name: str,
        targets: list[str],
        nodes: dict[str, TopologyNode],
        edges: list[TopologyEdge],
    ) -> int:
        def add_node(node_id: str, name: str, layer: str) -> None:
            nodes[node_id] = TopologyNode(node_id, name, layer)

        def relayer(node_id: str, layer: str) -> None:
            node = nodes.get(node_id)
            if node:
                nodes[node_id] = TopologyNode(node.id, node.name, layer)

        def link(source: str, target: str, label: str = "", kind: str = "sync") -> None:
            if source in nodes and target in nodes:
                edges.append(TopologyEdge(source, target, label, kind))

        changed = 0
        if style_id == "microservices":
            layer = f"{style_name}：服务拆分"
            for node_id in targets:
                if nodes.get(node_id) and nodes[node_id].layer == "业务服务层":
                    relayer(node_id, layer)
                    changed += 1
            return changed

        if style_id == "event_driven":
            layer = f"{style_name}：事件流"
            if "event_bus" not in nodes:
                add_node("event_bus", "事件总线", layer)
            relayer("event_bus", layer)
            event_targets = [node_id for node_id in targets if node_id in nodes]
            for node_id in event_targets:
                if nodes[node_id].layer == "业务服务层" or "服务拆分" in nodes[node_id].layer or "服务" in nodes[node_id].name:
                    link(node_id, "event_bus", "发布事件", "event")
                if node_id != "event_bus" and nodes[node_id].layer in {"异步事件层", "治理层"}:
                    relayer(node_id, layer)
                    link("event_bus", node_id, "订阅", "event")
            for node_id in ["notify", "monitoring", "audit"]:
                if node_id in nodes:
                    link("event_bus", node_id, "订阅", "event")
            return max(1, len(event_targets))

        if style_id == "cqrs":
            layer = f"{style_name}：读写分离"
            add_node("cqrs_query_service", "查询服务", layer)
            add_node("cqrs_projector", "投影器", layer)
            add_node("cqrs_read_model", "读模型", layer)
            write_targets = [node_id for node_id in targets if node_id in nodes]
            for node_id in write_targets:
                if nodes[node_id].layer == "数据层" or "库" in nodes[node_id].name:
                    relayer(node_id, layer)
                else:
                    link(node_id, "cqrs_projector", "同步投影")
            link("cqrs_projector", "cqrs_read_model", "生成读模型")
            link("cqrs_query_service", "cqrs_read_model", "查询")
            return 1 + len(write_targets)

        if style_id == "pipe_filter":
            layer = f"{style_name}：处理流水线"
            previous = ""
            for node_id in targets:
                if node_id not in nodes:
                    continue
                relayer(node_id, layer)
                if previous:
                    link(previous, node_id, "流水线")
                previous = node_id
                changed += 1
            return changed

        if style_id == "serverless":
            layer = f"{style_name}：弹性任务"
            for node_id in targets:
                if node_id in nodes:
                    relayer(node_id, layer)
                    changed += 1
            return changed

        layer = f"{style_name}：局部结构"
        for node_id in targets:
            if node_id in nodes and nodes[node_id].layer == "业务服务层":
                relayer(node_id, layer)
                changed += 1
        return changed

    def _resolve_responsibility_targets(
        self,
        style_id: str,
        apply_to: list[str],
        nodes: dict[str, TopologyNode],
    ) -> list[str]:
        resolved: list[str] = []

        def add_target(node_id: str) -> None:
            if node_id in nodes and node_id not in resolved:
                resolved.append(node_id)

        capability_map = self._domain_topology_spec().get("capabilities", {})
        for item in apply_to:
            text = str(item).strip()
            if not text:
                continue
            exact_id = self._component_id(text)
            add_target(exact_id)
            capability = capability_map.get(text)
            if capability:
                for name in list(capability.get("components", [])) + list(capability.get("stores", [])):
                    add_target(self._component_id(name))
            for node_id, node in nodes.items():
                if node.layer == "架构模式职责":
                    continue
                if text in node.name or node.name in text:
                    add_target(node_id)

        if style_id == "event_driven":
            for node_id in ["event_bus", "message_service", "message", "notify", "live", "interaction", "transcode"]:
                add_target(node_id)
        elif style_id == "microservices":
            for node_id, node in nodes.items():
                if node.layer == "业务服务层":
                    add_target(node_id)
        elif style_id == "cqrs":
            for node_id in ["message_db", "status_cache", "feed_cache", "search", "cache", "feature_store"]:
                add_target(node_id)
        elif style_id == "pipe_filter":
            for node_id in ["transcode", "object_store", "media", "replay"]:
                add_target(node_id)
        elif style_id == "serverless":
            for node_id in ["notify", "transcode", "ai"]:
                add_target(node_id)

        return resolved

    @staticmethod
    def _normalize_style_id(style_id: str, style_name: str = "") -> str:
        raw = str(style_id or "").strip().lower().replace("-", "_")
        if raw:
            aliases = {
                "eventdriven": "event_driven",
                "event_driven_architecture": "event_driven",
                "microservice": "microservices",
                "microservice_architecture": "microservices",
                "microservices_architecture": "microservices",
                "pipe_and_filter": "pipe_filter",
                "pipeline": "pipe_filter",
            }
            return aliases.get(raw, raw)
        name = str(style_name or "")
        if "事件" in name:
            return "event_driven"
        if "微服务" in name:
            return "microservices"
        if "CQRS" in name.upper() or "读写" in name:
            return "cqrs"
        if "管道" in name or "过滤器" in name or "流水线" in name:
            return "pipe_filter"
        if "Serverless" in name or "无服务器" in name:
            return "serverless"
        return ""

    @staticmethod
    def _primary_style_role(style_id: str) -> str:
        roles = {
            "event_driven": "核心：异步解耦与事件分发",
            "microservices": "核心：服务边界与独立部署",
            "layered": "核心：分层治理与职责隔离",
            "cqrs": "核心：读写分离与查询优化",
            "pipe_filter": "核心：流水线处理",
        }
        return roles.get(style_id, "核心：主导整体结构")

    def _primary_apply_targets(self, style_id: str, nodes: dict[str, TopologyNode]) -> list[str]:
        if style_id == "event_driven":
            return ["事件总线", "消息服务", "通知服务", "转码服务"]
        if style_id == "microservices":
            return [node.name for node in nodes.values() if node.layer == "业务服务层"][:8]
        if style_id == "cqrs":
            return ["消息库", "状态缓存", "搜索索引", "缓存集群"]
        if style_id == "pipe_filter":
            return ["转码服务", "对象存储", "数据管道"]
        return [node.name for node in nodes.values() if node.layer in {"业务服务层", "异步事件层"}][:6]

    def _add_local_capability_nodes(self, capabilities, add) -> None:
        if "CDN" in capabilities:
            add("cdn", "CDN", "接入层")
        if "负载均衡" in capabilities:
            add("lb", "负载均衡", "接入层")
        if "用户" in capabilities:
            add("user", "用户服务", "业务服务层")
            add("user_db", "用户库", "数据层")
        if "课程" in capabilities:
            add("course", "课程服务", "业务服务层")
            add("course_db", "课程库", "数据层")
        if "直播" in capabilities:
            add("live", "直播服务", "业务服务层")
            add("live_gateway", "直播网关", "接入层")
        if "录播" in capabilities:
            add("replay", "回放服务", "业务服务层")
            add("replay_db", "回放库", "数据层")
        if "作业" in capabilities:
            add("homework", "作业服务", "业务服务层")
            add("homework_db", "作业库", "数据层")
        if "考试" in capabilities:
            add("exam", "考试服务", "业务服务层")
            add("exam_db", "考试库", "数据层")
        if "关系" in capabilities:
            add("relation", "关系服务", "业务服务层")
            add("graph_db", "关系图谱", "数据层")
        if "内容" in capabilities:
            add("content", "内容服务", "业务服务层")
            add("content_db", "内容库", "数据层")
            add("search", "搜索索引", "数据层")
        if "互动" in capabilities:
            add("interaction", "互动服务", "业务服务层")
            add("interaction_db", "互动库", "数据层")
        if "评论" in capabilities:
            add("comment", "评论服务", "业务服务层")
        if "私信" in capabilities:
            add("message", "私信服务", "业务服务层")
            add("message_db", "消息库", "数据层")
        if "Feed" in capabilities:
            add("feed", "Feed服务", "业务服务层")
            add("feed_cache", "Feed缓存", "数据层")
        if "推荐" in capabilities:
            add("recommend", "推荐服务", "业务服务层")
            add("feature_store", "特征库", "数据层")
        if "审核" in capabilities:
            add("moderation", "审核服务", "治理层")
        if "通知" in capabilities:
            add("notify", "通知服务", "业务服务层")
        if "缓存" in capabilities:
            add("cache", "缓存集群", "数据层")
        if "消息队列" in capabilities or "事件总线" in capabilities:
            add("event_bus", "事件总线", "异步事件层")
        if "媒体" in capabilities:
            add("media", "媒体服务", "业务服务层")
        if "转码" in capabilities:
            add("transcode", "转码服务", "异步事件层")
        if "对象存储" in capabilities:
            add("object_store", "对象存储", "数据层")
        if "AI" in capabilities:
            add("ai", "AI服务", "业务服务层")
        if "商品" in capabilities:
            add("product", "商品服务", "业务服务层")
            add("product_db", "商品库", "数据层")
            add("search", "搜索索引", "数据层")
        if "购物车" in capabilities:
            add("cart", "购物车服务", "业务服务层")
            add("cart_cache", "购物车缓存", "数据层")
        if "订单" in capabilities:
            add("order", "订单服务", "业务服务层")
            add("order_db", "订单库", "数据层")
        if "支付" in capabilities:
            add("payment", "支付服务", "业务服务层")
            add("payment_db", "支付库", "数据层")
        if "库存" in capabilities:
            add("inventory", "库存服务", "业务服务层")
            add("inventory_db", "库存库", "数据层")
        if "促销" in capabilities:
            add("promotion", "促销服务", "业务服务层")
        if "秒杀" in capabilities:
            add("flash_sale", "秒杀服务", "业务服务层")
            add("event_bus", "事件总线", "异步事件层")
            add("cache", "缓存集群", "数据层")
        if "物流" in capabilities:
            add("logistics", "物流服务", "业务服务层")
            add("logistics_db", "物流库", "数据层")
        if "退款" in capabilities:
            add("refund", "退款服务", "业务服务层")
        if "风控" in capabilities:
            add("risk_control", "风控服务", "治理层")
        if "灰度" in capabilities:
            add("config_center", "配置中心", "治理层")
            add("service_registry", "服务注册", "治理层")

    def _add_llm_expected_nodes(self, features: ExtractedFeatures, add) -> None:
        expectations = features.topology_expectations or {}
        for name in expectations.get("must_have_components", []):
            clean_name = str(name).strip()
            if clean_name:
                add(self._component_id(clean_name), clean_name, self._component_layer(clean_name))

    def _add_llm_capability_nodes(self, capabilities: list[str], add) -> None:
        capability_map = self._domain_topology_spec().get("capabilities", {})
        for capability in capabilities:
            item = capability_map.get(capability)
            if not item:
                continue
            for component in item.get("components", []):
                add(self._component_id(component), component, self._component_layer(component))
            for store in item.get("stores", []):
                add(self._component_id(store), store, "数据层")

    def _connect_llm_expected_topology(
        self,
        features: ExtractedFeatures,
        nodes: dict[str, TopologyNode],
        link,
        notes: list[str],
    ) -> None:
        expectations = features.topology_expectations or {}
        for relation in expectations.get("must_have_relations", []):
            text = str(relation).strip()
            if "->" not in text:
                continue
            source, target = [part.strip() for part in text.split("->", 1)]
            if source and target:
                link(self._component_id(source), self._component_id(target), "依赖")

        service_nodes = [
            node_id for node_id, node in nodes.items()
            if node.layer == "业务服务层"
        ]
        for service in service_nodes:
            link("gateway", service, "API")
        notes.append("拓扑连接：已按 DeepSeek must_have_relations 和服务入口建立基础链路")

    def _apply_graph_knowledge(self, graph_knowledge, nodes, edges, add, link, notes) -> None:
        components = graph_knowledge.get("components", [])
        stores = graph_knowledge.get("stores", [])
        graph_edges = graph_knowledge.get("edges", [])
        if not components and not stores:
            return

        notes.append("拓扑图谱增强：已从 Neo4j 检索领域能力、组件和依赖关系")
        for component in components:
            node_id = self._component_id(component)
            add(node_id, component, self._component_layer(component))
        for store in stores:
            node_id = self._component_id(store)
            add(node_id, store, "数据层")
        for edge in graph_edges:
            source = self._component_id(edge["source"])
            target = self._component_id(edge["target"])
            label = edge.get("label", "依赖")
            kind = edge.get("kind", "sync")
            link(source, target, label, kind)

    def _relevant_graph_knowledge(
        self,
        graph_knowledge: dict,
        features: ExtractedFeatures,
        graph_primary: bool,
    ) -> dict:
        if graph_primary:
            return graph_knowledge

        allowed_names = self._expected_graph_enhancement_names(features)
        if not allowed_names:
            return {"components": [], "stores": [], "edges": [], "scenarios": [], "capabilities": []}

        components = [
            str(item).strip()
            for item in graph_knowledge.get("components", [])
            if str(item).strip() in allowed_names
        ]
        stores = [
            str(item).strip()
            for item in graph_knowledge.get("stores", [])
            if str(item).strip() in allowed_names
        ]
        retained_names = set(components) | set(stores)
        edges = []
        for edge in graph_knowledge.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if source in retained_names and target in retained_names:
                edges.append(edge)

        return {
            "components": components,
            "stores": stores,
            "edges": edges,
            "scenarios": list(graph_knowledge.get("scenarios", [])),
            "capabilities": [
                str(item).strip()
                for item in graph_knowledge.get("capabilities", [])
                if str(item).strip() in set(self.extract_business_capabilities("", features, []))
            ],
        }

    def _expected_graph_enhancement_names(self, features: ExtractedFeatures) -> set[str]:
        expected_caps = self.extract_business_capabilities("", features, [])
        names = set(self._components_for_capabilities(expected_caps))
        expectations = features.topology_expectations or {}
        names.update(str(item).strip() for item in expectations.get("must_have_components", []) if str(item).strip())
        names.update(self._explicit_quality_infrastructure(features))
        return {name for name in names if name}

    @staticmethod
    def _explicit_quality_infrastructure(features: ExtractedFeatures) -> list[str]:
        expectations = features.topology_expectations or {}
        return [
            str(item).strip()
            for item in expectations.get("quality_infrastructure", [])
            if str(item).strip()
        ]

    def _ensure_base_infrastructure(self, features: ExtractedFeatures, add, notes: list[str], graph_primary: bool) -> None:
        add("client", "客户端", "接入层")
        add("gateway", "API网关", "接入层")
        if graph_primary:
            notes.append("拓扑校验补齐：接入入口统一为客户端和 API 网关")

        if features.quality_attributes.get("concurrency", 0) >= 0.65:
            add("lb", "负载均衡", "接入层")
            if graph_primary:
                notes.append("拓扑校验补齐：高并发需求需要负载均衡")

        if features.data_flow == "event_stream" or features.quality_attributes.get("realtime", 0) >= 0.65:
            add("event_bus", "事件总线", "异步事件层")
            if graph_primary:
                notes.append("拓扑校验补齐：事件流或实时需求需要事件总线")

    def _ensure_quality_infrastructure(self, features: ExtractedFeatures, add, notes: list[str]) -> None:
        qualities = features.quality_attributes
        if qualities.get("concurrency", 0) >= 0.65:
            add("cache", "缓存集群", "数据层")
            notes.append("拓扑校验补齐：高并发场景补充缓存集群")
        if qualities.get("reliability", 0) >= 0.65:
            add("monitoring", "监控服务", "治理层")
            add("audit", "审计服务", "治理层")
            notes.append("拓扑校验补齐：可靠性需求补充监控与审计")
        if qualities.get("scalability", 0) >= 0.65:
            add("service_registry", "服务注册", "治理层")
            notes.append("拓扑校验补齐：扩展性需求补充服务注册")

    def _connect_graph_services(
        self,
        nodes: dict[str, TopologyNode],
        edges: list[TopologyEdge],
        link,
        notes: list[str],
    ) -> None:
        service_nodes = [
            node_id for node_id, node in nodes.items()
            if node.layer == "业务服务层" and node_id not in {"gateway"}
        ]
        for service in service_nodes:
            if not self._has_edge(edges, "gateway", service):
                link("gateway", service, "API")

        event_sources = [
            node_id for node_id, node in nodes.items()
            if node.layer == "业务服务层" and node_id not in {"notify"}
        ]
        if "event_bus" in nodes:
            for source in event_sources:
                if source in {"gateway"}:
                    continue
                if not self._has_edge(edges, source, "event_bus") and self._should_emit_event(source):
                    link(source, "event_bus", "事件", "event")
            for target in ["notify", "monitoring", "audit"]:
                if target in nodes and not self._has_edge(edges, "event_bus", target):
                    link("event_bus", target, "订阅", "event")

        if "cache" in nodes:
            for service in service_nodes[:4]:
                if not self._has_edge(edges, "cache", service):
                    link("cache", service, "热点")

        notes.append("拓扑校验完成：已补齐服务入口、事件订阅、缓存热点和边合法性")

    def _connect_common(self, nodes: dict[str, TopologyNode], edges: list[TopologyEdge], link) -> None:
        if "cdn" in nodes:
            link("client", "cdn", "访问")
            if "lb" in nodes:
                link("cdn", "lb", "转发")
            else:
                link("cdn", "gateway", "转发")
        elif "lb" in nodes:
            link("client", "lb", "访问")
        else:
            link("client", "gateway", "访问")

        if "lb" in nodes:
            link("lb", "gateway", "路由")

        for service in [
            "user", "relation", "content", "interaction", "comment", "message", "feed",
            "recommend", "media", "ai", "course", "live", "replay", "homework", "exam",
        ]:
            link("gateway", service, "API")
        link("live_gateway", "live", "推流")

        pairs = [
            ("user", "user_db", "读写"),
            ("course", "course_db", "课程"),
            ("relation", "graph_db", "关系"),
            ("content", "content_db", "内容"),
            ("content", "search", "索引"),
            ("interaction", "interaction_db", "互动"),
            ("comment", "content_db", "评论"),
            ("message", "message_db", "消息"),
            ("feed", "feed_cache", "动态"),
            ("recommend", "feature_store", "特征"),
            ("media", "object_store", "文件"),
            ("replay", "replay_db", "回放"),
            ("homework", "homework_db", "作业"),
            ("exam", "exam_db", "考试"),
        ]
        for source, target, label in pairs:
            link(source, target, label)

        for source in ["content", "interaction", "comment", "message", "relation", "media", "live", "course", "homework", "exam"]:
            link(source, "event_bus", "事件", "event")

        for target in ["feed", "recommend", "notify", "moderation", "transcode", "replay"]:
            link("event_bus", target, "订阅", "event")

        link("recommend", "feed", "排序")
        link("relation", "feed", "关注")
        link("cache", "feed", "加速")
        link("cache", "content", "热点")
        link("transcode", "object_store", "存储")
        link("live", "media", "音视频")
        link("media", "replay", "回放")
        link("cache", "live", "热点")
        link("cdn", "live_gateway", "分发")

    def _validate_social(self, capabilities, nodes, add, link, notes) -> None:
        if "社交" not in capabilities:
            return
        required = {
            "feed": ("Feed服务", "业务服务层"),
            "relation": ("关系服务", "业务服务层"),
            "moderation": ("审核服务", "治理层"),
            "feature_store": ("特征库", "数据层"),
            "event_bus": ("事件总线", "异步事件层"),
        }
        for node_id, (name, layer) in required.items():
            if node_id not in nodes:
                add(node_id, name, layer)
                notes.append(f"拓扑规则补全：社交媒体场景需要{name}")
        link("content", "event_bus", "发布", "event")
        link("interaction", "event_bus", "行为", "event")
        link("event_bus", "feed", "更新", "event")
        link("event_bus", "recommend", "训练", "event")
        link("moderation", "content", "审核")

    def _validate_education(self, capabilities, nodes, add, link, notes) -> None:
        if "教育" not in capabilities:
            return
        required = {
            "course": ("课程服务", "业务服务层"),
            "live": ("直播服务", "业务服务层"),
            "live_gateway": ("直播网关", "接入层"),
            "replay": ("回放服务", "业务服务层"),
            "interaction": ("互动服务", "业务服务层"),
            "media": ("媒体服务", "业务服务层"),
            "object_store": ("对象存储", "数据层"),
            "event_bus": ("事件总线", "异步事件层"),
            "cache": ("缓存集群", "数据层"),
            "cdn": ("CDN", "接入层"),
        }
        for node_id, (name, layer) in required.items():
            if node_id not in nodes:
                add(node_id, name, layer)
                notes.append(f"拓扑规则补全：在线教育场景需要{name}")

        if "录播" in capabilities and "transcode" not in nodes:
            add("transcode", "转码服务", "异步事件层")
            notes.append("拓扑规则补全：录播回放需要转码服务")
        if "课后互动" in capabilities and "notify" not in nodes:
            add("notify", "通知服务", "业务服务层")

        link("client", "cdn", "访问")
        link("cdn", "live_gateway", "直播分发")
        link("live_gateway", "live", "推流")
        link("gateway", "course", "课程")
        link("gateway", "interaction", "互动")
        link("live", "event_bus", "课堂事件", "event")
        link("interaction", "event_bus", "互动事件", "event")
        link("event_bus", "replay", "生成回放", "event")
        link("event_bus", "notify", "课后通知", "event")
        link("live", "media", "音视频")
        link("media", "object_store", "存储")
        link("transcode", "object_store", "录播")
        link("replay", "object_store", "读取")
        link("cache", "live", "热点加速")

    def _validate_commerce(self, capabilities, nodes, add, link, notes) -> None:
        capability_text = " ".join(capabilities)
        if not any(item in capability_text for item in ["电商", "商品", "购物车", "订单", "支付", "库存", "秒杀", "物流"]):
            return
        required = {
            "product": ("商品服务", "业务服务层"),
            "product_db": ("商品库", "数据层"),
            "cart": ("购物车服务", "业务服务层"),
            "cart_cache": ("购物车缓存", "数据层"),
            "order": ("订单服务", "业务服务层"),
            "order_db": ("订单库", "数据层"),
            "payment": ("支付服务", "业务服务层"),
            "payment_db": ("支付库", "数据层"),
            "inventory": ("库存服务", "业务服务层"),
            "inventory_db": ("库存库", "数据层"),
            "event_bus": ("事件总线", "异步事件层"),
            "cache": ("缓存集群", "数据层"),
            "monitoring": ("监控服务", "治理层"),
        }
        if "秒杀" in capability_text:
            required["flash_sale"] = ("秒杀服务", "业务服务层")
        if "物流" in capability_text:
            required["logistics"] = ("物流服务", "业务服务层")
            required["logistics_db"] = ("物流库", "数据层")
        if "促销" in capability_text:
            required["promotion"] = ("促销服务", "业务服务层")
        if "退款" in capability_text:
            required["refund"] = ("退款服务", "业务服务层")
        if "风控" in capability_text:
            required["risk_control"] = ("风控服务", "治理层")
        if "灰度" in capability_text:
            required["config_center"] = ("配置中心", "治理层")
            required["service_registry"] = ("服务注册", "治理层")

        for node_id, (name, layer) in required.items():
            if node_id not in nodes:
                add(node_id, name, layer)
                notes.append(f"拓扑规则补全：电商交易场景需要{name}")

        link("gateway", "product", "商品")
        link("gateway", "cart", "购物车")
        link("gateway", "order", "下单")
        link("gateway", "payment", "支付")
        link("gateway", "inventory", "库存")
        link("product", "product_db", "商品")
        link("product", "search", "检索")
        link("cart", "cart_cache", "缓存")
        link("order", "order_db", "订单")
        link("order", "payment", "支付")
        link("order", "inventory", "扣库存")
        link("payment", "payment_db", "交易")
        link("inventory", "inventory_db", "库存")
        link("order", "event_bus", "订单事件", "event")
        link("inventory", "event_bus", "库存事件", "event")
        link("event_bus", "notify", "通知", "event")
        link("cache", "product", "热点")
        link("cache", "cart", "热点")
        if "flash_sale" in nodes:
            link("gateway", "flash_sale", "秒杀")
            link("flash_sale", "cache", "热点库存")
            link("flash_sale", "event_bus", "削峰", "event")
            link("event_bus", "order", "异步下单", "event")
        if "promotion" in nodes:
            link("gateway", "promotion", "促销")
            link("promotion", "cache", "活动缓存")
        if "logistics" in nodes:
            link("event_bus", "logistics", "物流通知", "event")
            link("logistics", "logistics_db", "轨迹")
        if "refund" in nodes:
            link("gateway", "refund", "售后")
            link("refund", "order_db", "订单")
        if "risk_control" in nodes:
            link("risk_control", "payment", "风控")
        if "config_center" in nodes and "service_registry" in nodes:
            link("config_center", "service_registry", "灰度配置")

    def _validate_and_repair_topology(self, capabilities, features, nodes, add, link, notes, graph_primary: bool) -> None:
        repairs = []
        qualities = features.quality_attributes
        if qualities.get("concurrency", 0) >= 0.75:
            for node_id, name, layer in [("lb", "负载均衡", "接入层"), ("cache", "缓存集群", "数据层"), ("event_bus", "事件总线", "异步事件层")]:
                if node_id not in nodes:
                    add(node_id, name, layer)
                    repairs.append(node_id)
            link("lb", "gateway", "路由")
        if qualities.get("reliability", 0) >= 0.7:
            for node_id, name in [("monitoring", "监控服务"), ("audit", "审计服务")]:
                if node_id not in nodes:
                    add(node_id, name, "治理层")
                    repairs.append(node_id)
        if qualities.get("scalability", 0) >= 0.7 and "service_registry" not in nodes:
            add("service_registry", "服务注册", "治理层")
            repairs.append("service_registry")

        commerce_caps = {"商品浏览", "商品管理", "购物车", "订单管理", "支付结算", "库存管理", "库存一致性", "秒杀活动", "物流跟踪"}
        if commerce_caps & set(capabilities):
            expected = ["product", "cart", "order", "payment", "inventory"]
            covered = sum(1 for node_id in expected if node_id in nodes)
            notes.append(f"拓扑自检：电商核心能力覆盖率 {covered}/{len(expected)}")

        if repairs:
            notes.append("拓扑质量属性补齐：根据 DeepSeek 质量属性补充 " + "、".join(dict.fromkeys(repairs)))

    def _prune_irrelevant_topology(
        self,
        capabilities: list[str],
        features: ExtractedFeatures,
        graph_knowledge: dict,
        nodes: dict[str, TopologyNode],
        edges: list[TopologyEdge],
        notes: list[str],
        graph_primary: bool,
    ) -> tuple[dict[str, TopologyNode], list[TopologyEdge]]:
        expected_caps = self.extract_business_capabilities("", features, [])
        expected_components = set(self._components_for_capabilities(expected_caps))
        expected_components.update(str(item).strip() for item in (features.topology_expectations or {}).get("must_have_components", []) if str(item).strip())
        if graph_primary:
            expected_components.update(str(item).strip() for item in graph_knowledge.get("components", []) if str(item).strip())
            expected_components.update(str(item).strip() for item in graph_knowledge.get("stores", []) if str(item).strip())

        if not expected_components:
            return nodes, edges

        allowed_ids = {self._component_id(name) for name in expected_components}
        always_keep = {"client", "gateway", "lb", "cdn"}
        if graph_primary or not self._has_graph_knowledge(graph_knowledge):
            quality_names = self._expected_quality_infrastructure(features)
        else:
            quality_names = self._explicit_quality_infrastructure(features)
        quality_keep = {self._component_id(name) for name in quality_names}
        if "通知提醒" in expected_caps or "采购提醒" in expected_caps or "通知" in capabilities:
            quality_keep.add("notify")
        if features.data_flow == "event_stream" or any(item in expected_caps for item in ["库存一致性", "秒杀活动"]):
            quality_keep.add("event_bus")

        keep_ids = allowed_ids | always_keep | quality_keep
        pruned_nodes = {node_id: node for node_id, node in nodes.items() if node_id in keep_ids}
        pruned_edges = [
            edge for edge in edges
            if edge.source in pruned_nodes and edge.target in pruned_nodes
        ]
        removed = [node.name for node_id, node in nodes.items() if node_id not in pruned_nodes and node.layer in {"业务服务层", "数据层", "异步事件层", "治理层"}]
        if removed:
            notes.append("拓扑相关性裁剪：移除与当前业务能力无关的节点 " + "、".join(removed[:12]))
        return pruned_nodes, pruned_edges

    def _build_view_diagrams(self, nodes: list[TopologyNode], edges: list[TopologyEdge]) -> dict[str, str]:
        node_map = {node.id: node for node in nodes}
        overview_nodes, overview_edges = self._aggregate_dense_edges(nodes, edges)
        return {
            "总览图": self._render_mermaid(overview_nodes, overview_edges),
            "完整图": self._render_mermaid(nodes, edges),
            "业务链路图": self._render_mermaid(*self._project_view(nodes, edges, "business", node_map)),
            "数据流图": self._render_mermaid(*self._project_view(nodes, edges, "data", node_map)),
            "支撑设施图": self._render_mermaid(*self._project_view(nodes, edges, "support", node_map)),
        }

    def _build_structured_view_graphs(self, nodes: list[TopologyNode], edges: list[TopologyEdge]) -> dict[str, dict]:
        node_map = {node.id: node for node in nodes}
        overview_nodes, overview_edges = self._aggregate_dense_edges(nodes, edges)
        business_nodes, business_edges = self._project_view(nodes, edges, "business", node_map)
        data_nodes, data_edges = self._project_view(nodes, edges, "data", node_map)
        support_nodes, support_edges = self._project_view(nodes, edges, "support", node_map)
        return {
            "总览图": self._serialize_graph(overview_nodes, overview_edges),
            "完整图": self._serialize_graph(nodes, edges),
            "业务链路图": self._serialize_graph(business_nodes, business_edges),
            "数据流图": self._serialize_graph(data_nodes, data_edges),
            "支撑设施图": self._serialize_graph(support_nodes, support_edges),
        }

    def _serialize_graph(self, nodes: list[TopologyNode], edges: list[TopologyEdge]) -> dict:
        node_map = {node.id: node for node in nodes}
        layer_order = self._ordered_layers(nodes)
        return {
            "nodes": [
                {"id": node.id, "label": node.name, "layer": node.layer}
                for node in nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "kind": edge.kind,
                    "category": self._edge_category(edge, node_map),
                }
                for edge in edges
                if edge.source in node_map and edge.target in node_map
            ],
            "layers": layer_order,
        }

    def _project_view(
        self,
        nodes: list[TopologyNode],
        edges: list[TopologyEdge],
        view: str,
        node_map: dict[str, TopologyNode],
    ) -> tuple[list[TopologyNode], list[TopologyEdge]]:
        selected_edges: list[TopologyEdge] = []
        for edge in edges:
            category = self._edge_category(edge, node_map)
            if view == "business" and category in {"sync", "event", "responsibility"}:
                if self._edge_touches_layer(edge, node_map, {"业务服务层", "接入层", "异步事件层", "架构模式职责"}):
                    selected_edges.append(edge)
            elif view == "data" and category == "data":
                selected_edges.append(edge)
            elif view == "support" and category in {"support", "responsibility"}:
                selected_edges.append(edge)

        selected_node_ids = {edge.source for edge in selected_edges} | {edge.target for edge in selected_edges}
        if view == "business":
            selected_node_ids.update(
                node.id for node in nodes
                if node.layer in {"接入层", "业务服务层", "异步事件层", "架构模式职责"} or self._is_architecture_layer(node.layer)
            )
        elif view == "data":
            selected_node_ids.update(node.id for node in nodes if node.layer == "数据层")
        elif view == "support":
            selected_node_ids.update(node.id for node in nodes if node.layer in {"治理层", "架构模式职责"})

        selected_nodes = [node for node in nodes if node.id in selected_node_ids]
        return selected_nodes or nodes, selected_edges or edges

    def _aggregate_dense_edges(
        self,
        nodes: list[TopologyNode],
        edges: list[TopologyEdge],
    ) -> tuple[list[TopologyNode], list[TopologyEdge]]:
        node_map = {node.id: node for node in nodes}
        aggregate_nodes: dict[str, TopologyNode] = {node.id: node for node in nodes}
        aggregate_edges: list[TopologyEdge] = []
        consumed: set[int] = set()

        # Overview diagrams replace dense fan-in/fan-out groups with layer aggregate nodes.
        for hub_id, hub in node_map.items():
            grouped_in: dict[str, list[tuple[int, TopologyEdge]]] = {}
            grouped_out: dict[str, list[tuple[int, TopologyEdge]]] = {}
            for index, edge in enumerate(edges):
                if self._edge_category(edge, node_map) == "responsibility":
                    continue
                if edge.target == hub_id and edge.source in node_map:
                    grouped_in.setdefault(node_map[edge.source].layer, []).append((index, edge))
                if edge.source == hub_id and edge.target in node_map:
                    grouped_out.setdefault(node_map[edge.target].layer, []).append((index, edge))

            for layer, group in grouped_in.items():
                if len(group) < 3 or layer == "架构模式职责":
                    continue
                aggregate_id = self._aggregate_node_id(hub_id, layer, "in")
                aggregate_nodes[aggregate_id] = TopologyNode(aggregate_id, self._aggregate_label(layer), layer)
                aggregate_edges.append(TopologyEdge(aggregate_id, hub_id, self._aggregate_edge_label(group), group[0][1].kind))
                consumed.update(index for index, _edge in group)

            for layer, group in grouped_out.items():
                if len(group) < 3 or layer == "架构模式职责":
                    continue
                aggregate_id = self._aggregate_node_id(hub_id, layer, "out")
                aggregate_nodes[aggregate_id] = TopologyNode(aggregate_id, self._aggregate_label(layer), layer)
                aggregate_edges.append(TopologyEdge(hub_id, aggregate_id, self._aggregate_edge_label(group), group[0][1].kind))
                consumed.update(index for index, _edge in group)

        for index, edge in enumerate(edges):
            if index not in consumed:
                aggregate_edges.append(edge)

        reachable = {edge.source for edge in aggregate_edges} | {edge.target for edge in aggregate_edges}
        overview_nodes = [node for node in aggregate_nodes.values() if node.id in reachable or node.id in {"client", "gateway"}]
        return overview_nodes or nodes, aggregate_edges

    def _render_mermaid(self, nodes: list[TopologyNode], edges: list[TopologyEdge]) -> str:
        layer_names = self._ordered_layers(nodes)
        lines = ["flowchart TD"]
        node_map = {node.id: node for node in nodes}

        for layer in layer_names:
            layer_nodes = [node for node in nodes if node.layer == layer]
            if not layer_nodes:
                continue
            lines.append(f"  subgraph {self._safe_id(layer)}[{layer}]")
            for node in layer_nodes:
                lines.append(f"    {node.id}{self._node_label(node.name)}")
            lines.append("  end")

        seen_edges = set()
        for edge in edges:
            if edge.source not in node_map or edge.target not in node_map:
                continue
            key = (edge.source, edge.target, edge.label)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            arrow = "-.->" if edge.kind in {"event", "responsibility"} else "-->"
            lines.append(f"  %% edge-category:{self._edge_category(edge, node_map)}")
            if edge.label:
                lines.append(f"  {edge.source} {arrow}|{self._escape_label(edge.label)}| {edge.target}")
            else:
                lines.append(f"  {edge.source} {arrow} {edge.target}")

        return "\n".join(lines)

    @classmethod
    def _ordered_layers(cls, nodes: list[TopologyNode]) -> list[str]:
        present = list(dict.fromkeys(node.layer for node in nodes))
        fixed_order = ["接入层", "业务服务层", "异步事件层", "治理层", "数据层"]
        architecture_layers = [layer for layer in present if cls._is_architecture_layer(layer)]
        result: list[str] = []
        for layer in ["接入层"]:
            if layer in present:
                result.append(layer)
        result.extend(layer for layer in architecture_layers if layer not in result)
        for layer in fixed_order[1:]:
            if layer in present and layer not in result:
                result.append(layer)
        result.extend(layer for layer in present if layer not in result)
        return result

    @staticmethod
    def _is_architecture_layer(layer: str) -> bool:
        return "：" in layer and any(
            token in layer
            for token in ["架构", "微服务", "事件驱动", "CQRS", "管道", "Serverless", "分层", "六边形", "整洁"]
        )

    @staticmethod
    def _edge_touches_layer(edge: TopologyEdge, node_map: dict[str, TopologyNode], layers: set[str]) -> bool:
        return node_map.get(edge.source, TopologyNode("", "", "")).layer in layers or node_map.get(edge.target, TopologyNode("", "", "")).layer in layers

    @staticmethod
    def _aggregate_node_id(hub_id: str, layer: str, direction: str) -> str:
        digest = hashlib.sha1(f"{hub_id}:{layer}:{direction}".encode("utf-8")).hexdigest()[:8]
        return f"agg_{digest}"

    @staticmethod
    def _aggregate_label(layer: str) -> str:
        labels = {
            "业务服务层": "业务服务组",
            "接入层": "接入组件组",
            "异步事件层": "异步组件组",
            "治理层": "治理组件组",
            "数据层": "数据存储组",
        }
        return labels.get(layer, f"{layer}组件组")

    @staticmethod
    def _aggregate_edge_label(group: list[tuple[int, TopologyEdge]]) -> str:
        labels = [edge.label for _index, edge in group if edge.label]
        if not labels:
            return f"聚合 {len(group)} 条"
        primary = list(dict.fromkeys(labels))[0]
        return f"{primary} 等 {len(group)} 条"

    def _edge_category(self, edge: TopologyEdge, node_map: dict[str, TopologyNode]) -> str:
        if edge.kind == "responsibility":
            return "responsibility"
        if edge.kind == "event":
            return "event"
        source = node_map.get(edge.source)
        target = node_map.get(edge.target)
        if source and target and (source.layer == "数据层" or target.layer == "数据层"):
            return "data"
        if source and target and (source.layer == "治理层" or target.layer == "治理层"):
            return "support"
        if edge.source in {"monitoring", "audit", "service_registry", "config_center"} or edge.target in {"monitoring", "audit", "service_registry", "config_center"}:
            return "support"
        return "sync"

    def _dedupe_graph(
        self,
        nodes: dict[str, TopologyNode],
        edges: list[TopologyEdge],
        notes: list[str],
    ) -> tuple[dict[str, TopologyNode], list[TopologyEdge]]:
        canonical_by_name: dict[str, str] = {}
        aliases: dict[str, str] = {}
        deduped_nodes: dict[str, TopologyNode] = {}

        for node_id, node in nodes.items():
            canonical_id = self._canonical_node_id(node_id, node.name)
            if self._is_singleton(node.name) and node.name in canonical_by_name:
                aliases[node_id] = canonical_by_name[node.name]
                notes.append(f"拓扑规则去重：{node.name} 作为全局单例合并展示")
                continue
            if canonical_id in deduped_nodes:
                aliases[node_id] = canonical_id
                if deduped_nodes[canonical_id].name == node.name:
                    notes.append(f"拓扑规则去重：合并重复节点 {node.name}")
                continue
            canonical_by_name[node.name] = canonical_id
            aliases[node_id] = canonical_id
            deduped_nodes[canonical_id] = TopologyNode(canonical_id, node.name, node.layer)

        deduped_edges: list[TopologyEdge] = []
        seen_edges: set[tuple[str, str, str, str]] = set()
        for edge in edges:
            source = aliases.get(edge.source, self._canonical_node_id(edge.source))
            target = aliases.get(edge.target, self._canonical_node_id(edge.target))
            if source == target or source not in deduped_nodes or target not in deduped_nodes:
                continue
            key = (source, target, edge.label, edge.kind)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            deduped_edges.append(TopologyEdge(source, target, edge.label, edge.kind))

        return deduped_nodes, deduped_edges

    @staticmethod
    def _has_graph_knowledge(graph_knowledge: dict) -> bool:
        return bool(
            graph_knowledge.get("components")
            or graph_knowledge.get("stores")
            or graph_knowledge.get("edges")
        )

    def _graph_coverage_sufficient(
        self,
        requirement: str,
        features: ExtractedFeatures,
        graph_knowledge: dict,
        extra_capabilities: list[str],
    ) -> bool:
        if not self._has_graph_knowledge(graph_knowledge):
            return False
        coverage = self.assess_coverage(requirement, features, graph_knowledge, extra_capabilities)
        dimensions = coverage.get("dimensions", {})
        capability_score = dimensions.get("business_capability", {}).get("score", 0)
        component_score = dimensions.get("component", {}).get("score", 0)
        relation_score = dimensions.get("relation", {}).get("score", 1)
        return coverage.get("score", 0) >= 0.75 and capability_score >= 0.65 and component_score >= 0.65 and relation_score >= 0.4

    def _topology_capabilities(
        self,
        requirement: str,
        features: ExtractedFeatures,
        graph_knowledge: dict,
        extra_capabilities: list[str],
        graph_primary: bool,
    ) -> list[str]:
        capabilities = self.extract_capabilities(requirement, features)
        if graph_primary:
            capabilities.extend(str(item).strip() for item in graph_knowledge.get("capabilities", []) if str(item).strip())
        capabilities.extend(extra_capabilities)
        return list(dict.fromkeys(item for item in capabilities if item))

    @staticmethod
    def _has_edge(edges: list[TopologyEdge], source: str, target: str) -> bool:
        return any(edge.source == source and edge.target == target for edge in edges)

    @staticmethod
    def _should_emit_event(node_id: str) -> bool:
        return node_id not in {"user", "course"}

    @classmethod
    def _canonical_node_id(cls, node_id: str, name: str | None = None) -> str:
        if name and name in cls.CANONICAL_COMPONENT_IDS:
            return cls.CANONICAL_COMPONENT_IDS[name]
        return node_id

    @staticmethod
    def _safe_id(value: str) -> str:
        return "layer_" + str(abs(hash(value)))

    @staticmethod
    def _escape_label(value: str) -> str:
        return value.replace('"', "'")

    @classmethod
    def _node_label(cls, value: str) -> str:
        escaped = cls._escape_label(value)
        if any(token in escaped for token in ["<", ">", "|", "\n"]):
            return f'["{escaped}"]'
        return f"[{escaped}]"

    @classmethod
    def _component_id(cls, name: str) -> str:
        if name in cls.CANONICAL_COMPONENT_IDS:
            return cls.CANONICAL_COMPONENT_IDS[name]
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
        return f"kg_{digest}"

    @classmethod
    def _is_singleton(cls, name: str) -> bool:
        return name in cls.SINGLETON_COMPONENTS or name in cls.CANONICAL_COMPONENT_IDS

    @staticmethod
    def _component_layer(name: str) -> str:
        if name in ["CDN", "负载均衡", "API网关", "直播网关", "设备网关", "视频网关"]:
            return "接入层"
        if name in ["消息队列", "事件总线", "任务队列", "数据管道", "实时计算", "离线分析", "转码服务"]:
            return "异步事件层"
        if name in ["审核服务", "审计服务", "风控服务", "防作弊服务", "监控服务", "配置中心"]:
            return "治理层"
        return "业务服务层"
