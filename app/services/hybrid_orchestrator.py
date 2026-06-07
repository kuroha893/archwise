from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.architecture_matcher import ArchitectureMatcherAgent
from app.agents.evaluation_generator import EvaluationGeneratorAgent
from app.agents.requirement_analysis import RequirementAnalysisAgent
from app.models.schemas import ArchitectureStyle, CandidateEvaluation, ExtractedFeatures, RecommendationResponse
from app.services.composition_recommender import CompositionRecommender
from app.services.exceptions import DeepSeekServiceError, RequirementParsingError
from app.services.knowledge_service import KnowledgeService
from app.services.llm_client import LLMClient
from app.services.report_formatter import ReportFormatter
from app.services.rule_engine import RuleDecision, RuleEngine
from app.services.topology_generator import TopologyGenerator


@dataclass
class ReasoningContext:
    requirement: str
    top_k: int
    styles: list[ArchitectureStyle]
    style_map: dict[str, ArchitectureStyle]
    trace: list[str]
    decision_trace: dict[str, Any]
    composition_recommendation: dict[str, Any]
    topology_fast_mode: bool
    topology_llm_timeout_seconds: float | None
    topology_repair_max_rounds: int


class RecommendationGraphState(TypedDict, total=False):
    ctx: ReasoningContext
    features: ExtractedFeatures
    graph_matches: list[tuple[str, float, str]]
    case_matches: list[dict[str, Any]]
    llm_candidates: list[CandidateEvaluation]
    local_candidates: list[CandidateEvaluation]
    candidates: list[CandidateEvaluation]
    llm_composition: dict[str, Any]
    composition: dict[str, Any]
    review_notes: list[str]
    rule_decision: RuleDecision
    rule_notes: list[str]
    captured_case_id: str


class TopologyRepairGraphState(TypedDict, total=False):
    ctx: ReasoningContext
    features: ExtractedFeatures
    current_graph: dict[str, Any]
    original_graph: dict[str, Any]
    llm_capabilities: list[str]
    repair_trace: list[dict[str, Any]]
    round_index: int
    coverage: dict[str, Any]
    patch: dict[str, Any] | None
    stop: bool


class HybridReasoningOrchestrator:
    """LangChain-style chain orchestration for rules + LLM + knowledge graph reasoning."""

    TOPOLOGY_COVERAGE_THRESHOLD = 0.75
    TOPOLOGY_REPAIR_MAX_ROUNDS = int(os.getenv("TOPOLOGY_REPAIR_MAX_ROUNDS", "1"))
    TOPOLOGY_LLM_TIMEOUT_SECONDS = float(os.getenv("TOPOLOGY_LLM_TIMEOUT_SECONDS", "12"))
    TOPOLOGY_FAST_MODE = os.getenv("TOPOLOGY_FAST_MODE", "true").lower() not in {"0", "false", "no", "off"}

    def __init__(
        self,
        matcher: ArchitectureMatcherAgent,
        requirement_agent: RequirementAnalysisAgent,
        evaluator: EvaluationGeneratorAgent,
        llm_client: LLMClient,
        rule_engine: RuleEngine,
        knowledge_service: KnowledgeService,
    ) -> None:
        self.matcher = matcher
        self.requirement_agent = requirement_agent
        self.evaluator = evaluator
        self.llm_client = llm_client
        self.rule_engine = rule_engine
        self.knowledge_service = knowledge_service
        self.graph_service = knowledge_service.graph_service
        self.topology_generator = TopologyGenerator()
        self.composition_recommender = CompositionRecommender()

    async def run(
        self,
        requirement: str,
        styles: list[ArchitectureStyle],
        top_k: int,
        topology_options: dict | None = None,
    ) -> RecommendationResponse:
        ctx = self._context(requirement, styles, top_k, topology_options)
        state = await self._run_recommendation_graph(ctx)
        features = state["features"]
        candidates = state["candidates"]
        composition = state["composition"]
        review_notes = state["review_notes"]

        report = await self.evaluator.generate(requirement, features, candidates, ctx.style_map)
        if not report:
            message = self.llm_client.last_error or "DeepSeek 未返回可用评估报告。"
            ctx.trace.append(f"评估生成 Agent 终止：{message}")
            raise DeepSeekServiceError(f"评估报告生成失败：{message}")
        ctx.trace.append("评估生成 Agent 完成 DeepSeek 报告生成")

        matrix = [self._matrix_row(item) for item in candidates]
        ctx.decision_trace = self._build_decision_trace(
            features=features,
            graph_matches=state["graph_matches"],
            candidates=candidates,
            review_notes=review_notes,
            composition=composition,
            rule_decision=state["rule_decision"],
            rule_notes=state["rule_notes"],
            local_candidates=state["local_candidates"],
            case_matches=state["case_matches"],
            captured_case_id=state["captured_case_id"],
        )
        ctx.composition_recommendation = composition
        ctx.trace.append("架构组合推荐由本地组合推荐器生成，并保留 DeepSeek 组合建议作为证据")
        topology_payload = await self._build_topologies(ctx, features, candidates)

        return RecommendationResponse(
            requirement=requirement,
            features=features,
            candidates=candidates,
            final_recommendation=candidates[0],
            report=report,
            comparison_matrix=matrix,
            topology_diagrams=topology_payload["diagrams"],
            topology_graphs=topology_payload["graphs"],
            trace=ctx.trace,
            decision_trace=ctx.decision_trace,
            composition_recommendation=composition,
        )

    async def stream(
        self,
        requirement: str,
        styles: list[ArchitectureStyle],
        top_k: int,
        topology_options: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        ctx = self._context(requirement, styles, top_k, topology_options)
        try:
            state = await self._run_recommendation_graph(ctx)
        except RequirementParsingError as exc:
            yield self._sse(
                "error",
                {
                    "message": str(exc),
                    "trace": ctx.trace,
                },
            )
            yield self._sse("done", {"ok": False})
            return
        except DeepSeekServiceError as exc:
            yield self._sse(
                "error",
                {
                    "message": str(exc),
                    "trace": ctx.trace,
                },
            )
            yield self._sse("done", {"ok": False})
            return
        except RuntimeError as exc:
            yield self._sse(
                "error",
                {
                    "message": str(exc),
                    "trace": ctx.trace,
                },
            )
            yield self._sse("done", {"ok": False})
            return

        features = state["features"]
        candidates = state["candidates"]
        composition = state["composition"]
        review_notes = state["review_notes"]

        yield self._sse(
            "features",
            {
                "requirement": requirement,
                "features": features.model_dump(),
                "trace": ctx.trace,
            },
        )

        matrix = [self._matrix_row(item) for item in candidates]
        # Build the UI trace only after every reasoning node has contributed evidence.
        ctx.decision_trace = self._build_decision_trace(
            features=features,
            graph_matches=state["graph_matches"],
            candidates=candidates,
            review_notes=review_notes,
            composition=composition,
            rule_decision=state["rule_decision"],
            rule_notes=state["rule_notes"],
            local_candidates=state["local_candidates"],
            case_matches=state["case_matches"],
            captured_case_id=state["captured_case_id"],
        )
        ctx.composition_recommendation = composition
        ctx.trace.append("架构组合推荐由本地组合推荐器生成，并保留 DeepSeek 组合建议作为证据")
        yield self._sse(
            "recommendation",
            {
                "requirement": requirement,
                "features": features.model_dump(),
                "candidates": [item.model_dump() for item in candidates],
                "final_recommendation": candidates[0].model_dump(),
                "comparison_matrix": matrix,
                "trace": ctx.trace,
                "decision_trace": ctx.decision_trace,
                "composition_recommendation": composition,
            },
        )

        yield self._sse("report_delta", {"delta": ReportFormatter.build_report_prefix(requirement, features, candidates)})

        streamed = False
        report_buffer: list[str] = []
        last_report_flush = time.perf_counter()
        async for token in self.llm_client.stream_report(requirement, features, candidates):
            streamed = True
            report_buffer.append(token)
            buffered_text = "".join(report_buffer)
            should_flush = len(buffered_text) >= 120 or time.perf_counter() - last_report_flush >= 0.18
            if should_flush:
                yield self._sse("report_delta", {"delta": buffered_text})
                report_buffer = []
                last_report_flush = time.perf_counter()

        if report_buffer:
            yield self._sse("report_delta", {"delta": "".join(report_buffer)})

        if not streamed:
            message = self.llm_client.last_error or "DeepSeek 未返回流式报告内容。"
            ctx.trace.append(f"评估生成 Agent 终止：{message}")
            yield self._sse(
                "error",
                {
                    "message": f"评估报告生成失败：{message}",
                    "trace": ctx.trace,
                },
            )
            yield self._sse("done", {"ok": False})
            return

        yield self._sse("report_delta", {"delta": ReportFormatter.build_report_suffix(candidates[0], ctx.style_map)})
        yield self._sse("report_delta", {"delta": ReportFormatter.build_report_footer(candidates[0])})
        yield self._sse("done", {"ok": True})

    async def stream_topology(
        self,
        requirement: str,
        styles: list[ArchitectureStyle],
        features: ExtractedFeatures,
        final_recommendation: CandidateEvaluation,
        composition_recommendation: dict[str, Any] | None = None,
        decision_trace: dict[str, Any] | None = None,
        topology_options: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        ctx = self._context(requirement, styles, 1, topology_options)
        ctx.trace.append("拓扑生成流启动：复用推荐流已解析的需求特征和最终推荐架构")
        ctx.composition_recommendation = composition_recommendation or {}
        ctx.decision_trace = decision_trace or {}
        candidates = [final_recommendation]
        topology_task = asyncio.create_task(self._build_topologies(ctx, features, candidates))
        while not topology_task.done():
            yield self._sse(
                "heartbeat",
                {
                    "message": "架构图生成中：正在检索 Neo4j、检查覆盖率并生成 Mermaid 拓扑",
                    "trace": ctx.trace,
                    "decision_trace": ctx.decision_trace,
                },
            )
            await asyncio.sleep(2)
        topology_payload = await self._resolve_topology_task(topology_task, ctx, features, candidates)
        yield self._sse(
            "topology",
            {
                "topology_diagrams": topology_payload["diagrams"],
                "topology_graphs": topology_payload["graphs"],
                "trace": ctx.trace,
                "decision_trace": ctx.decision_trace,
            },
        )
        yield self._sse("done", {"ok": True})

    async def _resolve_topology_task(
        self,
        topology_task: asyncio.Task[dict[str, Any]],
        ctx: ReasoningContext,
        features: ExtractedFeatures,
        candidates: list[CandidateEvaluation],
    ) -> dict[str, Any]:
        try:
            return await topology_task
        except Exception as exc:
            ctx.trace.append(f"拓扑生成任务异常，使用结构化基础拓扑兜底：{exc}")
            topology_diagrams, topology_graphs, notes = self.topology_generator.generate_graph_views(
                ctx.requirement,
                features,
                candidates[0],
                extra_capabilities=[],
                graph_knowledge={},
                composition_recommendation=ctx.composition_recommendation,
            )
            ctx.trace.extend(notes)
            return {
                "diagrams": {f"{candidates[0].name}{name}": diagram for name, diagram in topology_diagrams.items()},
                "graphs": {f"{candidates[0].name}{name}": graph for name, graph in topology_graphs.items()},
            }

    async def _build_topologies(
        self,
        ctx: ReasoningContext,
        features: ExtractedFeatures,
        candidates: list[CandidateEvaluation],
        topology_prep_task: asyncio.Task[tuple[dict[str, Any], list[dict[str, Any]], list[str]]] | None = None,
    ) -> dict[str, Any]:
        if topology_prep_task is None:
            graph_knowledge, repair_trace, topology_capabilities = await self._prepare_topology_knowledge(ctx, features)
        else:
            graph_knowledge, repair_trace, topology_capabilities = await topology_prep_task
        topology_diagrams, topology_graphs, notes = self.topology_generator.generate_graph_views(
            ctx.requirement,
            features,
            candidates[0],
            extra_capabilities=topology_capabilities,
            graph_knowledge=graph_knowledge,
            composition_recommendation=ctx.composition_recommendation,
        )
        if topology_capabilities:
            ctx.trace.append("DeepSeek 主解析拓扑业务能力：" + "、".join(topology_capabilities[:8]))
        if graph_knowledge.get("scenarios"):
            ctx.trace.append("Neo4j 拓扑知识命中场景：" + "、".join(graph_knowledge["scenarios"][:5]))
        if graph_knowledge.get("capabilities"):
            ctx.trace.append("Neo4j 拓扑知识命中能力：" + "、".join(graph_knowledge["capabilities"][:8]))
        ctx.decision_trace["topology_evidence"] = {
            "scenarios": graph_knowledge.get("scenarios", []),
            "capabilities": graph_knowledge.get("capabilities", []),
            "components": graph_knowledge.get("components", []),
            "stores": graph_knowledge.get("stores", []),
            "vector_matches": graph_knowledge.get("vector_matches", []),
            "vector_error": graph_knowledge.get("vector_error", ""),
            "notes": notes,
            "react_repair": repair_trace,
        }
        ctx.trace.extend(notes)
        ctx.trace.append("拓扑生成器基于领域能力和规则校验生成结构化可交互拓扑")
        return {
            "diagrams": {f"{candidates[0].name}{name}": diagram for name, diagram in topology_diagrams.items()},
            "graphs": {f"{candidates[0].name}{name}": graph for name, graph in topology_graphs.items()},
        }

    async def _prepare_topology_knowledge(
        self,
        ctx: ReasoningContext,
        features: ExtractedFeatures,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        started = time.perf_counter()
        topology_capabilities = [
            str(item).strip()
            for item in features.business_capabilities
            if str(item).strip()
        ]
        retrieve_started = time.perf_counter()
        graph_knowledge = await self.graph_service.retrieve_topology_knowledge(
            ctx.requirement,
            features,
        )
        ctx.trace.append(f"拓扑耗时：Neo4j 知识检索 {time.perf_counter() - retrieve_started:.1f}s")
        repair_started = time.perf_counter()
        graph_knowledge, repair_trace = await self._repair_topology_knowledge(
            ctx,
            features,
            graph_knowledge,
            topology_capabilities,
        )
        ctx.trace.append(f"拓扑耗时：LLM 补全与临时合并 {time.perf_counter() - repair_started:.1f}s")
        ctx.trace.append(f"拓扑耗时：知识准备总计 {time.perf_counter() - started:.1f}s")
        return graph_knowledge, repair_trace, topology_capabilities

    async def _repair_topology_knowledge(
        self,
        ctx: ReasoningContext,
        features: ExtractedFeatures,
        graph_knowledge: dict,
        llm_capabilities: list[str],
    ) -> tuple[dict, list[dict[str, Any]]]:
        state = await self._run_topology_repair_graph(ctx, features, graph_knowledge, llm_capabilities)
        return state["current_graph"], state["repair_trace"]

    async def _run_topology_repair_graph(
        self,
        ctx: ReasoningContext,
        features: ExtractedFeatures,
        graph_knowledge: dict[str, Any],
        llm_capabilities: list[str],
    ) -> TopologyRepairGraphState:
        # The topology graph keeps coverage scoring separate from LLM patch creation.
        graph = StateGraph(TopologyRepairGraphState)
        graph.add_node("gap_review_agent", self._topology_gap_review_agent)
        graph.add_node("coverage_agent", self._topology_coverage_agent)
        graph.add_node("patch_agent", self._topology_patch_agent)
        graph.add_node("merge_agent", self._topology_merge_agent)
        graph.set_entry_point("gap_review_agent")
        graph.add_edge("gap_review_agent", "coverage_agent")
        graph.add_conditional_edges(
            "coverage_agent",
            self._route_topology_repair,
            {
                "patch": "patch_agent",
                "done": END,
            },
        )
        graph.add_edge("patch_agent", "merge_agent")
        graph.add_conditional_edges(
            "merge_agent",
            self._route_topology_merge,
            {
                "coverage": "coverage_agent",
                "done": END,
            },
        )
        compiled = graph.compile()
        return await compiled.ainvoke(
            {
                "ctx": ctx,
                "features": features,
                "current_graph": graph_knowledge,
                "original_graph": graph_knowledge,
                "llm_capabilities": llm_capabilities,
                "repair_trace": [],
                "round_index": 0,
                "patch": None,
                "stop": False,
            }
        )

    async def _topology_gap_review_agent(self, state: TopologyRepairGraphState) -> TopologyRepairGraphState:
        ctx = state["ctx"]
        features = state["features"]
        current_graph = state["current_graph"]
        repair_trace = list(state["repair_trace"])
        gap_started = time.perf_counter()
        try:
            gap_patch = await self._maybe_wait_for(
                self.llm_client.review_topology_coverage_gap(
                    ctx.requirement,
                    features,
                    state["current_graph"],
                    request_timeout=ctx.topology_llm_timeout_seconds,
                ),
                ctx.topology_llm_timeout_seconds,
            )
        except TimeoutError:
            gap_patch = None
            timeout_label = self._timeout_label(ctx.topology_llm_timeout_seconds)
            repair_trace.append(
                {
                    "round": 0,
                    "action": "llm_gap_review_timeout",
                    "message": f"拓扑完整性复核超过 {timeout_label}，已跳过以保证响应速度。",
                }
            )
        ctx.trace.append(f"拓扑耗时：DeepSeek 漏项复核 {time.perf_counter() - gap_started:.1f}s")
        if gap_patch and self._patch_has_write_items(gap_patch):
            before_coverage = self.topology_generator.assess_coverage(
                ctx.requirement,
                features,
                current_graph,
                extra_capabilities=state["llm_capabilities"],
            )
            trial_patch = gap_patch
            current_graph = self.topology_generator.merge_knowledge_patch(current_graph, trial_patch)
            after_coverage = self.topology_generator.assess_coverage(
                ctx.requirement,
                features,
                current_graph,
                extra_capabilities=state["llm_capabilities"],
            )
            # Trial graph updates the current response; durable writes are deferred to normalization.
            if ctx.topology_fast_mode:
                neo4j_result = {
                    "ok": False,
                    "skipped": True,
                    "reason": "拓扑快速模式：补丁已临时用于本次架构图，跳过 embedding 规范化和 Neo4j 写入。",
                }
            elif self._patch_has_write_items(trial_patch):
                self._schedule_background_topology_write(
                    ctx,
                    features,
                    trial_patch,
                    state["original_graph"],
                    before_coverage,
                    "漏项复核补丁",
                )
                neo4j_result = {
                    "ok": False,
                    "pending": True,
                    "reason": "知识库进化已转入后台：embedding 规范化通过后再写入 Neo4j。",
                }
            else:
                neo4j_result = {
                    "ok": False,
                    "skipped": True,
                    "reason": "补丁没有达到永久写入条件的节点或关系。",
                }
            repair_trace.append(
                {
                    "round": 0,
                    "action": "llm_gap_review_merged",
                    "raw_patch": gap_patch,
                    "trial_patch": trial_patch,
                    "write_patch": trial_patch if not ctx.topology_fast_mode else {},
                    "normalization": [],
                    "temporary_items": [],
                    "semantic_available": None,
                    "neo4j": neo4j_result,
                    "coverage_before": before_coverage,
                    "coverage_after": after_coverage,
                }
            )
            gap_components = list(trial_patch.get("components", [])) + list(trial_patch.get("stores", []))
            ctx.trace.append(
                "拓扑完整性复核：DeepSeek 对照原始需求补充漏项 "
                + ("、".join(gap_components[:8]) if gap_components else "见能力映射")
            )
        return {"current_graph": current_graph, "repair_trace": repair_trace}

    async def _topology_coverage_agent(self, state: TopologyRepairGraphState) -> TopologyRepairGraphState:
        ctx = state["ctx"]
        features = state["features"]
        repair_trace = list(state["repair_trace"])
        round_index = int(state["round_index"]) + 1
        coverage = self.topology_generator.assess_coverage(
            ctx.requirement,
            features,
            state["current_graph"],
            extra_capabilities=state["llm_capabilities"],
        )
        repair_trace.append(
            {
                "round": round_index,
                "action": "coverage_check",
                "coverage": coverage,
            }
        )
        stop = (
            round_index > ctx.topology_repair_max_rounds
            or not self._coverage_requires_repair(coverage)
        )
        return {
            "round_index": round_index,
            "coverage": coverage,
            "repair_trace": repair_trace,
            "stop": stop,
        }

    async def _topology_patch_agent(self, state: TopologyRepairGraphState) -> TopologyRepairGraphState:
        ctx = state["ctx"]
        features = state["features"]
        coverage = state["coverage"]
        repair_trace = list(state["repair_trace"])
        round_index = state["round_index"]
        patch_started = time.perf_counter()
        try:
            patch = await self._maybe_wait_for(
                self.llm_client.propose_topology_knowledge_patch(
                    ctx.requirement,
                    features,
                    coverage,
                    state["current_graph"],
                    request_timeout=ctx.topology_llm_timeout_seconds,
                ),
                ctx.topology_llm_timeout_seconds,
            )
        except TimeoutError:
            patch = None
            timeout_label = self._timeout_label(ctx.topology_llm_timeout_seconds)
            repair_trace.append(
                {
                    "round": round_index,
                    "action": "llm_patch_timeout",
                    "message": f"DeepSeek ReAct 补全超过 {timeout_label}，已跳过以保证响应速度。",
                }
            )
        ctx.trace.append(f"拓扑耗时：第 {round_index} 轮 DeepSeek ReAct 补全 {time.perf_counter() - patch_started:.1f}s")
        if not patch:
            repair_trace.append(
                {
                    "round": round_index,
                    "action": "llm_patch_unavailable",
                    "message": "DeepSeek 补全不可用或返回格式不合法，本次不写入 Neo4j。",
                }
            )
            return {"patch": None, "repair_trace": repair_trace, "stop": True}
        return {"patch": patch, "repair_trace": repair_trace}

    async def _topology_merge_agent(self, state: TopologyRepairGraphState) -> TopologyRepairGraphState:
        ctx = state["ctx"]
        features = state["features"]
        coverage = state["coverage"]
        round_index = state["round_index"]
        repair_trace = list(state["repair_trace"])
        raw_patch = state.get("patch")
        if not raw_patch:
            return {"stop": True, "repair_trace": repair_trace}
        trial_patch = raw_patch
        normalization_report: list[dict[str, Any]] = []
        patch_capability_names = {
            str(item.get("name", "")).strip()
            for item in trial_patch.get("capabilities", [])
            if isinstance(item, dict)
        }
        patch_names = set(trial_patch.get("components", [])) | set(trial_patch.get("stores", []))
        patch_relations = {
            f"{edge.get('source')}->{edge.get('target')}"
            for edge in trial_patch.get("edges", [])
            if isinstance(edge, dict) and edge.get("source") and edge.get("target")
        }
        # Reject broad patches that do not address the exact missing coverage items.
        if not (
            patch_capability_names & set(coverage.get("missing_capabilities", []))
            or patch_names & set(coverage.get("missing_components", []))
            or patch_names & set(coverage.get("missing_quality_infrastructure", []))
            or patch_relations & set(coverage.get("missing_relations", []))
        ):
            repair_trace.append(
                {
                    "round": round_index,
                    "action": "llm_patch_rejected",
                    "message": "DeepSeek 补丁未覆盖缺失能力、组件或关系，已拒绝泛化补全。",
                    "missing_capabilities": coverage.get("missing_capabilities", []),
                    "missing_components": coverage.get("missing_components", []),
                    "missing_relations": coverage.get("missing_relations", []),
                    "raw_patch": raw_patch,
                    "trial_patch": trial_patch,
                    "write_patch": trial_patch,
                    "normalization": normalization_report,
                }
            )
            return {"repair_trace": repair_trace, "stop": True}

        missing_capabilities = [item for item in coverage.get("missing_capabilities", []) if item]
        if missing_capabilities:
            covered_caps = {
                item
                for item in patch_capability_names
                if item in missing_capabilities
            }
            if len(covered_caps) < len(set(missing_capabilities)):
                repair_trace.append(
                    {
                        "round": round_index,
                        "action": "llm_patch_rejected",
                        "message": "DeepSeek 补丁没有把缺失能力按能力组逐项补齐，已拒绝。",
                        "missing_capabilities": missing_capabilities,
                        "trial_patch": trial_patch,
                        "write_patch": trial_patch,
                        "normalization": normalization_report,
                    }
                )
                return {"repair_trace": repair_trace, "stop": True}

        trial_graph = self.topology_generator.merge_knowledge_patch(state["current_graph"], trial_patch)
        refreshed_coverage = self.topology_generator.assess_coverage(
            ctx.requirement,
            features,
            trial_graph,
            extra_capabilities=state["llm_capabilities"],
        )
        if refreshed_coverage["score"] <= coverage["score"]:
            repair_trace.append(
                {
                    "round": round_index,
                    "action": "llm_patch_rejected",
                    "message": "DeepSeek 补丁试合并后未提升多维覆盖率，未写入 Neo4j。",
                    "coverage_after": refreshed_coverage,
                    "raw_patch": raw_patch,
                    "trial_patch": trial_patch,
                    "write_patch": trial_patch,
                    "normalization": normalization_report,
                }
            )
            return {"repair_trace": repair_trace, "stop": True}

        if ctx.topology_fast_mode:
            neo4j_result = {
                "ok": False,
                "skipped": True,
                "reason": "拓扑快速模式：补丁已临时用于本次架构图，跳过 embedding 规范化和 Neo4j 写入。",
            }
        elif self._patch_has_write_items(trial_patch):
            self._schedule_background_topology_write(
                ctx,
                features,
                trial_patch,
                trial_graph,
                coverage,
                f"第 {round_index} 轮 ReAct 补丁",
            )
            neo4j_result = {
                "ok": False,
                "pending": True,
                "reason": "知识库进化已转入后台：embedding 规范化通过后再写入 Neo4j。",
            }
        else:
            neo4j_result = {
                "ok": False,
                "skipped": True,
                "reason": "补丁没有达到永久写入条件的节点或关系。",
            }
        repair_trace.append(
            {
                "round": round_index,
                "action": "llm_patch_merged",
                "raw_patch": raw_patch,
                "trial_patch": trial_patch,
                "write_patch": trial_patch if not ctx.topology_fast_mode else {},
                "normalization": normalization_report,
                "temporary_items": [],
                "semantic_available": None,
                "neo4j": neo4j_result,
                "coverage_after": refreshed_coverage,
            }
        )
        ctx.trace.append(
            "拓扑 ReAct 补全：覆盖率 "
            f"{coverage['score']} -> {refreshed_coverage['score']}，"
            f"补充组件 {', '.join(trial_patch.get('components', [])[:6]) or '见能力映射'}"
        )
        return {
            "current_graph": trial_graph,
            "repair_trace": repair_trace,
            "coverage": refreshed_coverage,
            "stop": refreshed_coverage["score"] >= self.TOPOLOGY_COVERAGE_THRESHOLD,
        }

    @staticmethod
    def _route_topology_repair(state: TopologyRepairGraphState) -> str:
        return "done" if state.get("stop") else "patch"

    @staticmethod
    def _route_topology_merge(state: TopologyRepairGraphState) -> str:
        return "done" if state.get("stop") else "coverage"

    def _schedule_background_topology_write(
        self,
        ctx: ReasoningContext,
        features: ExtractedFeatures,
        write_candidate_patch: dict[str, Any],
        graph_knowledge: dict[str, Any],
        coverage: dict[str, Any],
        label: str,
    ) -> None:
        if ctx.topology_fast_mode or not self._patch_has_write_items(write_candidate_patch):
            return

        ctx.trace.append(f"知识库进化后台任务已启动：{label} 将执行 embedding 规范化和 Neo4j 写入")
        # Background normalization keeps the SSE response responsive while Neo4j evolves.
        task = asyncio.create_task(
            self._background_normalize_and_merge_topology_patch(
                requirement=ctx.requirement,
                features=features,
                patch=write_candidate_patch,
                graph_knowledge=graph_knowledge,
                coverage=coverage,
                label=label,
            )
        )
        task.add_done_callback(self._consume_background_task_result)

    async def _background_normalize_and_merge_topology_patch(
        self,
        requirement: str,
        features: ExtractedFeatures,
        patch: dict[str, Any],
        graph_knowledge: dict[str, Any],
        coverage: dict[str, Any],
        label: str,
    ) -> None:
        started = time.perf_counter()
        normalization = await self.graph_service.normalize_topology_patch(
            patch,
            requirement,
            features,
            graph_knowledge,
            coverage,
        )
        write_patch = normalization.get("write_patch", normalization.get("trial_patch", patch))
        if self._patch_has_write_items(write_patch):
            await asyncio.to_thread(
                self.graph_service.merge_topology_patch,
                requirement,
                features,
                write_patch,
            )
        elapsed = time.perf_counter() - started
        print(f"[topology-background] {label} normalized and merged in {elapsed:.1f}s")

    @staticmethod
    def _consume_background_task_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            print(f"[topology-background] knowledge evolution failed: {exc}")

    async def _run_recommendation_graph(self, ctx: ReasoningContext) -> RecommendationGraphState:
        graph = StateGraph(RecommendationGraphState)
        graph.add_node("requirement_analysis_agent", self._graph_requirement_analysis)
        graph.add_node("case_retrieval_agent", self._graph_case_retrieval)
        graph.add_node("architecture_matching_agent", self._graph_architecture_matching)
        graph.add_node("rule_validation_agent", self._graph_rule_validation)
        graph.add_node("composition_agent", self._graph_composition)
        graph.add_node("case_capture_agent", self._graph_case_capture)
        graph.set_entry_point("requirement_analysis_agent")
        graph.add_edge("requirement_analysis_agent", "case_retrieval_agent")
        graph.add_edge("case_retrieval_agent", "architecture_matching_agent")
        graph.add_edge("architecture_matching_agent", "rule_validation_agent")
        graph.add_edge("rule_validation_agent", "composition_agent")
        graph.add_edge("composition_agent", "case_capture_agent")
        graph.add_edge("case_capture_agent", END)
        compiled = graph.compile()
        return await compiled.ainvoke({"ctx": ctx})

    async def _graph_requirement_analysis(self, state: RecommendationGraphState) -> RecommendationGraphState:
        ctx = state["ctx"]
        features, graph_matches = await self._analyze(ctx)
        return {"features": features, "graph_matches": graph_matches}

    async def _graph_case_retrieval(self, state: RecommendationGraphState) -> RecommendationGraphState:
        ctx = state["ctx"]
        features = state["features"]
        ctx.trace.append("案例检索 Agent 从 Chroma 可信案例库召回相似历史需求")
        case_matches = await self.knowledge_service.retrieve_trusted_cases(ctx.requirement, features, top_k=3)
        if case_matches:
            summary = "、".join(f"{item['title']}({item['similarity']:.2f})" for item in case_matches)
            ctx.trace.append(f"案例检索 Agent 命中可信案例：{summary}")
        else:
            ctx.trace.append("案例检索 Agent 未命中达到阈值的可信案例，本次不注入案例材料")
        return {"case_matches": case_matches}

    async def _graph_architecture_matching(self, state: RecommendationGraphState) -> RecommendationGraphState:
        ctx = state["ctx"]
        candidates, composition, review_notes = await self._match_with_deepseek(
            ctx,
            state["features"],
            ctx.top_k,
            state["case_matches"],
        )
        return {
            "llm_candidates": candidates,
            "candidates": candidates,
            "llm_composition": composition,
            "review_notes": review_notes,
        }

    async def _graph_rule_validation(self, state: RecommendationGraphState) -> RecommendationGraphState:
        ctx = state["ctx"]
        features = state["features"]
        decision = self.rule_engine.evaluate(features)
        ctx.trace.append("规则校验 Agent 执行 data/rules.json 硬约束与偏好规则")
        local_candidates = self.matcher.match(
            features,
            ctx.styles,
            top_k=max(ctx.top_k, 6),
            preferred_style_ids=decision.preferred_style_ids,
            rejected_style_ids=decision.rejected_style_ids,
        )
        merged = self._merge_candidate_sets(state["candidates"], local_candidates, decision, ctx.top_k)
        candidates, rule_notes = self.rule_engine.validate_candidates(features, merged, decision)
        if decision.fired_rule_ids:
            ctx.trace.append("规则校验 Agent 命中规则：" + "、".join(decision.fired_rule_ids))
        else:
            ctx.trace.append("规则校验 Agent 未命中硬约束规则，仅记录本地匹配证据")
        ctx.trace.extend(rule_notes)
        return {
            "rule_decision": decision,
            "rule_notes": rule_notes,
            "local_candidates": local_candidates,
            "candidates": candidates[: ctx.top_k],
        }

    async def _graph_composition(self, state: RecommendationGraphState) -> RecommendationGraphState:
        ctx = state["ctx"]
        local_composition = self.composition_recommender.recommend(
            ctx.requirement,
            state["features"],
            state["candidates"],
        )
        composition = {
            **local_composition,
            "source": "composition_recommender",
            "llm_suggestion": state["llm_composition"],
        }
        return {"composition": composition}

    async def _graph_case_capture(self, state: RecommendationGraphState) -> RecommendationGraphState:
        ctx = state["ctx"]
        record = await self.knowledge_service.capture_candidate_case(
            ctx.requirement,
            state["features"],
            state["candidates"],
        )
        ctx.trace.append("案例记忆 Agent 已将本次推荐写入候选案例，等待人工确认")
        return {"captured_case_id": record.id}

    async def _analyze(self, ctx: ReasoningContext):
        ctx.trace.append("需求解析 Agent 接收自然语言需求")
        try:
            features = await self.requirement_agent.analyze(ctx.requirement)
        except RequirementParsingError as exc:
            ctx.trace.append(f"需求解析 Agent 终止：{exc}")
            raise
        ctx.trace.append("DeepSeek 主解析完成结构化需求特征，并通过 Pydantic Schema 校验")

        ctx.trace.append("需求解析结果进入案例检索、架构匹配和规则校验节点")
        ctx.trace.append("知识图谱保留给拓扑知识检索、ReAct 补全和 Neo4j 写入")
        return features, []

    async def _match_with_deepseek(
        self,
        ctx: ReasoningContext,
        features: ExtractedFeatures,
        top_k: int,
        case_matches: list[dict[str, Any]],
    ) -> tuple[list[CandidateEvaluation], dict[str, Any], list[str]]:
        ctx.trace.append("架构匹配 Agent 调用 DeepSeek 生成候选架构、评分和组合建议")
        styles = [style.model_dump() for style in ctx.styles]
        result = await self.llm_client.recommend_architectures(
            ctx.requirement,
            features,
            styles,
            top_k,
            case_references=case_matches,
        )
        if not result:
            message = self.llm_client.last_error or "DeepSeek 未返回可用候选架构 JSON。"
            ctx.trace.append(f"架构匹配 Agent 终止：{message}")
            raise DeepSeekServiceError(f"架构匹配失败：{message}")
        candidates, composition_payload = result
        review_notes = list(composition_payload.pop("review_notes", []))
        ctx.trace.append("DeepSeek 架构匹配 Agent 完成候选架构排序")
        if review_notes:
            ctx.trace.extend(f"DeepSeek 架构匹配说明：{note}" for note in review_notes)
        return candidates[:top_k], composition_payload, review_notes

    @staticmethod
    def _merge_candidate_sets(
        llm_candidates: list[CandidateEvaluation],
        local_candidates: list[CandidateEvaluation],
        decision: RuleDecision,
        top_k: int,
    ) -> list[CandidateEvaluation]:
        merged: dict[str, CandidateEvaluation] = {
            candidate.style_id: candidate.model_copy(deep=True)
            for candidate in llm_candidates
        }
        local_map = {candidate.style_id: candidate for candidate in local_candidates}
        # Local rule preferences can add omitted styles, then normal ranking still decides order.
        for style_id in decision.preferred_style_ids:
            if style_id in merged or style_id not in local_map:
                continue
            candidate = local_map[style_id].model_copy(deep=True)
            candidate.matched_reasons.append("本地匹配器根据规则偏好补入该候选")
            merged[style_id] = candidate
        for candidate in local_candidates:
            if len(merged) >= top_k:
                break
            if candidate.style_id in merged:
                continue
            merged[candidate.style_id] = candidate.model_copy(deep=True)
        candidates = list(merged.values())
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    @staticmethod
    def _drivers_from_features(features: ExtractedFeatures) -> list[str]:
        mapping = {
            "concurrency": "高并发",
            "realtime": "实时性",
            "reliability": "高可用",
            "scalability": "弹性伸缩",
            "data_intensity": "数据密集",
            "ai_reasoning": "AI 推理",
        }
        drivers = [label for key, label in mapping.items() if features.quality_attributes.get(key, 0) >= 0.65]
        if features.data_flow == "event_stream":
            drivers.append("事件流")
        if features.data_flow == "pipeline":
            drivers.append("数据管道")
        if features.data_flow == "transactional":
            drivers.append("事务处理")
        return drivers

    @staticmethod
    def _coverage_missing_items(coverage: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        for key in [
            "missing_capabilities",
            "missing_components",
            "missing_relations",
            "missing_quality_infrastructure",
        ]:
            missing.extend(coverage.get(key, []))
        return missing

    @classmethod
    def _coverage_requires_repair(cls, coverage: dict[str, Any]) -> bool:
        business_missing = bool(
            coverage.get("missing_capabilities")
            or coverage.get("missing_components")
            or coverage.get("missing_relations")
        )
        quality_missing = bool(coverage.get("missing_quality_infrastructure"))
        if business_missing:
            return True
        if quality_missing and coverage.get("score", 0) < cls.TOPOLOGY_COVERAGE_THRESHOLD:
            return True
        return False

    @staticmethod
    def _patch_has_write_items(patch: dict[str, Any]) -> bool:
        return bool(
            patch.get("capabilities")
            or patch.get("components")
            or patch.get("stores")
            or patch.get("edges")
        )

    @classmethod
    def _context(
        cls,
        requirement: str,
        styles: list[ArchitectureStyle],
        top_k: int,
        topology_options: dict | None = None,
    ) -> ReasoningContext:
        topology_options = topology_options or {}
        fast_mode = topology_options.get("fast_mode")
        timeout = topology_options.get("llm_timeout_seconds")
        rounds = topology_options.get("repair_max_rounds")
        normalized_timeout = cls._normalize_topology_timeout(timeout)
        return ReasoningContext(
            requirement=requirement,
            top_k=top_k,
            styles=styles,
            style_map={style.id: style for style in styles},
            trace=[
                "HybridReasoningOrchestrator 启动 LangChain 风格链式编排",
                "拓扑生成配置："
                f"{'快速模式' if (cls.TOPOLOGY_FAST_MODE if fast_mode is None else bool(fast_mode)) else '精细模式'}，"
                f"LLM 超时 {cls._timeout_label(normalized_timeout)}，"
                f"补全轮数 {int(rounds if rounds is not None else cls.TOPOLOGY_REPAIR_MAX_ROUNDS)}",
            ],
            decision_trace={},
            composition_recommendation={},
            topology_fast_mode=cls.TOPOLOGY_FAST_MODE if fast_mode is None else bool(fast_mode),
            topology_llm_timeout_seconds=normalized_timeout,
            topology_repair_max_rounds=max(
                0,
                min(3, int(rounds if rounds is not None else cls.TOPOLOGY_REPAIR_MAX_ROUNDS)),
            ),
        )

    @classmethod
    def _normalize_topology_timeout(cls, timeout: Any) -> float | None:
        if timeout is None:
            return max(3.0, float(cls.TOPOLOGY_LLM_TIMEOUT_SECONDS))
        try:
            value = float(timeout)
        except (TypeError, ValueError):
            return max(3.0, float(cls.TOPOLOGY_LLM_TIMEOUT_SECONDS))
        if value <= 0:
            return None
        return max(3.0, value)

    @staticmethod
    async def _maybe_wait_for(awaitable, timeout: float | None):
        if timeout is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=timeout)

    @staticmethod
    def _timeout_label(timeout: float | None) -> str:
        if timeout is None:
            return "无上限"
        return f"{timeout:.0f}s"

    @staticmethod
    def _build_decision_trace(
        features,
        graph_matches,
        candidates,
        review_notes,
        composition,
        rule_decision: RuleDecision,
        rule_notes: list[str],
        local_candidates: list[CandidateEvaluation],
        case_matches: list[dict[str, Any]],
        captured_case_id: str,
    ) -> dict[str, Any]:
        winner = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        score_gap = round(winner.score - runner_up.score, 1) if runner_up else None
        return {
            "requirement_features": {
                "domain": features.domain,
                "data_flow": features.data_flow,
                "keywords": features.keywords,
                "business_capabilities": features.business_capabilities,
                "architecture_drivers": features.architecture_drivers,
                "topology_expectations": features.topology_expectations,
                "quality_attributes": features.quality_attributes,
                "constraints": features.constraints,
                "ambiguity_notes": features.ambiguity_notes,
            },
            "rule_evidence": {
                "enabled": True,
                "reasons": rule_decision.reasons,
                "validation_notes": rule_notes,
                "fired_rule_ids": rule_decision.fired_rule_ids,
                "preferred_style_ids": rule_decision.preferred_style_ids,
                "rejected_style_ids": rule_decision.rejected_style_ids,
            },
            "case_memory_evidence": {
                "retrieved": [
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "similarity": item["similarity"],
                        "expected_styles": item["expected_styles"],
                    }
                    for item in case_matches
                ],
                "captured_candidate_case_id": captured_case_id,
                "policy": "仅可信案例参与检索注入；运行时案例先进入候选状态。",
            },
            "graph_evidence": [
                {"style_id": style_id, "score": score, "reason": reason}
                for style_id, score, reason in graph_matches[:8]
            ],
            "local_matcher_evidence": [
                {
                    "style_id": item.style_id,
                    "name": item.name,
                    "score": item.score,
                    "raw_score": item.raw_score,
                    "matched_reasons": item.matched_reasons,
                    "deductions": item.deductions,
                }
                for item in local_candidates[:8]
            ],
            "score_evidence": [
                {
                    "style_id": item.style_id,
                    "name": item.name,
                    "score": item.score,
                    "raw_score": item.raw_score,
                    "role": item.recommendation_role,
                    "confidence": item.confidence,
                    "matched_reasons": item.matched_reasons,
                    "deductions": item.deductions,
                    "risks": item.risks,
                }
                for item in candidates
            ],
            "llm_review": review_notes,
            "composition_evidence": composition,
            "final_reason": (
                f"{winner.name} 得分 {winner.score}/100，定位为{winner.recommendation_role}。"
                + (f"相对 {runner_up.name} 领先 {score_gap} 分。" if runner_up else "")
            ),
        }

    @staticmethod
    def _matrix_row(candidate):
        scores = candidate.quality_scores
        return {
            "架构风格": candidate.name,
            "综合评分": candidate.score,
            "推荐定位": candidate.recommendation_role,
            "置信度": candidate.confidence,
            "扩展性": HybridReasoningOrchestrator._stars(scores.get("scalability", 0)),
            "性能": HybridReasoningOrchestrator._stars(scores.get("performance", 0)),
            "可靠性": HybridReasoningOrchestrator._stars(scores.get("reliability", 0)),
            "可维护性": HybridReasoningOrchestrator._stars(scores.get("modifiability", 0)),
            "实时性": HybridReasoningOrchestrator._stars(scores.get("realtime", 0)),
            "复杂度友好度": HybridReasoningOrchestrator._complexity_label(scores.get("complexity", 0)),
            "扣分原因": "；".join(candidate.deductions) if candidate.deductions else "无明显扣分",
        }

    @staticmethod
    def _stars(value: float) -> str:
        count = max(1, min(5, round(value * 5)))
        return "★" * count + "☆" * (5 - count)

    @staticmethod
    def _complexity_label(value: float) -> str:
        if value >= 0.75:
            return "较低"
        if value >= 0.5:
            return "中等"
        return "较高"

    @staticmethod
    def _sse(event: str, data: object) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
