from __future__ import annotations

import asyncio

from app.models.schemas import ArchitectureStyle, ExtractedFeatures
from app.services.knowledge_normalizer import KnowledgeNormalizer
from app.services.langchain_embeddings import LLMClientEmbeddings
from app.services.llm_client import LLMClient
from app.services.neo4j_aura import Neo4jAuraService


QUALITY_TO_SCENARIO = {
    "concurrency": ["高并发", "即时通信", "高并发异步处理", "突发流量"],
    "realtime": ["实时消息", "即时通信", "交易通知", "IoT 事件流"],
    "reliability": ["金融业务", "订单交易", "审计系统"],
    "scalability": ["快速迭代产品", "复杂业务平台", "多团队协作", "弹性伸缩系统"],
    "data_intensity": ["数据处理", "ETL", "日志分析", "批处理流水线"],
    "ai_reasoning": ["AI 诊断", "专家系统", "多智能体推理"],
}


class KnowledgeGraphService:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()
        self.neo4j = Neo4jAuraService()
        self.embeddings = LLMClientEmbeddings(self.llm_client)
        self.normalizer = KnowledgeNormalizer(self.llm_client)

    def build_graph(self, styles: list[ArchitectureStyle]) -> dict[str, list[dict[str, str]]]:
        if self.neo4j.configured:
            try:
                graph = self.neo4j.fetch_graph()
                if graph["nodes"]:
                    return graph
            except Exception:
                pass

        nodes: dict[str, dict[str, str]] = {}
        edges: list[dict[str, str]] = []

        for style in styles:
            style_node = f"style:{style.id}"
            nodes[style_node] = {"id": style_node, "label": style.name, "type": "architecture_style"}
            nodes[f"category:{style.category}"] = {"id": f"category:{style.category}", "label": style.category, "type": "category"}
            edges.append({"source": style_node, "target": f"category:{style.category}", "relation": "BELONGS_TO"})

            for scenario in style.suitable_for:
                scenario_node = f"scenario:{scenario}"
                nodes[scenario_node] = {"id": scenario_node, "label": scenario, "type": "scenario"}
                edges.append({"source": style_node, "target": scenario_node, "relation": "SUITABLE_FOR"})

            for quality, score in style.quality_scores.items():
                quality_node = f"quality:{quality}"
                nodes[quality_node] = {"id": quality_node, "label": quality, "type": "quality_attribute"}
                edges.append({"source": style_node, "target": quality_node, "relation": f"HAS_SCORE:{score}"})

        return {"nodes": list(nodes.values()), "edges": edges}

    def build_topology_graph(self) -> dict[str, list[dict[str, object]]]:
        """Domain topology knowledge graph for visualization. Neo4j first, JSON seed fallback."""
        if self.neo4j.configured:
            try:
                graph = self.neo4j.fetch_topology_graph()
                if graph["nodes"]:
                    return graph
            except Exception:
                pass
        return self._seed_topology_graph()

    @staticmethod
    def _seed_topology_graph() -> dict[str, list[dict[str, object]]]:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "data" / "domain_topology.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"nodes": [], "edges": []}

        nodes: dict[str, dict[str, object]] = {}
        edges: list[dict[str, object]] = []

        def add_node(node_id: str, label: str, node_type: str) -> None:
            if node_id not in nodes:
                nodes[node_id] = {"id": node_id, "label": label, "type": node_type}

        capabilities = data.get("capabilities", {})
        store_names = {store for spec in capabilities.values() for store in spec.get("stores", [])}

        for scenario in data.get("scenarios", []):
            sid = f"domainscenario:{scenario['id']}"
            add_node(sid, scenario.get("name", scenario["id"]), "domainscenario")
            for capability in scenario.get("capabilities", []):
                cid = f"businesscapability:{capability}"
                add_node(cid, capability, "businesscapability")
                edges.append({"source": sid, "target": cid, "relation": "REQUIRES", "label": "需要", "kind": "sync"})

        for capability, spec in capabilities.items():
            cid = f"businesscapability:{capability}"
            add_node(cid, capability, "businesscapability")
            for component in spec.get("components", []):
                comp_id = f"architecturecomponent:{component}"
                add_node(comp_id, component, "architecturecomponent")
                edges.append({"source": cid, "target": comp_id, "relation": "IMPLEMENTED_BY", "label": "实现", "kind": "sync"})
            for store in spec.get("stores", []):
                store_id = f"datastore:{store}"
                add_node(store_id, store, "datastore")
                edges.append({"source": cid, "target": store_id, "relation": "USES_STORE", "label": "使用", "kind": "sync"})

        for dependency in data.get("dependencies", []):
            if not isinstance(dependency, (list, tuple)) or len(dependency) < 2:
                continue
            source, target = str(dependency[0]), str(dependency[1])
            label = str(dependency[2]) if len(dependency) > 2 else "依赖"
            if source.startswith("*") or target.startswith("*"):
                continue
            is_store = target in store_names
            source_id = f"architecturecomponent:{source}"
            target_id = f"datastore:{target}" if is_store else f"architecturecomponent:{target}"
            add_node(source_id, source, "architecturecomponent")
            add_node(target_id, target, "datastore" if is_store else "architecturecomponent")
            edges.append({
                "source": source_id,
                "target": target_id,
                "relation": "STORES_IN" if is_store else "DEPENDS_ON",
                "label": label,
                "kind": "event" if ("事件" in label or "通知" in label) else "sync",
            })

        return {"nodes": list(nodes.values()), "edges": edges}

    def list_styles(self) -> list[ArchitectureStyle]:
        return self.neo4j.fetch_styles()

    def add_style(self, style: ArchitectureStyle) -> ArchitectureStyle:
        result = self.neo4j.sync_styles([style])
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or result))
        return style

    def retrieve_styles(self, features: ExtractedFeatures, styles: list[ArchitectureStyle]) -> list[tuple[str, float, str]]:
        """Retrieve styles through graph-like Style-Scenario-Quality relations.

        The demo uses in-memory graph data. If Neo4j is configured later, this
        method is the seam to replace with Cypher queries while keeping the
        orchestrator unchanged.
        """
        matches: list[tuple[str, float, str]] = []
        active_quality = [
            quality for quality, value in features.quality_attributes.items()
            if value >= 0.6
        ]

        for style in styles:
            score = 0.0
            reasons: list[str] = []
            scenario_text = " ".join(style.suitable_for)
            keyword_text = " ".join(features.keywords)

            for quality in active_quality:
                architecture_quality = self._quality_name(quality)
                quality_score = style.quality_scores.get(architecture_quality, 0)
                if quality_score >= 0.68:
                    score += quality_score * 0.2
                    reasons.append(f"质量属性 {quality} 匹配")

                for scenario in QUALITY_TO_SCENARIO.get(quality, []):
                    if scenario in scenario_text:
                        score += 0.16
                        reasons.append(f"适用场景 {scenario} 匹配")

            for scenario in style.suitable_for:
                if scenario in keyword_text or scenario in features.domain:
                    score += 0.18
                    reasons.append(f"领域场景 {scenario} 匹配")

            if score > 0:
                matches.append((style.id, round(score, 3), "、".join(list(dict.fromkeys(reasons))[:3])))

        matches.sort(key=lambda item: item[1], reverse=True)
        return matches

    def neo4j_status(self) -> dict[str, object]:
        return self.neo4j.status()

    async def sync_to_neo4j(self, styles: list[ArchitectureStyle]) -> dict[str, object]:
        style_result = await asyncio.to_thread(self.neo4j.sync_styles, styles)
        topology_result = await asyncio.to_thread(self.neo4j.sync_domain_topology)
        singleton_result = await asyncio.to_thread(self.neo4j.sync_singleton_components)
        vector_result = await self.reindex_topology_vectors()
        return {
            "ok": bool(style_result.get("ok") and topology_result.get("ok") and singleton_result.get("ok") and vector_result.get("ok")),
            "styles": style_result,
            "domain_topology": topology_result,
            "singletons": singleton_result,
            "vectors": vector_result,
        }

    async def rebuild_domain_topology(self) -> dict[str, object]:
        topology_result = await asyncio.to_thread(self.neo4j.rebuild_domain_topology)
        singleton_result = await asyncio.to_thread(self.neo4j.sync_singleton_components)
        vector_result = await self.reindex_topology_vectors()
        return {
            "ok": bool(topology_result.get("ok") and singleton_result.get("ok") and vector_result.get("ok")),
            "domain_topology": topology_result,
            "singletons": singleton_result,
            "vectors": vector_result,
        }

    async def reindex_topology_vectors(self) -> dict[str, object]:
        if not self.neo4j.configured:
            return {"ok": False, "error": "Neo4j AuraDB is not configured."}
        records = await asyncio.to_thread(self.neo4j.fetch_topology_node_records)
        flat_records = [record for items in records.values() for record in items]
        if not flat_records:
            return {"ok": True, "vectors_indexed": 0}
        texts = [self._topology_record_text(record) for record in flat_records]
        vectors = await self.embeddings.aembed_documents(texts)
        if len(vectors) != len(flat_records):
            raise RuntimeError("Embedding vector count does not match Neo4j topology records.")
        dimension = len(vectors[0])
        payload = [
            {
                "label": record["label"],
                "name": record["name"],
                "text": text,
                "embedding": vector,
            }
            for record, text, vector in zip(flat_records, texts, vectors)
        ]
        return await asyncio.to_thread(self.neo4j.upsert_topology_embeddings, payload, dimension)

    def merge_topology_patch(self, requirement: str, features: ExtractedFeatures, patch: dict[str, object]) -> dict[str, object]:
        return self.neo4j.merge_topology_patch(
            requirement,
            features.keywords,
            patch,
            business_capabilities=features.business_capabilities,
            domain=features.domain,
        )

    async def normalize_topology_patch(
        self,
        patch: dict[str, object],
        requirement: str,
        features: ExtractedFeatures,
        graph_knowledge: dict[str, object] | None = None,
        coverage: dict[str, object] | None = None,
    ) -> dict[str, object]:
        existing_records = self._topology_records_for_normalization(graph_knowledge or {}, coverage or {})
        return await self.normalizer.normalize_patch(
            patch,
            existing_records,
            requirement=requirement,
            features=features,
        )

    async def detect_duplicate_like_nodes(self) -> dict[str, object]:
        records = self.neo4j.fetch_topology_node_records() if self.neo4j.configured else {
            "BusinessCapability": [],
            "ArchitectureComponent": [],
            "DataStore": [],
        }
        findings = await self.normalizer.detect_duplicates(records)
        return {
            "ok": True,
            "configured": self.neo4j.configured,
            "findings": findings,
            "count": len(findings),
        }

    async def retrieve_topology_knowledge(self, requirement: str, features: ExtractedFeatures) -> dict[str, object]:
        query_embedding = None
        vector_error = ""
        if self.neo4j.configured:
            try:
                query_embedding = await self.embeddings.aembed_query(self._topology_query_text(requirement, features))
            except RuntimeError as exc:
                vector_error = str(exc)
        if self.neo4j.configured:
            result = self.neo4j.retrieve_topology_knowledge(
                requirement,
                features.keywords,
                features.quality_attributes,
                features.business_capabilities,
                features.domain,
                query_embedding=query_embedding,
            )
            if vector_error:
                result["vector_error"] = vector_error
            if result.get("components") or result.get("stores"):
                return result
        return {"components": [], "stores": [], "edges": [], "scenarios": [], "capabilities": []}

    @staticmethod
    def _topology_query_text(requirement: str, features: ExtractedFeatures) -> str:
        # Keep the vector query aligned with the same fields used for topology generation.
        return "\n".join(
            [
                f"需求: {requirement}",
                f"领域: {features.domain}",
                "关键词: " + "、".join(features.keywords),
                "业务能力: " + "、".join(features.business_capabilities),
                "架构驱动: " + "、".join(features.architecture_drivers),
                f"数据流: {features.data_flow}",
            ]
        )

    @staticmethod
    def _topology_record_text(record: dict[str, object]) -> str:
        context = record.get("context", {}) if isinstance(record.get("context"), dict) else {}
        return "\n".join(
            [
                f"节点类型: {record.get('label', '')}",
                f"节点名称: {record.get('name', '')}",
                "关联场景: " + "、".join(context.get("scenarios", []) or []),
                "关联能力: " + "、".join(context.get("capabilities", []) or []),
                "关联组件: " + "、".join(context.get("components", []) or []),
                "数据存储: " + "、".join(context.get("stores", []) or []),
                "上下游: " + "、".join(context.get("neighbors", []) or []),
            ]
        )

    def _topology_records_for_normalization(
        self,
        graph_knowledge: dict[str, object],
        coverage: dict[str, object],
    ) -> dict[str, list[dict[str, object]]]:
        # Include retrieved graph knowledge as temporary context for semantic normalization.
        records = self.neo4j.fetch_topology_node_records() if self.neo4j.configured else {
            "BusinessCapability": [],
            "ArchitectureComponent": [],
            "DataStore": [],
        }

        def append(label: str, name: object, context: dict[str, object]) -> None:
            clean_name = str(name).strip()
            if not clean_name:
                return
            if any(item.get("name") == clean_name for item in records[label]):
                return
            records[label].append({"label": label, "name": clean_name, "context": context})

        for capability in list(graph_knowledge.get("capabilities", []) or []):
            append(
                "BusinessCapability",
                capability,
                {
                    "scenarios": list(graph_knowledge.get("scenarios", []) or []),
                    "capabilities": [capability],
                    "components": list(graph_knowledge.get("components", []) or []),
                    "stores": list(graph_knowledge.get("stores", []) or []),
                },
            )

        component_names = list(graph_knowledge.get("components", []) or [])
        for component in component_names:
            append(
                "ArchitectureComponent",
                component,
                {
                    "scenarios": list(graph_knowledge.get("scenarios", []) or []),
                    "capabilities": list(graph_knowledge.get("capabilities", []) or []),
                    "neighbors": component_names,
                },
            )

        for store in list(graph_knowledge.get("stores", []) or []):
            append(
                "DataStore",
                store,
                {
                    "scenarios": list(graph_knowledge.get("scenarios", []) or []),
                    "capabilities": list(graph_knowledge.get("capabilities", []) or []),
                    "components": list(graph_knowledge.get("components", []) or []),
                },
            )
        return records

    @staticmethod
    def _quality_name(feature_name: str) -> str:
        mapping = {
            "concurrency": "scalability",
            "realtime": "realtime",
            "reliability": "reliability",
            "scalability": "scalability",
            "data_intensity": "performance",
            "ai_reasoning": "modifiability",
        }
        return mapping.get(feature_name, feature_name)
