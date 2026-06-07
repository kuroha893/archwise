from __future__ import annotations

import math
from typing import Any

from app.models.schemas import ExtractedFeatures
from app.services.langchain_embeddings import LLMClientEmbeddings
from app.services.llm_client import LLMClient


class KnowledgeNormalizer:
    """Semantic gate before LLM-generated topology patches are written to Neo4j."""

    LABELS = ("BusinessCapability", "ArchitectureComponent", "DataStore")
    PATCH_KEYS = {
        "BusinessCapability": "capabilities",
        "ArchitectureComponent": "components",
        "DataStore": "stores",
    }
    HIGH_CONFIDENCE_THRESHOLD = 0.88
    UNCERTAIN_THRESHOLD = 0.72
    EMBEDDING_WEIGHT = 0.75
    CONTEXT_WEIGHT = 0.25
    TOP_K = 5

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()
        self.embeddings = LLMClientEmbeddings(self.llm_client)

    async def normalize_patch(
        self,
        patch: dict[str, Any],
        existing_records: dict[str, list[dict[str, Any]]] | None = None,
        requirement: str = "",
        features: ExtractedFeatures | None = None,
    ) -> dict[str, Any]:
        records = self._prepare_records(existing_records or {})
        patch_nodes = self._extract_patch_nodes(patch, requirement, features)
        semantic_report: list[dict[str, Any]] = []

        if not patch_nodes:
            return {
                "trial_patch": self._empty_patch(patch),
                "write_patch": self._empty_patch(patch),
                "report": semantic_report,
                "temporary_items": [],
                "semantic_available": False,
            }

        embeddings_available = await self._attach_embeddings(patch_nodes, records)
        decisions: dict[str, dict[str, str]] = {label: {} for label in self.LABELS}
        temporary: dict[str, set[str]] = {label: set() for label in self.LABELS}

        for node in patch_nodes:
            label = node["label"]
            matches = self._top_matches(node, records.get(label, [])) if embeddings_available else []
            decision = await self._decide_node(requirement, node, matches, embeddings_available)
            canonical = decision["canonical"] or node["name"]
            decisions[label][node["name"]] = canonical
            if decision["action"] == "temporary":
                temporary[label].add(node["name"])
            semantic_report.append(decision)

        # The trial patch powers the current diagram; write_patch excludes temporary nodes.
        trial_patch = self._rewrite_patch(patch, decisions, temporary, include_temporary=True)
        write_patch = self._rewrite_patch(patch, decisions, temporary, include_temporary=False)
        return {
            "trial_patch": trial_patch,
            "write_patch": write_patch,
            "report": semantic_report,
            "temporary_items": [
                {"label": label, "name": name}
                for label, names in temporary.items()
                for name in sorted(names)
            ],
            "semantic_available": embeddings_available,
        }

    async def detect_duplicates(
        self,
        existing_records: dict[str, list[dict[str, Any]]],
        limit_per_label: int = 50,
    ) -> list[dict[str, Any]]:
        records = self._prepare_records(existing_records)
        all_records = [record for items in records.values() for record in items]
        try:
            embeddings = await self.embeddings.aembed_documents([record["text"] for record in all_records])
        except RuntimeError:
            return []
        for record, embedding in zip(all_records, embeddings):
            record["embedding"] = embedding

        findings: list[dict[str, Any]] = []
        for label, items in records.items():
            for index, left in enumerate(items):
                for right in items[index + 1:]:
                    score = self._combined_score(left, right)
                    if score < self.UNCERTAIN_THRESHOLD:
                        continue
                    findings.append(
                        {
                            "label": label,
                            "left": left["name"],
                            "right": right["name"],
                            "score": round(score, 3),
                            "reason": "embedding 语义相似度与图谱上下文相似度达到疑似重复阈值",
                        }
                    )
        findings.sort(key=lambda item: item["score"], reverse=True)
        return findings[:limit_per_label]

    async def _attach_embeddings(
        self,
        patch_nodes: list[dict[str, Any]],
        records: dict[str, list[dict[str, Any]]],
    ) -> bool:
        texts = [node["text"] for node in patch_nodes]
        record_index: list[dict[str, Any]] = []
        for node in patch_nodes:
            for record in records.get(node["label"], []):
                record_index.append(record)
                texts.append(record["text"])
        try:
            # Embed candidate and existing nodes together to keep similarity scores comparable.
            embeddings = await self.embeddings.aembed_documents(texts)
        except RuntimeError:
            return False
        patch_count = len(patch_nodes)
        for node, embedding in zip(patch_nodes, embeddings[:patch_count]):
            node["embedding"] = embedding
        for record, embedding in zip(record_index, embeddings[patch_count:]):
            record["embedding"] = embedding
        return True

    async def _decide_node(
        self,
        requirement: str,
        node: dict[str, Any],
        matches: list[dict[str, Any]],
        embeddings_available: bool,
    ) -> dict[str, Any]:
        if not embeddings_available:
            return self._report(node, node["name"], "temporary", 0.0, "Embedding 服务不可用，不执行永久写入去重判定")
        if not matches:
            return self._report(node, node["name"], "created", 1.0, "同类型 Neo4j 候选为空，作为新节点写入")

        best = matches[0]
        if best["score"] >= self.HIGH_CONFIDENCE_THRESHOLD:
            return self._report(
                node,
                best["name"],
                "merged",
                best["score"],
                "embedding 语义召回与图谱上下文均达到高置信合并阈值",
                matches,
            )
        if best["score"] < self.UNCERTAIN_THRESHOLD:
            return self._report(
                node,
                node["name"],
                "created",
                best["score"],
                "同类型候选语义距离较远，作为新节点写入",
                matches,
            )

        # Only the uncertain band asks the LLM to adjudicate semantic equivalence.
        adjudication = await self.llm_client.adjudicate_semantic_merge(requirement, node, matches)
        if not adjudication:
            return self._report(
                node,
                node["name"],
                "temporary",
                best["score"],
                "相似度处于不确定区间且 LLM 二次判定不可用，仅本次临时使用",
                matches,
            )
        decision = adjudication["decision"]
        canonical = adjudication.get("canonical") or node["name"]
        if decision == "merge" and canonical not in {item["name"] for item in matches}:
            decision = "temporary"
            canonical = node["name"]
        if decision == "temporary":
            canonical = node["name"]
        action = {"merge": "merged", "create": "created", "temporary": "temporary"}[decision]
        return self._report(
            node,
            canonical,
            action,
            max(best["score"], float(adjudication.get("confidence", 0))),
            adjudication.get("reason") or "LLM 二次判定完成",
            matches,
        )

    def _top_matches(self, node: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        matches = []
        for record in records:
            if "embedding" not in record:
                continue
            score = self._combined_score(node, record)
            matches.append(
                {
                    "label": record["label"],
                    "name": record["name"],
                    "score": round(score, 3),
                    "embedding_score": round(self._cosine(node["embedding"], record["embedding"]), 3),
                    "context_score": round(self._context_overlap(node["context"], record["context"]), 3),
                    "context": record["context"],
                }
            )
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[: self.TOP_K]

    def _combined_score(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        embedding_score = self._cosine(left.get("embedding", []), right.get("embedding", []))
        context_score = self._context_overlap(left.get("context", {}), right.get("context", {}))
        return self.EMBEDDING_WEIGHT * embedding_score + self.CONTEXT_WEIGHT * context_score

    @staticmethod
    def _context_overlap(left: dict[str, list[str]], right: dict[str, list[str]]) -> float:
        scores = []
        for key in ["scenarios", "capabilities", "components", "stores", "neighbors"]:
            left_set = {str(item).strip() for item in left.get(key, []) if str(item).strip()}
            right_set = {str(item).strip() for item in right.get(key, []) if str(item).strip()}
            if not left_set or not right_set:
                continue
            scores.append(len(left_set & right_set) / len(left_set | right_set))
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return max(0.0, min(1.0, dot / (left_norm * right_norm)))

    def _extract_patch_nodes(
        self,
        patch: dict[str, Any],
        requirement: str,
        features: ExtractedFeatures | None,
    ) -> list[dict[str, Any]]:
        context_base = {
            "scenarios": [features.domain] if features and features.domain else [],
            "capabilities": list(features.business_capabilities if features else []),
            "components": [],
            "stores": [],
            "neighbors": [],
        }
        nodes: list[dict[str, Any]] = []
        capability_specs = [item for item in patch.get("capabilities", []) if isinstance(item, dict)]
        for item in capability_specs:
            name = self._clean(item.get("name"))
            if not name:
                continue
            context = {
                **context_base,
                "capabilities": [name],
                "components": self._dedupe(item.get("components", [])),
                "stores": self._dedupe(item.get("stores", [])),
            }
            nodes.append(self._node("BusinessCapability", name, context, requirement))

        capability_by_component: dict[str, list[str]] = {}
        capability_by_store: dict[str, list[str]] = {}
        for item in capability_specs:
            cap_name = self._clean(item.get("name"))
            for component in self._dedupe(item.get("components", [])):
                capability_by_component.setdefault(component, []).append(cap_name)
            for store in self._dedupe(item.get("stores", [])):
                capability_by_store.setdefault(store, []).append(cap_name)

        edge_neighbors: dict[str, list[str]] = {}
        for edge in patch.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = self._clean(edge.get("source"))
            target = self._clean(edge.get("target"))
            if source and target:
                edge_neighbors.setdefault(source, []).append(target)
                edge_neighbors.setdefault(target, []).append(source)

        for component in self._dedupe(
            list(patch.get("components", []))
            + [item for spec in capability_specs for item in spec.get("components", [])]
            + list(edge_neighbors.keys())
        ):
            context = {
                **context_base,
                "capabilities": self._dedupe(capability_by_component.get(component, [])),
                "neighbors": self._dedupe(edge_neighbors.get(component, [])),
            }
            nodes.append(self._node("ArchitectureComponent", component, context, requirement))

        for store in self._dedupe(
            list(patch.get("stores", []))
            + [item for spec in capability_specs for item in spec.get("stores", [])]
        ):
            context = {
                **context_base,
                "capabilities": self._dedupe(capability_by_store.get(store, [])),
                "components": [
                    component
                    for component, capabilities in capability_by_component.items()
                    if set(capabilities) & set(capability_by_store.get(store, []))
                ],
            }
            nodes.append(self._node("DataStore", store, context, requirement))
        return self._dedupe_nodes(nodes)

    def _prepare_records(self, existing_records: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
        prepared = {label: [] for label in self.LABELS}
        for label in self.LABELS:
            for item in existing_records.get(label, []):
                if isinstance(item, str):
                    item = {"label": label, "name": item, "context": {}}
                name = self._clean(item.get("name"))
                if not name:
                    continue
                context = self._normalize_context(item.get("context", {}))
                prepared[label].append(self._node(label, name, context, ""))
        return prepared

    def _node(self, label: str, name: str, context: dict[str, Any], requirement: str) -> dict[str, Any]:
        normalized_context = self._normalize_context(context)
        node = {
            "label": label,
            "name": name,
            "context": normalized_context,
        }
        node["text"] = self._node_text(node, requirement)
        return node

    @staticmethod
    def _node_text(node: dict[str, Any], requirement: str) -> str:
        context = node["context"]
        parts = [
            f"节点类型: {node['label']}",
            f"节点名称: {node['name']}",
            "关联场景: " + "、".join(context.get("scenarios", [])),
            "关联能力: " + "、".join(context.get("capabilities", [])),
            "关联组件: " + "、".join(context.get("components", [])),
            "数据存储: " + "、".join(context.get("stores", [])),
            "上下游: " + "、".join(context.get("neighbors", [])),
        ]
        if requirement:
            parts.append("用户需求: " + requirement[:240])
        return "\n".join(parts)

    @staticmethod
    def _normalize_context(context: dict[str, Any]) -> dict[str, list[str]]:
        return {
            key: KnowledgeNormalizer._dedupe(context.get(key, []))
            for key in ["scenarios", "capabilities", "components", "stores", "neighbors"]
        }

    def _rewrite_patch(
        self,
        patch: dict[str, Any],
        decisions: dict[str, dict[str, str]],
        temporary: dict[str, set[str]],
        include_temporary: bool,
    ) -> dict[str, Any]:
        def canonical(label: str, value: Any) -> str:
            name = self._clean(value)
            if not name:
                return ""
            return decisions.get(label, {}).get(name, name)

        def allowed(label: str, value: Any) -> bool:
            return include_temporary or self._clean(value) not in temporary.get(label, set())

        capabilities = []
        for item in patch.get("capabilities", []):
            if not isinstance(item, dict):
                continue
            name = self._clean(item.get("name"))
            if not name or not allowed("BusinessCapability", name):
                continue
            capabilities.append(
                {
                    "name": canonical("BusinessCapability", name),
                    "components": self._dedupe(
                        canonical("ArchitectureComponent", component)
                        for component in item.get("components", [])
                        if allowed("ArchitectureComponent", component)
                    ),
                    "stores": self._dedupe(
                        canonical("DataStore", store)
                        for store in item.get("stores", [])
                        if allowed("DataStore", store)
                    ),
                }
            )

        components = self._dedupe(
            canonical("ArchitectureComponent", component)
            for component in patch.get("components", [])
            if allowed("ArchitectureComponent", component)
        )
        stores = self._dedupe(
            canonical("DataStore", store)
            for store in patch.get("stores", [])
            if allowed("DataStore", store)
        )
        edges = []
        for edge in patch.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = self._clean(edge.get("source"))
            target = self._clean(edge.get("target"))
            if not source or not target:
                continue
            if not allowed("ArchitectureComponent", source) or not allowed("ArchitectureComponent", target):
                continue
            edges.append(
                {
                    "source": canonical("ArchitectureComponent", source),
                    "target": canonical("ArchitectureComponent", target),
                    "label": self._clean(edge.get("label")) or "依赖",
                    "kind": "event" if edge.get("kind") == "event" else "sync",
                }
            )
        return {
            **patch,
            "capabilities": self._dedupe_capabilities(capabilities),
            "components": components,
            "stores": stores,
            "edges": self._dedupe_edges(edges),
        }

    @staticmethod
    def _report(
        node: dict[str, Any],
        canonical: str,
        action: str,
        score: float,
        reason: str,
        matches: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "label": node["label"],
            "original": node["name"],
            "canonical": canonical,
            "action": action,
            "score": round(score, 3),
            "reason": reason,
            "matches": matches or [],
        }

    @staticmethod
    def _empty_patch(patch: dict[str, Any]) -> dict[str, Any]:
        return {**patch, "capabilities": [], "components": [], "stores": [], "edges": []}

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _dedupe(items) -> list[str]:
        result = []
        for item in items:
            if item is None:
                continue
            clean = str(item).strip()
            if clean:
                result.append(clean)
        return list(dict.fromkeys(result))

    @staticmethod
    def _dedupe_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        result = []
        for node in nodes:
            key = (node["label"], node["name"])
            if key in seen:
                continue
            seen.add(key)
            result.append(node)
        return result

    @staticmethod
    def _dedupe_capabilities(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in items:
            name = item["name"]
            if name not in merged:
                merged[name] = {"name": name, "components": [], "stores": []}
            merged[name]["components"] = KnowledgeNormalizer._dedupe(
                merged[name]["components"] + item.get("components", [])
            )
            merged[name]["stores"] = KnowledgeNormalizer._dedupe(
                merged[name]["stores"] + item.get("stores", [])
            )
        return list(merged.values())

    @staticmethod
    def _dedupe_edges(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        result = []
        for edge in items:
            key = (edge["source"], edge["target"], edge["label"], edge["kind"])
            if key in seen:
                continue
            seen.add(key)
            result.append(edge)
        return result
