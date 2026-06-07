from __future__ import annotations

import os
import json
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.models.schemas import ArchitectureStyle


class Neo4jAuraService:
    """Optional Neo4j AuraDB adapter.

    The app keeps working without this dependency/configuration. When AuraDB is
    configured, this adapter can sync the architecture knowledge graph and read
    graph data back from the cloud database.
    """

    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI", "")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.vector_top_k = int(os.getenv("NEO4J_TOPOLOGY_VECTOR_TOP_K", "6"))

    @property
    def configured(self) -> bool:
        return bool(self.uri and self.user and self.password)

    def status(self) -> dict[str, Any]:
        base = {
            "configured": self.configured,
            "uri": self.uri,
            "database": self.database,
            "mode": "auradb" if self.configured else "optional",
        }
        if not self.configured:
            return base
        try:
            with self._driver() as driver:
                driver.verify_connectivity()
            return {**base, "ok": True, "error": None}
        except Exception as exc:
            return {**base, "ok": False, "error": str(exc)}

    def sync_styles(self, styles: Iterable[ArchitectureStyle]) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "error": "Neo4j AuraDB is not configured."}

        styles = list(styles)
        with self._driver() as driver:
            with driver.session(database=self.database) as session:
                self._create_constraints(session)
                for style in styles:
                    session.execute_write(self._merge_style, style)
        return {"ok": True, "styles_synced": len(styles)}

    def fetch_styles(self) -> list[ArchitectureStyle]:
        if not self.configured:
            return []

        query = """
        MATCH (s:ArchitectureStyle)
        OPTIONAL MATCH (s)-[:SUITABLE_FOR]->(scenario:Scenario)
        OPTIONAL MATCH (s)-[score:HAS_SCORE]->(quality:QualityAttribute)
        RETURN s,
               collect(DISTINCT scenario.name) AS suitable_for,
               collect(DISTINCT {name: quality.name, value: score.value}) AS quality_scores
        ORDER BY s.id
        """
        styles: list[ArchitectureStyle] = []
        with self._driver() as driver:
            with driver.session(database=self.database) as session:
                for record in session.run(query):
                    node = record["s"]
                    missing = [
                        field
                        for field in ["id", "name", "category", "description", "strengths", "weaknesses", "topology", "rules_json"]
                        if node.get(field) is None
                    ]
                    if missing:
                        raise ValueError(f"Neo4j ArchitectureStyle missing required fields: {node.get('id')}, {missing}")
                    quality_payload = {
                        item["name"]: float(item["value"])
                        for item in record.get("quality_scores", [])
                        if item and item.get("name") and item.get("value") is not None
                    }
                    styles.append(
                        ArchitectureStyle(
                            id=node["id"],
                            name=node["name"],
                            category=node["category"],
                            description=node["description"],
                            suitable_for=[item for item in record.get("suitable_for", []) if item],
                            quality_scores=quality_payload,
                            strengths=list(node["strengths"]),
                            weaknesses=list(node["weaknesses"]),
                            topology=node["topology"],
                            rules=json.loads(node["rules_json"]),
                        )
                    )
        return styles

    def sync_domain_topology(self, knowledge_file: Path | None = None) -> dict[str, Any]:
        return self.rebuild_domain_topology(knowledge_file)

    def rebuild_domain_topology(self, knowledge_file: Path | None = None) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "error": "Neo4j AuraDB is not configured."}

        knowledge_file = knowledge_file or Path("data/domain_topology.json")
        data = json.loads(knowledge_file.read_text(encoding="utf-8"))
        with self._driver() as driver:
            with driver.session(database=self.database) as session:
                self._create_constraints(session)
                session.execute_write(self._reset_domain_topology)
                session.execute_write(self._merge_domain_topology, data)
        return {
            "ok": True,
            "mode": "rebuilt",
            "scenarios_synced": len(data.get("scenarios", [])),
            "capabilities_synced": len(data.get("capabilities", {})),
        }

    def sync_singleton_components(self, knowledge_file: Path | None = None) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "error": "Neo4j AuraDB is not configured."}

        knowledge_file = knowledge_file or Path("data/domain_topology.json")
        data = json.loads(knowledge_file.read_text(encoding="utf-8"))
        singletons = list(dict.fromkeys(data.get("singleton_components", [])))
        if not singletons:
            return {"ok": True, "singletons_synced": 0}

        query = """
        UNWIND $singletons AS name
        OPTIONAL MATCH (component:ArchitectureComponent {name: name})
        FOREACH (_ IN CASE WHEN component IS NULL THEN [] ELSE [1] END |
          SET component.singleton = true
        )
        WITH name
        OPTIONAL MATCH (store:DataStore {name: name})
        FOREACH (_ IN CASE WHEN store IS NULL THEN [] ELSE [1] END |
          SET store.singleton = true
        )
        """
        with self._driver() as driver:
            with driver.session(database=self.database) as session:
                session.run(query, singletons=singletons).consume()

        return {"ok": True, "singletons_synced": len(singletons)}

    def merge_topology_patch(
        self,
        requirement: str,
        keywords: list[str],
        patch: dict[str, Any],
        business_capabilities: list[str] | None = None,
        domain: str = "",
    ) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "error": "Neo4j AuraDB is not configured."}

        scenario_ids = [str(patch.get("scenario_id", "")).strip()] if str(patch.get("scenario_id", "")).strip() else []
        if not scenario_ids:
            scenario_ids = self._match_domain_scenarios(requirement, keywords, business_capabilities or [], domain)
        if not scenario_ids:
            scenario_ids = ["llm_learned"]
        scenario_name = patch.get("scenario_name") or domain or "LLM 补全场景"
        try:
            with self._driver() as driver:
                with driver.session(database=self.database) as session:
                    self._create_constraints(session)
                    session.execute_write(self._merge_topology_patch, scenario_ids, scenario_name, patch)
            return {
                "ok": True,
                "scenario_ids": scenario_ids,
                "capabilities": len(patch.get("capabilities", [])),
                "components": len(patch.get("components", [])),
                "stores": len(patch.get("stores", [])),
                "edges": len(patch.get("edges", [])),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def fetch_topology_node_names(self) -> dict[str, list[str]]:
        if not self.configured:
            return {"BusinessCapability": [], "ArchitectureComponent": [], "DataStore": []}

        query = """
        MATCH (n)
        WHERE n:BusinessCapability OR n:ArchitectureComponent OR n:DataStore
        RETURN labels(n) AS labels, n.name AS name
        """
        names = {"BusinessCapability": [], "ArchitectureComponent": [], "DataStore": []}
        try:
            with self._driver() as driver:
                with driver.session(database=self.database) as session:
                    for record in session.run(query):
                        name = record.get("name")
                        labels = record.get("labels") or []
                        if not name:
                            continue
                        for label in names:
                            if label in labels:
                                names[label].append(name)
            return {label: self._dedupe(items) for label, items in names.items()}
        except Exception:
            return {"BusinessCapability": [], "ArchitectureComponent": [], "DataStore": []}

    def fetch_topology_node_records(self) -> dict[str, list[dict[str, Any]]]:
        if not self.configured:
            return {"BusinessCapability": [], "ArchitectureComponent": [], "DataStore": []}

        queries = {
            "BusinessCapability": """
            MATCH (n:BusinessCapability)
            OPTIONAL MATCH (scenario:DomainScenario)-[:REQUIRES]->(n)
            OPTIONAL MATCH (n)-[:IMPLEMENTED_BY]->(component:ArchitectureComponent)
            OPTIONAL MATCH (n)-[:USES_STORE]->(store:DataStore)
            RETURN n.name AS name,
                   collect(DISTINCT scenario.name) AS scenarios,
                   collect(DISTINCT component.name) AS components,
                   collect(DISTINCT store.name) AS stores,
                   [] AS capabilities,
                   [] AS neighbors
            """,
            "ArchitectureComponent": """
            MATCH (n:ArchitectureComponent)
            OPTIONAL MATCH (capability:BusinessCapability)-[:IMPLEMENTED_BY]->(n)
            OPTIONAL MATCH (scenario:DomainScenario)-[:REQUIRES]->(capability)
            OPTIONAL MATCH (n)-[:STORES_IN]->(store:DataStore)
            OPTIONAL MATCH (n)-[:DEPENDS_ON]-(neighbor:ArchitectureComponent)
            RETURN n.name AS name,
                   collect(DISTINCT scenario.name) AS scenarios,
                   collect(DISTINCT capability.name) AS capabilities,
                   [] AS components,
                   collect(DISTINCT store.name) AS stores,
                   collect(DISTINCT neighbor.name) AS neighbors
            """,
            "DataStore": """
            MATCH (n:DataStore)
            OPTIONAL MATCH (capability:BusinessCapability)-[:USES_STORE]->(n)
            OPTIONAL MATCH (component:ArchitectureComponent)-[:STORES_IN]->(n)
            OPTIONAL MATCH (scenario:DomainScenario)-[:REQUIRES]->(capability)
            RETURN n.name AS name,
                   collect(DISTINCT scenario.name) AS scenarios,
                   collect(DISTINCT capability.name) AS capabilities,
                   collect(DISTINCT component.name) AS components,
                   [] AS stores,
                   [] AS neighbors
            """,
        }
        records = {"BusinessCapability": [], "ArchitectureComponent": [], "DataStore": []}
        try:
            with self._driver() as driver:
                with driver.session(database=self.database) as session:
                    for label, query in queries.items():
                        for record in session.run(query):
                            name = record.get("name")
                            if not name:
                                continue
                            records[label].append(
                                {
                                    "label": label,
                                    "name": name,
                                    "context": {
                                        "scenarios": self._dedupe(list(record.get("scenarios", []) or [])),
                                        "capabilities": self._dedupe(list(record.get("capabilities", []) or [])),
                                        "components": self._dedupe(list(record.get("components", []) or [])),
                                        "stores": self._dedupe(list(record.get("stores", []) or [])),
                                        "neighbors": self._dedupe(list(record.get("neighbors", []) or [])),
                                    },
                                }
                            )
            return records
        except Exception:
            return {"BusinessCapability": [], "ArchitectureComponent": [], "DataStore": []}

    def upsert_topology_embeddings(self, records: list[dict[str, Any]], dimension: int) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "error": "Neo4j AuraDB is not configured."}
        if dimension <= 0:
            raise ValueError("Neo4j topology vector dimension must be positive.")
        allowed_labels = {"BusinessCapability", "ArchitectureComponent", "DataStore"}
        grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in allowed_labels}
        for record in records:
            label = str(record.get("label", "")).strip()
            name = str(record.get("name", "")).strip()
            embedding = record.get("embedding")
            if label not in allowed_labels or not name:
                raise ValueError(f"Invalid Neo4j topology embedding record: {record}")
            if not isinstance(embedding, list) or len(embedding) != dimension:
                raise ValueError(f"Invalid Neo4j topology embedding vector for {label}:{name}")
            grouped[label].append(
                {
                    "name": name,
                    "text": str(record.get("text", "")),
                    "embedding": [float(item) for item in embedding],
                }
            )

        with self._driver() as driver:
            with driver.session(database=self.database) as session:
                self._create_constraints(session)
                self._create_vector_indexes(session, dimension)
                for label, items in grouped.items():
                    if items:
                        session.execute_write(self._set_topology_embeddings, label, items)
        return {"ok": True, "vectors_indexed": len(records), "dimension": dimension}

    def retrieve_topology_knowledge(
        self,
        requirement: str,
        keywords: list[str],
        qualities: dict[str, float],
        business_capabilities: list[str] | None = None,
        domain: str = "",
        query_embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        if not self.configured:
            return {"components": [], "stores": [], "edges": [], "scenarios": [], "capabilities": []}

        active_qualities = [name for name, value in qualities.items() if value >= 0.6]
        business_capabilities = [item for item in (business_capabilities or []) if item]
        scenario_ids = [] if business_capabilities else self._match_domain_scenarios("", keywords, [], domain)
        scenario_query = """
        MATCH (scenario:DomainScenario)
        WHERE scenario.id IN $scenario_ids
        OPTIONAL MATCH (scenario)-[requires:REQUIRES]->(cap:BusinessCapability)
        WITH collect(DISTINCT scenario.name) AS scenarios,
             collect(DISTINCT cap.name) AS scenario_capabilities
        WITH scenarios, scenario_capabilities,
             CASE
               WHEN size($business_capabilities) = 0 THEN scenario_capabilities
               ELSE [cap IN scenario_capabilities WHERE cap IN $business_capabilities]
             END AS selected_capabilities
        OPTIONAL MATCH (cap:BusinessCapability)
        WHERE cap.name IN selected_capabilities
        OPTIONAL MATCH (cap)-[impl:IMPLEMENTED_BY]->(component:ArchitectureComponent)
        WHERE impl.scenario_id IN $scenario_ids OR impl.scenario_id IS NULL
        OPTIONAL MATCH (cap)-[uses:USES_STORE]->(store:DataStore)
        WHERE uses.scenario_id IN $scenario_ids OR uses.scenario_id IS NULL
        RETURN scenarios,
               selected_capabilities AS capabilities,
               collect(DISTINCT component.name) AS components,
               collect(DISTINCT store.name) AS stores
        """
        capability_query = """
        MATCH (cap:BusinessCapability)
        WHERE cap.name IN $business_capabilities
        OPTIONAL MATCH (cap)-[impl:IMPLEMENTED_BY]->(component:ArchitectureComponent)
        WHERE size($scenario_ids) = 0 OR impl.scenario_id IN $scenario_ids OR impl.scenario_id IS NULL
        OPTIONAL MATCH (cap)-[uses:USES_STORE]->(store:DataStore)
        WHERE size($scenario_ids) = 0 OR uses.scenario_id IN $scenario_ids OR uses.scenario_id IS NULL
        RETURN collect(DISTINCT cap.name) AS capabilities,
               collect(DISTINCT component.name) AS components,
               collect(DISTINCT store.name) AS stores
        """
        quality_query = """
        MATCH (q:QualityAttribute)-[:REQUIRES_COMPONENT]->(component:ArchitectureComponent)
        WHERE q.name IN $active_qualities
        RETURN collect(DISTINCT component.name) AS components
        """
        # Combine scenario, explicit capability, quality, and vector matches before fetching edges.
        edge_query = """
        MATCH (a)-[r:DEPENDS_ON|STORES_IN]->(b)
        WHERE a.name IN $components
          AND b.name IN $targets
          AND (size($scenario_ids) = 0 OR r.scenario_id IN $scenario_ids OR r.capability IN $capabilities)
        RETURN collect(DISTINCT {source: a.name, target: b.name, label: coalesce(r.label, "依赖"), kind: coalesce(r.kind, "sync")}) AS edges
        """
        try:
            with self._driver() as driver:
                with driver.session(database=self.database) as session:
                    scenario_record = session.run(
                        scenario_query,
                        scenario_ids=scenario_ids,
                        business_capabilities=business_capabilities,
                    ).single()
                    capability_record = session.run(
                        capability_query,
                        scenario_ids=scenario_ids,
                        business_capabilities=business_capabilities,
                    ).single()
                    quality_record = session.run(quality_query, active_qualities=active_qualities).single()
                    scenario_record = scenario_record or {}
                    capability_record = capability_record or {}
                    quality_record = quality_record or {}
                    vector_records: list[dict[str, Any]] = []
                    vector_error = ""
                    if query_embedding:
                        vector_records, vector_error = self._query_topology_vectors(session, query_embedding)
                    components = self._dedupe(
                        list(scenario_record.get("components", []) or [])
                        + list(capability_record.get("components", []) or [])
                        + list(quality_record.get("components", []) or [])
                        + [record["name"] for record in vector_records if record["label"] == "ArchitectureComponent"]
                    )
                    stores = self._dedupe(
                        list(scenario_record.get("stores", []) or [])
                        + list(capability_record.get("stores", []) or [])
                        + [record["name"] for record in vector_records if record["label"] == "DataStore"]
                    )
                    selected_capabilities = self._dedupe(
                        list(scenario_record.get("capabilities", []) or [])
                        + list(capability_record.get("capabilities", []) or [])
                        + [record["name"] for record in vector_records if record["label"] == "BusinessCapability"]
                    )
                    edge_record = session.run(
                        edge_query,
                        components=components,
                        targets=components + stores,
                        scenario_ids=scenario_ids,
                        capabilities=selected_capabilities,
                    ).single()
                    return {
                        "components": [item for item in components if item],
                        "stores": [item for item in stores if item],
                        "edges": [item for item in (edge_record["edges"] if edge_record else []) if item["source"] and item["target"]],
                        "scenarios": [item for item in scenario_record.get("scenarios", []) if item],
                        "capabilities": selected_capabilities,
                        "scenario_ids": scenario_ids,
                        "vector_matches": vector_records,
                        "vector_error": vector_error,
                    }
        except Exception:
            return {"components": [], "stores": [], "edges": [], "scenarios": [], "capabilities": []}

    @staticmethod
    def _match_domain_scenarios(
        requirement: str,
        keywords: list[str],
        business_capabilities: list[str] | None = None,
        domain: str = "",
    ) -> list[str]:
        text = requirement + " " + " ".join(keywords) + " " + " ".join(business_capabilities or []) + " " + domain
        rules = {
            "instant_messaging": ["即时通讯", "即时通信", "聊天", "消息", "万人在线", "同时在线", "视频通话", "长连接"],
            "social_media": ["社交", "发帖", "点赞", "评论", "私信", "内容推荐"],
            "online_education": ["在线教育", "上课", "课程", "直播", "录播", "回放", "课后互动"],
            "ecommerce": ["电商", "商品", "购物车", "下单", "支付", "促销", "订单", "退款", "秒杀", "物流"],
            "iot": ["物联网", "设备", "传感器", "采集", "远程控制", "告警"],
            "big_data": ["大数据", "TB", "实时计算", "离线分析", "数据可视化", "批量处理", "ETL"],
            "finance_payment": ["金融", "支付", "交易", "对账", "清算", "安全"],
            "game_backend": ["游戏", "玩家", "实时对战", "道具交易", "社交互动"],
            "healthcare": ["医院", "挂号", "患者", "科室", "医生"],
            "logistics": ["物流", "快递", "轨迹", "位置", "异常告警"],
            "short_video": ["短视频", "视频", "转码", "播放", "推荐算法"],
            "smart_city": ["智慧城市", "摄像头", "传感器", "实时监控", "应急指挥"],
            "blog": ["博客", "文章", "分类"],
            "supply_chain": ["供应链", "多企业", "订单跟踪", "库存管理"],
            "exam": ["在线考试", "考试", "防作弊", "自动阅卷", "成绩统计"],
            "industrial_control": ["工业", "生产线", "设备数据", "故障预警", "远程维护"],
            "library": ["图书", "借阅", "归还"],
            "travel_booking": ["旅游", "酒店", "机票", "门票", "预订"],
            "ai_image": ["AI 绘画", "提示词", "生成图片", "图片存储"],
            "lab_reagent": ["实验室", "试剂", "申领", "领用", "出库", "库存余量", "低值试剂", "采购提醒"],
        }
        scored = []
        for scenario_id, scenario_keywords in rules.items():
            hits = [keyword for keyword in scenario_keywords if keyword in text]
            if hits:
                scored.append((scenario_id, len(hits)))
        if not scored:
            return []
        scored.sort(key=lambda item: item[1], reverse=True)
        best_score = scored[0][1]
        return [scenario_id for scenario_id, score in scored if score == best_score or score >= 2][:3]

    @staticmethod
    def _dedupe(items: list[Any]) -> list[Any]:
        return list(dict.fromkeys(item for item in items if item))

    def fetch_graph(self) -> dict[str, list[dict[str, str]]]:
        if not self.configured:
            return {"nodes": [], "edges": []}

        query = """
        MATCH (s:ArchitectureStyle)
        OPTIONAL MATCH (s)-[r]->(n)
        RETURN s, collect({rel: type(r), node: n, value: r.value}) AS links
        """
        nodes: dict[str, dict[str, str]] = {}
        edges: list[dict[str, str]] = []
        with self._driver() as driver:
            with driver.session(database=self.database) as session:
                for record in session.run(query):
                    style = record["s"]
                    style_id = f"style:{style['id']}"
                    nodes[style_id] = {"id": style_id, "label": style["name"], "type": "architecture_style"}
                    for link in record["links"]:
                        node = link["node"]
                        relation = link["rel"]
                        if node is None or relation is None:
                            continue
                        node_id = self._node_id(node)
                        nodes[node_id] = {
                            "id": node_id,
                            "label": node.get("name", node.get("category", node_id)),
                            "type": next(iter(node.labels)).lower(),
                        }
                        rel_label = relation if link["value"] is None else f"{relation}:{link['value']}"
                        edges.append({"source": style_id, "target": node_id, "relation": rel_label})
        return {"nodes": list(nodes.values()), "edges": edges}

    def fetch_topology_graph(self) -> dict[str, list[dict[str, Any]]]:
        """Read the domain topology knowledge graph (scenarios/capabilities/components/stores)."""
        if not self.configured:
            return {"nodes": [], "edges": []}

        node_query = """
        MATCH (n)
        WHERE n:DomainScenario OR n:BusinessCapability OR n:ArchitectureComponent OR n:DataStore
        RETURN n
        """
        edge_query = """
        MATCH (a)-[r:REQUIRES|IMPLEMENTED_BY|USES_STORE|STORES_IN|DEPENDS_ON]->(b)
        WHERE (a:DomainScenario OR a:BusinessCapability OR a:ArchitectureComponent OR a:DataStore)
          AND (b:DomainScenario OR b:BusinessCapability OR b:ArchitectureComponent OR b:DataStore)
        RETURN a, type(r) AS rel, r.label AS label, r.kind AS kind, b
        """
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        def register(node) -> str:
            node_id = self._node_id(node)
            if node_id not in nodes:
                labels = list(node.labels)
                nodes[node_id] = {
                    "id": node_id,
                    "label": node.get("name") or node.get("id") or node_id,
                    "type": (labels[0].lower() if labels else "node"),
                }
            return node_id

        try:
            with self._driver() as driver:
                with driver.session(database=self.database) as session:
                    for record in session.run(node_query):
                        register(record["n"])
                    for record in session.run(edge_query):
                        source_id = register(record["a"])
                        target_id = register(record["b"])
                        relation = record["rel"]
                        edges.append(
                            {
                                "source": source_id,
                                "target": target_id,
                                "relation": relation,
                                "label": record.get("label") or relation,
                                "kind": record.get("kind") or "sync",
                            }
                        )
            return {"nodes": list(nodes.values()), "edges": edges}
        except Exception:
            return {"nodes": [], "edges": []}

    @contextmanager
    def _driver(self):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Please install neo4j driver: pip install -r requirements.txt") from exc

        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            yield driver
        finally:
            driver.close()

    @staticmethod
    def _create_constraints(session) -> None:
        constraints = [
            "CREATE CONSTRAINT architecture_style_id IF NOT EXISTS FOR (n:ArchitectureStyle) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT quality_name IF NOT EXISTS FOR (n:QualityAttribute) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT scenario_name IF NOT EXISTS FOR (n:Scenario) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (n:Category) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT domain_scenario_id IF NOT EXISTS FOR (n:DomainScenario) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT capability_name IF NOT EXISTS FOR (n:BusinessCapability) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT component_name IF NOT EXISTS FOR (n:ArchitectureComponent) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT datastore_name IF NOT EXISTS FOR (n:DataStore) REQUIRE n.name IS UNIQUE",
        ]
        for statement in constraints:
            session.run(statement)

    @staticmethod
    def _create_vector_indexes(session, dimension: int) -> None:
        # Fixed index names let reindexing stay idempotent across repeated syncs.
        index_specs = {
            "BusinessCapability": "topology_business_capability_embedding",
            "ArchitectureComponent": "topology_architecture_component_embedding",
            "DataStore": "topology_data_store_embedding",
        }
        for label, index_name in index_specs.items():
            session.run(
                f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {dimension},
                  `vector.similarity_function`: 'cosine'
                }}}}
                """
            )

    @staticmethod
    def _merge_style(tx, style: ArchitectureStyle) -> None:
        tx.run(
            """
            MERGE (s:ArchitectureStyle {id: $id})
            SET s.name = $name,
                s.category = $category,
                s.description = $description,
                s.strengths = $strengths,
                s.weaknesses = $weaknesses,
                s.topology = $topology,
                s.rules_json = $rules_json
            MERGE (c:Category {name: $category})
            MERGE (s)-[:BELONGS_TO]->(c)
            """,
            id=style.id,
            name=style.name,
            category=style.category,
            description=style.description,
            strengths=style.strengths,
            weaknesses=style.weaknesses,
            topology=style.topology,
            rules_json=json.dumps(style.rules, ensure_ascii=False, sort_keys=True),
        )
        for scenario in style.suitable_for:
            tx.run(
                """
                MATCH (s:ArchitectureStyle {id: $id})
                MERGE (sc:Scenario {name: $scenario})
                MERGE (s)-[:SUITABLE_FOR]->(sc)
                """,
                id=style.id,
                scenario=scenario,
            )
        for quality, score in style.quality_scores.items():
            tx.run(
                """
                MATCH (s:ArchitectureStyle {id: $id})
                MERGE (q:QualityAttribute {name: $quality})
                MERGE (s)-[r:HAS_SCORE]->(q)
                SET r.value = $score
                """,
                id=style.id,
                quality=quality,
                score=score,
            )

    @staticmethod
    def _set_topology_embeddings(tx, label: str, items: list[dict[str, Any]]) -> None:
        if label not in {"BusinessCapability", "ArchitectureComponent", "DataStore"}:
            raise ValueError(f"Unsupported topology vector label: {label}")
        tx.run(
            f"""
            UNWIND $items AS item
            MATCH (n:{label} {{name: item.name}})
            SET n.embedding = item.embedding,
                n.embedding_text = item.text,
                n.embedding_updated_at = datetime()
            """,
            items=items,
        )

    def _query_topology_vectors(self, session, embedding: list[float]) -> tuple[list[dict[str, Any]], str]:
        if not embedding:
            return [], ""
        index_specs = {
            "BusinessCapability": "topology_business_capability_embedding",
            "ArchitectureComponent": "topology_architecture_component_embedding",
            "DataStore": "topology_data_store_embedding",
        }
        matches: list[dict[str, Any]] = []
        errors: list[str] = []
        for label, index_name in index_specs.items():
            try:
                records = session.run(
                    """
                    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                    YIELD node, score
                    RETURN node.name AS name, score
                    """,
                    index_name=index_name,
                    top_k=self.vector_top_k,
                    embedding=embedding,
                )
                for record in records:
                    name = record.get("name")
                    if name:
                        matches.append({"label": label, "name": name, "score": round(float(record.get("score", 0)), 4)})
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        matches.sort(key=lambda item: item["score"], reverse=True)
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in matches:
            deduped.setdefault((item["label"], item["name"]), item)
        return list(deduped.values())[: self.vector_top_k], "；".join(errors)

    @staticmethod
    def _reset_domain_topology(tx) -> None:
        tx.run(
            """
            MATCH (n)
            WHERE n:DomainScenario
               OR n:BusinessCapability
               OR n:ArchitectureComponent
               OR n:DataStore
            DETACH DELETE n
            """
        )

    @staticmethod
    def _merge_domain_topology(tx, data: dict[str, Any]) -> None:
        capabilities = data.get("capabilities", {})
        component_layers = data.get("component_layers", {})
        singleton_components = set(data.get("singleton_components", []))
        capability_scenarios: dict[str, list[dict[str, str]]] = {}
        store_names = {
            store
            for spec in capabilities.values()
            for store in spec.get("stores", [])
        }

        # Scenario scoped relationships preserve why a capability/component pair was learned.
        for scenario in data.get("scenarios", []):
            tx.run(
                """
                MERGE (s:DomainScenario {id: $id})
                SET s.name = $name, s.keywords = $keywords
                """,
                id=scenario["id"],
                name=scenario["name"],
                keywords=scenario.get("keywords", []),
            )
            for capability_name in scenario.get("capabilities", []):
                capability_scenarios.setdefault(capability_name, []).append(
                    {"id": scenario["id"], "name": scenario["name"]}
                )
                tx.run(
                    """
                    MATCH (s:DomainScenario {id: $scenario_id})
                    MERGE (c:BusinessCapability {name: $capability})
                    MERGE (s)-[r:REQUIRES]->(c)
                    SET r.scenario_id = $scenario_id,
                        r.scenario_name = $scenario_name
                    """,
                    scenario_id=scenario["id"],
                    scenario_name=scenario["name"],
                    capability=capability_name,
                )

        for capability_name, spec in capabilities.items():
            tx.run("MERGE (:BusinessCapability {name: $name})", name=capability_name)
            scoped_scenarios = capability_scenarios.get(capability_name) or [{"id": "", "name": ""}]
            for component_name in spec.get("components", []):
                for scoped in scoped_scenarios:
                    tx.run(
                        """
                        MATCH (c:BusinessCapability {name: $capability})
                        MERGE (component:ArchitectureComponent {name: $component})
                        SET component.layer = $layer,
                            component.singleton = $singleton
                        MERGE (c)-[r:IMPLEMENTED_BY]->(component)
                        SET r.scenario_id = $scenario_id,
                            r.scenario_name = $scenario_name,
                            r.capability = $capability
                        """,
                        capability=capability_name,
                        component=component_name,
                        layer=component_layers.get(component_name, "业务服务层"),
                        singleton=component_name in singleton_components,
                        scenario_id=scoped["id"],
                        scenario_name=scoped["name"],
                    )
            for store_name in spec.get("stores", []):
                for scoped in scoped_scenarios:
                    tx.run(
                        """
                        MATCH (c:BusinessCapability {name: $capability})
                        MERGE (store:DataStore {name: $store})
                        SET store.singleton = $singleton
                        MERGE (c)-[uses:USES_STORE]->(store)
                        SET uses.scenario_id = $scenario_id,
                            uses.scenario_name = $scenario_name,
                            uses.capability = $capability
                        WITH c, store
                        MATCH (component:ArchitectureComponent)<-[impl:IMPLEMENTED_BY]-(c)
                        WHERE impl.scenario_id = $scenario_id OR impl.scenario_id = ""
                        MERGE (component)-[stores:STORES_IN]->(store)
                        SET stores.scenario_id = $scenario_id,
                            stores.scenario_name = $scenario_name,
                            stores.capability = $capability,
                            stores.label = "存储",
                            stores.kind = "sync"
                        """,
                        capability=capability_name,
                        store=store_name,
                        singleton=store_name in singleton_components,
                        scenario_id=scoped["id"],
                        scenario_name=scoped["name"],
                    )

        for quality, components in data.get("quality_components", {}).items():
            tx.run("MERGE (:QualityAttribute {name: $name})", name=quality)
            for component_name in components:
                tx.run(
                    """
                    MATCH (q:QualityAttribute {name: $quality})
                    MERGE (component:ArchitectureComponent {name: $component})
                    SET component.layer = $layer,
                        component.singleton = $singleton
                    MERGE (q)-[:REQUIRES_COMPONENT]->(component)
                    """,
                    quality=quality,
                    component=component_name,
                    layer=component_layers.get(component_name, "业务服务层"),
                    singleton=component_name in singleton_components,
                )

        for source, target, label in data.get("dependencies", []):
            if source.startswith("*") or target.startswith("*"):
                continue
            scoped = Neo4jAuraService._dependency_scopes(source, target, capabilities, capability_scenarios)
            if not scoped:
                continue
            for scope in scoped:
                if target in store_names:
                    tx.run(
                        """
                        MERGE (a:ArchitectureComponent {name: $source})
                        SET a.layer = $source_layer,
                            a.singleton = $source_singleton
                        MERGE (b:DataStore {name: $target})
                        SET b.singleton = $target_singleton
                        MERGE (a)-[r:STORES_IN]->(b)
                        SET r.label = $label,
                            r.kind = $kind,
                            r.scenario_id = $scenario_id,
                            r.scenario_name = $scenario_name,
                            r.capability = $capability
                        """,
                        source=source,
                        target=target,
                        label=label,
                        kind="event" if "事件" in label or "通知" in label else "sync",
                        source_layer=component_layers.get(source, "业务服务层"),
                        source_singleton=source in singleton_components,
                        target_singleton=target in singleton_components,
                        **scope,
                    )
                else:
                    tx.run(
                        """
                        MERGE (a:ArchitectureComponent {name: $source})
                        SET a.layer = $source_layer,
                            a.singleton = $source_singleton
                        MERGE (b:ArchitectureComponent {name: $target})
                        SET b.layer = $target_layer,
                            b.singleton = $target_singleton
                        MERGE (a)-[r:DEPENDS_ON]->(b)
                        SET r.label = $label,
                            r.kind = $kind,
                            r.scenario_id = $scenario_id,
                            r.scenario_name = $scenario_name,
                            r.capability = $capability
                        """,
                        source=source,
                        target=target,
                        label=label,
                        kind="event" if "事件" in label or "通知" in label else "sync",
                        source_layer=component_layers.get(source, "业务服务层"),
                        target_layer=component_layers.get(target, "业务服务层"),
                        source_singleton=source in singleton_components,
                        target_singleton=target in singleton_components,
                        **scope,
                    )

    @staticmethod
    def _merge_topology_patch(tx, scenario_ids: list[str], scenario_name: str, patch: dict[str, Any]) -> None:
        # Learned patches attach to one or more scenario scopes instead of global edges only.
        for scenario_id in scenario_ids:
            tx.run(
                """
                MERGE (s:DomainScenario {id: $id})
                SET s.name = coalesce(s.name, $name)
                """,
                id=scenario_id,
                name=scenario_name,
            )

        component_names = set(patch.get("components", []))
        store_names = set(patch.get("stores", []))
        for capability in patch.get("capabilities", []):
            if not isinstance(capability, dict) or not capability.get("name"):
                continue
            capability_name = capability["name"]
            component_names.update(capability.get("components", []))
            store_names.update(capability.get("stores", []))
            tx.run("MERGE (:BusinessCapability {name: $name})", name=capability_name)
            for scenario_id in scenario_ids:
                tx.run(
                    """
                    MATCH (s:DomainScenario {id: $scenario_id})
                    MATCH (c:BusinessCapability {name: $capability})
                    MERGE (s)-[r:REQUIRES]->(c)
                    SET r.scenario_id = $scenario_id,
                        r.scenario_name = $scenario_name
                    """,
                    scenario_id=scenario_id,
                    scenario_name=scenario_name,
                    capability=capability_name,
                )
            for component_name in capability.get("components", []):
                for scenario_id in scenario_ids:
                    tx.run(
                        """
                        MATCH (c:BusinessCapability {name: $capability})
                        MERGE (component:ArchitectureComponent {name: $component})
                        MERGE (c)-[r:IMPLEMENTED_BY]->(component)
                        SET r.scenario_id = $scenario_id,
                            r.scenario_name = $scenario_name,
                            r.capability = $capability
                        """,
                        capability=capability_name,
                        component=component_name,
                        scenario_id=scenario_id,
                        scenario_name=scenario_name,
                    )
            for store_name in capability.get("stores", []):
                for scenario_id in scenario_ids:
                    tx.run(
                        """
                        MATCH (c:BusinessCapability {name: $capability})
                        MERGE (store:DataStore {name: $store})
                        MERGE (c)-[uses:USES_STORE]->(store)
                        SET uses.scenario_id = $scenario_id,
                            uses.scenario_name = $scenario_name,
                            uses.capability = $capability
                        WITH c, store
                        MATCH (component:ArchitectureComponent)<-[impl:IMPLEMENTED_BY]-(c)
                        WHERE impl.scenario_id = $scenario_id
                        MERGE (component)-[stores:STORES_IN]->(store)
                        SET stores.scenario_id = $scenario_id,
                            stores.scenario_name = $scenario_name,
                            stores.capability = $capability,
                            stores.label = "存储",
                            stores.kind = "sync"
                        """,
                        capability=capability_name,
                        store=store_name,
                        scenario_id=scenario_id,
                        scenario_name=scenario_name,
                    )

        for component_name in component_names:
            tx.run("MERGE (:ArchitectureComponent {name: $name})", name=component_name)
        for store_name in store_names:
            tx.run("MERGE (:DataStore {name: $name})", name=store_name)

        for edge in patch.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if not source or not target:
                continue
            for scenario_id in scenario_ids:
                if target in store_names:
                    tx.run(
                        """
                        MERGE (a:ArchitectureComponent {name: $source})
                        MERGE (b:DataStore {name: $target})
                        MERGE (a)-[r:STORES_IN]->(b)
                        SET r.label = $label,
                            r.kind = $kind,
                            r.scenario_id = $scenario_id,
                            r.scenario_name = $scenario_name,
                            r.capability = $capability
                        """,
                        source=source,
                        target=target,
                        label=edge.get("label", "依赖"),
                        kind=edge.get("kind", "sync"),
                        scenario_id=scenario_id,
                        scenario_name=scenario_name,
                        capability=edge.get("capability", ""),
                    )
                else:
                    tx.run(
                        """
                        MERGE (a:ArchitectureComponent {name: $source})
                        MERGE (b:ArchitectureComponent {name: $target})
                        MERGE (a)-[r:DEPENDS_ON]->(b)
                        SET r.label = $label,
                            r.kind = $kind,
                            r.scenario_id = $scenario_id,
                            r.scenario_name = $scenario_name,
                            r.capability = $capability
                        """,
                        source=source,
                        target=target,
                        label=edge.get("label", "依赖"),
                        kind=edge.get("kind", "sync"),
                        scenario_id=scenario_id,
                        scenario_name=scenario_name,
                        capability=edge.get("capability", ""),
                    )

    @staticmethod
    def _dependency_scopes(
        source: str,
        target: str,
        capabilities: dict[str, Any],
        capability_scenarios: dict[str, list[dict[str, str]]],
    ) -> list[dict[str, str]]:
        scenario_caps: dict[str, dict[str, Any]] = {}
        for capability_name, spec in capabilities.items():
            names = set(spec.get("components", [])) | set(spec.get("stores", []))
            for scenario in capability_scenarios.get(capability_name, []):
                scenario_caps.setdefault(
                    scenario["id"],
                    {"scenario_id": scenario["id"], "scenario_name": scenario["name"], "source_caps": [], "target_caps": []},
                )
                if source in names:
                    scenario_caps[scenario["id"]]["source_caps"].append(capability_name)
                if target in names:
                    scenario_caps[scenario["id"]]["target_caps"].append(capability_name)

        scopes: list[dict[str, str]] = []
        for item in scenario_caps.values():
            if not item["source_caps"] or not item["target_caps"]:
                continue
            scopes.append(
                {
                    "scenario_id": item["scenario_id"],
                    "scenario_name": item["scenario_name"],
                    "capability": item["source_caps"][0],
                }
            )
        return Neo4jAuraService._dedupe_scope_dicts(scopes)

    @staticmethod
    def _dedupe_scope_dicts(items: list[dict[str, str]]) -> list[dict[str, str]]:
        seen = set()
        result = []
        for item in items:
            key = (item.get("scenario_id", ""), item.get("capability", ""))
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _node_id(node) -> str:
        labels = list(node.labels)
        label = labels[0] if labels else "Node"
        key = node.get("id") or node.get("name") or node.get("category") or str(node.element_id)
        return f"{label.lower()}:{key}"
