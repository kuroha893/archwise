from __future__ import annotations

from app.models.schemas import ArchitectureStyle, CandidateEvaluation, ExtractedFeatures


class ArchitectureMatcherAgent:
    """Agent 2: match extracted features with architecture style knowledge."""

    ATTRIBUTE_MAP = {
        "concurrency": ["scalability", "performance"],
        "realtime": ["realtime", "performance"],
        "reliability": ["reliability"],
        "scalability": ["scalability", "modifiability"],
        "data_intensity": ["performance", "modifiability"],
        "ai_reasoning": ["modifiability"],
    }

    FLOW_BONUS = {
        "event_stream": {"event_driven": 0.22, "microservices": 0.12, "cqrs": 0.08, "serverless": 0.05},
        "pipeline": {"pipe_filter": 0.28, "serverless": 0.08, "microservices": 0.06},
        "transactional": {"layered": 0.08, "microservices": 0.06, "cqrs": 0.12, "soa": 0.06},
        "request_response": {"layered": 0.1, "mvc": 0.08, "hexagonal": 0.05},
    }

    FLOW_PRIORITY = {
        "event_stream": {"event_driven": 4, "microservices": 3, "cqrs": 2, "serverless": 1},
        "pipeline": {"pipe_filter": 4, "serverless": 2, "microservices": 1},
        "transactional": {"cqrs": 3, "layered": 2, "microservices": 1},
        "request_response": {"layered": 3, "mvc": 2, "hexagonal": 1},
    }

    def match(
        self,
        features: ExtractedFeatures,
        styles: list[ArchitectureStyle],
        top_k: int = 3,
        preferred_style_ids: list[str] | None = None,
        rejected_style_ids: list[str] | None = None,
    ) -> list[CandidateEvaluation]:
        preferred_style_ids = preferred_style_ids or []
        rejected_style_ids = rejected_style_ids or []
        candidates = [
            self._score_style(features, style, preferred_style_ids, rejected_style_ids)
            for style in styles
        ]
        priority = self.FLOW_PRIORITY.get(features.data_flow, {})
        candidates.sort(key=lambda item: (item.raw_score, priority.get(item.style_id, 0)), reverse=True)
        selected = candidates[:top_k]
        self._normalize_scores(selected)
        return selected

    def _score_style(
        self,
        features: ExtractedFeatures,
        style: ArchitectureStyle,
        preferred_style_ids: list[str],
        rejected_style_ids: list[str],
    ) -> CandidateEvaluation:
        score = 0.0
        reasons: list[str] = []
        risks: list[str] = []
        deductions: list[str] = []

        for feature, weight in features.quality_attributes.items():
            for quality in self.ATTRIBUTE_MAP.get(feature, []):
                score += weight * style.quality_scores.get(quality, 0) * 0.18
                if weight >= 0.65 and style.quality_scores.get(quality, 0) >= 0.7:
                    reasons.append(f"{style.name} 的 {quality} 能力契合需求中的 {feature} 信号")

        keywords_text = " ".join(features.keywords)
        for keyword in style.rules.get("prefer", []):
            if keyword.lower() in keywords_text.lower():
                score += 0.08
                reasons.append(f"命中知识库适用关键词：{keyword}")

        for keyword in style.rules.get("avoid", []):
            if keyword.lower() in keywords_text.lower():
                score -= 0.07
                risks.append(f"需求包含与该风格不完全匹配的信号：{keyword}")

        flow_bonus = self.FLOW_BONUS.get(features.data_flow, {}).get(style.id, 0)
        if flow_bonus:
            score += flow_bonus
            reasons.append(f"数据流类型 {features.data_flow} 与 {style.name} 匹配")

        if features.constraints.get("requires_future_extension") and style.quality_scores.get("modifiability", 0) >= 0.78:
            score += 0.08
            reasons.append("后续扩展诉求与该风格的可修改性匹配")

        if features.constraints.get("requires_high_availability") and style.quality_scores.get("reliability", 0) < 0.65:
            score -= 0.06
            risks.append("可靠性目标较高，但该风格需要额外高可用设计弥补")
            deductions.append("可靠性能力低于高可用诉求")

        if style.quality_scores.get("complexity", 1) < 0.45:
            risks.append("实现与运维复杂度较高，需要配套治理能力")
            deductions.append("工程治理复杂度较高")

        if style.id in preferred_style_ids:
            score += 0.14
            reasons.append("规则引擎或知识图谱将该风格列为优先候选")

        if style.id in rejected_style_ids:
            score -= 0.35
            risks.append("规则引擎硬约束认为该风格不适合作为优先方案")
            deductions.append("触发规则引擎排除或降权")

        score += self._context_fit_adjustment(features, style, deductions, risks)

        if not reasons:
            reasons.append("作为基线候选，用于与其他架构风格进行对比")

        return CandidateEvaluation(
            style_id=style.id,
            name=style.name,
            score=0,
            raw_score=round(score, 4),
            matched_reasons=list(dict.fromkeys(reasons))[:5],
            risks=list(dict.fromkeys(risks))[:4],
            deductions=list(dict.fromkeys(deductions))[:4],
            quality_scores=style.quality_scores,
        )

    def _context_fit_adjustment(
        self,
        features: ExtractedFeatures,
        style: ArchitectureStyle,
        deductions: list[str],
        risks: list[str],
    ) -> float:
        qualities = features.quality_attributes
        style_scores = style.quality_scores
        adjustment = 0.0

        # These adjustments encode cross-quality conflicts that raw style scores cannot express.
        if qualities.get("realtime", 0) >= 0.8 and style_scores.get("realtime", 0) < 0.6:
            adjustment -= 0.07 if style.id == "microservices" else 0.12
            deductions.append("强实时需求与架构实时能力不匹配")
            risks.append("实时链路需要额外组件补强")

        if qualities.get("concurrency", 0) >= 0.8 and style_scores.get("scalability", 0) < 0.65:
            adjustment -= 0.12
            deductions.append("高并发诉求下横向扩展能力不足")
            risks.append("高峰流量下可能需要额外拆分和缓存削峰")

        if qualities.get("reliability", 0) >= 0.75 and style_scores.get("reliability", 0) < 0.7:
            adjustment -= 0.08
            deductions.append("可靠性评分未达到关键业务阈值")

        if qualities.get("scalability", 0) >= 0.75 and style_scores.get("modifiability", 0) < 0.7:
            adjustment -= 0.07
            deductions.append("后续扩展和快速迭代支撑不足")

        if qualities.get("concurrency", 0) >= 0.8 and qualities.get("scalability", 0) >= 0.7 and style.id == "microservices":
            adjustment += 0.06

        capability_scope = len([item for item in features.business_capabilities if str(item).strip()])
        low_pressure_scope = (
            capability_scope
            and capability_scope <= 7
            and qualities.get("concurrency", 0) < 0.6
            and qualities.get("realtime", 0) < 0.6
            and qualities.get("scalability", 0) < 0.65
        )
        if low_pressure_scope and style.id == "microservices":
            adjustment -= 0.1
            deductions.append("业务能力范围较小，微服务拆分收益不足")
            risks.append("当前需求规模偏小，微服务会放大部署和治理成本")

        if features.data_flow == "event_stream" and style.id not in {"event_driven", "microservices", "cqrs", "serverless"}:
            adjustment -= 0.1
            deductions.append("事件流场景适配度不足")

        if features.data_flow == "pipeline" and style.id != "pipe_filter":
            adjustment -= 0.05
            deductions.append("流水线处理不是该风格核心优势")

        if features.data_flow == "transactional" and style.id in {"event_driven", "serverless", "blackboard"}:
            adjustment -= 0.06
            deductions.append("强事务场景需要额外一致性设计")

        simple_system = (
            qualities.get("concurrency", 0) <= 0.25
            and qualities.get("realtime", 0) <= 0.25
            and qualities.get("scalability", 0) <= 0.35
        )
        if simple_system and style.id in {"microservices", "event_driven", "cqrs", "soa", "clean"}:
            adjustment -= 0.16
            deductions.append("需求规模较小，采用该风格可能过度设计")
            risks.append("团队需要承担不必要的拆分和治理成本")

        if style_scores.get("complexity", 1) < 0.45 and qualities.get("scalability", 0) < 0.6:
            adjustment -= 0.05
            deductions.append("复杂度投入与当前扩展收益不成比例")

        return adjustment

    @staticmethod
    def _normalize_scores(candidates: list[CandidateEvaluation]) -> None:
        if not candidates:
            return

        raw_values = [item.raw_score for item in candidates]
        min_raw = min(raw_values)
        max_raw = max(raw_values)
        spread = max_raw - min_raw

        for index, item in enumerate(candidates):
            if spread < 0.08:
                score = 92 - index * 4.5
            else:
                relative = (item.raw_score - min_raw) / spread
                score = 72 + relative * 24
                score -= index * 1.2

            score -= min(len(item.deductions), 4) * 1.3
            score = max(55, min(98, score))
            item.score = round(score, 1)

        candidates.sort(key=lambda item: item.score, reverse=True)
        if len(candidates) == 1:
            candidates[0].recommendation_role = "核心推荐"
            candidates[0].confidence = "高"
            return

        top_score = candidates[0].score
        second_score = candidates[1].score
        gap = top_score - second_score

        for index, item in enumerate(candidates):
            if index == 0:
                item.recommendation_role = "核心推荐" if gap >= 3 else "核心推荐/组合候选"
                item.confidence = "高" if gap >= 8 else "中高" if gap >= 3 else "中"
            elif index == 1:
                item.recommendation_role = "备选方案" if gap >= 3 else "组合备选"
                item.confidence = "中高" if item.score >= 85 else "中"
            else:
                item.recommendation_role = "专项补充"
                item.confidence = "中" if item.score >= 78 else "中低"
