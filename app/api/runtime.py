from __future__ import annotations

import os

from app.knowledge.repository import KnowledgeRepository
from app.services.knowledge_client import KnowledgeServiceClient
from app.services.recommendation_service import RecommendationService

knowledge_service_url = os.getenv("KNOWLEDGE_SERVICE_URL", "").strip()

# This flag selects the service boundary used by the reasoning app.
if knowledge_service_url:
    knowledge_service = KnowledgeServiceClient(knowledge_service_url)
    recommendation_service = RecommendationService(knowledge_service=knowledge_service)
else:
    repository = KnowledgeRepository()
    recommendation_service = RecommendationService(repository)
    knowledge_service = recommendation_service.knowledge_service
