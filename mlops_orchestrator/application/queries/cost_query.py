from __future__ import annotations
from mlops_orchestrator.domain.ports.infrastructure_ports import CostManagementPort


class CostQuery:
    """Query FinOps cost metrics and optimization recommendations."""

    def __init__(self, cost_port: CostManagementPort) -> None:
        self._cost_port = cost_port

    async def get_project_metrics(self, project_id: str) -> dict[str, float]:
        metrics = await self._cost_port.get_project_metrics(project_id)
        return {
            "cost_per_tb_scanned": metrics.cost_per_tb_scanned,
            "cost_per_1000_queries": metrics.cost_per_1000_queries,
            "cost_per_user": metrics.cost_per_user,
            "gpu_idle_pct": metrics.gpu_idle_pct,
        }

    async def get_recommendations(self, project_id: str) -> list[dict[str, object]]:
        recs = await self._cost_port.get_recommendations(project_id)
        return [
            {
                "type": r.recommendation_type,
                "description": r.description,
                "estimated_savings": r.estimated_savings,
                "priority": r.priority,
            }
            for r in recs
        ]
