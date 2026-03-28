from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class CostMetrics:
    """Aggregated cost metrics for a project or resource."""
    cost_per_tb_scanned: float = 0.0
    cost_per_1000_queries: float = 0.0
    cost_per_user: float = 0.0
    gpu_idle_pct: float = 0.0

    @property
    def has_waste(self) -> bool:
        return self.gpu_idle_pct > 30.0

@dataclass(frozen=True)
class CostBreakdown:
    """Itemized cost breakdown for a resource."""
    compute_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.compute_cost + self.storage_cost + self.network_cost

@dataclass(frozen=True)
class CostRecommendation:
    """A single FinOps optimization recommendation."""
    recommendation_type: str
    description: str
    estimated_savings: float
    priority: Literal["low", "medium", "high", "critical"]
