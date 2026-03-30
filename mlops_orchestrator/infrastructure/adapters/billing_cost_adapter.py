"""Real GCP cost adapter using BigQuery billing export and Cloud Billing Budgets API."""
from __future__ import annotations

import asyncio

from mlops_orchestrator.domain.value_objects.cost_metrics import (
    CostBreakdown,
    CostMetrics,
    CostRecommendation,
)
from mlops_orchestrator.infrastructure.adapters.retry import with_retry


class BigQueryCostAdapter:
    """Real cost adapter backed by BigQuery billing export.

    Implements CostManagementPort.

    Requires:
    - A BigQuery billing export table (standard or detailed usage cost).
    - The MLOPS_BILLING_TABLE env var set to the fully-qualified table
      (e.g. "project.dataset.gcp_billing_export_v1_XXXXXX").
    """

    def __init__(self, project: str, billing_table: str) -> None:
        self._project = project
        self._billing_table = billing_table
        from google.cloud import bigquery

        self._bq_client = bigquery.Client(project=project)

    @with_retry(max_attempts=3)
    async def get_resource_costs(
        self, resource_id: str, start_date: str, end_date: str
    ) -> CostBreakdown:
        from google.cloud import bigquery

        query = f"""
            SELECT
                IFNULL(SUM(CASE WHEN service.description LIKE '%Compute%' THEN cost ELSE 0 END), 0) AS compute_cost,
                IFNULL(SUM(CASE WHEN service.description LIKE '%Storage%' THEN cost ELSE 0 END), 0) AS storage_cost,
                IFNULL(SUM(CASE WHEN service.description LIKE '%Network%' THEN cost ELSE 0 END), 0) AS network_cost
            FROM `{self._billing_table}`
            WHERE usage_start_time >= @start_date
              AND usage_start_time < @end_date
              AND resource.name LIKE @resource_pattern
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
                bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
                bigquery.ScalarQueryParameter(
                    "resource_pattern", "STRING", f"%{resource_id}%"
                ),
            ]
        )
        result = await asyncio.to_thread(
            self._bq_client.query, query, job_config=job_config
        )
        rows = await asyncio.to_thread(lambda: list(result.result()))
        if rows:
            row = rows[0]
            return CostBreakdown(
                compute_cost=float(row.compute_cost),
                storage_cost=float(row.storage_cost),
                network_cost=float(row.network_cost),
            )
        return CostBreakdown()

    @with_retry(max_attempts=3)
    async def get_project_metrics(self, project_id: str) -> CostMetrics:
        from google.cloud import bigquery

        query = f"""
            SELECT
                IFNULL(SUM(CASE WHEN service.description LIKE '%BigQuery%' THEN cost ELSE 0 END)
                    / GREATEST(SUM(CASE WHEN service.description LIKE '%BigQuery%'
                        AND sku.description LIKE '%Byte%' THEN usage.amount ELSE 0 END) / 1e12, 1), 0)
                    AS cost_per_tb_scanned,
                IFNULL(SUM(CASE WHEN service.description LIKE '%BigQuery%' THEN cost ELSE 0 END)
                    / GREATEST(SUM(CASE WHEN service.description LIKE '%BigQuery%'
                        AND sku.description LIKE '%Quer%' THEN usage.amount ELSE 0 END) / 1000, 1), 0)
                    AS cost_per_1000_queries,
                IFNULL(SUM(cost), 0) AS total_cost
            FROM `{self._billing_table}`
            WHERE project.id = @project_id
              AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            ]
        )
        result = await asyncio.to_thread(
            self._bq_client.query, query, job_config=job_config
        )
        rows = await asyncio.to_thread(lambda: list(result.result()))

        # GPU idle percentage from Vertex AI workload metrics
        gpu_idle = await self._get_gpu_idle_pct(project_id)

        if rows:
            row = rows[0]
            return CostMetrics(
                cost_per_tb_scanned=float(row.cost_per_tb_scanned),
                cost_per_1000_queries=float(row.cost_per_1000_queries),
                cost_per_user=0.0,
                gpu_idle_pct=gpu_idle,
            )
        return CostMetrics(gpu_idle_pct=gpu_idle)

    @with_retry(max_attempts=3)
    async def get_recommendations(self, project_id: str) -> list[CostRecommendation]:
        from google.cloud import bigquery

        recommendations: list[CostRecommendation] = []

        # Check for idle GPU spend
        gpu_idle = await self._get_gpu_idle_pct(project_id)
        if gpu_idle > 30.0:
            recommendations.append(
                CostRecommendation(
                    recommendation_type="gpu_rightsizing",
                    description=f"GPU idle {gpu_idle:.0f}% over last 30 days. "
                    "Consider using preemptible/spot VMs or scaling down accelerator count.",
                    estimated_savings=gpu_idle * 0.5,
                    priority="high" if gpu_idle > 60 else "medium",
                )
            )

        # Check for unused endpoints
        query = f"""
            SELECT resource.name, SUM(cost) AS cost
            FROM `{self._billing_table}`
            WHERE project.id = @project_id
              AND service.description LIKE '%Vertex AI%'
              AND sku.description LIKE '%Endpoint%'
              AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            GROUP BY resource.name
            HAVING cost > 10
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            ]
        )
        result = await asyncio.to_thread(
            self._bq_client.query, query, job_config=job_config
        )
        rows = await asyncio.to_thread(lambda: list(result.result()))
        for row in rows:
            recommendations.append(
                CostRecommendation(
                    recommendation_type="idle_endpoint",
                    description=f"Endpoint {row.name} cost ${float(row.cost):.2f} in 7 days with possible low traffic.",
                    estimated_savings=float(row.cost) * 0.7,
                    priority="medium",
                )
            )

        return recommendations

    async def _get_gpu_idle_pct(self, project_id: str) -> float:
        """Query GPU utilization from billing data to estimate idle percentage."""
        from google.cloud import bigquery

        query = f"""
            SELECT
                IFNULL(100 - AVG(
                    CASE WHEN sku.description LIKE '%GPU%' AND usage.amount > 0
                    THEN usage.amount ELSE NULL END
                ) * 100, 0) AS gpu_idle_pct
            FROM `{self._billing_table}`
            WHERE project.id = @project_id
              AND service.description LIKE '%Compute%'
              AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("project_id", "STRING", project_id),
            ]
        )
        try:
            result = await asyncio.to_thread(
                self._bq_client.query, query, job_config=job_config
            )
            rows = await asyncio.to_thread(lambda: list(result.result()))
            if rows and rows[0].gpu_idle_pct is not None:
                return float(rows[0].gpu_idle_pct)
        except Exception:
            pass
        return 0.0
