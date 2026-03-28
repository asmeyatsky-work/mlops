from __future__ import annotations
from mlops_orchestrator.domain.entities.agent import Agent, AgentRole


class AgentRegistry:
    """In-memory registry of specialist agents."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.id] = agent

    def get_by_id(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def get_by_role(self, role: AgentRole) -> list[Agent]:
        return [a for a in self._agents.values() if a.role == role]

    def get_available(self) -> list[Agent]:
        return [a for a in self._agents.values() if a.status == "IDLE"]

    def get_for_capability(self, capability: str) -> list[Agent]:
        return [a for a in self._agents.values() if a.can_handle(capability)]

    def all_agents(self) -> list[Agent]:
        return list(self._agents.values())

    @classmethod
    def create_default_swarm(cls) -> AgentRegistry:
        """Create registry pre-populated with the 7 PRD-specified agent roles."""
        registry = cls()
        role_configs = {
            AgentRole.ORCHESTRATOR: (
                ("task_decomposition", "coordination", "aggregation"),
                ("create_dataset", "train_model", "deploy_to_vertex", "deploy_to_gke", "configure_monitoring"),
            ),
            AgentRole.ARCHITECT: (
                ("infrastructure", "cloud_region", "data_flow"),
                ("create_dataset",),
            ),
            AgentRole.DATA_ENGINEER: (
                ("etl", "sql", "data_validation", "bigquery"),
                ("create_dataset",),
            ),
            AgentRole.VALIDATION: (
                ("testing", "bias_analysis", "performance"),
                (),
            ),
            AgentRole.DEPLOYMENT: (
                ("kubernetes", "vertex_endpoint", "kserve"),
                ("deploy_to_vertex", "deploy_to_gke"),
            ),
            AgentRole.FINOPS: (
                ("cost_optimization", "right_sizing", "spot_instances"),
                (),
            ),
            AgentRole.SECURITY: (
                ("iam", "rbac", "metadata_audit", "vpc_sc"),
                (),
            ),
        }
        for role, (capabilities, tools) in role_configs.items():
            agent = Agent.create(role=role, capabilities=capabilities, permitted_tools=tools)
            registry.register(agent)
        return registry
