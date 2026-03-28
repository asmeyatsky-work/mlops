"""
MLOps Orchestrator Application Layer.

Architectural Intent:
- Contains use cases (commands/queries), DTOs, session state, and orchestration
- Depends only on the domain layer
- Orchestrates domain objects via ports
- Uses DAG-based orchestration for multi-step workflows
"""
