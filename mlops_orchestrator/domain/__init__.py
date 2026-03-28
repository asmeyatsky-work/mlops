"""
MLOps Orchestrator Domain Layer.

Architectural Intent:
- Contains all business rules, entities, value objects, domain events, and service logic
- ZERO dependencies on infrastructure, frameworks, or AI providers
- All external dependencies abstracted behind Protocol-based ports
- Domain models are immutable (frozen dataclasses)
"""
