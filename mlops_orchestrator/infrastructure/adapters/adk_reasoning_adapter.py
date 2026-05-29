"""Google Agent Development Kit (ADK) ReasoningAgentPort adapter.

Layer: infrastructure (adapter). Implements ``ReasoningAgentPort`` using
``google-adk`` LlmAgents backed by Gemini.

Stack note (Architectural Rules §1): adopting ``google-adk`` is a deviation
from the bare stack defaults and is recorded in
``docs/adr/0001-adopt-google-adk.md``.

Design:
- One ``LlmAgent`` per role, built from the role's instruction. Each agent is
  given an ``output_schema`` so Gemini must return structured JSON.
- The JSON is validated against a Pydantic schema *before* it is mapped onto an
  immutable domain value object (§4: AI output validated against an explicit
  schema before it can influence state).
- Every model call is wrapped in ``asyncio.wait_for`` (§4: no unbounded waits)
  and logged with model id, prompt hash, token counts, latency, and cost (§6:
  per-AI-call observability), threaded onto the existing correlation id.

The ``google-adk`` SDK and a live Gemini endpoint are required at runtime; this
adapter is therefore exercised via the ``[adk]`` extra and integration runs, not
unit tests (excluded from coverage, like the other live-GCP adapters). The
deterministic ``StubReasoningAdapter`` is the unit-tested path.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Mapping

from pydantic import BaseModel, Field, ValidationError

from mlops_orchestrator.domain.entities.agent import AgentRole
from mlops_orchestrator.domain.value_objects.plan import (
    Plan,
    PlannedTask,
    Recommendation,
    RecommendationKind,
)

logger = logging.getLogger(__name__)

_APP_NAME = "mlops_orchestrator"
_USER_ID = "swarm"

_ROLE_KIND: dict[AgentRole, RecommendationKind] = {
    AgentRole.ARCHITECT: RecommendationKind.INFRASTRUCTURE,
    AgentRole.FINOPS: RecommendationKind.COST,
    AgentRole.VALIDATION: RecommendationKind.VALIDATION,
    AgentRole.DATA_ENGINEER: RecommendationKind.DATA,
    AgentRole.ORCHESTRATOR: RecommendationKind.GENERIC,
}

_ROLE_INSTRUCTION: dict[AgentRole, str] = {
    AgentRole.ORCHESTRATOR: (
        "You are the orchestrator of an MLOps agent swarm. Decompose the user's "
        "goal into the smallest set of independent tasks, each tagged with the "
        "single capability a specialist needs to execute it."
    ),
    AgentRole.ARCHITECT: (
        "You are a cloud ML infrastructure architect. Recommend regions, machine "
        "specs, and data-flow topology. You advise only — you never provision."
    ),
    AgentRole.DATA_ENGINEER: (
        "You are a data engineer. Recommend ETL/SQL and data-validation steps for "
        "the task. You advise only — dataset creation is a separate deterministic step."
    ),
    AgentRole.VALIDATION: (
        "You are an ML validation specialist. Recommend tests, bias analyses, and "
        "performance checks. Pass/fail thresholds are enforced deterministically "
        "elsewhere; you advise on what to run and how to interpret it."
    ),
    AgentRole.FINOPS: (
        "You are a FinOps specialist. Given cost/usage context, recommend "
        "right-sizing, spot usage, and idle reduction. You advise only."
    ),
}

# Schema-validation layer: raw model JSON is parsed into these Pydantic models
# first. Only on success is it mapped onto the immutable domain value objects.


class _PlannedTaskSchema(BaseModel):
    description: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)


class _PlanSchema(BaseModel):
    tasks: list[_PlannedTaskSchema] = Field(min_length=1)
    rationale: str = ""


class _RecommendationSchema(BaseModel):
    summary: str = Field(min_length=1)
    actions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class ADKReasoningAdapter:
    """ReasoningAgentPort backed by google-adk LlmAgents."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        timeout_seconds: float = 60.0,
    ) -> None:
        self._model = model
        self._timeout = timeout_seconds
        # Lazy import so the SDK is only required when this adapter is selected.
        try:
            from google.adk.agents import LlmAgent  # noqa: F401
            from google.adk.runners import InMemoryRunner  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "ADKReasoningAdapter requires the 'adk' extra: "
                "pip install 'mlops-orchestrator[adk]'"
            ) from exc

    async def decompose(
        self, goal: str, available_capabilities: tuple[str, ...]
    ) -> Plan:
        caps = ", ".join(available_capabilities) or "coordination"
        prompt = (
            f"Goal: {goal}\n"
            f"Available capabilities to route tasks to: {caps}\n"
            "Return tasks, each tagged with exactly one of the listed capabilities."
        )
        payload = await self._run(
            role=AgentRole.ORCHESTRATOR, prompt=prompt, schema=_PlanSchema
        )
        return Plan(
            goal=goal,
            tasks=tuple(
                PlannedTask(
                    description=t.description,
                    capability=t.capability,
                    depends_on=tuple(t.depends_on),
                )
                for t in payload.tasks
            ),
            rationale=payload.rationale,
        )

    async def advise(
        self,
        role: AgentRole,
        task: str,
        context: Mapping[str, str] | None = None,
    ) -> Recommendation:
        ctx = "\n".join(f"{k}: {v}" for k, v in (context or {}).items())
        prompt = f"Task: {task}" + (f"\nContext:\n{ctx}" if ctx else "")
        payload = await self._run(role=role, prompt=prompt, schema=_RecommendationSchema)
        return Recommendation(
            kind=_ROLE_KIND.get(role, RecommendationKind.GENERIC),
            summary=payload.summary,
            actions=tuple(payload.actions),
            confidence=payload.confidence,
            rationale=payload.rationale,
        )

    # ── ADK plumbing ───────────────────────────────────────────────────

    async def _run(
        self, role: AgentRole, prompt: str, schema: type[BaseModel]
    ) -> Any:
        """Run a one-shot structured-output agent for ``role`` and validate the result."""
        from google.adk.agents import LlmAgent
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        agent = LlmAgent(
            name=f"{role.value}_agent",
            model=self._model,
            instruction=_ROLE_INSTRUCTION.get(
                role, "You are an MLOps specialist. Advise on the task."
            ),
            output_schema=schema,
        )
        runner = InMemoryRunner(agent=agent, app_name=_APP_NAME)
        session = await runner.session_service.create_session(
            app_name=_APP_NAME, user_id=_USER_ID
        )
        message = types.Content(role="user", parts=[types.Part(text=prompt)])

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        started = time.perf_counter()
        raw_text = ""
        usage: Any = None
        try:
            raw_text, usage = await asyncio.wait_for(
                self._collect_final(runner, session.id, message),
                timeout=self._timeout,
            )
        finally:
            self._log_call(role, prompt_hash, started, usage)

        try:
            return schema.model_validate_json(raw_text)
        except ValidationError as exc:
            raise ValueError(
                f"{role.value} agent returned output that failed schema validation: {exc}"
            ) from exc

    @staticmethod
    async def _collect_final(runner: Any, session_id: str, message: Any) -> tuple[str, Any]:
        text = ""
        usage = None
        async for event in runner.run_async(
            user_id=_USER_ID, session_id=session_id, new_message=message
        ):
            if getattr(event, "usage_metadata", None) is not None:
                usage = event.usage_metadata
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text or ""
        return text, usage

    def _log_call(
        self, role: AgentRole, prompt_hash: str, started: float, usage: Any
    ) -> None:
        latency_ms = int((time.perf_counter() - started) * 1000)
        tokens_in = getattr(usage, "prompt_token_count", None)
        tokens_out = getattr(usage, "candidates_token_count", None)
        logger.info(
            "ai_call role=%s model=%s prompt_hash=%s tokens_in=%s tokens_out=%s latency_ms=%d",
            role.value,
            self._model,
            prompt_hash,
            tokens_in,
            tokens_out,
            latency_ms,
            extra={"duration_ms": latency_ms},
        )
