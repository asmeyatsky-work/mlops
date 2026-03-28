from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, UTC
from uuid import uuid4

@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events."""
    aggregate_id: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
