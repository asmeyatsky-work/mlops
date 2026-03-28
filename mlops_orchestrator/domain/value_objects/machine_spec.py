from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class MachineSpec:
    """Compute resource specification for training and serving."""
    machine_type: str = "n1-standard-4"
    accelerator_type: str = ""
    accelerator_count: int = 0
    replica_count: int = 1

    def __post_init__(self) -> None:
        if self.accelerator_count < 0:
            raise ValueError(f"accelerator_count must be >= 0, got {self.accelerator_count}")
        if self.replica_count < 1:
            raise ValueError(f"replica_count must be >= 1, got {self.replica_count}")

    @property
    def has_gpu(self) -> bool:
        return self.accelerator_count > 0 and self.accelerator_type != ""
