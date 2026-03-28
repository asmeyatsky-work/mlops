from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class MachineSpec:
    """Compute resource specification for training and serving."""
    machine_type: str = "n1-standard-4"
    accelerator_type: str = ""
    accelerator_count: int = 0
    replica_count: int = 1

    @property
    def has_gpu(self) -> bool:
        return self.accelerator_count > 0 and self.accelerator_type != ""
