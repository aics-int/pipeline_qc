from dataclasses import dataclass
from enum import Enum

class ResultStatus(Enum):
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass
class SegmentationResult:
    fov_id: int
    status: ResultStatus
    message: str = ""

    def __str__(self):
        return f"FOV {self.fov_id}: {self.status.value}\n{self.message}"