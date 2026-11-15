from dataclasses import dataclass


@dataclass(frozen=True)
class FaceEnrollment:
    name: str
    sample_count: int
