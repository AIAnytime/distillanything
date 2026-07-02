"""Distill Anything — the open-source lifecycle framework for building specialized models.

Generate data from teachers -> distill students -> evaluate -> benchmark -> deploy.
"""

from distillanything.config import (
    DataConfig,
    DistillConfig,
    LossConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
)
from distillanything.student import Student

__version__ = "0.1.0"

__all__ = [
    "Student",
    "DistillConfig",
    "TeacherConfig",
    "StudentConfig",
    "DataConfig",
    "LossConfig",
    "TrainConfig",
    "__version__",
]
