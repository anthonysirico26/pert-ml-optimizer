from typing import Literal
from pydantic import BaseModel, Field


class TaskFeatures(BaseModel):
    task_type: Literal["design", "development", "testing", "review", "deployment", "research"]
    complexity: Literal["low", "medium", "high"]
    team_size: int = Field(..., ge=1, le=20)
    dependency_count: int = Field(..., ge=0, le=20)
    is_external: int = Field(..., ge=0, le=1)
    optimistic: float = Field(..., gt=0)
    most_likely: float = Field(..., gt=0)
    pessimistic: float = Field(..., gt=0)

    @property
    def pert_expected(self) -> float:
        return (self.optimistic + 4 * self.most_likely + self.pessimistic) / 6

    @property
    def pert_variance(self) -> float:
        return ((self.pessimistic - self.optimistic) / 6) ** 2

    def to_feature_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "complexity": self.complexity,
            "team_size": self.team_size,
            "dependency_count": self.dependency_count,
            "is_external": self.is_external,
            "optimistic": self.optimistic,
            "most_likely": self.most_likely,
            "pessimistic": self.pessimistic,
            "pert_expected": self.pert_expected,
            "pert_variance": self.pert_variance,
        }


class PredictionResponse(BaseModel):
    pert_expected: float
    ml_prediction: float
    divergence_days: float
    divergence_pct: float
    risk_flag: bool
    interpretation: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"