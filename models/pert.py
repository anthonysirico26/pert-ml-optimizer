from dataclasses import dataclass, field
from typing import Dict, List
import math


@dataclass
class Task:
    task_id: str
    name: str
    optimistic: float
    most_likely: float
    pessimistic: float
    dependencies: List[str] = field(default_factory=list)

    @property
    def expected_duration(self) -> float:
        return (self.optimistic + 4 * self.most_likely + self.pessimistic) / 6

    @property
    def variance(self) -> float:
        return ((self.pessimistic - self.optimistic) / 6) ** 2

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)

    @property
    def risk_score(self) -> float:
        if self.expected_duration == 0:
            return 0.0
        return min(self.std_dev / self.expected_duration, 1.0)


@dataclass
class PERTResult:
    task_id: str
    expected_duration: float
    variance: float
    std_dev: float
    risk_score: float
    is_critical: bool = False
    early_start: float = 0.0
    early_finish: float = 0.0
    late_start: float = 0.0
    late_finish: float = 0.0
    slack: float = 0.0


class PERTEngine:
    def __init__(self, tasks: List[Task]):
        self.tasks: Dict[str, Task] = {t.task_id: t for t in tasks}
        self.results: Dict[str, PERTResult] = {}

    def analyze(self) -> Dict[str, PERTResult]:
        self._forward_pass()
        self._backward_pass()
        self._compute_slack()
        return self.results

    def _forward_pass(self):
        visited = set()

        def visit(task_id: str):
            if task_id in visited:
                return
            task = self.tasks[task_id]
            for dep in task.dependencies:
                visit(dep)

            early_start = 0.0
            for dep in task.dependencies:
                dep_result = self.results[dep]
                early_start = max(early_start, dep_result.early_finish)

            early_finish = early_start + task.expected_duration

            self.results[task_id] = PERTResult(
                task_id=task_id,
                expected_duration=task.expected_duration,
                variance=task.variance,
                std_dev=task.std_dev,
                risk_score=task.risk_score,
                early_start=early_start,
                early_finish=early_finish,
            )
            visited.add(task_id)

        for task_id in self.tasks:
            visit(task_id)

    def _backward_pass(self):
        project_duration = max(r.early_finish for r in self.results.values())

        for task_id, result in self.results.items():
            result.late_finish = project_duration

        order = list(self.results.keys())
        for task_id in reversed(order):
            task = self.tasks[task_id]
            result = self.results[task_id]
            result.late_start = result.late_finish - result.expected_duration

            for dep_id in task.dependencies:
                dep_result = self.results[dep_id]
                dep_result.late_finish = min(dep_result.late_finish, result.late_start)
                dep_result.late_start = dep_result.late_finish - dep_result.expected_duration

    def _compute_slack(self):
        for result in self.results.values():
            result.slack = result.late_start - result.early_start
            result.is_critical = abs(result.slack) < 1e-6

    @property
    def critical_path(self) -> List[str]:
        return [tid for tid, r in self.results.items() if r.is_critical]

    @property
    def project_duration(self) -> float:
        if not self.results:
            return 0.0
        return max(r.early_finish for r in self.results.values())

    @property
    def project_variance(self) -> float:
        return sum(self.results[tid].variance for tid in self.critical_path)

    @property
    def project_std_dev(self) -> float:
        return math.sqrt(self.project_variance)