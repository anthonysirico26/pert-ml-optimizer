import pytest
from models.pert import Task, PERTEngine


def test_expected_duration():
    task = Task("t1", "Design", optimistic=2, most_likely=4, pessimistic=9, dependencies=[])
    expected = (2 + 4 * 4 + 9) / 6
    assert abs(task.expected_duration - expected) < 1e-6


def test_variance():
    task = Task("t1", "Design", optimistic=2, most_likely=4, pessimistic=9, dependencies=[])
    expected_var = ((9 - 2) / 6) ** 2
    assert abs(task.variance - expected_var) < 1e-6


def test_risk_score_bounded():
    task = Task("t1", "Design", optimistic=1, most_likely=5, pessimistic=20, dependencies=[])
    assert 0 <= task.risk_score <= 1


def test_critical_path_linear():
    tasks = [
        Task("A", "Task A", optimistic=1, most_likely=2, pessimistic=3),
        Task("B", "Task B", optimistic=2, most_likely=3, pessimistic=4, dependencies=["A"]),
        Task("C", "Task C", optimistic=1, most_likely=2, pessimistic=3, dependencies=["B"]),
    ]
    engine = PERTEngine(tasks)
    engine.analyze()
    assert set(engine.critical_path) == {"A", "B", "C"}


def test_critical_path_parallel():
    tasks = [
        Task("A", "Short path", optimistic=1, most_likely=1, pessimistic=1),
        Task("B", "Long path",  optimistic=5, most_likely=5, pessimistic=5),
        Task("C", "Final task", optimistic=1, most_likely=1, pessimistic=1, dependencies=["A", "B"]),
    ]
    engine = PERTEngine(tasks)
    engine.analyze()

    assert "B" in engine.critical_path
    assert "C" in engine.critical_path
    assert "A" not in engine.critical_path


def test_project_duration():
    tasks = [
        Task("A", "Task A", optimistic=3, most_likely=3, pessimistic=3),
        Task("B", "Task B", optimistic=2, most_likely=2, pessimistic=2, dependencies=["A"]),
    ]
    engine = PERTEngine(tasks)
    engine.analyze()
    assert abs(engine.project_duration - 5.0) < 1e-6


def test_single_task():
    tasks = [Task("A", "Solo", optimistic=1, most_likely=3, pessimistic=5)]
    engine = PERTEngine(tasks)
    results = engine.analyze()
    assert "A" in results
    assert results["A"].is_critical
    assert abs(results["A"].slack) < 1e-6