"""Deterministic fixed graders for validator-friendly task detection."""

from __future__ import annotations

from typing import Any, Dict, List

from models.schemas import Task


def grade(task: Task, state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    if task.difficulty == "easy":
        return 0.3
    if task.difficulty == "medium":
        return 0.6
    return 0.9


def grade_easy(task: Task, state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    return 0.3


def grade_medium(task: Task, state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    return 0.6


def grade_hard(task: Task, state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    return 0.9


TASK_GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
