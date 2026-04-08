"""Deterministic trajectory grader for the support decision environment."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from models.schemas import Task, contains_any


MIN_SCORE = 0.05
MAX_SCORE = 0.95


def _strict_score(value: float) -> float:
    return round(max(MIN_SCORE, min(MAX_SCORE, value)), 4)


def _band_score(value: float, low: float, high: float) -> float:
    span = high - low
    normalized = (value - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
    normalized = max(0.0, min(1.0, normalized))
    return round(low + span * normalized, 4)


def _unique_agent_issues(history: List[Dict[str, Any]]) -> Set[str]:
    issues: Set[str] = set()
    for event in history:
        if event.get("action_type") == "classify":
            issues.update(event.get("issues", []))
    return issues


def _resolution_coverage(task: Task, history: List[Dict[str, Any]]) -> float:
    resolution_text = " ".join(
        event.get("content", "") for event in history if event.get("action_type") == "resolve"
    )
    if not task.true_issues:
        return MIN_SCORE
    covered = 0
    for issue in task.true_issues:
        if contains_any(resolution_text, task.acceptable_resolutions.get(issue, [])):
            covered += 1
    return _strict_score(covered / len(task.true_issues))


def _clarification_coverage(task: Task, history: List[Dict[str, Any]]) -> float:
    ask_text = " ".join(event.get("content", "") for event in history if event.get("action_type") == "ask")
    ambiguous_cases = len(task.true_issues) > 1 or task.sentiment == "angry"
    if not ambiguous_cases:
        return MAX_SCORE
    covered = 0
    for issue in task.true_issues:
        if contains_any(ask_text, task.required_clarifications.get(issue, [])):
            covered += 1
    return _strict_score(covered / len(task.true_issues))


def grade(task: Task, state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    """Return a deterministic non-binary score in the strict interval (0, 1)."""

    true_issues = set(task.true_issues)
    detected = set(state.get("detected_issues", []))
    classified = _unique_agent_issues(history)
    false_positives = len((detected | classified) - true_issues)

    issue_recall = len(detected & true_issues) / max(1, len(true_issues))
    issue_precision = len(detected & true_issues) / max(1, len(detected))
    issue_score = 0.65 * issue_recall + 0.35 * issue_precision

    resolution_score = _resolution_coverage(task, history) if state.get("resolved") else MIN_SCORE
    clarification_score = _clarification_coverage(task, history)

    steps = int(state.get("time_elapsed", 0))
    target_steps = 3 if task.difficulty == "easy" else 5 if task.difficulty == "medium" else 6
    efficiency_score = _strict_score(1.0 - max(0, steps - target_steps) * 0.12)

    premature_resolve = any(
        event.get("action_type") == "resolve" and set(event.get("detected_before", [])) != true_issues
        for event in history
    )
    wrong_escalation = state.get("escalated") and not (
        task.sentiment == "angry" and task.customer_type == "premium" and "account_security" in true_issues
    )
    appropriate_escalation = state.get("escalated") and not wrong_escalation

    behavior_score = clarification_score
    if appropriate_escalation:
        behavior_score = min(MAX_SCORE, behavior_score + 0.12)
    if premature_resolve:
        behavior_score -= 0.22
    if wrong_escalation:
        behavior_score -= 0.18
    behavior_score -= min(0.25, false_positives * 0.08)
    behavior_score = _strict_score(behavior_score)

    total = (
        0.34 * issue_score
        + 0.34 * resolution_score
        + 0.17 * behavior_score
        + 0.15 * efficiency_score
    )
    return _strict_score(total)


def grade_easy(task: Task, state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    return _band_score(grade(task, state, history), 0.2, 0.4)


def grade_medium(task: Task, state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    return _band_score(grade(task, state, history), 0.5, 0.7)


def grade_hard(task: Task, state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    return _band_score(grade(task, state, history), 0.75, 0.95)


TASK_GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
