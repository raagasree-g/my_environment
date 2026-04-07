"""Deterministic trajectory grader for the support decision environment."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from models.schemas import Task, contains_any


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
        return 0.0
    covered = 0
    for issue in task.true_issues:
        if contains_any(resolution_text, task.acceptable_resolutions.get(issue, [])):
            covered += 1
    return covered / len(task.true_issues)


def _clarification_coverage(task: Task, history: List[Dict[str, Any]]) -> float:
    ask_text = " ".join(event.get("content", "") for event in history if event.get("action_type") == "ask")
    ambiguous_cases = len(task.true_issues) > 1 or task.sentiment == "angry"
    if not ambiguous_cases:
        return 1.0
    covered = 0
    for issue in task.true_issues:
        if contains_any(ask_text, task.required_clarifications.get(issue, [])):
            covered += 1
    return covered / len(task.true_issues)


def grade(task: Task, state: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    """Return a deterministic score in [0, 1] based on the full trajectory."""

    true_issues = set(task.true_issues)
    detected = set(state.get("detected_issues", []))
    classified = _unique_agent_issues(history)
    false_positives = len((detected | classified) - true_issues)

    issue_recall = len(detected & true_issues) / max(1, len(true_issues))
    issue_precision = len(detected & true_issues) / max(1, len(detected))
    issue_score = 0.65 * issue_recall + 0.35 * issue_precision

    resolution_score = _resolution_coverage(task, history) if state.get("resolved") else 0.0
    clarification_score = _clarification_coverage(task, history)

    steps = int(state.get("time_elapsed", 0))
    target_steps = 3 if task.difficulty == "easy" else 5 if task.difficulty == "medium" else 6
    efficiency_score = max(0.0, 1.0 - max(0, steps - target_steps) * 0.12)

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
        behavior_score = min(1.0, behavior_score + 0.12)
    if premature_resolve:
        behavior_score -= 0.22
    if wrong_escalation:
        behavior_score -= 0.18
    behavior_score -= min(0.25, false_positives * 0.08)
    behavior_score = max(0.0, min(1.0, behavior_score))

    total = (
        0.34 * issue_score
        + 0.34 * resolution_score
        + 0.17 * behavior_score
        + 0.15 * efficiency_score
    )
    return round(max(0.0, min(1.0, total)), 4)
