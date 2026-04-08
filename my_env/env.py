"""OpenEnv-style customer support decision environment (UPGRADED)."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from models.schemas import SupportState, Task, action_to_dict, contains_any, normalize_issue_text
from tasks import TASKS, TASK_GRADERS, TASK_REGISTRY


def _normalize_reward(raw_reward: float) -> float:
    """Map shaped step reward into the validator-safe strict interval (0, 1)."""

    return max(0.05, min(0.95, (raw_reward + 1.0) / 2.0))


class CustomerSupportEnv:
    _cycle_index = 0
    action_space = {"type": "classify | ask | resolve | escalate", "content": "string"}

    def __init__(self, task: str = "cycle"):
        if task != "cycle" and task not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task}'")
        self.task_name = task
        self._cycle_tasks = task == "cycle"
        self.tasks = TASKS
        self.current_task: Optional[Dict[str, Any]] = None
        initial_task_name = self.tasks[0]["name"] if self._cycle_tasks else task
        self.task: Task = TASK_REGISTRY[initial_task_name]
        self.grader = TASK_GRADERS[initial_task_name]
        self._state: Optional[SupportState] = None
        self._done = False
        self._last_score = 0.0

    def _select_task(self) -> None:
        if self._cycle_tasks:
            current = self.tasks[CustomerSupportEnv._cycle_index % len(self.tasks)]
            CustomerSupportEnv._cycle_index += 1
            self.task_name = current["name"]
            self.current_task = current
        else:
            self.current_task = next(item for item in self.tasks if item["name"] == self.task_name)
        self.task = self.current_task["task"]
        self.grader = self.current_task["grader"]

    def _public_observation(self) -> Dict[str, Any]:
        observation = self._state.to_public_observation()
        observation["task_name"] = self.task_name
        return observation

    def reset(self) -> Dict[str, Any]:
        self._select_task()
        self._state = SupportState(
            customer_query=self.task.customer_query,
            true_issues=list(self.task.true_issues),
            detected_issues=[],
            customer_type=self.task.customer_type,
            sentiment=self.task.sentiment,
            conversation_history=[],
            time_elapsed=0,
            resolved=False,
            escalated=False,
        )
        self._done = False
        self._last_score = 0.0
        return self._public_observation()

    def state(self) -> Dict[str, Any]:
        if self._state is None:
            self.reset()
        state = self._state.to_full_state()
        state["task_name"] = self.task_name
        return state

    def close(self) -> None:
        self._state = None
        self._done = False

    def step(self, action: Dict[str, str]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._state is None:
            self.reset()

        if self._done:
            terminal_score = self._last_score or 0.05
            return self._public_observation(), terminal_score, True, {"score": terminal_score, "task_name": self.task_name}

        parsed = action_to_dict(action)
        action_type = parsed["type"]
        content = parsed["content"]

        detected_before = list(self._state.detected_issues)

        # Time-scaled penalty creates efficiency pressure.
        reward = -0.02 * (self._state.time_elapsed + 1)

        messages: List[str] = []
        inferred_issues: List[str] = []

        if action_type not in {"classify", "ask", "resolve", "escalate"}:
            reward -= 0.4
            messages.append("Invalid action type.")

        elif not content:
            reward -= 0.2
            messages.append("Empty action.")

        elif action_type == "classify":
            inferred_issues = normalize_issue_text(content)
            reward += self._handle_classify(inferred_issues, messages)

        elif action_type == "ask":
            reward += self._handle_ask(content, messages)

        elif action_type == "resolve":
            reward += self._handle_resolve(content, messages)

        elif action_type == "escalate":
            reward += self._handle_escalate(content, messages)

        self._state.time_elapsed += 1

        if self._state.time_elapsed > self.task.max_steps:
            reward -= 0.25
            messages.append("Exceeded step limit.")
            self._done = True

        normalized_reward = _normalize_reward(reward)

        event = {
            "step": self._state.time_elapsed,
            "action_type": action_type,
            "content": content,
            "detected_before": detected_before,
            "issues": inferred_issues,
            "raw_reward": round(reward, 4),
            "shaped_reward": round(normalized_reward, 4),
            "feedback": messages,
        }

        self._state.conversation_history.append(event)

        grader = self.current_task["grader"]
        self._last_score = grader(
            self.task,
            self._state.to_full_state(),
            self._state.conversation_history,
        )
        event["reward"] = self._last_score

        self._done = True

        return self._public_observation(), self._last_score, self._done, {
            "score": self._last_score,
            "task_name": self.task_name,
            "raw_reward": round(reward, 4),
            "shaped_reward": round(normalized_reward, 4),
            "feedback": messages,
        }

    # -----------------------------
    # CLASSIFY: precision-focused
    # -----------------------------
    def _handle_classify(self, issues: List[str], messages: List[str]) -> float:
        if not issues:
            return -0.25

        reward = 0.0
        true_issues = set(self.task.true_issues)

        # Over-classification penalty.
        if len(issues) > len(true_issues):
            extra = len(issues) - len(true_issues)
            reward -= 0.15 * extra
            messages.append("Over-classification penalty.")

        for issue in issues:
            if issue in true_issues and issue not in self._state.detected_issues:
                self._state.detected_issues.append(issue)
                reward += 0.3
            elif issue in true_issues:
                reward -= 0.08
            else:
                reward -= 0.25

        # Duplicate penalty.
        if len(set(issues)) < len(issues):
            reward -= 0.1
            messages.append("Duplicate issue guesses.")

        if set(self._state.detected_issues) == true_issues:
            reward += 0.15

        return reward

    # -----------------------------
    # ASK: quality + reasoning
    # -----------------------------
    def _handle_ask(self, content: str, messages: List[str]) -> float:
        reward = 0.0
        matched = []

        for issue in self.task.true_issues:
            if contains_any(content, self.task.required_clarifications.get(issue, [])):
                matched.append(issue)

        # Force meaningful questions.
        if matched and len(content.split()) > 6:
            reward += 0.2 + 0.08 * len(set(matched))
        else:
            reward -= 0.18

        # Emotional intelligence bonus.
        if self.task.sentiment == "angry" and contains_any(content, ["sorry", "understand", "urgent"]):
            reward += 0.08

        # Spam penalty.
        if len(self._state.conversation_history) >= 4:
            reward -= 0.1

        return reward

    # -----------------------------
    # RESOLVE: graded, not binary
    # -----------------------------
    def _handle_resolve(self, content: str, messages: List[str]) -> float:
        true_issues = set(self.task.true_issues)
        detected = set(self._state.detected_issues)

        if detected != true_issues:
            return -0.4

        covered = []
        for issue in true_issues:
            if contains_any(content, self.task.acceptable_resolutions.get(issue, [])):
                covered.append(issue)

        coverage = len(set(covered)) / len(true_issues)

        # Hallucination penalty.
        if "refund" in content and "refund_request" not in true_issues:
            return -0.25

        if coverage == 1.0:
            self._state.resolved = True
            return 0.75

        if coverage > 0:
            return 0.3 * coverage - 0.15

        return -0.3

    # -----------------------------
    # ESCALATE: context-aware
    # -----------------------------
    def _handle_escalate(self, content: str, messages: List[str]) -> float:
        self._state.escalated = True

        appropriate = (
            self.task.customer_type == "premium"
            or "security" in self.task.true_issues
        )

        if appropriate and contains_any(content, ["security", "fraud", "urgent", "specialist"]):
            return 0.25

        return -0.25


def make_env(task: str = "cycle") -> CustomerSupportEnv:
    return CustomerSupportEnv(task)
