"""Simple deterministic OpenEnv-style customer support environment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from graders.grader import grade_easy, grade_hard, grade_medium
from models.schemas import SupportState, Task, action_to_dict
from tasks import TASK_REGISTRY


class CustomerSupportEnv:
    _cycle_index = 0
    action_space = {"type": "classify | ask | resolve | escalate", "content": "string"}

    def __init__(self, task: str = "cycle"):
        self._fixed_task = None if task == "cycle" else task
        self.tasks = [
            {"name": "easy", "task": TASK_REGISTRY["easy"], "grader": self.grade_easy},
            {"name": "medium", "task": TASK_REGISTRY["medium"], "grader": self.grade_medium},
            {"name": "hard", "task": TASK_REGISTRY["hard"], "grader": self.grade_hard},
        ]
        self.task_index = 0
        self.current_task: Optional[Dict[str, Any]] = None
        self.task_name = ""
        self.task: Optional[Task] = None
        self._state: Optional[SupportState] = None
        self._last_score = 0.0

    def grade_easy(self, action_message: str) -> float:
        return float(grade_easy(self.task, self.get_state(), self._history_with_message(action_message)))

    def grade_medium(self, action_message: str) -> float:
        return float(grade_medium(self.task, self.get_state(), self._history_with_message(action_message)))

    def grade_hard(self, action_message: str) -> float:
        return float(grade_hard(self.task, self.get_state(), self._history_with_message(action_message)))

    def _history_with_message(self, action_message: str) -> list[Dict[str, Any]]:
        history = list(self._state.conversation_history) if self._state is not None else []
        if action_message:
            history.append({"content": action_message})
        return history

    def _select_task(self) -> None:
        if self._fixed_task is not None:
            self.current_task = next(item for item in self.tasks if item["name"] == self._fixed_task)
        else:
            self.current_task = self.tasks[CustomerSupportEnv._cycle_index % len(self.tasks)]
            CustomerSupportEnv._cycle_index += 1
            self.task_index += 1
        self.task_name = self.current_task["name"]
        self.task = self.current_task["task"]

    def _public_observation(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "current_task": self.task_name,
            "customer_query": self.task.customer_query,
            "conversation_history": list(self._state.conversation_history),
        }

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
        self._last_score = 0.0
        return self._public_observation()

    def state(self) -> Dict[str, Any]:
        if self._state is None:
            self.reset()
        return {"current_task": self.current_task["name"]}

    def get_state(self) -> Dict[str, Any]:
        return self.state()

    def close(self) -> None:
        self._state = None

    def step(self, action: Dict[str, str]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._state is None:
            self.reset()

        parsed = action_to_dict(action)
        content = parsed.get("content", "")
        self._state.time_elapsed += 1
        self._state.conversation_history.append(
            {
                "step": self._state.time_elapsed,
                "action_type": parsed.get("type", ""),
                "content": content,
            }
        )

        score = float(self.current_task["grader"](content))
        self._last_score = score

        observation = self._public_observation()
        info = {
            "score": score,
            "task_name": self.current_task["name"],
        }
        return observation, score, True, info


def make_env(task: str = "cycle") -> CustomerSupportEnv:
    return CustomerSupportEnv(task)
