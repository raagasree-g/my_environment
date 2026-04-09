from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env import CustomerSupportEnv
from models import CustomerSupportAction, CustomerSupportObservation


class CustomerSupportOpenEnv(Environment[CustomerSupportAction, CustomerSupportObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._env = CustomerSupportEnv()
        self._state = State(episode_id=str(uuid4()), step_count=0, current_task="")

    def _to_observation(self, observation: dict, reward: float | None, done: bool) -> CustomerSupportObservation:
        return CustomerSupportObservation(
            task_name=observation["task_name"],
            current_task=observation["current_task"],
            customer_query=observation["customer_query"],
            conversation_history=observation.get("conversation_history", []),
            reward=reward,
            done=done,
            metadata={"current_task": observation["current_task"]},
        )

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> CustomerSupportObservation:
        observation = self._env.reset()
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_task=observation["current_task"],
        )
        return self._to_observation(observation, reward=None, done=False)

    def step(self, action: CustomerSupportAction, timeout_s: float | None = None, **kwargs) -> CustomerSupportObservation:
        message = action.message or action.content or "test response"
        action_type = action.type or "classify"
        observation, reward, done, info = self._env.step({"type": action_type, "content": message})
        self._state.step_count += 1
        self._state.current_task = info["task_name"]
        return self._to_observation(observation, reward=reward, done=done)

    @property
    def state(self) -> State:
        if not self._state.current_task:
            current = self._env.get_state().get("current_task", "")
            self._state.current_task = current
        return self._state

    def close(self) -> None:
        self._env.close()
