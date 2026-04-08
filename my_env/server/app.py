import os
from threading import Lock
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
import uvicorn

from env import CustomerSupportEnv
from tasks import TASKS

app = FastAPI()
DEFAULT_SESSION_ID = "default"
_envs: Dict[str, CustomerSupportEnv] = {
    DEFAULT_SESSION_ID: CustomerSupportEnv(task=os.getenv("TASK_NAME", "easy")),
}
_lock = Lock()


def _get_env(session_id: str = DEFAULT_SESSION_ID) -> CustomerSupportEnv:
    with _lock:
        if session_id not in _envs:
            _envs[session_id] = CustomerSupportEnv(task=os.getenv("TASK_NAME", "easy"))
        return _envs[session_id]


@app.get("/")
def root():
    return {
        "message": "OpenEnv server running",
        "tasks": sorted(TASKS),
        "endpoints": ["/reset", "/step", "/state"],
        "default_session_id": DEFAULT_SESSION_ID,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "intelligent-customer-support-decision-system",
        "description": "Multi-step customer support RL environment with task-specific grading.",
        "tasks": sorted(TASKS),
    }


@app.get("/schema")
def schema():
    return {
        "action": {"type": "classify | ask | resolve | escalate", "content": "string"},
        "observation": {
            "customer_query": "string",
            "detected_issues": "list[string]",
            "customer_type": "normal | premium",
            "sentiment": "calm | angry",
            "conversation_history": "list[object]",
            "time_elapsed": "int",
            "resolved": "bool",
            "escalated": "bool",
        },
        "state": {
            "true_issues": "list[string]",
            "customer_query": "string",
            "detected_issues": "list[string]",
            "customer_type": "normal | premium",
            "sentiment": "calm | angry",
            "conversation_history": "list[object]",
            "time_elapsed": "int",
            "resolved": "bool",
            "escalated": "bool",
        },
    }


@app.post("/mcp")
def mcp(payload: Dict[str, Any] | None = None):
    request_id = (payload or {}).get("id")
    return {"jsonrpc": "2.0", "id": request_id, "result": {"status": "ok"}}


@app.post("/reset")
def reset(payload: Dict[str, Any] | None = None):
    payload = payload or {}
    task = payload.get("task", os.getenv("TASK_NAME", "easy"))
    if task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'")
    session_id = str(payload.get("session_id") or uuid4())
    new_env = CustomerSupportEnv(task=task)
    observation = new_env.reset()
    with _lock:
        _envs[session_id] = new_env
    return {"session_id": session_id, "observation": observation}


@app.post("/step")
def step(payload: Dict[str, Any]):
    session_id = str(payload.get("session_id") or DEFAULT_SESSION_ID)
    action = payload.get("action", payload)
    if not isinstance(action, dict):
        raise HTTPException(status_code=400, detail="Action must be an object")
    env = _get_env(session_id)
    observation, reward, done, info = env.step(action)
    return {
        "session_id": session_id,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str = DEFAULT_SESSION_ID):
    return _get_env(session_id).state()


def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
