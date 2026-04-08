import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
import uvicorn

from env import CustomerSupportEnv
from tasks import TASKS

app = FastAPI()
env = CustomerSupportEnv(task=os.getenv("TASK_NAME", "easy"))

@app.get("/")
def root():
    return {
        "message": "OpenEnv server running",
        "tasks": sorted(TASKS),
        "endpoints": ["/reset", "/step", "/state"],
    }


@app.post("/reset")
def reset(payload: Dict[str, Any] | None = None):
    global env
    task = (payload or {}).get("task", env.task_name)
    if task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'")
    env = CustomerSupportEnv(task=task)
    return {"observation": env.reset()}


@app.post("/step")
def step(payload: Dict[str, Any]):
    action = payload.get("action", payload)
    if not isinstance(action, dict):
        raise HTTPException(status_code=400, detail="Action must be an object")
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()

def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
