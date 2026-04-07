from pydantic import BaseModel

class Observation(BaseModel):
    state: dict

class Action(BaseModel):
    type: str
    content: str

class Reward(BaseModel):
    value: float