from openenv.core.env_server.types import Action, Observation


class CustomerSupportAction(Action):
    message: str = ""
    type: str = "classify"
    content: str = ""


class CustomerSupportObservation(Observation):
    task_name: str
    current_task: str
    customer_query: str
    message: str = ""
    conversation_history: list[dict] = []


Action = CustomerSupportAction
Observation = CustomerSupportObservation
