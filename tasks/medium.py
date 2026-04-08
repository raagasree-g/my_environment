from graders.medium import grade as grade_medium
from models.schemas import Task

name = "medium"
grader = grade_medium


TASK = Task(
    task_id="medium_shipping_refund_address",
    difficulty="medium",
    customer_query=(
        "My order still has not arrived and I might have entered my old apartment address. "
        "If it is not coming today, I want my money back."
    ),
    true_issues=["shipping_delay", "address_correction", "refund_request"],
    customer_type="normal",
    sentiment="calm",
    required_clarifications={
        "shipping_delay": ["order", "tracking", "delivery", "carrier", "eta"],
        "address_correction": ["address", "apartment", "delivery address", "old address", "new address"],
        "refund_request": ["refund", "return", "money back", "eligibility"],
    },
    acceptable_resolutions={
        "shipping_delay": ["track", "carrier", "expedite", "delivery update", "reship"],
        "address_correction": ["update address", "correct address", "reroute", "contact carrier"],
        "refund_request": ["refund", "return authorization", "refund eligibility", "credit"],
    },
    max_steps=8,
    grader=grade_medium,
)

TASK_DEF = {
    "name": name,
    "query": TASK.customer_query,
    "grader": grader,
    "task": TASK,
}

query = TASK.customer_query
TASK_REGISTRY = {name: TASK}
TASKS = [TASK_DEF]
tasks = TASKS
