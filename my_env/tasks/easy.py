from graders.easy import grade as grade_easy
from models.schemas import Task

name = "easy"
task_name = name
TASK_NAME = name
enabled = True
ENABLED = True
grader = grade_easy
GRADER = grader


TASK = Task(
    task_id="easy_billing_overcharge",
    difficulty="easy",
    customer_query=(
        "I was charged twice for my monthly subscription yesterday. "
        "Can you fix the extra charge?"
    ),
    true_issues=["billing_overcharge"],
    customer_type="normal",
    sentiment="calm",
    required_clarifications={
        "billing_overcharge": ["invoice", "charge", "transaction", "receipt", "date", "amount"],
    },
    acceptable_resolutions={
        "billing_overcharge": ["refund", "reverse", "credit", "billing adjustment", "remove duplicate"],
    },
    max_steps=6,
    grader=grade_easy,
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
