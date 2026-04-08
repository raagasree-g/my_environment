from graders.easy import grade as grade_easy
from models.schemas import Task

name = "easy"
grader = grade_easy


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
TASKS = {name: TASK}
tasks = [TASK_DEF]
