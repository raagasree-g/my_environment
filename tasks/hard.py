from graders.hard import grade as grade_hard
from models.schemas import Task

name = "hard"
grader = grade_hard


TASK = Task(
    task_id="hard_security_defect_warranty_angry",
    difficulty="hard",
    customer_query=(
        "This is ridiculous. Your headphones died after two weeks, and now I see a login "
        "from another city on my account. I do not care about your troubleshooting script, "
        "just send me a new pair and make sure nobody buys anything with my card."
    ),
    true_issues=["product_defect", "warranty_claim", "account_security"],
    customer_type="premium",
    sentiment="angry",
    required_clarifications={
        "product_defect": ["symptom", "headphones", "serial", "troubleshooting", "device"],
        "warranty_claim": ["purchase date", "warranty", "replacement", "serial", "receipt"],
        "account_security": ["login", "city", "unauthorized", "secure", "card", "password", "2fa"],
    },
    acceptable_resolutions={
        "product_defect": ["replacement", "diagnostic", "defect", "return label", "troubleshooting"],
        "warranty_claim": ["warranty replacement", "warranty claim", "rma", "receipt", "serial"],
        "account_security": ["secure account", "reset password", "2fa", "lock payments", "fraud review", "revoke sessions"],
    },
    misleading_signals=[
        "The customer rejects troubleshooting, but a defect/warranty workflow still needs evidence.",
        "The replacement request is not only a shipping issue.",
        "Premium angry customers deserve urgency, but escalation alone is incomplete.",
    ],
    max_steps=9,
    grader=grade_hard,
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
