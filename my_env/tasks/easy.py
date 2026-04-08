from graders.easy import grade as grade_easy
from graders.hard import grade as grade_hard
from graders.medium import grade as grade_medium
from models.schemas import Task


TASKS = {
    "easy": Task(
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
    ),
    "medium": Task(
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
    ),
    "hard": Task(
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
    ),
}

TASK_GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

TASKS_WITH_GRADERS = {
    name: {"task": task, "grader": TASK_GRADERS[name]}
    for name, task in TASKS.items()
}

# Backward-compatible single-task export for older imports.
TASK = TASKS["easy"]
