"""Core schemas for the customer support OpenEnv environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


Issue = Literal[
    "billing_overcharge",
    "login_failure",
    "shipping_delay",
    "refund_request",
    "subscription_cancellation",
    "account_security",
    "product_defect",
    "warranty_claim",
    "address_correction",
]

CustomerType = Literal["normal", "premium"]
Sentiment = Literal["calm", "angry"]
ActionType = Literal["classify", "ask", "resolve", "escalate"]


@dataclass(frozen=True)
class Task:
    """A deterministic support case definition."""

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    customer_query: str
    true_issues: List[Issue]
    customer_type: CustomerType
    sentiment: Sentiment
    required_clarifications: Dict[Issue, List[str]]
    acceptable_resolutions: Dict[Issue, List[str]]
    misleading_signals: List[str] = field(default_factory=list)
    max_steps: int = 8


@dataclass
class SupportState:
    customer_query: str
    true_issues: List[Issue]
    detected_issues: List[Issue]
    customer_type: CustomerType
    sentiment: Sentiment
    conversation_history: List[Dict[str, Any]]
    time_elapsed: int
    resolved: bool
    escalated: bool

    def to_public_observation(self) -> Dict[str, Any]:
        """Observation returned to agents. Hidden ground truth is intentionally omitted."""

        return {
            "customer_query": self.customer_query,
            "detected_issues": list(self.detected_issues),
            "customer_type": self.customer_type,
            "sentiment": self.sentiment,
            "conversation_history": list(self.conversation_history),
            "time_elapsed": self.time_elapsed,
            "resolved": self.resolved,
            "escalated": self.escalated,
        }

    def to_full_state(self) -> Dict[str, Any]:
        return {
            "customer_query": self.customer_query,
            "true_issues": list(self.true_issues),
            "detected_issues": list(self.detected_issues),
            "customer_type": self.customer_type,
            "sentiment": self.sentiment,
            "conversation_history": list(self.conversation_history),
            "time_elapsed": self.time_elapsed,
            "resolved": self.resolved,
            "escalated": self.escalated,
        }


@dataclass
class StepResult:
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


def normalize_issue_text(text: str) -> List[str]:
    """Extract canonical issue ids from free-form action content."""

    text_l = (text or "").lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "billing_overcharge": ["billing_overcharge", "overcharge", "charged_twice", "double_charge", "extra_charge"],
        "login_failure": ["login_failure", "cannot_login", "can't_login", "signin", "sign_in", "password_reset"],
        "shipping_delay": ["shipping_delay", "late_delivery", "delivery_delay", "tracking", "shipment"],
        "refund_request": ["refund_request", "refund", "money_back", "return_payment"],
        "subscription_cancellation": ["subscription_cancellation", "cancel_subscription", "cancellation", "unsubscribe"],
        "account_security": ["account_security", "security", "unauthorized", "fraud", "suspicious_login"],
        "product_defect": ["product_defect", "defective", "broken", "faulty", "does_not_work"],
        "warranty_claim": ["warranty_claim", "warranty", "replacement_under_warranty"],
        "address_correction": ["address_correction", "wrong_address", "change_address", "address_update"],
    }
    found: List[str] = []
    for issue, terms in aliases.items():
        if any(term in text_l for term in terms):
            found.append(issue)
    return found


def action_to_dict(action: Any) -> Dict[str, str]:
    if isinstance(action, dict):
        return {
            "type": str(action.get("type", "")).strip().lower(),
            "content": str(action.get("content", "")).strip(),
        }
    return {"type": "", "content": str(action).strip()}


def contains_any(text: str, terms: List[str]) -> bool:
    lowered = (text or "").lower()
    return any(term.lower() in lowered for term in terms)
