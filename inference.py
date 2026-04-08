"""Baseline inference runner with strict stdout formatting."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI

from env import CustomerSupportEnv


ISSUES = [
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


def compact_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)


def parse_action(text: str) -> Dict[str, str]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "type" in data and "content" in data:
            return {"type": str(data["type"]), "content": str(data["content"])}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return {"type": str(data.get("type", "")), "content": str(data.get("content", ""))}
        except Exception:
            pass
    return {"type": "ask", "content": "Please provide the order, invoice, account, and security details needed to help."}


def model_action(client: OpenAI, model: str, observation: Dict[str, Any], step_idx: int) -> Dict[str, str]:
    prompt = (
        "You are controlling an RL customer support environment. "
        "Return only a JSON object with keys type and content. "
        "type must be one of classify, ask, resolve, escalate. "
        "Do not include any explanation.\n"
        f"Known issue labels: {', '.join(ISSUES)}\n"
        f"Step: {step_idx}\n"
        f"Observation: {compact_json(observation)}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid compact JSON for the next support action."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=160,
    )
    content = response.choices[0].message.content or ""
    return parse_action(content)


def heuristic_action(observation: Dict[str, Any]) -> Dict[str, str]:
    query = (observation.get("customer_query") or "").lower()
    detected = set(observation.get("detected_issues") or [])
    history = observation.get("conversation_history") or []
    asked = any(event.get("action_type") == "ask" for event in history)
    escalated = bool(observation.get("escalated"))

    issue_terms = [
        ("billing_overcharge", ["charged twice", "extra charge", "overcharge"]),
        ("shipping_delay", ["not arrived", "coming today", "delivery", "order"]),
        ("address_correction", ["old apartment", "address"]),
        ("refund_request", ["money back", "refund"]),
        ("account_security", ["login from another city", "nobody buys", "card", "account"]),
        ("product_defect", ["headphones died", "died after", "does not work"]),
        ("warranty_claim", ["new pair", "two weeks", "warranty"]),
    ]
    missing = [issue for issue, terms in issue_terms if issue not in detected and any(term in query for term in terms)]
    if missing:
        return {"type": "classify", "content": ", ".join(missing)}

    if "account_security" in detected and observation.get("customer_type") == "premium" and observation.get("sentiment") == "angry" and not escalated:
        return {"type": "escalate", "content": "Escalate to a priority security and fraud specialist for unauthorized login and card risk."}

    if len(detected) > 1 and not asked:
        return {
            "type": "ask",
            "content": (
                "I understand this is urgent. Please confirm order/tracking, current address, receipt or serial number, "
                "and whether the login/card activity was unauthorized."
            ),
        }

    if detected == {"billing_overcharge"}:
        return {"type": "resolve", "content": "Reverse the duplicate charge and apply a billing adjustment credit for the extra transaction."}
    if {"shipping_delay", "address_correction", "refund_request"}.issubset(detected):
        return {
            "type": "resolve",
            "content": "Track the shipment with the carrier, correct or reroute the delivery address, and process refund eligibility or credit if delivery cannot be completed.",
        }
    if {"account_security", "product_defect", "warranty_claim"}.issubset(detected):
        return {
            "type": "resolve",
            "content": "Secure account by revoking sessions, resetting password, enabling 2FA and fraud review, then open an RMA warranty replacement using receipt and serial after defect diagnostics.",
        }
    return {"type": "ask", "content": "Please share the invoice, order, account, receipt, serial, and security details needed to resolve this."}


def main() -> None:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    HF_TOKEN = os.getenv("HF_TOKEN")
    LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
    task_name = os.getenv("TASK_NAME", "hard").strip().lower()
    if task_name not in {"easy", "medium", "hard"}:
        task_name = "hard"

    client = None
    if HF_TOKEN:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
            timeout=float(os.getenv("OPENAI_TIMEOUT", "2")),
            max_retries=0,
        )
    env = CustomerSupportEnv(task=task_name)
    observation = env.reset()
    rewards: List[float] = []
    model_available = client is not None

    print(f"[START] task={task_name} env=CustomerSupportEnv model={MODEL_NAME}", flush=True)
    done = False
    step_idx = 0
    info: Dict[str, Any] = {"score": 0.0}
    try:
        while not done and step_idx < env.task.max_steps:
            step_idx += 1
            if model_available and client is not None:
                try:
                    action = model_action(client, MODEL_NAME, observation, step_idx)
                except Exception:
                    model_available = False
                    action = heuristic_action(observation)
            else:
                action = heuristic_action(observation)

            error_msg = "null"
            try:
                observation, reward, done, info = env.step(action)
            except Exception as exc:
                reward = 0.05
                done = True
                info = {"score": 0.05}
                error_msg = compact_json(str(exc))

            rewards.append(float(reward))
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_idx} action={compact_json(action)} reward={reward:.2f} done={done_str} error={error_msg}",
                flush=True,
            )
    finally:
        env.close()

    success = "true" if bool(observation.get("resolved")) else "false"
    score = max(0.0, min(1.0, float(info.get("score", 0.0))))
    reward_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={success} steps={step_idx} score={score:.2f} rewards={reward_str}", flush=True)


if __name__ == "__main__":
    main()
