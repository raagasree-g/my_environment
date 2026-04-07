---
title: my-env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
---

# Intelligent Customer Support Decision System

This OpenEnv submission implements a realistic multi-step reinforcement learning environment for customer support decision-making. The agent must infer hidden customer issues from an initial query, ask useful clarification questions when uncertainty remains, classify issues, resolve the case, and escalate only when escalation is justified.

The environment intentionally keeps `true_issues` out of the observations returned by `reset()` and `step()`. Full hidden state is available through `state()` for validators and graders.

## State and Observation

Full state includes:

- `customer_query`: original customer message
- `true_issues`: hidden ground-truth issue list
- `detected_issues`: issues found by the agent
- `customer_type`: `normal` or `premium`
- `sentiment`: `calm` or `angry`
- `conversation_history`: structured action, reward, and feedback records
- `time_elapsed`: number of steps taken
- `resolved`: whether the case is resolved
- `escalated`: whether the case has been escalated

Public observations include all fields except `true_issues`.

## Action Space

Actions are structured dictionaries:

```json
{
  "type": "classify | ask | resolve | escalate",
  "content": "string"
}
```

Examples:

```json
{"type":"classify","content":"shipping_delay, address_correction, refund_request"}
{"type":"ask","content":"Please confirm the tracking number and the correct delivery address."}
{"type":"resolve","content":"Track with carrier, reroute to corrected address, and process refund eligibility if delivery fails."}
```

## Tasks

The project includes three deterministic tasks:

- Easy: one clear billing overcharge issue
- Medium: shipping delay, address correction, and refund request in a slightly ambiguous query
- Hard: angry premium customer with product defect, warranty claim, and account security risk, plus misleading signals

## Reward Design

Rewards are trajectory-shaped rather than binary:

- Correct new issue detection receives positive reward.
- Meaningful clarification questions receive positive reward, especially when they address actual uncertainty.
- Complete resolutions receive high reward only after all true issues have been detected.
- Appropriate premium security escalation receives positive reward.
- Wrong classifications, repeated actions, empty actions, premature resolution, unnecessary escalation, and inefficient excess steps are penalized.

The deterministic grader returns a score in `[0, 1]` using issue precision/recall, resolution coverage, clarification quality, escalation behavior, and efficiency.

## Why This Environment Is Hard

The case is partially observable: agents see the customer message, visible support context, and their own history, but not `true_issues`. They must infer hidden issue combinations from ambiguous wording and update that belief across steps.

The task requires multi-step reasoning rather than one-shot classification. A strong trajectory may classify some issues, ask targeted clarifying questions, decide whether escalation is justified, and only then resolve with coverage for every true issue.

There are trade-offs between asking and resolving. Asking too little causes premature or incomplete resolution, while asking too much wastes steps and reduces efficiency. Escalation is useful for premium security risk, but harmful when used as a generic escape hatch.

## Failure Modes

- Over-classification: guessing extra issue labels to maximize recall is penalized through false-positive and over-classification costs.
- Hallucinated actions: resolutions that introduce unsupported actions, such as refunds for non-refund cases, receive negative reward.
- Premature resolution: trying to resolve before all true issues are detected is penalized and lowers trajectory score.
- Inefficient loops: repeated guesses, low-information questions, empty actions, and excess steps reduce reward.

## Why RL

This environment is a sequential decision problem. The quality of an action depends on what the agent has already inferred, asked, and attempted, so rewards are shaped across the trajectory rather than assigned only to a final answer.

Multiple valid trajectories can succeed. For example, an agent may ask one broad high-quality clarification or several narrower ones, and hard cases may benefit from escalation before final resolution. RL is a natural fit because the agent must learn the timing and ordering of classify, ask, resolve, and escalate actions under partial observability.

## Setup and Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run baseline inference:

```bash
python inference.py
```

Optional environment variables:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token
export TASK_NAME=hard
```

Docker:

```bash
docker build -t customer-support-openenv .
docker run --rm -e API_BASE_URL="$API_BASE_URL" -e MODEL_NAME="$MODEL_NAME" -e HF_TOKEN="$HF_TOKEN" customer-support-openenv
```

The inference script prints only:

```text
[START] task=... env=... model=...
[STEP] step=1 action=... reward=0.00 done=false error=null
[END] success=true steps=N score=0.X rewards=...
```
