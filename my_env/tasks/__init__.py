from .easy import TASK as EASY_TASK, TASK_DEF as EASY_TASK_DEF
from .medium import TASK as MEDIUM_TASK, TASK_DEF as MEDIUM_TASK_DEF
from .hard import TASK as HARD_TASK, TASK_DEF as HARD_TASK_DEF


TASK_REGISTRY = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}

TASK_GRADERS = {
    "easy": EASY_TASK.grader,
    "medium": MEDIUM_TASK.grader,
    "hard": HARD_TASK.grader,
}

TASKS_WITH_GRADERS = {
    name: {"task": task, "grader": TASK_GRADERS[name]}
    for name, task in TASK_REGISTRY.items()
}

TASKS = [EASY_TASK_DEF, MEDIUM_TASK_DEF, HARD_TASK_DEF]
tasks = TASKS
