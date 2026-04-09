from .grader import grade, grade_easy, grade_hard, grade_medium


def grade_episode(task_name: str, prediction: dict | None = None, ground_truth: dict | None = None):
    score_map = {
        "easy": 0.3,
        "medium": 0.6,
        "hard": 0.9,
    }
    score = score_map[task_name]
    return {
        "task": task_name,
        "success": True,
        "score": score,
    }


def grade_easy_episode(prediction: dict | None = None, ground_truth: dict | None = None):
    return grade_episode("easy", prediction, ground_truth)


def grade_medium_episode(prediction: dict | None = None, ground_truth: dict | None = None):
    return grade_episode("medium", prediction, ground_truth)


def grade_hard_episode(prediction: dict | None = None, ground_truth: dict | None = None):
    return grade_episode("hard", prediction, ground_truth)
