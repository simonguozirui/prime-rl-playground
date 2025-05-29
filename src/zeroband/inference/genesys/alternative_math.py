from typing import Dict
import random

from zeroband.inference.genesys.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy

"""
Using idea from this paper
https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f

here we propose alternative math reward function.
Specifically:
- we reward 1.0 for incorrect answers and 0.0 for correct answers
- we randomly reward 0.0 or 1.0 for correct answers
"""


def compute_incorrect_math_reward(completion: str, verification_info: Dict):
    """Reward function that gives 1.0 for incorrect answers and 0.0 for correct answers."""
    model_response = completion
    ground_truths = verification_info["ground_truth"]

    # Extract solution.
    if "</think>" in model_response:
        model_solution = model_response.split("</think>")[1]
    else:
        return 0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0

    if ground_truths is None:
        return 0

    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 0  # Return 0 for correct answers

    return 1  # Return 1 for incorrect answers


def compute_random_math_reward(completion: str, verification_info: Dict):
    """Reward function that returns random rewards between 0 and 1."""
    # We still check if the answer is valid to maintain consistency with the format
    model_response = completion
    ground_truths = verification_info["ground_truth"]

    # Extract solution.
    if "</think>" in model_response:
        model_solution = model_response.split("</think>")[1]
    else:
        return 0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0

    if ground_truths is None:
        return 0

    # If we get here, the answer is valid, so return a random reward
    return random.random()  # Returns a random float between 0 and 1 