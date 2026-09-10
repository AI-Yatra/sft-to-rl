"""
Verifiable reward for the Countdown task (TinyZero/Jiayi-Pan style).

Given a target integer and a list of numbers, the model must produce an
arithmetic equation that uses each number exactly once (via +, -, *, /) and
evaluates to the target. This is a genuinely verifiable reward -- no reward
model, no human labels, just "did the equation check out."

Expected completion format:
    <think>
    ...free-form reasoning...
    </think>
    <answer>
    (numbers combined with + - * / and parens)
    </answer>

Reward breakdown per completion:
    0.0  -- no <answer>...</answer> block, or expression doesn't parse
    0.1  -- parses, but doesn't use exactly the given numbers (each once)
    0.2  -- uses the numbers correctly, but doesn't evaluate to target
    1.0  -- uses the numbers correctly AND evaluates to target
"""
import ast
import re
from collections import Counter

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Load,
)


def _safe_eval(expr: str):
    """Evaluate a pure arithmetic expression via the AST, no builtins/eval()."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"disallowed expression node: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
    return eval(compile(tree, "<countdown>", "eval"))


def _extract_numbers_used(expr: str):
    tree = ast.parse(expr, mode="eval")
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]


def score_completion(completion: str, target: int, nums: list) -> float:
    match = ANSWER_RE.search(completion)
    if not match:
        return 0.0
    expr = match.group(1).strip()
    if not expr:
        return 0.0

    try:
        used = _extract_numbers_used(expr)
    except (SyntaxError, ValueError):
        return 0.0

    used_ints = [int(n) for n in used if float(n).is_integer()]
    if len(used_ints) != len(used) or Counter(used_ints) != Counter(int(n) for n in nums):
        return 0.1

    try:
        result = _safe_eval(expr)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return 0.2

    if abs(result - target) < 1e-4:
        return 1.0
    return 0.2


def countdown_reward(completions, target=None, nums=None, **kwargs):
    """GRPOTrainer-compatible reward_func: completions is a list of strings
    (or list of chat-message lists -- handled below), target/nums are lists
    aligned to completions (TRL broadcasts per-example dataset columns)."""
    texts = []
    for c in completions:
        if isinstance(c, list):  # chat-format completion
            texts.append(c[-1]["content"])
        else:
            texts.append(c)

    rewards = []
    for text, t, n in zip(texts, target, nums):
        rewards.append(score_completion(text, t, n))
    return rewards


if __name__ == "__main__":
    good = "<think>3*4=12, 12+2=14</think><answer>3*4+2</answer>"
    wrong_numbers = "<think>...</think><answer>3*5+2</answer>"
    wrong_target = "<think>...</think><answer>3*4-2</answer>"
    no_answer = "<think>I don't know</think>"

    assert score_completion(good, 14, [3, 4, 2]) == 1.0
    assert score_completion(wrong_numbers, 14, [3, 4, 2]) == 0.1
    assert score_completion(wrong_target, 14, [3, 4, 2]) == 0.2
    assert score_completion(no_answer, 14, [3, 4, 2]) == 0.0
    print("All self-tests passed.")
