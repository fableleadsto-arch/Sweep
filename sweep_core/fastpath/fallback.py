import math
from typing import Iterable, List


def rms_energy(samples: Iterable[float]) -> float:
    values = list(samples)
    if not values:
        return 0.0
    total = math.fsum(s * s for s in values)
    return math.sqrt(total / len(values))


def moving_average(values: Iterable[float], window: int) -> List[float]:
    if window < 1:
        raise ValueError("window must be >= 1")
    data = list(values)
    n = len(data)
    out: List[float] = [0.0] * n
    if n == 0:
        return out
    rolling = 0.0
    for i in range(n):
        rolling += data[i]
        if i >= window:
            rolling -= data[i - window]
        count = i + 1 if i < window - 1 else window
        out[i] = rolling / count
    return out


def frame_diff_score(prev: bytes, curr: bytes) -> float:
    max_len = max(len(prev), len(curr))
    if max_len == 0:
        return 0.0
    diff = max_len - min(len(prev), len(curr))
    for a, b in zip(prev, curr):
        if a != b:
            diff += 1
    return diff / max_len


def dot_product(a: Iterable[float], b: Iterable[float]) -> float:
    vec_a, vec_b = list(a), list(b)
    if len(vec_a) != len(vec_b):
        raise ValueError("vectors must have equal length")
    return math.fsum(x * y for x, y in zip(vec_a, vec_b))


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    vec_a, vec_b = list(a), list(b)
    if not vec_a or len(vec_a) != len(vec_b):
        return 0.0
    dot = math.fsum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(math.fsum(x * x for x in vec_a))
    norm_b = math.sqrt(math.fsum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
