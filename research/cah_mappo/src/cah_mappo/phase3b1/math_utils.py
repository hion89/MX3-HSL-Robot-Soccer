"""Dependency-free return semantics, shared by tests and the real trainer."""
import math


def gae(rewards, values, next_values, bootstrap, continuation, gamma=.99, lam=.95):
    n = len(rewards)
    if not n or any(len(x) != n for x in (values, next_values, bootstrap, continuation)):
        raise ValueError('GAE shapes disagree')
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('Invalid GAE coefficients')
    if any(x not in (0, 1) for x in [*bootstrap, *continuation]):
        raise ValueError('Masks must be binary')
    if not all(math.isfinite(v) for v in [*rewards, *values, *next_values]):
        raise ValueError('Nonfinite rollout')
    advantages = [0.0] * n
    tail = 0.0
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * bootstrap[t] * next_values[t] - values[t]
        tail = delta + gamma * lam * continuation[t] * tail
        advantages[t] = tail
    return advantages, [a + v for a, v in zip(advantages, values)]
