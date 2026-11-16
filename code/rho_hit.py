"""Hitting-time robustness rho_hit(C) of the d-dimensional hypercube (Zuckerman, ITCS 2026, Def. 1.8-1.9).
Standard eps-adversary: each transition prob 1/d may become anything in [(1-eps)/d, (1+eps)/d], rows stay stochastic.
Coordinate permutations fixing the target 0 act transitively on each Hamming sphere, so the optimal (MDP) value
depends only on the distance k, and at distance k the adversary only chooses the total mass q on the k neighbours
toward 0, within [k(1-eps)/d, k(1+eps)/d] and with 1-q within [(d-k)(1-eps)/d, (d-k)(1+eps)/d].
Bellman on {0..d} with both extreme choices per k, solved by policy iteration (exact linear solves in log-safe form).
    python rho_hit.py
"""
import math
import sys

import numpy as np


def hit_from_antipode(d, eps, policy=None):
    """Expected hitting time of 0 from d for a birth-death chain with toward-prob q_k; returns (H, policy)."""
    lo = [max(k * (1 - eps) / d, 1 - (d - k) * (1 + eps) / d) for k in range(d + 1)]
    hi = [min(k * (1 + eps) / d, 1 - (d - k) * (1 - eps) / d) for k in range(d + 1)]
    pol = policy or ['lo'] * (d + 1)
    for _ in range(50):
        q = [lo[k] if pol[k] == 'lo' else hi[k] for k in range(d + 1)]
        # T_k = expected time to go from k to k-1:  T_d = 1/q_d ; T_k = (1 + (1-q_k) T_{k+1}) / q_k
        T = [0.0] * (d + 2)
        T[d] = 1.0 / q[d]
        for k in range(d - 1, 0, -1):
            T[k] = (1.0 + (1 - q[k]) * T[k + 1]) / q[k]
        V = [0.0] * (d + 1)
        for k in range(1, d + 1):
            V[k] = V[k - 1] + T[k]
        new = ['lo' if k == d or V[min(k + 1, d)] >= V[k - 1] else 'hi' for k in range(d + 1)]
        if new == pol:
            break
        pol = new
    return V[d], pol


def rho(d, C):
    base, _ = hit_from_antipode(d, 0.0)
    a, b = 0.0, 1.0
    for _ in range(60):
        m = (a + b) / 2
        if hit_from_antipode(d, m)[0] <= C * base:
            a = m
        else:
            b = m
    return a, base


if __name__ == '__main__':
    print('d   t_hit/2^d   rho(2)   d*rho(2) (limit 1)   d*rho(4) (limit 2)')
    for d in (8, 16, 32, 64, 128, 256, 512):
        r2, base = rho(d, 2.0)
        r4, _ = rho(d, 4.0)
        _, pol = hit_from_antipode(d, r2)
        print(d, f'{base / 2 ** d:.4f}', f'{r2:.5f}', f'{d * r2:.4f}', f'{d * r4:.4f}', 'policy all lo:', set(pol[1:]) == {'lo'})
        sys.stdout.flush()
