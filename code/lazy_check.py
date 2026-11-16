"""Lazy walk on Q_d (hold 1/2, each neighbour 1/(2d)) under the standard eps-adversary (every entry, the hold
included, may move by a factor in [1-eps, 1+eps]). The extremal adversary: toward mass k(1-eps)/(2d), away (d-k)(1+eps)/(2d), the hold
absorbs the rest; w_j = C(d,j) r^j with r = (1+eps)/(1-eps), W = (2/(1-eps))^d, T_1 = 2(W-1)/(1+eps), and
t_hit^+/t_hit -> e^c at eps = c/d, so d*rho_hit(C) -> ln C.
Checks: (1) reduced chain vs closed form; (2) reduced chain vs value iteration on the full cube (d <= 7);
(3) d*rho(2), d*rho(4) against ln 2, ln 4.
    python lazy_check.py
"""
import math

import numpy as np


def reduced(d, eps):
    T = [0.0] * (d + 2)
    for k in range(d, 0, -1):
        q, p = k * (1 - eps) / (2 * d), (d - k) * (1 + eps) / (2 * d)
        T[k] = (1.0 + p * T[k + 1]) / q
    return sum(T[1:d + 1]), T[1]


def full_cube(d, eps, tol=1e-11):
    N = 1 << d
    nb = np.array([[v ^ (1 << i) for i in range(d)] for v in range(N)])
    lo_n, hi_n = (1 - eps) / (2 * d), (1 + eps) / (2 * d)
    lo_h, hi_h = (1 - eps) / 2, (1 + eps) / 2
    V = np.zeros(N)
    for _ in range(10 ** 6):
        cand = np.concatenate([V[nb], V[:, None]], axis=1)          # neighbours then the hold
        lo = np.array([lo_n] * d + [lo_h]); hi = np.array([hi_n] * d + [hi_h])
        order = np.argsort(-cand, axis=1)
        sv = np.take_along_axis(cand, order, axis=1)
        slo, shi = lo[order], hi[order]
        slack = 1.0 - lo.sum()
        room = shi - slo
        before = np.cumsum(room, axis=1) - room
        add = np.minimum(room, np.maximum(0.0, slack - before))
        newV = 1.0 + ((slo + add) * sv).sum(axis=1)
        newV[0] = 0.0
        if np.max(np.abs(newV - V)) < tol * max(1.0, newV.max()):
            return newV.max()
        V = newV
    return V.max()


def rho(d, C):
    base = reduced(d, 0.0)[0]
    a, b = 0.0, 0.999
    for _ in range(60):
        m = (a + b) / 2
        a, b = (m, b) if reduced(d, m)[0] <= C * base else (a, m)
    return a


if __name__ == '__main__':
    for d in (4, 6, 7):
        for eps in (0.0, 0.2):
            V, T1 = reduced(d, eps)
            W = (2 / (1 - eps)) ** d
            print('d', d, 'eps', eps, f'reduced {V:.6f} full {full_cube(d, eps):.6f}  T1 {T1:.6f} closed {2 * (W - 1) / (1 + eps):.6f}')
    print('d  d*rho(2) (ln2=0.6931)  d*rho(4) (ln4=1.3863)')
    for d in (16, 64, 256, 512):
        print(d, f'{d * rho(d, 2.0):.4f}', f'{d * rho(d, 4.0):.4f}')
