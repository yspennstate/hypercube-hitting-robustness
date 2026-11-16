"""Check the symmetry reduction of rho_hit.py on the full cube Q_d (no reduction): value iteration for the standard
eps-adversary maximising E H_0, per-edge bounds [(1-eps)/d, (1+eps)/d], greedy fill per vertex (the optimal response
to a value vector: give every neighbour its lower bound, then pour the slack onto neighbours in decreasing value).
Compares max_v V(v) with the reduced antipode value.
    python full_cube_check.py
"""
import numpy as np

from rho_hit import hit_from_antipode


def full_cube(d, eps, iters=200000, tol=1e-10):
    N = 1 << d
    nb = np.array([[v ^ (1 << i) for i in range(d)] for v in range(N)])
    lo, hi = (1 - eps) / d, (1 + eps) / d
    V = np.zeros(N)
    for _ in range(iters):
        vals = V[nb]                                   # N x d
        order = np.argsort(-vals, axis=1)
        sv = np.take_along_axis(vals, order, axis=1)
        slack = 1.0 - d * lo                           # = eps
        add = np.minimum(hi - lo, np.maximum(0.0, slack - (hi - lo) * np.arange(d)))
        newV = 1.0 + (lo * sv + add * sv).sum(axis=1)
        newV[0] = 0.0
        if np.max(np.abs(newV - V)) < tol * max(1.0, newV.max()):
            V = newV
            break
        V = newV
    return V.max()


if __name__ == '__main__':
    for d in (3, 4, 5, 6, 7, 8):
        for eps in (0.0, 0.1, 0.3):
            full = full_cube(d, eps)
            red, _ = hit_from_antipode(d, eps)
            print(d, eps, f'full {full:.6f}  reduced {red:.6f}  rel.diff {abs(full - red) / red:.2e}')
