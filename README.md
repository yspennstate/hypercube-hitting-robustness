# Hitting-time robustness of the hypercube

Yitzchak Shmalo, Einstein Institute of Mathematics, The Hebrew University of Jerusalem.

Zuckerman (Markov chain robustness, ITCS 2026, LIPIcs 362, Paper 118, doi:10.4230/LIPIcs.ITCS.2026.118) defined the hitting-time robustness of a random walk. It is the largest eps for which an adversary may change every transition probability by a factor in [1-eps, 1+eps], choosing the change from the whole history, and still not raise the maximum expected hitting time by more than a constant factor. He listed hypercubes among the graphs whose robustness is open.

For the simple random walk on {0,1}^d, the strongest such adversary with eps = c/d raises the maximum expected hitting time by a factor tending to 2^c. So the largest eps for which the factor stays at most C is (log_2 C + o(1))/d. For the lazy walk the factor tends to e^c and the threshold is (ln C + o(1))/d. Both are of the order of the reciprocal of the diameter. The upper bound O(1/d) also follows from Zuckerman's Theorem 4.5; the matching lower bound and the constants are new. Zuckerman conjectured that the robustness of every graph is at least of order one over its diameter (Conjecture 1.16); counterexamples on other graphs have been proposed in doi:10.5281/zenodo.23575572.

## Contents

- `paper/main.tex`, `paper/main.pdf`: the paper.
- `code/rho_hit.py`: solves the birth-death recursion of Lemma 2.1 exactly and finds rho_hit(2), rho_hit(4) by bisection for d = 8 to 512 (the table in Section 4). Run `python3 rho_hit.py`; needs numpy, under a minute.
- `code/full_cube_check.py`: value iteration over all 2^d vertices with the per-edge bounds, without the symmetry reduction, compared with the recursion for d = 3 to 8. Run `python3 full_cube_check.py` from the `code` folder.
- `code/lazy_check.py`: the same checks for the lazy walk, against the closed form W = (2/(1-eps))^d.
- `SHA256SUMS`: sha256 of each file above.
