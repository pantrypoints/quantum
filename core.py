"""Classical core: ADB joint-loss model, exact enumeration, sampling baselines."""
import itertools
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd                                   # <-- added (pd_row uses it)
from scipy.linalg import expm
from scipy.stats import qmc, t as tdist

CAP = np.array([4, 4, 1, 1])
NS = 100
TARGETS = ["expected_principal_loss_m", "joint_attachment_prob", "portfolio_loss_exceedance_prob",
           "portfolio_stop_loss_m", "both_exhausted_prob"]
S = np.array(list(itertools.product(range(5), range(5), range(2), range(2))))
_add = np.minimum(S[:, None, :] + S[None, :, :], CAP)
ADD = ((_add[..., 0] * 5 + _add[..., 1]) * 2 + _add[..., 2]) * 2 + _add[..., 3]

TAILS = {"KG": (.03883, .02331, .01790, .01184), "TJ": (.03925, .01743, .01332, .01043)}

def cp_rates(t):
    t1, t2, t3, t4 = t
    p0, p1, p2, p3 = 1 - t1, t1 - t2, t2 - t3, t3 - t4
    Lam = -np.log(p0); l1 = p1 / p0
    l2 = p2 / p0 - l1 ** 2 / 2
    l3 = p3 / p0 - l1 * l2 - l1 ** 3 / 6
    return np.array([l1, l2, l3, Lam - l1 - l2 - l3])

@dataclass
class Params:
    lam1: np.ndarray
    lam2: np.ndarray
    c: float
    q1: float = .02
    q2: float = .02
    rain_mult: float = 1.0
    cr_share: float = .25
    eq_lim: tuple = (70., 70.)
    pr_lim: tuple = (10., 10.)
    years: int = 3
    T: float = 80.

def adb_params(scn="adb-base"):
    p = Params(cp_rates(TAILS["KG"]), cp_rates(TAILS["TJ"]), 0.0)
    p.c = .5 * (.075 * p.lam1.sum() + .074 * p.lam2.sum())
    if scn == "adb-independent":  p = replace(p, c=0.0, cr_share=0.0)
    elif scn == "adb-common-shock": p = replace(p, c=2 * p.c)
    elif scn == "adb-rain-stress":  p = replace(p, rain_mult=2.0)
    elif scn == "adb-rain-cover":   p = replace(p, eq_lim=(60., 60.), pr_lim=(20., 20.))
    return p

SCENARIOS = ["adb-base", "adb-independent", "adb-common-shock", "adb-rain-stress", "adb-rain-cover"]

def eq_dist(p):
    L1, L2 = p.lam1.sum(), p.lam2.sum()
    c = min(p.c, L1, L2)
    w1, w2 = p.lam1 / L1, p.lam2 / L2
    G = np.zeros((25, 25))
    for a in range(5):
        for b in range(5):
            s = a * 5 + b
            for k in range(1, 5):
                G[s, min(a + k, 4) * 5 + b] += (L1 - c) * w1[k - 1]
                G[s, a * 5 + min(b + k, 4)] += (L2 - c) * w2[k - 1]
                for j in range(1, 5):
                    G[s, min(a + k, 4) * 5 + min(b + j, 4)] += c * w1[k - 1] * w2[j - 1]
    np.fill_diagonal(G, 0)
    G -= np.diag(G.sum(1))
    return expm(G)[0]

def rain_dist(p):
    m1, m2 = -np.log(1 - p.q1) * p.rain_mult, -np.log(1 - p.q2) * p.rain_mult
    cr = p.cr_share * min(m1, m2)
    P00 = np.exp(-(m1 + m2 - cr))
    P10 = max(np.exp(-m2) - P00, 0)
    P01 = max(np.exp(-m1) - P00, 0)
    return np.array([P00, P01, P10, max(1 - P00 - P10 - P01, 0)])

def annual_dist(p):
    return np.kron(eq_dist(p), rain_dist(p))

def losses(p):
    a, b, r1, r2 = S.T
    return p.eq_lim[0] * a / 4 + p.pr_lim[0] * r1, p.eq_lim[1] * b / 4 + p.pr_lim[1] * r2

def target_matrix(p):
    L1, L2 = losses(p); L = L1 + L2
    f1, f2 = p.eq_lim[0] + p.pr_lim[0], p.eq_lim[1] + p.pr_lim[1]
    return np.c_[L, (L1 > 0) & (L2 > 0), L > p.T + 1e-9, np.maximum(L - p.T, 0),
                 (L1 >= f1 - 1e-9) & (L2 >= f2 - 1e-9)].astype(float)

def term_dist(p, A=None):
    A = annual_dist(p) if A is None else A
    cur = np.zeros(NS); cur[0] = 1
    for _ in range(p.years):
        cur = np.bincount(ADD.ravel(), weights=np.outer(cur, A).ravel(), minlength=NS)
    return cur

def _fold(ys):
    st = np.zeros(ys.shape[0], int)
    for j in range(ys.shape[1]):
        st = ADD[st, ys[:, j]]
    return st

def _inv(cdf, u):
    return np.minimum(np.searchsorted(cdf, u), NS - 1)

def exact(p, n=None, rng=None):
    return dict(value=term_dist(p) @ target_matrix(p), ci=np.zeros(5), queries=0)

def mc(p, n, rng):
    A, V = annual_dist(p), target_matrix(p)
    x = V[_fold(_inv(np.cumsum(A), rng.random((n, p.years))))]
    return dict(value=x.mean(0), ci=1.96 * x.std(0, ddof=1) / np.sqrt(n), queries=n)

def sobol(p, n, rng, reps=8):
    A, V = annual_dist(p), target_matrix(p); cdf = np.cumsum(A)
    m = max(int(np.floor(np.log2(n / reps))), 1)
    means = np.array([V[_fold(_inv(cdf, qmc.Sobol(p.years, scramble=True, seed=rng).random_base2(m)))].mean(0)
                      for _ in range(reps)])
    return dict(value=means.mean(0), ci=tdist.ppf(.975, reps - 1) * means.std(0, ddof=1) / np.sqrt(reps),
                queries=reps * 2 ** m)

def stratified(p, n, rng):
    A, V = annual_dist(p), target_matrix(p); p0 = A[0]
    cond = A.copy(); cond[0] = 0; cond /= cond.sum()
    cc, cf = np.cumsum(cond), np.cumsum(A)
    nk = max(n // p.years, 2); est, var = np.zeros(5), np.zeros(5)
    for k in range(1, p.years + 1):
        ys = np.empty((nk, p.years - k + 1), int)
        ys[:, 0] = _inv(cc, rng.random(nk))
        if ys.shape[1] > 1: ys[:, 1:] = _inv(cf, rng.random((nk, ys.shape[1] - 1)))
        x = V[_fold(ys)]
        w = p0 ** (k - 1) * (1 - p0)
        est += w * x.mean(0); var += w ** 2 * x.var(0, ddof=1) / nk
    return dict(value=est, ci=1.96 * np.sqrt(var), queries=nk * p.years)

def importance(p, n, rng):
    A, V = annual_dist(p), target_matrix(p)
    tilt = A * V[:, 0]; tilt /= tilt.sum()
    q = .5 * A + .5 * tilt
    ys = _inv(np.cumsum(q), rng.random((n, p.years)))
    w = np.prod(A[ys] / q[ys], axis=1)
    x = V[_fold(ys)] * w[:, None]
    return dict(value=x.mean(0), ci=1.96 * x.std(0, ddof=1) / np.sqrt(n), queries=n)

ENGINES = {"exact": exact, "Monte Carlo": mc, "Scrambled Sobol": sobol,
           "Stratified": stratified, "Importance": importance}

def benchmark(p, engines, n, reps, seed=0):
    truth = exact(p)["value"]; rows = []
    for name in engines:
        rng = np.random.default_rng(seed)
        est = np.array([ENGINES[name](p, n, rng)["value"] for _ in range(reps)])
        rmse = np.sqrt(((est - truth) ** 2).mean(0))
        rows.append(pd_row(name, rmse, truth))
    return truth, rows

def pd_row(name, rmse, truth):
    return {"engine": name, **{t: r for t, r in zip(TARGETS, rmse / np.where(truth > 0, truth, 1))}}

def qae_vs_mc(prob, rel_err):
    n_mc = (1 - prob) / (prob * rel_err ** 2)
    eps = rel_err * prob
    m_qae = 2 * np.pi * np.sqrt(prob * (1 - prob)) / eps
    return n_mc, m_qae