"""Present-pegged societal cycle layer.

A 4-phase cycle (Worker/Warrior/Thinker/Trader) pegged at (t_now, phi_now).
Fitted by 1-D span search over T in [12, 40] years. Falsifiable via held-out
accuracy and permutation p-value. Does NOT reproduce the stationary benchmark;
it is a non-stationary extension.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import core

PHASES = ['w', 'W', 't', 'T']  # Worker, Warrior, Thinker, Trader
PHASE_IDX = {p: i for i, p in enumerate(PHASES)}


@dataclass
class CycleFit:
    t_now: int
    phi_now: str
    T_star: float
    score_profile: np.ndarray
    T_grid: np.ndarray
    heldout_acc: float = None
    heldout_p: float = None


def phase_at(t, fit):
    """Phase at year t, pegged at (t_now, phi_now) with period T_star."""
    steps = round((fit.t_now - t) / (fit.T_star / 4))
    return PHASES[(PHASE_IDX[fit.phi_now] - int(steps)) % 4]


def phase_sequence(t_years, fit):
    return np.array([phase_at(t, fit) for t in t_years])


def implied_T(t_now, phi_now, t_past, phi_past):
    """Two-point completion: T implied by (present, one past event)."""
    dphi = (PHASE_IDX[phi_now] - PHASE_IDX[phi_past]) % 4
    if dphi == 0:
        return None
    return 4 * (t_now - t_past) / dphi


def score(T, t_now, phi_now, history, event_phase_sets, weights=None):
    """Fit score: weighted fraction of historical events consistent with phases."""
    if len(history) == 0:
        return 0.0
    steps = np.round((t_now - history['year'].values) / (T / 4)).astype(int)
    phases = np.array([PHASES[(PHASE_IDX[phi_now] - int(s)) % 4] for s in steps])
    ok = np.array([p in event_phase_sets.get(e, set())
                   for p, e in zip(phases, history['event'])])
    w = np.ones(len(ok)) if weights is None else weights
    return float((w * ok).sum() / w.sum())


def fit_cycle(t_now, phi_now, history, event_phase_sets,
              T_grid=np.arange(12, 40.1, 0.5), holdout_year=1800,
              perm_reps=500, seed=0):
    """Span search with held-out test + permutation p-value."""
    history = history.copy()
    history['year'] = history['year'].astype(int)
    train = history[history['year'] < holdout_year]
    test = history[history['year'] >= holdout_year]
    profile = np.array([score(T, t_now, phi_now, train, event_phase_sets)
                        for T in T_grid])
    T_star = float(T_grid[np.argmax(profile)]) if len(profile) else 25.0
    fit = CycleFit(t_now, phi_now, T_star, profile, T_grid)
    if len(test) > 0:
        test_score = score(T_star, t_now, phi_now, test, event_phase_sets)
        fit.heldout_acc = test_score
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(perm_reps):
            perm = rng.permutation(test['event'].values)
            null.append(score(T_star, t_now, phi_now,
                              test.assign(event=perm), event_phase_sets))
        fit.heldout_p = float((np.array(null) >= test_score).mean())
    return fit


def rate_multiplier(t, fit, sentiment, lam_base, sent_mult):
    ph = phase_at(t, fit)
    return lam_base[ph] * sent_mult.get(sentiment, 1.0)


def year_rates(base_params, t, sentiment, fit, lam_base, sent_mult):
    from dataclasses import replace
    m = rate_multiplier(t, fit, sentiment, lam_base, sent_mult)
    return replace(base_params,
                   lam1=base_params.lam1 * m,
                   lam2=base_params.lam2 * m,
                   c=base_params.c * m)


def term_dist_forward(base_params, fit, sentiments, lam_base, sent_mult):
    """Time-ordered convolution from t_now forward, using the fitted cycle."""
    cur = np.zeros(core.NS)
    cur[0] = 1
    for k, s in enumerate(sentiments):
        p_k = year_rates(base_params, fit.t_now + k, s, fit, lam_base, sent_mult)
        A_k = core.annual_dist(p_k)
        cur = np.bincount(core.ADD.ravel(),
                          weights=np.outer(cur, A_k).ravel(), minlength=core.NS)
    return cur


def targets_forward(base_params, fit, sentiments, lam_base, sent_mult):
    term = term_dist_forward(base_params, fit, sentiments, lam_base, sent_mult)
    return term @ core.target_matrix(base_params)


# ---------- data loading ----------

def load_history(country, path=None):
    if path is None:
        path = f"data/history_{country}.csv"
    df = pd.read_csv(path)
    return df


EVENT_PHASE_SETS = {
    'plague': {'w'}, 'crop_failure': {'w'}, 'revolution': {'w'},
    'war': {'W'}, 'conflict': {'W'},
    'science': {'t'}, 'philosophy': {'t'},
    'trade_boom': {'T'}, 'finance': {'T'},
}


DEFAULT_LAM_BASE = {'w': 1.0, 'W': 0.8, 't': 0.7, 'T': 0.7}
DEFAULT_SENT_MULT = {'+': 1.0, '-': 2.5}

