"""Pricing overlay (NOT part of the accuracy claim).

Four-law value decomposition: nominal (L1), real (L2), market (L4).
Law 3 is the identity P = V1 + V2 + V4 plus shares and balance flag.
All parameters are MODELLING ASSUMPTIONS with no data behind them.
"""
from dataclasses import dataclass, replace
import numpy as np
import core


@dataclass
class LawParams:
    Dm: float = 0.0
    beta: float = 0.5
    capital_share: float = 0.8
    effort: float = 1.0
    e_min: float = 0.25
    k: float = 0.25
    gamma: float = 1.0
    natural_cap: float = 0.35


def price(p, lp=None, term=None):
    lp = lp or LawParams()
    term = core.term_dist(p) if term is None else term
    L1, L2 = core.losses(p)
    f1 = p.eq_lim[0] + p.pr_lim[0]
    f2 = p.eq_lim[1] + p.pr_lim[1]
    s1 = (L1 > 0) & (L1 / f1 >= lp.Dm)
    s2 = (L2 > 0) & (L2 / f2 >= lp.Dm)
    L = L1 + L2
    V1 = float(term @ (L1 * s1 + L2 * s2))
    uplift = lp.beta * lp.capital_share * np.minimum(L / p.T, 1)
    F = uplift / max(lp.effort, lp.e_min)
    V2 = float(term @ (L * F))
    V4 = lp.k * (V1 + V2) * lp.gamma
    P = V1 + V2 + V4
    shares = np.array([V1, V2, V4]) / max(P, 1e-18)
    return dict(nominal=V1, real=V2, market=V4, premium=P,
                premium_per_year=P / p.years,
                rate_on_line=P / (f1 + f2), shares=shares,
                chain_valid_prob=float(term @ (s1 & s2)),
                mean_flow=float(term @ F),
                balanced=bool(shares[2] <= lp.natural_cap))


def variants(p):
    return {
        "base": p,
        "independent": replace(p, c=0.0, cr_share=0.0),
        "common-shock": replace(p, c=2 * p.c),
        "rain-stress": replace(p, rain_mult=2.0),
    }