"""Four-law layer. Value-weighted payoffs stay of the form E[f(c(X))], so any engine can estimate them.
Functional forms are MODELLING ASSUMPTIONS (no data behind them); treat parameters as sensitivities."""
from dataclasses import dataclass
import numpy as np
import core


@dataclass
class LawParams:
    Dm: float = 0.0            # L1: min acceptable payout fraction for a unit to count as 'valid'
    beta: float = 0.5          # L2: max uplift in value of a payout dollar at full-limit loss
    capital_share: float = 0.8 # L2: share of exposure that is capital (disaster risk hits capital most)
    effort: float = 1.0        # L2: friction E (toil, delay); floored at e_min
    e_min: float = 0.25
    k: float = 0.25            # L4: fair-exchange coefficient (service fee on output)
    gamma: float = 1.0         # L4: circulation coefficient
    natural_cap: float = 0.35  # L3: market share above this is flagged 'unnatural'


def price(p, lp=LawParams(), term=None):
    term = core.term_dist(p) if term is None else term
    L1, L2 = core.losses(p)
    f1, f2 = p.eq_lim[0] + p.pr_lim[0], p.eq_lim[1] + p.pr_lim[1]
    s1 = (L1 > 0) & (L1 / f1 >= lp.Dm)          # sigma step function per unit
    s2 = (L2 > 0) & (L2 / f2 >= lp.Dm)
    L = L1 + L2
    V1 = term @ (L1 * s1 + L2 * s2)                         # Law 1: nominal
    uplift = lp.beta * lp.capital_share * np.minimum(L / p.T, 1)   # V_up - V_down (V_down = 1)
    F = uplift / max(lp.effort, lp.e_min)                   # Law 2: flow
    V2 = term @ (L * F)                                     # real (value-weighted payoff)
    V4 = lp.k * (V1 + V2) * lp.gamma                        # Law 4: market price
    P = V1 + V2 + V4                                        # Law 3: consolidated
    shares = np.array([V1, V2, V4]) / P
    return dict(nominal=V1, real=V2, market=V4, premium=P, premium_per_year=P / p.years,
                rate_on_line=P / (f1 + f2), shares=shares, chain_valid_prob=float(term @ (s1 & s2)),
                mean_flow=float(term @ F), balanced=bool(shares[2] <= lp.natural_cap))


def variants(p):
    from dataclasses import replace
    return {"base": p, "independent": replace(p, c=0.0, cr_share=0.0),
            "common-shock": replace(p, c=2 * p.c), "rain-stress": replace(p, rain_mult=2.0)}
