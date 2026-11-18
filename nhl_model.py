"""NHL model: team ratings on regulation goals + goalies + skater injuries -> Poisson score grid.

Regulation (60 min) goals for each side ~ Poisson(lambda), with a small boost to tied scores
(real hockey has more regulation ties than independent Poissons give). A regulation tie goes
to overtime/shootout, the winner gets +1 goal in the final score (ESPN/bookmaker convention).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import model as M

ROOT = Path(__file__).resolve().parent
PARAMS_FILE = ROOT / "nhl_params.json"

DEFAULTS = dict(
    half_life=60.0, ridge=8.0, prior_shrink=0.5,
    hca_prior=0.15, b2b_prior=0.15,
    tie_infl=0.10,        # extra weight on regulation ties
    ot_k=0.5,             # how much team strength matters in OT/shootout
    goalie_coef=1.0,      # multiplies goalie goals-saved-above-average per game
    skater_coef=0.5,      # multiplies skater values
    skater_repl=0.6,      # game score per 60 min of a replacement-level skater
    goalie_k=1500.0,      # shrinkage of save% towards league average (in shots)
    shots_pg=28.0,        # shots against per game used to turn save% into goals
    market_weight=0.8,    # blend towards bookmaker-implied goal expectations (log space)
    recent_games=10,
)
MAXG = 12


def load_params() -> dict:
    p = dict(DEFAULTS)
    if PARAMS_FILE.exists():
        p.update(json.loads(PARAMS_FILE.read_text()))
    return p


# --------------------------------------------------------------------------- team ratings

def rating_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Games table in the shape model.fit_ratings expects (points = regulation goals)."""
    g = M.add_rest(games)
    g = g.assign(home_pts=g["home_reg"], away_pts=g["away_reg"])
    g.loc[g["home_pts"].isna() & g["completed"], "completed"] = False
    return g


def fit(games_rf: pd.DataFrame, as_of=None, prior=None, p=None) -> M.Ratings:
    p = p or DEFAULTS
    return M.fit_ratings(games_rf, as_of=as_of, prior=prior, half_life=p["half_life"], ridge=p["ridge"],
                         hca_prior=p["hca_prior"], b2b_prior=p["b2b_prior"])


def season_prior(prev_games: pd.DataFrame, p=None) -> M.Ratings:
    p = p or DEFAULTS
    r = M.fit_ratings(rating_frame(prev_games), half_life=90, ridge=4.0,
                      hca_prior=p["hca_prior"], b2b_prior=p["b2b_prior"])
    return r.shrunk(p["prior_shrink"])


def lambdas(r: M.Ratings, home: str, away: str, neutral=False, hb2b=0, ab2b=0) -> tuple[float, float]:
    m = r.margin(home, away, neutral, hb2b, ab2b)
    t = r.total(home, away)
    return max(0.3, (t + m) / 2), max(0.3, (t - m) / 2)


# --------------------------------------------------------------------------- goalies

def goalie_values(goalies: pd.DataFrame, as_of=None, p=None) -> tuple[pd.DataFrame, float]:
    """Goals saved above average per game for each goalie (shrunk), and the league save%."""
    p = p or DEFAULTS
    gl = goalies if as_of is None else goalies[goalies["start_utc"] < as_of]
    gl = gl[gl["sa"] > 0]
    if gl.empty:
        return pd.DataFrame(columns=["name", "gsaa"]).set_index(pd.Index([], name="player_id")), 0.90
    lg = float(gl["sv"].sum() / gl["sa"].sum())
    agg = gl.groupby("player_id").agg(name=("name", "last"), sv=("sv", "sum"), sa=("sa", "sum"),
                                      gp=("game_id", "nunique"))
    sv_shr = (agg["sv"] + p["goalie_k"] * lg) / (agg["sa"] + p["goalie_k"])
    agg["gsaa"] = (sv_shr - lg) * p["shots_pg"]
    return agg, lg


def starters(goalies: pd.DataFrame) -> pd.DataFrame:
    """The goalie with the most ice time for each team in each game."""
    if goalies.empty:
        return goalies
    return goalies.sort_values("toi").groupby(["game_id", "team_id"]).tail(1)


def recent_goalie_mix(starts: pd.DataFrame, team_id: str, as_of, n: int, gv: pd.DataFrame) -> float:
    s = starts[starts["team_id"] == team_id]
    if as_of is not None:
        s = s[s["start_utc"] < as_of]
    s = s.nlargest(n, "start_utc")
    if s.empty:
        return 0.0
    return float(np.mean([gv["gsaa"].get(pid, 0.0) for pid in s["player_id"]]))


NEW_GOALIE_GSAA = -0.10  # a goalie with no NHL shots this season: slightly below average


def goalie_delta(team_id: str, starter_id: str | None, starts, gv, as_of, p) -> tuple[float, str | None]:
    """Change in goals saved vs the team's recent goalie mix (+ = better than usual)."""
    if not starter_id:
        return 0.0, None
    mix = recent_goalie_mix(starts, team_id, as_of, p["recent_games"], gv)
    tonight = float(gv["gsaa"].get(starter_id, NEW_GOALIE_GSAA))
    return p["goalie_coef"] * (tonight - mix), None


# --------------------------------------------------------------------------- skaters

def skater_values(skaters: pd.DataFrame, as_of=None, p=None, coef=None) -> pd.DataFrame:
    p = p or DEFAULTS
    coef = p["skater_coef"] if coef is None else coef
    sk = skaters if as_of is None else skaters[skaters["start_utc"] < as_of]
    sk = sk[sk["toi"] > 0]
    if sk.empty:
        return pd.DataFrame(columns=["name", "value"]).set_index(pd.Index([], name="player_id"))
    agg = sk.groupby("player_id").agg(name=("name", "last"), gp=("game_id", "nunique"),
                                      toi=("toi", "sum"), gs=("gs", "sum"))
    agg["toi_pg"] = agg["toi"] / agg["gp"]
    per60 = 60 * agg["gs"] / agg["toi"].clip(lower=1)
    rel = agg["gp"] / (agg["gp"] + 5)
    agg["value"] = coef * (per60 - p["skater_repl"]).clip(lower=0) * agg["toi_pg"] / 60 * rel
    return agg


def skater_logs_for_availability(skaters: pd.DataFrame) -> pd.DataFrame:
    return skaters.assign(min=skaters["toi"])


# --------------------------------------------------------------------------- score grid

_K = np.arange(MAXG + 1)
_FACT = np.array([math.factorial(int(i)) for i in _K], float)
_H, _A = np.meshgrid(_K, _K, indexing="ij")
_DIAG = _H == _A
_HW = _H > _A
_TOT_REG = (_H + _A).astype(float)


def reg_matrix(lh: float, la: float, tie_infl: float) -> np.ndarray:
    ph = np.exp(-lh) * lh ** _K / _FACT
    pa = np.exp(-la) * la ** _K / _FACT
    reg = np.outer(ph, pa)
    reg[_DIAG] *= (1 + tie_infl)
    return reg / reg.sum()


def p_ot_home(lh: float, la: float, ot_k: float) -> float:
    return float(np.clip(0.5 + ot_k * (lh - la) / (lh + la), 0.2, 0.8))


def fast_p_home(lh, la, tie_infl, ot_k) -> float:
    reg = reg_matrix(lh, la, tie_infl)
    return float(reg[_HW].sum() + reg[_DIAG].sum() * p_ot_home(lh, la, ot_k))


def fast_p_over(lh, la, tie_infl, line) -> float:
    reg = reg_matrix(lh, la, tie_infl)
    final_tot = _TOT_REG + _DIAG  # a tie gets +1 goal for the OT/shootout winner
    return float(reg[final_tot > line].sum())


class HockeyGrid:
    """All final-score scenarios: regulation goals (RH, RA), final (FH, FA), OT flag, weight W."""

    def __init__(self, lh: float, la: float, tie_infl: float, ot_k: float):
        reg = reg_matrix(lh, la, tie_infl)
        pot = p_ot_home(lh, la, ot_k)
        nd = ~_DIAG
        dh, da = _H[_DIAG], _A[_DIAG]
        self.RH = np.concatenate([_H[nd], dh, dh]).astype(float)
        self.RA = np.concatenate([_A[nd], da, da]).astype(float)
        self.FH = np.concatenate([_H[nd], dh + 1, dh]).astype(float)
        self.FA = np.concatenate([_A[nd], da, da + 1]).astype(float)
        self.OT = np.concatenate([np.zeros(nd.sum(), bool), np.ones(2 * len(dh), bool)])
        self.W = np.concatenate([reg[nd], reg[_DIAG] * pot, reg[_DIAG] * (1 - pot)])
        self.lh, self.la = lh, la

    def p(self, mask) -> float:
        return float(self.W[mask].sum())

    def p_home_win(self) -> float:
        return self.p(self.FH > self.FA)

    def p_over(self, line: float) -> float:
        return self.p(self.FH + self.FA > line)


def devig2(d1: float | None, d2: float | None) -> float | None:
    if not d1 or not d2:
        return None
    a, b = 1 / d1, 1 / d2
    return a / (a + b)


def market_lambdas(p_home: float, line: float | None, q_over: float | None, tie_infl: float, ot_k: float,
                   start_total: float) -> tuple[float, float]:
    """Goal expectations that reproduce the bookmaker's win probability and total line."""
    q = 0.5 if q_over is None else q_over

    def share_for(S):
        lo, hi = 0.15, 0.85
        for _ in range(30):
            mid = (lo + hi) / 2
            if fast_p_home(S * mid, S * (1 - mid), tie_infl, ot_k) < p_home:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    if line is None:
        S = start_total
        r = share_for(S)
        return S * r, S * (1 - r)
    lo, hi = 2.5, 11.0
    for _ in range(25):
        S = (lo + hi) / 2
        r = share_for(S)
        if fast_p_over(S * r, S * (1 - r), tie_infl, line) < q:
            lo = S
        else:
            hi = S
    S = (lo + hi) / 2
    r = share_for(S)
    return S * r, S * (1 - r)


def blend(model_l: tuple[float, float], market_l: tuple[float, float] | None, w: float) -> tuple[float, float]:
    if market_l is None or w <= 0:
        return model_l
    return tuple(math.exp((1 - w) * math.log(a) + w * math.log(b)) for a, b in zip(model_l, market_l))


@dataclass
class Side:
    adj: float = 0.0                # goals (+ helps this team)
    notes: list = field(default_factory=list)   # (name, status, delta)
