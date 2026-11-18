"""Prediction model.

1. Team ratings: weighted ridge regression on points scored/allowed.
       points(A vs B) = mu + OFF_A + DEF_B + HCA*(+-0.5) + REST*(b2b effects)
   Recent games weigh more (exponential decay); early in the season ratings are
   pulled towards last season's (shrunk) ratings.
2. Player availability: every player gets a "value" in points of margin from his
   box-score production (Game Score per 36 min above replacement level).
   A team's rating already reflects who played recently, so the adjustment is
       value * (P(plays tonight) - share of the last 10 games he played).
   That handles both directions: a star who is out tonight costs points, a star
   returning from a long absence adds points.
3. Margin -> win probability via a normal distribution (sigma fitted in backtest).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Defaults; backtest.py prints values fitted on real data so they can be tuned.
DEFAULTS = dict(
    half_life=40.0,      # days: game weight halves every N days
    ridge=6.0,           # strength of pull towards prior ratings (in "games")
    prior_shrink=0.6,    # how much of last season's rating carries over
    sigma=12.5,          # SD of actual margin around predicted margin
    injury_coef=0.24,    # points of margin per unit of (GmSc/36 above replacement) * mpg/36 (fitted on 2025-26)
    replacement=6.0,     # Game Score per 36 min of a replacement-level player (fitted)
    recent_games=10,     # window for "how often did he play recently"
)

STATUS_AVAIL = {  # probability the player plays
    "out": 0.0, "suspension": 0.0, "doubtful": 0.2, "questionable": 0.5,
    "day-to-day": 0.6, "probable": 0.9, "available": 1.0,
}


def add_rest(games: pd.DataFrame) -> pd.DataFrame:
    """Flag back-to-backs (team also played the previous ET day)."""
    g = games.copy()
    long = pd.concat([
        g[["game_id", "date", "home_id"]].rename(columns={"home_id": "team"}),
        g[["game_id", "date", "away_id"]].rename(columns={"away_id": "team"}),
    ]).sort_values("date")
    long["prev"] = long.groupby("team")["date"].shift()
    long["b2b"] = [(d - p).days == 1 if isinstance(p, type(d)) else False
                   for d, p in zip(long["date"], long["prev"])]
    key = long.set_index(["game_id", "team"])["b2b"]
    g["home_b2b"] = [int(key.get((gid, t), False)) for gid, t in zip(g["game_id"], g["home_id"])]
    g["away_b2b"] = [int(key.get((gid, t), False)) for gid, t in zip(g["game_id"], g["away_id"])]
    return g


# --------------------------------------------------------------------------- team ratings

@dataclass
class Ratings:
    off: dict[str, float]
    deff: dict[str, float]
    mu: float = 114.0
    hca: float = 2.5
    b2b: float = 1.5
    n_games: int = 0

    def margin(self, home: str, away: str, neutral=False, home_b2b=0, away_b2b=0) -> float:
        o, d = self.off, self.deff
        m = (o.get(home, 0) + d.get(away, 0)) - (o.get(away, 0) + d.get(home, 0))
        if not neutral:
            m += self.hca
        return m + self.b2b * (away_b2b - home_b2b)

    def total(self, home: str, away: str) -> float:
        o, d = self.off, self.deff
        return 2 * self.mu + o.get(home, 0) + d.get(away, 0) + o.get(away, 0) + d.get(home, 0)

    def net(self, team: str) -> float:
        """Net rating vs average team on a neutral floor (points per game)."""
        return self.off.get(team, 0) - self.deff.get(team, 0)

    def shrunk(self, k: float) -> "Ratings":
        return Ratings({t: v * k for t, v in self.off.items()},
                       {t: v * k for t, v in self.deff.items()},
                       self.mu, self.hca, self.b2b)


def fit_ratings(games: pd.DataFrame, as_of: pd.Timestamp | None = None,
                prior: Ratings | None = None, half_life=None, ridge=None) -> Ratings:
    """Fit ratings on completed games that started before `as_of`."""
    half_life = DEFAULTS["half_life"] if half_life is None else half_life
    ridge = DEFAULTS["ridge"] if ridge is None else ridge
    g = games[games["completed"]]
    if as_of is not None:
        g = g[g["start_utc"] < as_of]
    teams = sorted(set(games["home_id"]) | set(games["away_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    P = 3 + 2 * n  # mu, hca, b2b, off[n], def[n]

    if g.empty:
        return prior or Ratings({t: 0.0 for t in teams}, {t: 0.0 for t in teams})

    ref = as_of if as_of is not None else g["start_utc"].max()
    age = (ref - g["start_utc"]).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / half_life) if half_life and half_life < 1e6 else np.ones(len(g))

    m = len(g)
    X = np.zeros((2 * m, P))
    y = np.zeros(2 * m)
    ww = np.repeat(w, 2)
    hi = g["home_id"].map(idx).to_numpy()
    ai = g["away_id"].map(idx).to_numpy()
    neutral = g["neutral"].to_numpy().astype(float)
    hb = g["home_b2b"].to_numpy() if "home_b2b" in g else np.zeros(m)
    ab = g["away_b2b"].to_numpy() if "away_b2b" in g else np.zeros(m)
    r = np.arange(m)
    # home team's points
    X[2 * r, 0] = 1
    X[2 * r, 1] = 0.5 * (1 - neutral)
    X[2 * r, 2] = 0.5 * (ab - hb)
    X[2 * r, 3 + hi] = 1
    X[2 * r, 3 + n + ai] = 1
    y[2 * r] = g["home_pts"].to_numpy()
    # away team's points
    X[2 * r + 1, 0] = 1
    X[2 * r + 1, 1] = -0.5 * (1 - neutral)
    X[2 * r + 1, 2] = 0.5 * (hb - ab)
    X[2 * r + 1, 3 + ai] = 1
    X[2 * r + 1, 3 + n + hi] = 1
    y[2 * r + 1] = g["away_pts"].to_numpy()

    sw = np.sqrt(ww)
    Xs, ys = X * sw[:, None], y * sw

    # Regularisation rows: team ratings -> prior, hca -> 2.5, b2b -> 1.5,
    # and sum-to-zero constraints so mu is the league average.
    reg_X, reg_y = [], []
    pri = prior or Ratings({}, {})
    for t, i in idx.items():
        for col, target in ((3 + i, pri.off.get(t, 0.0)), (3 + n + i, pri.deff.get(t, 0.0))):
            row = np.zeros(P); row[col] = math.sqrt(ridge)
            reg_X.append(row); reg_y.append(math.sqrt(ridge) * target)
    for col, target, lam in ((1, 2.5, 40.0), (2, 1.5, 40.0)):
        row = np.zeros(P); row[col] = math.sqrt(lam)
        reg_X.append(row); reg_y.append(math.sqrt(lam) * target)
    for start in (3, 3 + n):
        row = np.zeros(P); row[start:start + n] = 100.0
        reg_X.append(row); reg_y.append(0.0)

    A = np.vstack([Xs, np.array(reg_X)])
    b = np.concatenate([ys, np.array(reg_y)])
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return Ratings(
        off={t: float(beta[3 + i]) for t, i in idx.items()},
        deff={t: float(beta[3 + n + i]) for t, i in idx.items()},
        mu=float(beta[0]), hca=float(beta[1]), b2b=float(beta[2]), n_games=m,
    )


# --------------------------------------------------------------------------- players

def player_values(logs: pd.DataFrame, as_of: pd.Timestamp | None = None,
                  coef=None, replacement=None) -> pd.DataFrame:
    coef = DEFAULTS["injury_coef"] if coef is None else coef
    replacement = DEFAULTS["replacement"] if replacement is None else replacement
    lg = logs if as_of is None else logs[logs["start_utc"] < as_of]
    lg = lg[lg["min"] > 0]
    if lg.empty:
        return pd.DataFrame(columns=["player_id", "name", "gp", "mpg", "gmsc", "value"]).set_index("player_id")
    agg = lg.groupby("player_id").agg(name=("name", "last"), gp=("game_id", "nunique"),
                                      mins=("min", "sum"), gsum=("gmsc", "sum"), pts=("pts", "mean"))
    agg["mpg"] = agg["mins"] / agg["gp"]
    agg["gmsc"] = agg["gsum"] / agg["gp"]
    per36 = 36 * agg["gsum"] / agg["mins"].clip(lower=1)
    reliability = agg["gp"] / (agg["gp"] + 3)
    agg["value"] = coef * (per36 - replacement).clip(lower=0) * agg["mpg"] / 36 * reliability
    return agg[["name", "gp", "mpg", "gmsc", "pts", "value"]]


def recent_participation(logs: pd.DataFrame, team_id: str, as_of: pd.Timestamp | None,
                         n_games: int) -> tuple[dict[str, float], int]:
    """Share of the team's last n games each player appeared in."""
    lg = logs[logs["team_id"] == team_id]
    if as_of is not None:
        lg = lg[lg["start_utc"] < as_of]
    recent = lg.drop_duplicates("game_id").nlargest(n_games, "start_utc")["game_id"]
    if recent.empty:
        return {}, 0
    played = lg[lg["game_id"].isin(recent) & (lg["min"] > 0)]
    share = played.groupby("player_id")["game_id"].nunique() / len(recent)
    return share.to_dict(), len(recent)


@dataclass
class Availability:
    adj: float = 0.0
    notes: list[tuple[str, str, float]] = field(default_factory=list)  # (name, status, pts)


def availability_adjustment(team_id: str, logs: pd.DataFrame, values: pd.DataFrame,
                            plays_prob: dict[str, float], statuses: dict[str, str],
                            roster: set[str] | None, as_of: pd.Timestamp | None,
                            n_recent=None) -> Availability:
    """plays_prob: player_id -> P(plays). Missing players default to 1.0 if on roster."""
    n_recent = DEFAULTS["recent_games"] if n_recent is None else n_recent
    share, n_games = recent_participation(logs, team_id, as_of, n_recent)
    candidates = set(share) | (roster or set())
    out = Availability()
    for pid in candidates:
        if pid not in values.index:
            continue
        v = float(values.at[pid, "value"])
        if v <= 0:
            continue
        on_team = roster is None or pid in roster
        p_now = plays_prob.get(pid, 1.0) if on_team else 0.0
        delta = v * (p_now - share.get(pid, 0.0))
        out.adj += delta
        if abs(delta) >= 0.4:
            played = round(share.get(pid, 0.0) * n_games)
            if statuses.get(pid):
                status = statuses[pid]
            elif not on_team:
                status = "vairs nav komandā"
            elif played == 0:
                status = "jauns komandā / atgriežas"
            else:
                status = f"spēlēja {played}/{n_games} pēdējās"
            out.notes.append((values.at[pid, "name"], status, delta))
    out.notes.sort(key=lambda x: x[2])
    return out


def status_to_prob(status: str, return_date: str | None, game_date) -> float:
    p = STATUS_AVAIL.get((status or "").strip().lower(), 0.5)
    if p == 0.0 and return_date:
        try:
            if pd.Timestamp(return_date).date() <= game_date:
                p = 0.6  # expected back by then, still uncertain
        except ValueError:
            pass
    return p


def win_prob(margin: float, sigma=None) -> float:
    sigma = DEFAULTS["sigma"] if sigma is None else sigma
    return 0.5 * (1 + math.erf(margin / (sigma * math.sqrt(2))))
