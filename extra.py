"""Papildu likmes: both teams to score, first goal, 1st period / 1st half / 1st quarter markets (odds >= 1.7).

The bookmakers' prices for these markets are not in the ESPN feed, so the robot works them out from the
game's expected goals / points (the same numbers its normal bets use: the bookmakers' main lines, for NHL
blended 80/20 with the robot's model) and the shares measured on last season:
  NHL      both teams score (regulation), first goal, 1st period total / team / 1X2
  football both teams score, first goal, 1st half total / 1X2 (90-minute games)
  NBA      1st quarter and 1st half total and handicap
`python extra.py kalibret` measures the shares and checks every market (predicted vs. real) on last season;
markets the robot does not predict well are switched off. In every game the most likely bet whose estimated
odds are still >= 1.7 is offered.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARAMS_FILE = ROOT / "extra_params.json"
PAYOUT = 0.93
MIN_ODDS = 1.7
REFETCH = False  # set by `kalibret --refetch` (once, to fill old cached days with goal minutes)
DEFAULTS = {"nhl_p1_share": 0.31, "nhl_ot_goal": 0.6, "fb_h1_share": 0.44,
            "nba_q1_share": 0.25, "nba_q1_sd_t": 8.5, "nba_q1_m": 0.25, "nba_q1_sd_m": 7.0,
            "nba_h1_share": 0.505, "nba_h1_sd_t": 12.5, "nba_h1_m": 0.5, "nba_h1_sd_m": 9.0,
            "disabled": []}


def load_params() -> dict:
    p = dict(DEFAULTS)
    if PARAMS_FILE.exists():
        p.update(json.loads(PARAMS_FILE.read_text()))
    return p


# --------------------------------------------------------------------------- probabilities

def _pois(lam: float, kmax: int = 15) -> np.ndarray:
    k = np.arange(kmax + 1)
    return np.exp(-lam + k * math.log(max(lam, 1e-9)) - np.array([math.lgamma(x + 1) for x in k]))


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def goal_markets(home: str, away: str, lh: float, la: float, p: dict, kind: str) -> list[tuple[str, str, float]]:
    """(market, bet label, probability) for hockey ('nhl') or football ('fb'); lh/la = expected goals
    in regulation / 90 minutes."""
    out = []
    ph, pa = _pois(lh), _pois(la)
    # both teams score (regulation / 90 minutes)
    both = (1 - ph[0]) * (1 - pa[0])
    out += [("abas gūst", "abas komandas gūst vārtus", both), ("abas gūst", "abas komandas negūst vārtus", 1 - both)]
    # first goal of the game (hockey: overtime counts, the shootout does not)
    tot = lh + la
    none = math.exp(-tot)
    extra = none * (p["nhl_ot_goal"] if kind == "nhl" else 0.0)
    out += [("pirmie vārti", f"{home} gūst pirmos vārtus", lh / tot * (1 - none) + lh / tot * extra),
            ("pirmie vārti", f"{away} gūst pirmos vārtus", la / tot * (1 - none) + la / tot * extra)]
    # 1st period / 1st half
    share = p["nhl_p1_share"] if kind == "nhl" else p["fb_h1_share"]
    word = "1. periods" if kind == "nhl" else "1. puslaiks"
    h1, a1 = _pois(lh * share), _pois(la * share)
    grid = np.outer(h1, a1)
    s = np.add.outer(np.arange(len(h1)), np.arange(len(a1)))
    d = np.subtract.outer(np.arange(len(h1)), np.arange(len(a1)))
    for line in (0.5, 1.5, 2.5):
        over = float(grid[s > line].sum())
        out += [(f"{word} kopā", f"{word} virs {line:g}", over), (f"{word} kopā", f"{word} zem {line:g}", 1 - over)]
    for team, arr in ((home, h1), (away, a1)):
        out.append((f"{word} komanda", f"{word} {team} virs 0.5", float(1 - arr[0])))
    out += [(f"{word} 1X2", f"{word} {home} uzvar", float(grid[d > 0].sum())),
            (f"{word} 1X2", f"{word} neizšķirts", float(grid[d == 0].sum())),
            (f"{word} 1X2", f"{word} {away} uzvar", float(grid[d < 0].sum()))]
    return out


def nba_markets(home: str, away: str, margin: float, total: float, p: dict) -> list[tuple[str, str, float]]:
    """1st quarter / 1st half totals and handicaps from the game's margin and total (normal approximation)."""
    out = []
    for key, word in (("q1", "1. ceturtdaļa"), ("h1", "1. puslaiks")):
        mt, sdt = total * p[f"nba_{key}_share"], p[f"nba_{key}_sd_t"]
        mm, sdm = margin * p[f"nba_{key}_m"], p[f"nba_{key}_sd_m"]
        for x in range(-12, 13):
            line = round(mt) + x + 0.5
            over = 1 - _phi((line - mt) / sdt)
            out += [(f"{word} kopā", f"{word} virs {line:g}", over), (f"{word} kopā", f"{word} zem {line:g}", 1 - over)]
        for x in range(-12, 13):
            line = x + 0.5  # home team -line
            p_home = 1 - _phi((line - mm) / sdm)
            out += [(f"{word} handikaps", f"{word} {home} {-line:+g}", p_home),
                    (f"{word} handikaps", f"{word} {away} {line:+g}", 1 - p_home)]
    return out


def best_bet(markets: list[tuple[str, str, float]], disabled: list[str], sport: str) -> tuple[str, str, float] | None:
    """The most likely bet whose estimated odds are still >= 1.7 (markets that failed the check are skipped)."""
    ok = [m for m in markets if f"{sport}:{m[0]}" not in disabled and 0.25 <= m[2] <= PAYOUT / MIN_ODDS]
    return max(ok, key=lambda m: m[2]) if ok else None


def pick(rows: list[dict], sport: str) -> list[dict]:
    """rows: [{game_id, game, start_utc, riga_time, away_short, home_short, markets}] -> one bet per game."""
    p = load_params()
    out = []
    for r in rows:
        b = best_bet(r["markets"], p.get("disabled", []), sport)
        if not b:
            continue
        cat, label, pr = b
        x = {k: r[k] for k in ("game_id", "game", "start_utc", "riga_time") if k in r}
        x.update({"category": cat, "bet": label, "p": pr, "fair_odds": round(1 / pr, 2),
                  "est_odds": round(PAYOUT / pr, 2), "confidence": int(round(100 * pr)), "result": None})
        for k in ("away_short", "home_short"):
            if r.get(k):
                x[k] = r[k]
        out.append(x)
    out.sort(key=lambda x: x["start_utc"])
    return out


# --------------------------------------------------------------------------- settling

def settle_goal_bet(text: str, home: str, away: str, reg: tuple, first: str | None, p1: tuple | None
                    ) -> tuple[bool, bool]:
    """reg = (home, away) regulation / 90-min goals; first = 'home' | 'away' | None (no goal before the
    shootout); p1 = (home, away) goals in the 1st period / half."""
    t = text.strip()
    if t == "abas komandas gūst vārtus":
        return reg[0] > 0 and reg[1] > 0, False
    if t == "abas komandas negūst vārtus":
        return not (reg[0] > 0 and reg[1] > 0), False
    m = re.fullmatch(r"(\S+) gūst pirmos vārtus", t)
    if m:
        side = "home" if m.group(1) == home else "away"
        return first == side, False
    m = re.fullmatch(r"1\. (?:periods|puslaiks) (.+)", t)
    if not m or p1 is None or None in p1:
        raise ValueError("nav 1. perioda / puslaika rezultāta")
    rest, h, a = m.group(1), p1[0], p1[1]
    if (mm := re.fullmatch(r"(virs|zem) ([\d.]+)", rest)):
        over = h + a > float(mm.group(2))
        return over if mm.group(1) == "virs" else not over, False
    if (mm := re.fullmatch(r"(\S+) virs 0\.5", rest)):
        return (h if mm.group(1) == home else a) > 0, False
    if rest == "neizšķirts":
        return h == a, False
    if (mm := re.fullmatch(r"(\S+) uzvar", rest)):
        return (h > a) if mm.group(1) == home else (a > h), False
    raise ValueError(f"nepazīstama likme '{text}'")


def settle_nba_bet(text: str, home: str, away: str, hq: list, aq: list) -> tuple[bool, bool]:
    m = re.fullmatch(r"1\. (ceturtdaļa|puslaiks) (.+)", text.strip())
    if not m or not hq or not aq:
        raise ValueError("nav ceturtdaļu rezultātu")
    n = 1 if m.group(1) == "ceturtdaļa" else 2
    h, a, rest = sum(hq[:n]), sum(aq[:n]), m.group(2)
    if (mm := re.fullmatch(r"(virs|zem) ([\d.]+)", rest)):
        over = h + a > float(mm.group(2))
        return over if mm.group(1) == "virs" else not over, False
    if (mm := re.fullmatch(r"(\S+) ([+-][\d.]+)", rest)):
        diff = (h - a) if mm.group(1) == home else (a - h)
        v = diff + float(mm.group(2))
        return v > 0, v == 0
    raise ValueError(f"nepazīstama likme '{text}'")


def football_goal_info(g) -> tuple[str | None, tuple | None]:
    """First goal side and 1st-half goals from the scoreboard's goal list (None if it is incomplete)."""
    goals = g.goals if isinstance(getattr(g, "goals", None), list) else None
    if not goals:
        return (None, (0.0, 0.0)) if (g.home_reg == 0 and g.away_reg == 0) else (None, None)
    rows = []
    for x in goals:
        if x.get("so"):
            continue
        mm = re.match(r"(\d+)'(?:\+(\d+))?", str(x.get("clock") or ""))
        if not mm:
            return None, None
        minute = int(mm.group(1))
        side = "home" if str(x.get("team")) == str(g.home_id) else "away"
        rows.append((minute, int(mm.group(2) or 0), side))
    reg = [r for r in rows if r[0] <= 90]
    if (sum(r[2] == "home" for r in reg), sum(r[2] == "away" for r in reg)) != (g.home_reg, g.away_reg):
        return None, None  # e.g. own goals credited the other way: do not guess
    rows.sort()
    first = rows[0][2] if rows else None
    h1 = [r for r in rows if r[0] <= 45]
    return first, (float(sum(r[2] == "home" for r in h1)), float(sum(r[2] == "away" for r in h1)))


def nhl_first_goal(game_id: str, home_id: str) -> str | None:
    """'home' / 'away' from the game's play-by-play (None: no goal before the shootout)."""
    import espn
    s = espn.get_json("/summary", {"event": game_id}, ttl=None, base=espn.NHL)
    for pl in s.get("plays") or []:
        if pl.get("scoringPlay") and int((pl.get("period") or {}).get("number", 0) or 0) <= 4:
            return "home" if str((pl.get("team") or {}).get("id")) == str(home_id) else "away"
    return None


def settle(league: str, text: str, g) -> tuple[bool, bool, str]:
    """Settle one Papildu likmes bet on a finished game row of that league."""
    score = f"{g.away} {int(g.away_reg)} - {int(g.home_reg)} {g.home}"
    if league == "nba":
        w, push = settle_nba_bet(text, g.home, g.away, g.home_q, g.away_q)
        q1 = f"1. ceturtdaļa {int(g.away_q[0])} - {int(g.home_q[0])}"
        h1 = f"1. puslaiks {int(sum(g.away_q[:2]))} - {int(sum(g.home_q[:2]))}"
        return w, push, f"{score} ({q1}; {h1})"
    if league == "nhl":
        first = nhl_first_goal(g.game_id, g.home_id) if "pirmos vārtus" in text else None
        p1 = (g.home_p1, g.away_p1)
        w, push = settle_goal_bet(text, g.home, g.away, (g.home_reg, g.away_reg), first, p1)
        return w, push, f"{score} (pamatlaikā; 1. periods {int(g.away_p1)} - {int(g.home_p1)})"
    first, h1 = football_goal_info(g)
    if h1 is None:
        raise ValueError("nav vārtu minūšu")
    w, push = settle_goal_bet(text, g.home, g.away, (g.home_reg, g.away_reg), first, h1)
    return w, push, f"{score} (90 min.; 1. puslaiks {int(h1[1])} - {int(h1[0])})"


# --------------------------------------------------------------------------- calibration on last season

def _check(rows: list[tuple[str, float, bool]]) -> dict:
    """market -> predicted vs. real, for bets the page could offer (probability 0.40-0.70)."""
    df = pd.DataFrame(rows, columns=["market", "p", "won"])
    df = df[(df["p"] >= 0.40) & (df["p"] <= 0.70)]
    out = {}
    for m, g in df.groupby("market"):
        out[m] = {"n": int(len(g)), "pred": round(float(g["p"].mean()), 3), "real": round(float(g["won"].mean()), 3)}
    return out


def calibrate_nhl(p: dict) -> dict:
    import nhl_data as D
    import nhl_model as N
    games = D.load_season(2026)
    games = games[games["completed"] & games["home_reg"].notna() & games["home_p1"].notna()]
    reg_goals = (games["home_reg"] + games["away_reg"]).sum()
    p["nhl_p1_share"] = round(float((games["home_p1"] + games["away_p1"]).sum() / reg_goals), 3)
    _, _, odds = D.load_boxscores(games)
    g = games.merge(odds, on="game_id")
    g = g[g["close_ml_home"].notna() & g["close_ml_away"].notna() & g["close_total"].notna()]
    q = N.load_params()
    rows = []
    for x in g.itertuples():
        qo = N.devig2(x.close_over, x.close_under) if not (pd.isna(x.close_over) or pd.isna(x.close_under)) else 0.5
        lh, la = N.market_lambdas(N.devig2(x.close_ml_home, x.close_ml_away), x.close_total, qo,
                                  q["tie_infl"], q["ot_k"], x.close_total)
        for market, label, pr in goal_markets(x.home, x.away, lh, la, p, "nhl"):
            if "pirmos vārtus" in label:
                continue  # needs play-by-play; the share model is checked through the totals instead
            w, _ = settle_goal_bet(label, x.home, x.away, (x.home_reg, x.away_reg), None, (x.home_p1, x.away_p1))
            rows.append((market, pr, w))
    print(f"NHL: {len(g)} spēles, 1. perioda vārtu daļa {p['nhl_p1_share']:.1%}")
    return _check(rows)


def calibrate_football(p: dict) -> dict:
    import cl_bot as F
    import espn
    import nhl_model as N
    data_rows = []
    for base, season in (("https://site.api.espn.com/apis/site/v2/sports/soccer/eng.2", 2025),
                         ("https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions", 2025)):
        F.BASE = base
        start, end = F.SEASONS[season]
        espn.FORCE_ALL = REFETCH  # older cached scoreboards were stored without the goal list
        try:
            games = F.load_games(start, end)
        finally:
            espn.FORCE_ALL = False
        games = games[games["completed"] & games["home_reg"].notna() & ~games["extra_time"].fillna(False)]
        odds = F.closing_odds(games["game_id"].tolist())
        g = games.drop(columns=[c for c in odds.columns if c != "game_id"]).merge(odds, on="game_id")
        n_ok = 0
        for x in g.itertuples():
            first, h1 = football_goal_info(x)
            if h1 is None:
                continue
            n_ok += 1
            data_rows.append((x, first, h1))
        print(f"{base.rsplit('/', 1)[1]}: {len(g)} spēles, ar pilnām vārtu minūtēm {n_ok}")
    h1g = sum(h1[0] + h1[1] for _, _, h1 in data_rows)
    allg = sum(x.home_reg + x.away_reg for x, _, _ in data_rows)
    if allg:
        p["fb_h1_share"] = round(float(h1g / allg), 3)
        print(f"futbols: 1. puslaika vārtu daļa {p['fb_h1_share']:.1%}")
    rows = []
    for x, first, h1 in data_rows:
        p3 = F.devig3(x.ml_home, x.ml_draw, x.ml_away)
        if p3 is None or x.total_line is None or pd.isna(x.total_line):
            continue
        qo = N.devig2(x.over_odds, x.under_odds) if x.over_odds and x.under_odds else 0.5
        lh, la = F.solve_lambdas(p3, x.total_line, qo, 0.10)
        for market, label, pr in goal_markets(x.home, x.away, lh, la, p, "fb"):
            w, _ = settle_goal_bet(label, x.home, x.away, (x.home_reg, x.away_reg), first, h1)
            rows.append((market, pr, w))
    return _check(rows)


def calibrate_nba(p: dict) -> dict:
    import data
    games = data.load_season(2026)
    games = games[games["completed"] & games["home_q"].notna()]
    _, lines = data.load_player_logs(games)
    g = games.merge(lines, on="game_id").dropna(subset=["close_margin", "close_total"])
    hq, aq = np.array(g["home_q"].tolist()), np.array(g["away_q"].tolist())
    for key, n in (("q1", 1), ("h1", 2)):
        tot = hq[:, :n].sum(1) + aq[:, :n].sum(1)
        mar = hq[:, :n].sum(1) - aq[:, :n].sum(1)
        T, M = g["close_total"].to_numpy(), g["close_margin"].to_numpy()
        p[f"nba_{key}_share"] = round(float(np.sum(tot * T) / np.sum(T * T)), 4)
        p[f"nba_{key}_sd_t"] = round(float(np.std(tot - p[f"nba_{key}_share"] * T)), 2)
        p[f"nba_{key}_m"] = round(float(np.sum(mar * M) / np.sum(M * M)), 3)
        p[f"nba_{key}_sd_m"] = round(float(np.std(mar - p[f"nba_{key}_m"] * M)), 2)
    rows = []
    for x, h, a in zip(g.itertuples(), hq, aq):
        for market, label, pr in nba_markets(x.home, x.away, x.close_margin, x.close_total, p):
            if not 0.40 <= pr <= 0.70:
                continue
            w, push = settle_nba_bet(label, x.home, x.away, list(h), list(a))
            if not push:
                rows.append((market, pr, w))
    print(f"NBA: {len(g)} spēles; 1. ceturtdaļa {p['nba_q1_share']:.1%} no kopējiem punktiem (±{p['nba_q1_sd_t']}), "
          f"1. puslaiks {p['nba_h1_share']:.1%} (±{p['nba_h1_sd_t']})")
    return _check(rows)


def cmd_kalibret(args):
    p = load_params()
    checks = {}
    for name, fn in (("nba", calibrate_nba), ("nhl", calibrate_nhl), ("fb", calibrate_football)):
        try:
            checks[name] = fn(p)
        except Exception as e:  # one league's data problem must not stop the others
            print(f"{name}: kalibrēšana neizdevās: {e}")
    disabled = []
    print("\nTirgus: prognozēts -> realitātē (likmes, ko lapa varētu piedāvāt, varbūtība 40-70 %)")
    for lg, ch in checks.items():
        for m, c in ch.items():
            # switched off when the real rate is clearly away from the prediction (3 points or 1.4 standard errors)
            off = abs(c["pred"] - c["real"]) > max(0.03, 1.4 * math.sqrt(c["pred"] * (1 - c["pred"]) / max(c["n"], 1)))
            if off or c["n"] < 60:
                disabled.append(f"{lg}:{m}")
            print(f"  {lg:<8} {m:<26} {c['n']:>6}  {c['pred']:.1%} -> {c['real']:.1%}"
                  f"{'  ✗ izslēgts' if off else ('  ✗ par maz datu' if c['n'] < 60 else '')}")
    p["disabled"] = sorted(set(disabled))
    p["checks"] = checks
    PARAMS_FILE.write_text(json.dumps(p, indent=2, ensure_ascii=False))
    print(f"\nSaglabāts: {PARAMS_FILE.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("komanda", choices=["kalibret"])
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    global REFETCH
    REFETCH = args.refetch
    cmd_kalibret(args)


if __name__ == "__main__":
    sys.exit(main())
