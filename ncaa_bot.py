"""NCAA men's college basketball — the 10 best bets of the night (odds >= 2.1) + 3-5 Crazy Dazy.

Unlike NBA there are far too many games for 3 bets each, so the robot takes the best bet of every
game and keeps the 10 most likely ones (one per game). Not part of the combined statistics page.

Data: ESPN (all Division I games: groups=50). Model: the NBA rating method on ~360 teams (points,
home court, neutral sites, rest), calibrated on last season (`python ncaa_bot.py kalibret`); where
ESPN has a betting line (bigger games) the bookmakers' line is used. Settled on regulation
(2 halves) unless a bet says "(ieskaitot papildlaiku)".

  python ncaa_bot.py likmes | rezultati | eksports | kalibret
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import bot as core
import data
import espn
import model as M

RIGA = ZoneInfo("Europe/Riga")
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
EXTRA = {"groups": "50", "limit": "500"}
SEASONS = {2026: (date(2025, 11, 1), date(2026, 4, 10)), 2027: (date(2026, 11, 1), date(2027, 4, 10))}
PARAMS_FILE = core.ROOT / "ncaa_params.json"
DEFAULTS = dict(half_life=45.0, ridge=4.0, prior_shrink=0.5, hca_prior=3.0, sigma_margin=11.0, sigma_total=15.5,
                tie_k=1.5, ot_points=17.0, ot_margin=3.0, reg_offset=0.9)
TOP_N = 10
CRAZY_N = 5


def load_params() -> dict:
    p = dict(DEFAULTS)
    if PARAMS_FILE.exists():
        p.update(json.loads(PARAMS_FILE.read_text()))
    return p


def season_for(d: date) -> int:
    return d.year + 1 if d.month >= 8 else d.year


# --------------------------------------------------------------------------- data

def _parse(e: dict, day: date) -> dict | None:
    row = data._parse_event(e, day)  # NBA parser: teams, scores, records, odds
    if row is None:
        return None
    t = {c["homeAway"]: c for c in e["competitions"][0]["competitors"]}
    lh = [float(x.get("value", 0) or 0) for x in t["home"].get("linescores") or []]
    la = [float(x.get("value", 0) or 0) for x in t["away"].get("linescores") or []]
    for side in ("home", "away"):
        tm = t[side]["team"]
        row[f"{side}_short"] = tm.get("shortDisplayName") or tm.get("location") or row[side]
    if row["completed"] and len(lh) >= 2 and len(la) >= 2:  # college: two halves, then overtime
        row["home_reg"], row["away_reg"], row["ot"] = sum(lh[:2]), sum(la[:2]), len(lh) > 2
    else:
        row["home_reg"], row["away_reg"], row["ot"] = row["home_pts"], row["away_pts"], False
    return row


def load_games(start: date, end: date) -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    with ThreadPoolExecutor(4) as ex:
        boards = list(ex.map(lambda d: espn.scoreboard(d, base=BASE, extra=EXTRA), days))
    rows = [r for day, sb in zip(days, boards) for e in sb.get("events", [])
            if (r := _parse(e, day)) and r["season_type"] in data.NON_PRESEASON]
    df = pd.DataFrame(rows)
    return df.drop_duplicates("game_id").sort_values("start_utc").reset_index(drop=True) if not df.empty else df


def load_season(year: int, until: date | None = None) -> pd.DataFrame:
    start, end = SEASONS[year]
    return load_games(start, min(end, until or espn.today_et() + timedelta(days=2)))


NCAA = core.League("ncaa", "NCAA", "🏀", core.ROOT / "ncaa", core.ROOT / "NCAA.md", load_games, core._nba_settle)


# --------------------------------------------------------------------------- model

def fit(games, as_of, prior, p):
    return M.fit_ratings(games, as_of=as_of, prior=prior, half_life=p["half_life"], ridge=p["ridge"],
                         hca_prior=p["hca_prior"], b2b_prior=1.0)


def prior_for(year: int, p):
    if year - 1 not in SEASONS:
        return None
    prev = M.add_rest(load_season(year - 1))
    return M.fit_ratings(prev, half_life=90, ridge=2.0, hca_prior=p["hca_prior"], b2b_prior=1.0).shrunk(p["prior_shrink"])


def reg_settings(p) -> dict:
    return {k: p[k] for k in ("tie_k", "ot_points", "ot_margin", "reg_offset")}


def predict_day(day: date, p) -> pd.DataFrame:
    year = season_for(day)
    games = M.add_rest(load_season(year, until=day))
    todo = games[(games["date"] == day) & ~games["completed"]]
    if todo.empty:
        return pd.DataFrame()
    r = fit(games, todo["start_utc"].min(), prior_for(year, p), p)
    rows = []
    for g in todo.itertuples():
        m = r.margin(g.home_id, g.away_id, g.neutral, g.home_b2b, g.away_b2b)
        t = r.total(g.home_id, g.away_id)
        has_m = g.market_margin is not None and pd.notna(g.market_margin)
        has_t = g.market_total is not None and pd.notna(g.market_total)
        rows.append({
            "date": g.date, "game_id": g.game_id, "start_utc": g.start_utc,
            "riga_time": g.start_utc.tz_convert(RIGA).strftime("%d.%m %H:%M"),
            "home": g.home, "away": g.away, "home_name": g.home_name, "away_name": g.away_name,
            "home_short": g.home_short, "away_short": g.away_short,
            "home_record": g.home_record, "away_record": g.away_record,
            # bookmakers' line where ESPN has one (the simulation showed it is the better number)
            "margin_home": float(g.market_margin) if has_m else m, "model_margin_home": m,
            "total": float(g.market_total) if has_t else t, "model_total": t,
            "market_margin_home": float(g.market_margin) if has_m else None,
            "market_total": float(g.market_total) if has_t else None, "has_market": bool(has_m),
            "s_m": p["sigma_margin"], "s_t": p["sigma_total"], "reg": reg_settings(p),
        })
    return pd.DataFrame(rows)


def best_across_games(df: pd.DataFrame, min_odds: float, n: int) -> list[dict]:
    """The best bet of each game, then the n most likely (one per game; a 2nd per game only if needed)."""
    firsts, seconds = [], []
    for r in df.itertuples():
        picks = core.pick_for_game(r, min_odds)
        for i, x in enumerate(sorted(picks, key=lambda x: -x["p"])[:2]):
            x = {"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                 "riga_time": r.riga_time, **x, "market": bool(r.has_market), "result": None}
            (firsts if i == 0 else seconds).append(x)
    # at odds >= 2.1 nearly every game offers a line at ~44 %, so after the rounded safety the games with a
    # bookmakers' line come first (a firmer number than our own model), then the bookmakers' own probability
    key = lambda x: (-x["confidence"], not x["market"], -(x.get("data_score") or x["p"]))
    out = sorted(firsts, key=key)[:n]
    if len(out) < n:
        out += sorted(seconds, key=key)[: n - len(out)]
    return out


# --------------------------------------------------------------------------- commands

def cmd_likmes(args) -> None:
    p = load_params()
    NCAA.message.unlink(missing_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    day = espn.today_et()
    df = pd.DataFrame()
    for d in (day, day + timedelta(days=1)):
        df = predict_day(d, p)
        if not df.empty:
            df = df[df["start_utc"] > now]
        if not df.empty:
            day = d
            break
    if df.empty:
        print("Tuvākajā laikā nav neizspēlētu NCAA spēļu.")
        return
    best = best_across_games(df, args.min_odds, TOP_N)
    print(f"NCAA — {len(df)} spēles naktī uz {(day + timedelta(days=1)):%d.%m.%Y}; {len(best)} labākās likmes (koef. >= {args.min_odds:g}):")
    for i, x in enumerate(best, 1):
        print(f"  {i:>2}. {x['game']:<14} {x['bet']:<20} drošība {x['confidence']:>3}/100  ~{x['est_odds']:.2f}  "
              f"{'(bukm. līnija)' if x['market'] else '(robota modelis)'}")
    crazy = core.pick_crazy(df, core.CRAZY_MIN_ODDS, CRAZY_N)
    print("CRAZY DAZY: " + "; ".join(f"{x['game']} {x['bet']} (~{x['est_odds']:.2f})" for x in crazy))
    used = {x["game_id"] for x in best + crazy}
    games_info = {}
    for r in df.itertuples():
        if r.game_id not in used:
            continue
        line = f"{r.home} {-r.margin_home:+.1f}" if r.margin_home >= 0 else f"{r.away} {r.margin_home:+.1f}"
        games_info[r.game_id] = {
            "game_id": r.game_id, "away": r.away, "home": r.home, "away_name": r.away_name, "home_name": r.home_name,
            "away_short": r.away_short, "home_short": r.home_short,
            "away_record": r.away_record, "home_record": r.home_record, "start_utc": str(r.start_utc),
            "riga_time": r.riga_time, "robot_line": line, "robot_total": round(float(r.total), 1),
            "market_line": P_fmt(r), "injuries": [],
            "market_total": float(r.market_total) if r.has_market and pd.notna(r.market_total) else None,
        }
    f = core.save_night(day, best, games_info, crazy, args.min_odds, now, safe=[], safe_all=core.safe_all(df))
    print(f"Saglabāts: {f.relative_to(core.ROOT)}")


def P_fmt(r) -> str:
    if r.market_margin_home is None or pd.isna(r.market_margin_home):
        return "—"
    m = r.market_margin_home
    return f"{r.home} -{m:.1f}" if m >= 0 else f"{r.away} -{-m:.1f}"


def cmd_kalibret(args) -> None:
    """Walk-forward on last season (prior from the season before) -> ncaa_params.json"""
    p = dict(DEFAULTS)
    this = season_for(espn.today_et())
    tune = M.add_rest(load_season(this - 1))
    tune = tune[tune["completed"]]
    print(f"NCAA sezona {this - 1}: {len(tune)} spēles")
    best = None
    for hl, rg in itertools.product([45, 1e9], [2, 6]):
        q = {**p, "half_life": hl, "ridge": rg}
        pred, act, tot, atot = [], [], [], []
        days = sorted(tune["date"].unique())
        for day in days[14:]:  # the first two weeks have almost no information
            todays = tune[tune["date"] == day]
            r = fit(tune, todays["start_utc"].min(), None, q)
            for g in todays.itertuples():
                pred.append(r.margin(g.home_id, g.away_id, g.neutral, g.home_b2b, g.away_b2b))
                tot.append(r.total(g.home_id, g.away_id))
                act.append(g.home_pts - g.away_pts)
                atot.append(g.home_pts + g.away_pts)
        pred, act, tot, atot = map(np.array, (pred, act, tot, atot))
        mse = float(np.mean((pred - act) ** 2))
        print(f"  half_life={hl:g} ridge={rg}: vid. kļūda {np.mean(np.abs(pred - act)):.2f}, uzvarētāji {np.mean((pred > 0) == (act > 0)):.1%}")
        if best is None or mse < best[0]:
            best = (mse, q, pred, act, tot, atot)
    _, p, pred, act, tot, atot = best
    phi = lambda x: 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))

    def pooled(pr, ac, offs, s):
        ll = 0.0
        for o in offs:
            q = np.clip(1 - phi(o / s), 1e-6, 1 - 1e-6)
            y = ac > pr + o
            ll += -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
        return ll
    p["sigma_margin"] = float(min(np.arange(8, 18, 0.25), key=lambda s: pooled(pred, act, np.arange(-14.5, 15, 1), s)))
    p["sigma_total"] = float(min(np.arange(10, 24, 0.25), key=lambda s: pooled(tot, atot, np.arange(-20.5, 21, 1), s)))
    ot = tune["ot"].astype(bool)
    if ot.any():
        o = tune[ot]
        p["ot_points"] = round(float(((o.home_pts + o.away_pts) - (o.home_reg + o.away_reg)).mean()), 1)
        p["ot_margin"] = round(float((o.home_pts - o.away_pts).abs().mean()), 1)
    p["reg_offset"] = round(float(ot.mean() * p["ot_points"]), 2)
    print(f"  -> half_life={p['half_life']:g}, ridge={p['ridge']}, izkliede {p['sigma_margin']}/{p['sigma_total']}, "
          f"papildlaiks {ot.mean():.1%} (+{p['ot_points']} p.)")
    PARAMS_FILE.write_text(json.dumps({k: (float(v) if isinstance(v, np.floating) else v) for k, v in p.items()}, indent=2))


def main():
    ap = argparse.ArgumentParser(description="NCAA basketbola likmju robots")
    ap.add_argument("komanda", type=lambda s: s.lower().replace("ā", "a").replace("ū", "u"),
                    choices=["likmes", "rezultati", "eksports", "kalibret"])
    ap.add_argument("--min-odds", type=float, default=2.1)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    espn.FORCE_REFRESH = args.refresh
    core.configure(NCAA)
    {"kalibret": cmd_kalibret, "eksports": lambda a: core.export_all(), "likmes": cmd_likmes,
     "rezultati": core.cmd_rezultati}[args.komanda](args)


if __name__ == "__main__":
    sys.exit(main())
