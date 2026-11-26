"""EuroLeague prognozes — the same rules as NBA (3 bets >= 2.1, ★, Drošās >= 1.7, Crazy Dazy >= 4.0).

Data: the official EuroLeague feed (api-live.euroleague.net): schedule, results and quarter scores.
There are no bookmaker lines in that feed, so the probabilities come from the robot's own model
(team ratings, home court, rest), calibrated on the two previous seasons (`python el_bot.py kalibret`).
Bets are settled on regulation time (40 min) unless they say "(ieskaitot papildlaiku)".

  python el_bot.py likmes | rezultati | eksports | kalibret
History: el/history/, page data: el/export/, report: EL.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import date, timedelta
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import bot as core
import espn
import model as M
import odds

RIGA = ZoneInfo("Europe/Riga")
FEED = "https://api-live.euroleague.net/v2/competitions/E/seasons"
PARAMS_FILE = core.ROOT / "el_params.json"
DEFAULTS = dict(half_life=60.0, ridge=6.0, prior_shrink=0.5, hca_prior=3.0, sigma_margin=11.5, sigma_total=15.5,
                tie_k=1.7, ot_points=20.0, ot_margin=3.0, reg_offset=0.8)


def load_params() -> dict:
    p = dict(DEFAULTS)
    if PARAMS_FILE.exists():
        p.update(json.loads(PARAMS_FILE.read_text()))
    return p


def season_code(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1


# --------------------------------------------------------------------------- data

def _parse(g: dict) -> dict:
    loc, road = g["local"], g["road"]
    utc = pd.Timestamp(g["utcDate"])

    def reg(side):
        pt = side.get("partials") or {}
        q = [pt.get(f"partials{i}") for i in range(1, 5)]
        return float(sum(x for x in q if x is not None)) if all(x is not None for x in q) else None

    played = bool(g.get("played"))
    ot = bool((loc.get("partials") or {}).get("extraPeriods"))
    hp, ap = (float(loc["score"]), float(road["score"])) if played else (None, None)
    hr, ar = (reg(loc), reg(road)) if played else (None, None)
    if played and (hr is None or ar is None):
        hr, ar = hp, ap
    return {
        "game_id": g["identifier"], "date": utc.tz_convert(RIGA).date(), "start_utc": utc,
        "season": g["season"]["code"], "neutral": bool(g.get("isNeutralVenue")),
        "home_id": loc["club"]["code"], "away_id": road["club"]["code"],
        "home": loc["club"]["code"], "away": road["club"]["code"],
        "home_name": loc["club"].get("name", loc["club"]["code"]), "away_name": road["club"].get("name", road["club"]["code"]),
        "home_short": loc["club"].get("abbreviatedName") or loc["club"]["code"],
        "away_short": road["club"].get("abbreviatedName") or road["club"]["code"],
        "completed": played, "home_pts": hp, "away_pts": ap, "home_reg": hr, "away_reg": ar, "ot": played and ot,
        "home_record": None, "away_record": None,
    }


def load_season(year: int) -> pd.DataFrame:
    current = year == season_code(espn.today_et())
    raw = espn.get_json(f"/E{year}/games", ttl=600 if current else None, base=FEED)
    rows = [_parse(g) for g in raw.get("data", []) if g.get("local") and g.get("road")]
    df = pd.DataFrame(rows)
    return df.sort_values("start_utc").reset_index(drop=True) if not df.empty else df


def load_games(start: date, end: date) -> pd.DataFrame:
    """For settling: all games between two dates (Riga dates)."""
    years = {season_code(start), season_code(end)}
    df = pd.concat([load_season(y) for y in years], ignore_index=True)
    return df[(df["date"] >= start) & (df["date"] <= end)]


def _settle(text, g):
    return core._nba_settle(text, g)


EL = core.League("el", "EuroLeague", "🏀", core.ROOT / "el", core.ROOT / "EL.md", load_games, _settle, day_offset=0)


# --------------------------------------------------------------------------- model

def fit(games, as_of, prior, p):
    return M.fit_ratings(M.add_rest(games), as_of=as_of, prior=prior, half_life=p["half_life"], ridge=p["ridge"],
                         hca_prior=p["hca_prior"], b2b_prior=1.0)


def prior_for(year: int, p) -> M.Ratings | None:
    prev = load_season(year - 1)
    if prev.empty:
        return None
    return M.fit_ratings(M.add_rest(prev), half_life=90, ridge=2.0, hca_prior=p["hca_prior"], b2b_prior=1.0).shrunk(p["prior_shrink"])


def reg_settings(p) -> dict:
    return {k: p[k] for k in ("tie_k", "ot_points", "ot_margin", "reg_offset")}


def predict_day(day: date, p) -> pd.DataFrame:
    year = season_code(day)
    games = M.add_rest(load_season(year))
    todo = games[(games["date"] == day) & ~games["completed"]]
    if todo.empty:
        return pd.DataFrame()
    r = fit(games, todo["start_utc"].min(), prior_for(year, p), p)
    # bookmakers' lines from The Odds API (when ODDS_API_KEY is set): then, as for NBA, the bookmakers' spread and
    # total are used and the robot's model is shown next to them
    events = odds.fetch("el", "h2h,spreads,totals")
    rows = []
    for g in todo.itertuples():
        m = r.margin(g.home_id, g.away_id, g.neutral, g.home_b2b, g.away_b2b)
        t = r.total(g.home_id, g.away_id)
        mm = mt = None
        ev = odds.match(events, g.home_name, g.away_name, g.start_utc) if events else None
        if ev:
            ln = odds.lines_for(ev, g.home_name, g.away_name)
            if ln.get("spread_home") is not None:
                mm = -ln["spread_home"]
            elif ln.get("p_home") is not None:
                mm = p["sigma_margin"] * NormalDist().inv_cdf(min(max(ln["p_home"], 0.02), 0.98))
            mt = ln.get("total")
        rows.append({
            "date": g.date, "game_id": g.game_id, "start_utc": g.start_utc,
            "riga_time": g.start_utc.tz_convert(RIGA).strftime("%d.%m %H:%M"),
            "home": g.home, "away": g.away, "home_name": g.home_name, "away_name": g.away_name,
            "home_short": g.home_short, "away_short": g.away_short,
            "home_record": None, "away_record": None,
            "margin_home": mm if mm is not None else m, "model_margin_home": m,
            "total": mt if mt is not None else t, "model_total": t,
            "market_margin_home": mm, "market_total": mt, "has_market": mm is not None,
            "s_m": p["sigma_margin"], "s_t": p["sigma_total"], "reg": reg_settings(p),
            "home_net": r.net(g.home_id), "away_net": r.net(g.away_id),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- commands

def cmd_likmes(args) -> None:
    p = load_params()
    EL.message.unlink(missing_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    day = pd.Timestamp.now(tz=RIGA).date()
    df = pd.DataFrame()
    for d in (day, day + timedelta(days=1)):
        df = predict_day(d, p)
        if not df.empty:
            df = df[df["start_utc"] > now]
        if not df.empty:
            day = d
            break
    if df.empty:
        print("Šodien un rīt nav neizspēlētu EuroLeague spēļu.")
        return
    print(f"EUROLEAGUE LIKMES — {day:%d.%m.%Y}, min. koeficients {args.min_odds:g}")
    if df["has_market"].any():
        print("Bukmeikeru līnijas no The Odds API (kā NBA: likmes pēc bukmeikeru līnijām). Likmes vērtē pēc pamatlaika (40 min).")
    else:
        print("Bukmeikeru koeficientu nav: varbūtības ir no robota modeļa. Likmes vērtē pēc pamatlaika (40 min).")
    record, games_info = [], {}
    for r in df.itertuples():
        picks = core.pick_for_game(r, args.min_odds)
        mm = r.model_margin_home
        line = f"{r.home} {-mm:+.1f}" if mm >= 0 else f"{r.away} {mm:+.1f}"
        bk = (f"  |  bukmeikeri: {r.home if r.market_margin_home >= 0 else r.away} {-abs(r.market_margin_home):+.1f}"
              f", kopā {r.market_total}" if r.has_market else "")
        print(f"\n{r.away} @ {r.home}  {r.riga_time}  |  robots: {line}, kopā {r.model_total:.1f}{bk}")
        top = max(range(len(picks)), key=lambda i: picks[i].get("data_score", picks[i]["p"])) if picks else -1
        for i, x in enumerate(picks, 1):
            print(f"  {i}.{'★' if i - 1 == top else ' '}{x['bet']:<20} drošība {x['confidence']:>3}/100   "
                  f"godīgais koef. {x['fair_odds']:.2f}  (Optibet ~{x['est_odds']:.2f})   [{x['category']}]")
            record.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                           **x, "result": None})
        games_info[r.game_id] = {
            "game_id": r.game_id, "away": r.away, "home": r.home, "away_name": r.away_name, "home_name": r.home_name,
            "away_short": r.away_short, "home_short": r.home_short,
            "start_utc": str(r.start_utc), "riga_time": r.riga_time, "robot_line": line,
            "robot_total": round(float(r.model_total), 1),
            "market_line": (f"{r.home} {-r.market_margin_home:+.1f}" if r.has_market and r.market_margin_home >= 0
                            else f"{r.away} {r.market_margin_home:+.1f}" if r.has_market else "—"),
            "market_total": r.market_total if r.has_market and r.market_total is not None and pd.notna(r.market_total) else None,
            "injuries": [],
        }
    crazy = core.pick_crazy(df)
    safe = core.pick_safe(df)
    core.print_safe(safe)
    print(f"\nCRAZY DAZY: " + "; ".join(f"{x['game']} {x['bet']} (~{x['est_odds']:.2f})" for x in crazy))
    f = core.save_night(day, record, games_info, crazy, args.min_odds, now, safe=safe, safe_all=core.safe_all(df))
    print(f"\nSaglabāts: {f.relative_to(core.ROOT)}")


def cmd_kalibret(args) -> None:
    """Walk-forward on last season (prior from the season before) -> el_params.json"""
    p = dict(DEFAULTS)
    this = season_code(espn.today_et())
    tune, prev = load_season(this - 1), load_season(this - 2)
    tune = M.add_rest(tune[tune["completed"]])
    print(f"Sezona E{this - 1}: {len(tune)} spēles, prior no E{this - 2}: {len(prev)}")
    best = None
    for hl, rg, sh in itertools.product([30, 60, 1e9], [3, 6, 12], [0.3, 0.5, 0.7]):
        q = {**p, "half_life": hl, "ridge": rg, "prior_shrink": sh}
        prior = M.fit_ratings(M.add_rest(prev), half_life=90, ridge=2.0, hca_prior=q["hca_prior"], b2b_prior=1.0).shrunk(sh)
        pred, act, tot, atot = [], [], [], []
        for day, todays in tune.groupby("date"):
            r = fit(tune, todays["start_utc"].min(), prior, q)
            for g in todays.itertuples():
                pred.append(r.margin(g.home_id, g.away_id, g.neutral, g.home_b2b, g.away_b2b))
                tot.append(r.total(g.home_id, g.away_id))
                act.append(g.home_pts - g.away_pts)
                atot.append(g.home_pts + g.away_pts)
        mse = float(np.mean((np.array(pred) - np.array(act)) ** 2))
        if best is None or mse < best[0]:
            best = (mse, q, np.array(pred), np.array(act), np.array(tot), np.array(atot))
    mse, p, pred, act, tot, atot = best
    print(f"  labākie: half_life={p['half_life']:g}, ridge={p['ridge']}, prior_shrink={p['prior_shrink']}  (MAE {np.mean(np.abs(pred - act)):.2f})")
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
    reg_margin = (tune["home_reg"] - tune["away_reg"]).to_numpy()
    base0 = np.mean(phi((0.5 - pred) / p["sigma_margin"]) - phi((-0.5 - pred) / p["sigma_margin"]))
    p["tie_k"] = round(float(ot.mean() / max(base0, 1e-6)), 2)
    if ot.any():
        o = tune[ot]
        p["ot_points"] = round(float(((o.home_pts + o.away_pts) - (o.home_reg + o.away_reg)).mean()), 1)
        p["ot_margin"] = round(float((o.home_pts - o.away_pts).abs().mean()), 1)
    p["reg_offset"] = round(float(ot.mean() * p["ot_points"]), 2)
    acc = float(np.mean((pred > 0) == (act > 0)))
    print(f"  izkliede: starpība {p['sigma_margin']}, kopsumma {p['sigma_total']} | papildlaiks {ot.mean():.1%} "
          f"(tie_k {p['tie_k']}, +{p['ot_points']} p.) | mājas laukums {fit(tune, None, None, p).hca:+.1f} p.")
    print(f"  uzvarētāju uzminēja {acc:.1%}, vid. kļūda starpībā {np.mean(np.abs(pred - act)):.2f} p., "
          f"kopsummā {np.mean(np.abs(tot - atot)):.2f} p.")
    PARAMS_FILE.write_text(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in p.items()}, indent=2))
    print(f"  saglabāts {PARAMS_FILE.name}")


def main():
    ap = argparse.ArgumentParser(description="EuroLeague likmju robots")
    ap.add_argument("komanda", type=lambda s: s.lower().replace("ā", "a").replace("ū", "u"),
                    choices=["likmes", "rezultati", "eksports", "kalibret"])
    ap.add_argument("--min-odds", type=float, default=2.1)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    espn.FORCE_REFRESH = args.refresh
    core.configure(EL)
    if args.komanda == "kalibret":
        cmd_kalibret(args)
    elif args.komanda == "eksports":
        core.export_all()
    elif args.komanda == "likmes":
        cmd_likmes(args)
    else:
        core.cmd_rezultati(args)


if __name__ == "__main__":
    sys.exit(main())
