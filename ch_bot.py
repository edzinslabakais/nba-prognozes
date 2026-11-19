"""English Championship — up to 10 of the day's safest bets with odds >= 2.1 (one per game) + Crazy Dazy.

Same football model as the Champions League (cl_bot.py): expected goals solved from the bookmakers'
1X2 and over/under prices, every bet settled on 90 minutes. Not part of the combined statistics page.

  python ch_bot.py likmes | rezultati | eksports | kalibret
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import bot as core
import cl_bot as F  # the football engine; pointed at the Championship below
import espn

F.BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.2"
F.PARAMS_FILE = core.ROOT / "ch_params.json"
CH = core.League("ch", "Championship", "⚽", core.ROOT / "ch", core.ROOT / "CH.md", F.load_games, F.settle, day_offset=0)
F.CL = CH
TOP_N = 10
CRAZY_N = 3


def best_across_games(df, p, min_odds: float, n: int) -> list[dict]:
    """Each game's most likely bet, then the n most likely of those (a 2nd per game only if needed)."""
    firsts, seconds = [], []
    for r in df.itertuples():
        for i, x in enumerate(F.pick_for_game(r, p, min_odds)[:2]):  # already sorted by probability
            x = {"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                 "riga_time": r.riga_time, **x, "result": None}
            (firsts if i == 0 else seconds).append(x)
    out = sorted(firsts, key=lambda x: -x["p"])[:n]
    if len(out) < n:
        out += sorted(seconds, key=lambda x: -x["p"])[: n - len(out)]
    return out


def cmd_likmes(args) -> None:
    p = F.load_params()
    CH.message.unlink(missing_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    day = pd.Timestamp.now(tz=F.RIGA).date()
    df = F.predict_day(day, p)
    if not df.empty:
        df = df[df["start_utc"] > now]
    if df.empty:
        print("Šodien nav Championship spēļu ar bukmeikeru koeficientiem.")
        return
    best = best_across_games(df, p, args.min_odds, TOP_N)
    print(f"CHAMPIONSHIP — {day:%d.%m.%Y}: {len(df)} spēles, {len(best)} drošākās likmes (koef. >= {args.min_odds:g}, 90 min.)")
    for i, x in enumerate(best, 1):
        print(f"  {i:>2}. {x['game']:<10} {x['bet']:<34} drošība {x['confidence']:>3}/100  ~{x['est_odds']:.2f}")
    games_info = {}
    for r in df.itertuples():
        ph, pd_, pa = r.p3
        games_info[r.game_id] = {
            "game_id": r.game_id, "away": r.away, "home": r.home, "away_name": r.away_name, "home_name": r.home_name,
            "away_short": r.away_short, "home_short": r.home_short, "start_utc": str(r.start_utc),
            "riga_time": r.riga_time, "robot_line": f"{r.home} {ph:.0%} · X {pd_:.0%} · {r.away} {pa:.0%}",
            "robot_total": f"vārti {r.lh:.1f}–{r.la:.1f}", "market_line": "—",
            "market_total": (f"kopā {r.total_line:g}" if r.total_line else None), "injuries": [],
        }
    crazy = F.pick_crazy(df, p, n=CRAZY_N)
    used = {x["game_id"] for x in best + crazy}
    games_info = {k: v for k, v in games_info.items() if k in used}
    print("CRAZY DAZY: " + "; ".join(f"{x['game']} {x['bet']} (~{x['est_odds']:.2f})" for x in crazy))
    f = core.save_night(day, best, games_info, crazy, args.min_odds, now, safe=[])
    print(f"Saglabāts: {f.relative_to(core.ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="Championship likmju robots")
    ap.add_argument("komanda", type=lambda s: s.lower().replace("ā", "a").replace("ū", "u"),
                    choices=["likmes", "rezultati", "eksports", "kalibret"])
    ap.add_argument("--min-odds", type=float, default=2.1)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    espn.FORCE_REFRESH = args.refresh
    core.configure(CH)
    {"kalibret": F.cmd_kalibret, "eksports": lambda a: core.export_all(), "likmes": cmd_likmes,
     "rezultati": core.cmd_rezultati}[args.komanda](args)


if __name__ == "__main__":
    sys.exit(main())
