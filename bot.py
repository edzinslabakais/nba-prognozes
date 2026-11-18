"""Divas komandas prognožu spēlei.

  python bot.py likmes      # 3 likmes katrai šīs nakts spēlei (koef. >= 2.1) + drošība 1-100
  python bot.py rezultati   # pārbauda iepriekšējo likmju rezultātus + kopējā statistika

Likmes tiek saglabātas history/YYYY-MM-DD.json, rezultāti tiek ierakstīti tajos pašos failos.

Kā tiek izvēlētas likmes:
  * Katrai likmei robots aprēķina varbūtību (sk. bets.py) -> "drošība" = varbūtība %.
  * Optibet koeficientu robots nezina, tāpēc to novērtē kā  0.93 / varbūtība
    (bukmeikers patur ~7 %). Lai koeficients būtu >= 2.1, varbūtībai jābūt <= ~44 %.
  * No katras kategorijas (starpība/handikaps, kopējie punkti, komandas punkti)
    izvēlas līniju ar AUGSTĀKO varbūtību, kas vēl dod koeficientu >= 2.1.
    Ja varbūtības ir līdzīgas, priekšroka tai pusei, uz kuru robots "velk" vairāk
    nekā bukmeikeri.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import bets as B
import data
import espn
import predict as P

ROOT = Path(__file__).resolve().parent
HIST = ROOT / "history"
LINES = [x + 0.5 for x in range(-40, 40)]


# --------------------------------------------------------------------------- LIKMES

def _best(g: B.Grid, make, lines, pmax: float, lean_fn) -> tuple[B.Bet, float, float] | None:
    """Highest-probability line whose estimated bookmaker odds are still >= min odds."""
    best = None
    for L in lines:
        b = make(L)
        p, push = g.prob(b)
        if push > 0 or p > pmax or p < 0.15:
            continue
        score = p + 0.004 * lean_fn(L)
        if best is None or score > best[2]:
            best = (b, p, score)
    return best


def pick_for_game(r, min_odds: float) -> list[dict]:
    g = B.grid_for(r)
    pmax = B.PAYOUT / min_odds
    H, A = r.home, r.away
    # How much our own model leans compared with the final (market-blended) numbers
    lean_m = (r.model_margin_home - r.margin_home) if r.has_market else 0.0      # + => home
    has_mt = r.market_total is not None and pd.notna(r.market_total)
    lean_t = (r.model_total - r.total) if has_mt else 0.0                         # + => over

    cats: dict[str, list] = {"starpība": [], "kopā": [], "komandas punkti": []}
    for team, home in ((H, True), (A, False)):
        sgn = 1 if home else -1
        c = _best(g, lambda L, t=team, h=home: B.bet_handicap(t, h, L), LINES, pmax, lambda L, s=sgn: s * lean_m)
        if c:
            cats["starpība"].append(c)
        ml = B.bet_ml(team, home)
        p, _ = g.prob(ml)
        if 0.15 <= p <= pmax:
            cats["starpība"].append((ml, p, p + 0.004 * sgn * lean_m))
    t0 = round(r.total)
    tot_lines = [t0 + x + 0.5 for x in range(-40, 40)]
    for over in (True, False):
        s = 1 if over else -1
        c = _best(g, lambda L, o=over: B.bet_total(o, L), tot_lines, pmax, lambda L, s=s: s * lean_t)
        if c:
            cats["kopā"].append(c)
    for team, home in ((H, True), (A, False)):
        exp = (r.total + (r.margin_home if home else -r.margin_home)) / 2
        lean_team = (lean_t + (lean_m if home else -lean_m)) / 2
        tl = [round(exp) + x + 0.5 for x in range(-30, 30)]
        for over in (True, False):
            s = 1 if over else -1
            c = _best(g, lambda L, t=team, h=home, o=over: B.bet_team_total(t, h, o, L), tl, pmax,
                      lambda L, s=s, lt=lean_team: s * lt)
            if c:
                cats["komandas punkti"].append(c)

    picks = []
    for cat, cands in cats.items():
        if cands:
            b, p, _ = max(cands, key=lambda x: x[2])
            picks.append({"category": cat, "bet": b.label, "p": p})
    picks.sort(key=lambda x: -x["p"])
    for x in picks:
        x["fair_odds"] = round(1 / x["p"], 2)
        x["est_odds"] = round(B.PAYOUT / x["p"], 2)
        x["confidence"] = int(round(100 * x["p"]))
    return picks[:3]


def cmd_likmes(args) -> None:
    params = P.load_params()
    mw = params.get("market_weight_suggested", 0.9)
    now = pd.Timestamp.now(tz="UTC")
    day = espn.today_et()
    df = pd.DataFrame()
    for d in (day, day + timedelta(days=1)):
        df = P.predict([d], mw, params)
        if not df.empty:
            df = df[df["start_utc"] > now]
        if not df.empty:
            day = d
            break
    if df.empty:
        print("Tuvākajā laikā nav neizspēlētu NBA spēļu.")
        return

    print(f"NBA LIKMES — nakts uz {(day + timedelta(days=1)):%d.%m.%Y} (ASV datums {day:%Y-%m-%d}), "
          f"min. koeficients {args.min_odds:g}")
    print("Drošība = robota aprēķinātā varbūtība, ka likme izies (1-100).")
    record, games_info = [], {}
    for r in df.itertuples():
        picks = pick_for_game(r, args.min_odds)
        print("\n" + B.header(r))
        if not r.has_market:
            print("  (nav bukmeikeru līnijas — mazāk precīzs vērtējums)")
        for i, x in enumerate(picks, 1):
            print(f"  {i}. {x['bet']:<20} drošība {x['confidence']:>3}/100   "
                  f"godīgais koef. {x['fair_odds']:.2f}  (Optibet ~{x['est_odds']:.2f})   [{x['category']}]")
        inj = [n for n in (P._inj_text(r.inj_home, 2), P._inj_text(r.inj_away, 2)) if n]
        if inj:
            print("  savainojumi/sastāvs: " + " | ".join(inj))
        for x in picks:
            record.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}",
                           "start_utc": str(r.start_utc), **x, "result": None})
        robot_line = (f"{r.home} {-r.margin_home:+.1f}" if r.margin_home >= 0
                      else f"{r.away} {r.margin_home:+.1f}")
        games_info[r.game_id] = {
            "game_id": r.game_id, "away": r.away, "home": r.home,
            "away_name": r.away_name, "home_name": r.home_name,
            "away_record": r.away_record, "home_record": r.home_record,
            "start_utc": str(r.start_utc), "riga_time": r.riga_time,
            "robot_line": robot_line, "robot_total": round(float(r.total), 1),
            "market_line": P._fmt_line(r.market_margin_home, r.home, r.away),
            "market_total": r.market_total if r.market_total is not None and pd.notna(r.market_total) else None,
            "injuries": [f"{team}: {n}{' (' + s + ')' if s else ''} {d:+.1f}"
                         for team, a in ((r.home, r.inj_home), (r.away, r.inj_away))
                         for n, s, d in a.notes[:3]],
        }

    HIST.mkdir(exist_ok=True)
    f = HIST / f"{day:%Y-%m-%d}.json"
    old = json.loads(f.read_text()) if f.exists() else {"picks": [], "games": {}}
    # keep picks for games that have already started, replace the rest
    started = {p["game_id"] for p in old["picks"] if pd.Timestamp(p["start_utc"]) <= now}
    keep = [p for p in old["picks"] if p["game_id"] in started]
    record = keep + [p for p in record if p["game_id"] not in started]
    games_all = {**{k: v for k, v in old.get("games", {}).items() if k in started}, **games_info}
    f.write_text(json.dumps({"date": str(day), "min_odds": args.min_odds,
                             "generated_at": now.isoformat(timespec="minutes"),
                             "games": games_all, "picks": record},
                            indent=1, ensure_ascii=False))
    export_all()
    print(f"\nSaglabāts: {f.relative_to(ROOT)}  — rīt palaid: python bot.py rezultati")
    print("Ja Optibet līnija nedaudz atšķiras (piem. -7.5 vietā -8.5), ņem tuvāko ar koef. >= "
          f"{args.min_odds:g}; godīgais koef. parāda, cik likme ir vērta.")


# --------------------------------------------------------------------------- REZULTĀTI

class Actual:
    """Lets the same bet functions be evaluated on a finished game's real score."""

    def __init__(self, home_pts: float, away_pts: float):
        self.M = np.array([home_pts - away_pts])
        self.T = np.array([home_pts + away_pts])
        self.H, self.A = np.array([home_pts]), np.array([away_pts])


def settle(path: Path) -> tuple[dict, int]:
    rec = json.loads(path.read_text())
    day = pd.Timestamp(rec["date"]).date()
    games = data.load_games(day, day)
    if games.empty:
        return rec, 0
    by_id = {g.game_id: g for g in games.itertuples()}
    n_new = 0
    for p in rec["picks"]:
        if p.get("result"):
            continue
        g = by_id.get(p["game_id"])
        if g is None or not g.completed:
            continue
        try:
            bet = B.parse_bet(p["bet"], g)
        except ValueError as e:
            print(f"Nevar novērtēt likmi {p['game']} '{p['bet']}': {e}")
            continue
        win, push = bet.fn(Actual(g.home_pts, g.away_pts))
        p["result"] = "atgriezts" if push[0] else ("izgāja" if win[0] else "neizgāja")
        p["score"] = f"{g.away} {int(g.away_pts)} - {int(g.home_pts)} {g.home}"
        n_new += 1
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec, n_new


ICON = {"izgāja": "✅", "neizgāja": "❌", "atgriezts": "↩️", None: "⏳"}


def cmd_rezultati(args) -> None:
    files = sorted(HIST.glob("*.json")) if HIST.exists() else []
    if not files:
        print("Vēl nav saglabātu likmju. Vispirms palaid: python bot.py likmes")
        return
    espn.FORCE_REFRESH = True  # make sure last night's scoreboard is fresh
    recs = []
    for f in files:
        rec, _ = settle(f)
        recs.append(rec)
    export_all()

    # the most recent day that has at least one settled pick
    show = next((r for r in reversed(recs) if any(p.get("result") for p in r["picks"])), recs[-1])
    print(f"REZULTĀTI — ASV datums {show['date']}")
    for game, picks in pd.DataFrame(show["picks"]).groupby("game", sort=False):
        score = next((p for p in picks.get("score", pd.Series(dtype=object)) if isinstance(p, str)), "vēl nav beigusies")
        print(f"\n{game}: {score}")
        for p in picks.itertuples():
            res = p.result if isinstance(p.result, str) else None
            print(f"  {ICON[res]} {p.bet:<20} drošība {p.confidence:>3}/100   {res or 'gaida'}")
    day_stats = _stats(show["picks"])
    print(f"\nŠī diena: {day_stats}")

    all_picks = [p for r in recs for p in r["picks"]]
    print(f"Kopā ({len(recs)} dienas): {_stats(all_picks)}")
    done = [p for p in all_picks if p.get("result") in ("izgāja", "neizgāja")]
    if len(done) >= 20:
        print("\nKalibrācija (vai drošības % atbilst realitātei):")
        df = pd.DataFrame(done)
        df["hit"] = df["result"] == "izgāja"
        df["bucket"] = pd.cut(df["confidence"], [0, 35, 40, 45, 100], labels=["<35", "35-39", "40-44", "45+"])
        for b, d in df.groupby("bucket", observed=True):
            print(f"  drošība {b:>6}: {len(d):>3} likmes, izgāja {100*d['hit'].mean():.0f}% "
                  f"(gaidīts {d['confidence'].mean():.0f}%)")


def _stats(picks: list[dict]) -> str:
    settled = [p for p in picks if p.get("result") in ("izgāja", "neizgāja", "atgriezts")]
    if not settled:
        return "vēl nav noslēgtu likmju"
    hits = [p for p in settled if p["result"] == "izgāja"]
    pushes = [p for p in settled if p["result"] == "atgriezts"]
    exp = sum(p["p"] for p in settled if p["result"] != "atgriezts")
    profit = sum(p["est_odds"] for p in hits) + len(pushes) - len(settled)
    return (f"izgāja {len(hits)}/{len(settled)} ({100*len(hits)/len(settled):.0f}%), "
            f"robots gaidīja ~{exp:.1f} | ja katrai likmei 1 vienība pie ~Optibet koef.: "
            f"{profit:+.2f} vien.")


# --------------------------------------------------------------------------- export for the phone page

EXPORT = ROOT / "export"


def _summary(picks: list[dict]) -> dict:
    settled = [p for p in picks if p.get("result") in ("izgāja", "neizgāja", "atgriezts")]
    hits = [p for p in settled if p["result"] == "izgāja"]
    pushes = [p for p in settled if p["result"] == "atgriezts"]
    return {
        "picks": len(picks), "settled": len(settled), "hits": len(hits), "pushes": len(pushes),
        "expected_hits": round(sum(p["p"] for p in settled if p["result"] != "atgriezts"), 2),
        "profit_units": round(sum(p["est_odds"] for p in hits) + len(pushes) - len(settled), 2),
    }


def export_all() -> None:
    """Write export/nights/<date>.json + export/stats.json (what the phone page shows)."""
    files = sorted(HIST.glob("*.json")) if HIST.exists() else []
    (EXPORT / "nights").mkdir(parents=True, exist_ok=True)
    all_picks, nights = [], []
    for f in files:
        rec = json.loads(f.read_text())
        games = rec.get("games", {})
        out_games = []
        for gid, gp in pd.DataFrame(rec["picks"]).groupby("game_id", sort=False) if rec["picks"] else []:
            info = games.get(gid, {"game_id": gid, "away": gp.iloc[0]["game"].split("@")[0],
                                   "home": gp.iloc[0]["game"].split("@")[1],
                                   "start_utc": gp.iloc[0]["start_utc"]})
            picks = [{k: (None if isinstance(v, float) and math.isnan(v) else v)
                      for k, v in p.items() if k in ("bet", "category", "confidence", "fair_odds",
                                                      "est_odds", "result", "score")}
                     for p in gp.to_dict("records")]
            score = next((p.get("score") for p in picks if p.get("score")), None)
            out_games.append({**info, "score": score, "picks": picks})
        out_games.sort(key=lambda g: g.get("start_utc", ""))
        doc = {"date": rec["date"], "min_odds": rec.get("min_odds", 2.1),
               "generated_at": rec.get("generated_at"), "games": out_games,
               "summary": _summary(rec["picks"])}
        (EXPORT / "nights" / f"{rec['date']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        all_picks += rec["picks"]
        nights.append({"date": rec["date"], **_summary(rec["picks"])})
    buckets = []
    done = [p for p in all_picks if p.get("result") in ("izgāja", "neizgāja")]
    for lo, hi in ((0, 35), (35, 40), (40, 45), (45, 101)):
        b = [p for p in done if lo <= p["confidence"] < hi]
        if b:
            buckets.append({"range": f"{lo}-{min(hi, 100) - 1}", "n": len(b),
                            "hit_rate": round(sum(p["result"] == "izgāja" for p in b) / len(b), 3),
                            "expected": round(sum(p["p"] for p in b) / len(b), 3)})
    stats = {"overall": _summary(all_picks), "nights": nights, "calibration": buckets,
             "updated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes")}
    (EXPORT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description="NBA likmju robots")
    ap.add_argument("komanda", type=lambda s: s.lower().replace("ā", "a").replace("ū", "u"),
                    choices=["likmes", "rezultati", "eksports"])
    ap.add_argument("--min-odds", type=float, default=2.1)
    ap.add_argument("--refresh", action="store_true", help="svaigs savainojumu saraksts")
    args = ap.parse_args()
    espn.FORCE_REFRESH = args.refresh
    if args.komanda == "eksports":
        export_all()
        print(f"Eksportēts: {EXPORT.relative_to(ROOT)}/")
        return
    (cmd_likmes if args.komanda == "likmes" else cmd_rezultati)(args)


if __name__ == "__main__":
    sys.exit(main())
