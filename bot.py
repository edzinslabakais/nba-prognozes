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
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import bets as B
import data
import espn
import predict as P

ROOT = Path(__file__).resolve().parent
LINES = [x + 0.5 for x in range(-40, 40)]


# --------------------------------------------------------------------------- LIKMES

def _best(g: B.Grid, make, lines, pmax: float, lean_fn, pmin: float = 0.15) -> tuple[B.Bet, float, float] | None:
    """Highest-probability line whose estimated bookmaker odds are still >= min odds."""
    best = None
    for L in lines:
        b = make(L)
        p, push = g.prob(b)
        if push > 0 or p > pmax or p < pmin:
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
            picks.append({"category": cat, "bet": b.label, "p": p, "_bet": b})
    picks.sort(key=lambda x: -x["p"])
    add_data_view(picks, *nba_views(r))
    for x in picks:
        x["fair_odds"] = round(1 / x["p"], 2)
        x["est_odds"] = round(B.PAYOUT / x["p"], 2)
        x["confidence"] = int(round(100 * x["p"]))
    return picks[:3]


# --------------------------------------------------------------------------- "most likely by the data"

EDGE_WEIGHT = 0.5  # how much the robot's own view (vs the bookmakers') counts when choosing the ★ pick


def nba_views(r):
    """Score grids for the robot's own model and for the bookmakers alone."""
    own = B.Grid(r.model_margin_home, r.model_total, B.SIGMA["margin_model"], B.SIGMA["total_model"])
    if not r.has_market:
        return own, None
    mt = r.market_total if r.market_total is not None and pd.notna(r.market_total) else r.total
    return own, B.Grid(r.market_margin_home, mt, B.SIGMA["margin_market"], B.SIGMA["total_market"])


def add_data_view(picks: list[dict], own_grid, market_grid, prob=None) -> None:
    """Attach p_model / p_market to each pick and a data score used to choose the ★ pick:
    final probability + half of how much more likely our own data (form, injuries, rest, goalies)
    makes the bet than the bookmakers do."""
    prob = prob or (lambda g, b: g.prob(b)[0])
    for x in picks:
        b = x.pop("_bet")
        pm = prob(own_grid, b)
        x["p_model"] = round(pm, 4)
        if market_grid is not None:
            pk = prob(market_grid, b)
            x["p_market"] = round(pk, 4)
            x["data_score"] = round(x["p"] + EDGE_WEIGHT * (pm - pk), 4)
        else:
            x["p_market"] = None
            x["data_score"] = round(x["p"], 4)


# --------------------------------------------------------------------------- CRAZY DAZY

CRAZY_MIN_ODDS = 4.0
CRAZY_COUNT = 3


def _leans(r) -> tuple[float, float]:
    lean_m = (r.model_margin_home - r.margin_home) if r.has_market else 0.0      # + => home
    has_mt = r.market_total is not None and pd.notna(r.market_total)
    lean_t = (r.model_total - r.total) if has_mt else 0.0                         # + => over
    return lean_m, lean_t


def crazy_for_game(r, min_odds: float) -> tuple[B.Bet, float, float, str] | None:
    """The most likely bet in this game whose estimated odds are still >= min_odds (e.g. 4.0)."""
    g = B.grid_for(r)
    pmax, pmin = B.PAYOUT / min_odds, 0.08
    H, A = r.home, r.away
    lean_m, lean_t = _leans(r)
    cands: list[tuple[B.Bet, float, float, str]] = []

    def add(bet, lean, cat):
        p, push = g.prob(bet)
        if push == 0 and pmin <= p <= pmax:
            cands.append((bet, p, p + 0.004 * lean, cat))

    def add_best(c, cat):
        if c:
            cands.append((*c, cat))

    t_main = B.half(r.total)
    for team, home in ((H, True), (A, False)):
        sgn = 1 if home else -1
        add_best(_best(g, lambda L, t=team, h=home: B.bet_handicap(t, h, L), LINES, pmax,
                       lambda L, s=sgn: s * lean_m, pmin), "handikaps")
        add(B.bet_ml(team, home), sgn * lean_m, "uzvarētājs")
        for lo, hi in ((1, 5), (6, 10), (11, 15), (16, 20), (21, 999)):
            add(B.bet_band(team, home, lo, hi), sgn * lean_m, "uzvaras starpība")
        for over in (True, False):
            s = 1 if over else -1
            add(B.bet_combo([B.bet_ml(team, home), B.bet_total(over, t_main)]),
                sgn * lean_m + s * lean_t, "kombinācija")
        exp = (r.total + (r.margin_home if home else -r.margin_home)) / 2
        lean_team = (lean_t + (lean_m if home else -lean_m)) / 2
        tl = [round(exp) + x + 0.5 for x in range(-35, 35)]
        for over in (True, False):
            s = 1 if over else -1
            add_best(_best(g, lambda L, t=team, h=home, o=over: B.bet_team_total(t, h, o, L), tl, pmax,
                           lambda L, s=s, lt=lean_team: s * lt, pmin), "komandas punkti")
    tot_lines = [round(r.total) + x + 0.5 for x in range(-50, 50)]
    for over in (True, False):
        s = 1 if over else -1
        add_best(_best(g, lambda L, o=over: B.bet_total(o, L), tot_lines, pmax, lambda L, s=s: s * lean_t, pmin),
                 "kopā")
    return max(cands, key=lambda c: c[2]) if cands else None


def pick_crazy(df: pd.DataFrame, min_odds: float = CRAZY_MIN_ODDS, n: int = CRAZY_COUNT) -> list[dict]:
    """Best long shot per game, then the n games where it is most likely (one bet per game)."""
    rows = []
    for r in df.itertuples():
        c = crazy_for_game(r, min_odds)
        if c:
            b, p, score, cat = c
            rows.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                         "riga_time": r.riga_time, "category": cat, "bet": b.label, "p": p, "_score": score})
    rows.sort(key=lambda x: -x["_score"])
    out = []
    for x in rows[:n]:
        x.pop("_score")
        x["fair_odds"] = round(1 / x["p"], 2)
        x["est_odds"] = round(B.PAYOUT / x["p"], 2)
        x["confidence"] = int(round(100 * x["p"]))
        x["result"] = None
        out.append(x)
    return out


def cmd_likmes(args) -> None:
    L.message.unlink(missing_ok=True)
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
    print("Visas likmes vērtē pēc pamatlaika (48 min), ja vien nav rakstīts '(ieskaitot papildlaiku)'.")
    record, games_info = [], {}
    for r in df.itertuples():
        picks = pick_for_game(r, args.min_odds)
        print("\n" + B.header(r))
        if not r.has_market:
            print("  (nav bukmeikeru līnijas — mazāk precīzs vērtējums)")
        top = max(range(len(picks)), key=lambda i: picks[i].get("data_score", picks[i]["p"])) if picks else -1
        for i, x in enumerate(picks, 1):
            print(f"  {i}.{'★' if i - 1 == top else ' '}{x['bet']:<20} drošība {x['confidence']:>3}/100   "
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

    crazy = pick_crazy(df)
    print(f"\nCRAZY DAZY — 3 reālākās likmes ar koeficientu >= {CRAZY_MIN_ODDS:g}:")
    for i, x in enumerate(crazy, 1):
        print(f"  {i}. {x['game']:<10} {x['bet']:<26} drošība {x['confidence']:>3}/100   "
              f"godīgais koef. {x['fair_odds']:.2f}  (Optibet ~{x['est_odds']:.2f})   [{x['category']}]")

    f = save_night(day, record, games_info, crazy, args.min_odds, now)
    print(f"\nSaglabāts: {f.relative_to(ROOT)}  — rīt palaid: python bot.py rezultati")
    print("Ja Optibet līnija nedaudz atšķiras (piem. -7.5 vietā -8.5), ņem tuvāko ar koef. >= "
          f"{args.min_odds:g}; godīgais koef. parāda, cik likme ir vērta.")


def save_night(day, record: list, games_info: dict, crazy: list, min_odds: float, now) -> Path:
    """Write the night's picks to history (games that already started keep their old picks)."""
    L.hist.mkdir(parents=True, exist_ok=True)
    f = L.hist / f"{day:%Y-%m-%d}.json"
    old = json.loads(f.read_text()) if f.exists() else {"picks": [], "games": {}}
    started = {p["game_id"] for p in old["picks"] if pd.Timestamp(p["start_utc"]) <= now}
    keep = [p for p in old["picks"] if p["game_id"] in started]
    record = keep + [p for p in record if p["game_id"] not in started]
    games_all = {**{k: v for k, v in old.get("games", {}).items() if k in started}, **games_info}
    # once any Crazy Dazy game has started, the Crazy Dazy list for the night is frozen
    old_crazy = old.get("crazy", [])
    if any(c["game_id"] in started for c in old_crazy):
        crazy = old_crazy
    f.write_text(json.dumps({"date": str(day), "min_odds": min_odds,
                             "generated_at": now.isoformat(timespec="minutes"),
                             "games": games_all, "picks": record, "crazy": crazy},
                            indent=1, ensure_ascii=False))
    export_all()
    write_message(day.isoformat(), "likmes")
    return f


# --------------------------------------------------------------------------- REZULTĀTI

class Actual:
    """Lets the same bet functions be evaluated on a finished game's real score
    (regulation: M, T, H, A; final incl. overtime: FM, FT, FH, FA)."""

    def __init__(self, home_reg: float, away_reg: float, home_pts: float, away_pts: float):
        self.H, self.A = np.array([home_reg], float), np.array([away_reg], float)
        self.M, self.T = self.H - self.A, self.H + self.A
        self.FH, self.FA = np.array([home_pts], float), np.array([away_pts], float)
        self.FM, self.FT = self.FH - self.FA, self.FH + self.FA
        self.W = np.array([1.0])


@dataclass
class League:
    """Where a league keeps its history/export and how its finished bets are settled."""
    key: str
    title: str
    emoji: str
    base: Path
    report: Path
    load_games: Callable
    settle_bet: Callable          # (bet text, finished game row) -> (win, push, score text)
    crazy_min_odds: float = 4.0

    @property
    def hist(self) -> Path:
        return self.base / "history"

    @property
    def export(self) -> Path:
        return self.base / "export"

    @property
    def archive(self) -> Path:
        """Simulated past nights (archive.py)."""
        return self.base / "archive"

    @property
    def message(self) -> Path:
        return self.export / "message.txt"


def _nba_settle(text: str, g) -> tuple[bool, bool, str]:
    """Settled on the regulation result unless the bet says '(ieskaitot papildlaiku)'."""
    bet = B.parse_bet(text, g)
    win, push = bet.fn(Actual(g.home_reg, g.away_reg, g.home_pts, g.away_pts))
    score = f"{g.away} {int(g.away_reg)} - {int(g.home_reg)} {g.home}"
    if getattr(g, "ot", False):
        score += f" (pamatlaikā; ar papildlaiku {int(g.away_pts)} - {int(g.home_pts)})"
    return bool(win[0]), bool(push[0]), score


NBA = League("nba", "NBA", "🏀", ROOT, ROOT / "PROGNOZES.md", data.load_games, _nba_settle)
L = NBA


def configure(league: League) -> None:
    global L
    L = league


def settle(path: Path) -> tuple[dict, int]:
    rec = json.loads(path.read_text())
    day = pd.Timestamp(rec["date"]).date()
    games = L.load_games(day, day)
    if games.empty:
        return rec, 0
    by_id = {g.game_id: g for g in games.itertuples()}
    n_new = 0
    for p in rec["picks"] + rec.get("crazy", []):
        if p.get("result"):
            continue
        g = by_id.get(p["game_id"])
        if g is None or not g.completed:
            continue
        try:
            win, push, score = L.settle_bet(p["bet"], g)
        except ValueError as e:
            print(f"Nevar novērtēt likmi {p['game']} '{p['bet']}': {e}")
            continue
        p["result"] = "atgriezts" if push else ("izgāja" if win else "neizgāja")
        p["score"] = score
        n_new += 1
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec, n_new


ICON = {"izgāja": "✅", "neizgāja": "❌", "atgriezts": "↩️", None: "⏳"}


def cmd_rezultati(args) -> None:
    files = sorted(L.hist.glob("*.json")) if L.hist.exists() else []
    if not files:
        print("Vēl nav saglabātu likmju. Vispirms palaid: python bot.py likmes")
        return
    L.message.unlink(missing_ok=True)
    espn.FORCE_REFRESH = True  # make sure last night's scoreboard is fresh
    recs, new_dates = [], []
    for f in files:
        rec, n_new = settle(f)
        recs.append(rec)
        if n_new:
            new_dates.append(rec["date"])
    export_all()
    for d in new_dates[-1:]:
        write_message(d, "rezultati")

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
    if show.get("crazy"):
        print(f"\nCRAZY DAZY (koef. >= {L.crazy_min_odds:g}):")
        for p in show["crazy"]:
            res = p.get("result")
            print(f"  {ICON[res]} {p['game']:<10} {p['bet']:<26} drošība {p['confidence']:>3}/100   ~{p['est_odds']:.2f}   {res or 'gaida'}")
        all_crazy = [p for r in recs for p in r.get("crazy", [])]
        print(f"  Crazy Dazy šī diena: {_stats(show['crazy'])}")
        print(f"  Crazy Dazy kopā: {_stats(all_crazy)}")
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



def _summary(picks: list[dict]) -> dict:
    settled = [p for p in picks if p.get("result") in ("izgāja", "neizgāja", "atgriezts")]
    hits = [p for p in settled if p["result"] == "izgāja"]
    pushes = [p for p in settled if p["result"] == "atgriezts"]
    return {
        "picks": len(picks), "settled": len(settled), "hits": len(hits), "pushes": len(pushes),
        "expected_hits": round(sum(p["p"] for p in settled if p["result"] != "atgriezts"), 2),
        "profit_units": round(sum(p["est_odds"] for p in hits) + len(pushes) - len(settled), 2),
    }


MONTHS_LV = ["janvāris", "februāris", "marts", "aprīlis", "maijs", "jūnijs", "jūlijs", "augusts",
             "septembris", "oktobris", "novembris", "decembris"]


def month_label(m: str) -> str:
    y, mo = m.split("-")
    return f"{MONTHS_LV[int(mo) - 1].capitalize()} {y}"


def month_days(m: str, first_day: int = 1, last_day: int | None = None) -> str:
    """'1.–30.11.' — a month runs from its first to its last day (by game date);
    the month tracking started in begins on that day, e.g. '18.–30.11.'."""
    first = pd.Timestamp(m + "-01")
    last = last_day or (first + pd.offsets.MonthEnd(0)).day
    return f"{first_day}.–{last}.{first.month:02d}."


def _month_entry(m: str, sim: bool, acc: dict, live_start: str | None) -> dict:
    dates = sorted(acc["dates"])
    first_day, last_day = 1, None
    if sim:  # a simulated stretch covers exactly the days it has
        first_day, last_day = int(dates[0][8:]), int(dates[-1][8:])
    elif live_start and live_start[:7] == m:  # live tracking began during this month
        first_day = int(live_start[8:])
    return {"month": m, "label": month_label(m), "simulated": sim,
            "days": month_days(m, first_day, last_day),
            **{k: _summary(v) for k, v in acc.items() if k != "dates"}}


def _num(v, default):
    return default if v is None or (isinstance(v, float) and math.isnan(v)) else v


def export_all() -> None:
    """Write export/nights/<date>.json + export/stats.json (what the phone page shows)."""
    live = sorted(L.hist.glob("*.json")) if L.hist.exists() else []
    sims = sorted(L.archive.glob("*.json")) if L.archive.exists() else []
    live_dates = {f.stem for f in live}
    files = [f for f in sims if f.stem not in live_dates] + live
    (L.export / "nights").mkdir(parents=True, exist_ok=True)
    all_picks, all_crazy, all_top, nights = [], [], [], []
    month_acc: dict[tuple, dict] = {}
    for f in sorted(files, key=lambda f: f.stem):
        rec = json.loads(f.read_text())
        sim = bool(rec.get("simulated"))
        games = rec.get("games", {})
        out_games, night_top = [], []
        for gid, gp in pd.DataFrame(rec["picks"]).groupby("game_id", sort=False) if rec["picks"] else []:
            info = games.get(gid, {"game_id": gid, "away": gp.iloc[0]["game"].split("@")[0],
                                   "home": gp.iloc[0]["game"].split("@")[1],
                                   "start_utc": gp.iloc[0]["start_utc"]})
            raw = gp.to_dict("records")
            # ★ = best by the data score (falls back to probability for older nights)
            top_i = max(range(len(raw)), key=lambda i: (_num(raw[i].get("data_score"), raw[i]["p"]), -i))
            night_top.append(raw[top_i])
            picks = [{**{k: (None if isinstance(v, float) and math.isnan(v) else v)
                         for k, v in p.items() if k in ("bet", "category", "confidence", "fair_odds",
                                                         "est_odds", "result", "score", "p_model", "p_market")},
                      "top": i == top_i}
                     for i, p in enumerate(raw)]
            score = next((p.get("score") for p in picks if p.get("score")), None)
            out_games.append({**info, "score": score, "picks": picks})
        out_games.sort(key=lambda g: g.get("start_utc", ""))
        crazy = rec.get("crazy", [])
        scores = {g["game_id"]: g.get("score") for g in out_games}
        crazy_out = [{**{k: c.get(k) for k in ("game_id", "game", "riga_time", "category", "bet",
                                                "confidence", "fair_odds", "est_odds", "result")},
                      "score": c.get("score") or scores.get(c["game_id"])} for c in crazy]
        doc = {"date": rec["date"], "min_odds": rec.get("min_odds", 2.1),
               "generated_at": rec.get("generated_at"), "games": out_games,
               "summary": _summary(rec["picks"]),
               "crazy_min_odds": L.crazy_min_odds, "crazy": crazy_out, "crazy_summary": _summary(crazy),
               "top_summary": _summary(night_top), "simulated": sim}
        (L.export / "nights" / f"{rec['date']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        if not sim:  # the real track record counts only nights picked live
            all_picks += rec["picks"]
            all_crazy += crazy
            all_top += night_top
        nights.append({"date": rec["date"], "simulated": sim, **_summary(rec["picks"]),
                       "crazy": _summary(crazy), "top": _summary(night_top)})
        mon = month_acc.setdefault((rec["date"][:7], sim), {"all": [], "top": [], "crazy": [], "dates": []})
        mon["all"] += rec["picks"]
        mon["top"] += night_top
        mon["crazy"] += crazy
        mon["dates"].append(rec["date"])
    buckets = []
    done = [p for p in all_picks if p.get("result") in ("izgāja", "neizgāja")]
    for lo, hi in ((0, 35), (35, 40), (40, 45), (45, 101)):
        b = [p for p in done if lo <= p["confidence"] < hi]
        if b:
            buckets.append({"range": f"{lo}-{min(hi, 100) - 1}", "n": len(b),
                            "hit_rate": round(sum(p["result"] == "izgāja" for p in b) / len(b), 3),
                            "expected": round(sum(p["p"] for p in b) / len(b), 3)})
    stats = {"overall": _summary(all_picks), "crazy_overall": _summary(all_crazy), "top_overall": _summary(all_top),
             "nights": nights, "calibration": buckets,
             "months": [_month_entry(m, sim, acc, live_start=min(live_dates) if live_dates else None)
                        for (m, sim), acc in sorted(month_acc.items(), key=lambda kv: (kv[0][0], not kv[0][1]),
                                                    reverse=True)],
             "updated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes")}
    (L.export / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    write_markdown(stats)


RIGA = "Europe/Riga"
RES_ICON = {"izgāja": "✅", "neizgāja": "❌", "atgriezts": "↩️"}


def _night_label(us_date: str) -> str:
    d = pd.Timestamp(us_date) + pd.Timedelta(days=1)
    return f"{d:%d.%m.}"


def _load_night(us_date: str) -> dict | None:
    f = L.export / "nights" / f"{us_date}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _overall_line(o: dict) -> str:
    if not o.get("settled"):
        return "vēl nav noslēgtu likmju"
    return (f"izgāja {o['hits']}/{o['settled']} ({100 * o['hits'] / o['settled']:.0f}%), "
            f"robots gaidīja ~{o['expected_hits']:.1f}, vienības {o['profit_units']:+.2f}")


def write_message(us_date: str, mode: str) -> None:
    """Short phone message (Telegram) for tonight's picks or last night's results."""
    doc = _load_night(us_date)
    if not doc or not doc["games"]:
        return
    stats = json.loads((L.export / "stats.json").read_text())
    lines = []
    if mode == "likmes":
        lines.append(f"{L.emoji} {L.title} LIKMES — nakts uz {_night_label(us_date)} ({len(doc['games'])} spēles, koef. ≥ {doc['min_odds']:g})")
        lines.append("drošība 1-100 · ~aptuvenais Optibet koeficients")
        for g in doc["games"]:
            t = (g.get("riga_time") or " ").split(" ")[-1]
            lines.append(f"\n{g['away']} @ {g['home']}  {t}")
            for i, p in enumerate(g["picks"], 1):
                star = " ★" if p.get("top") else ""
                lines.append(f"{i}. {p['bet']} — {p['confidence']}/100 (~{p['est_odds']:.2f}){star}")
            if g.get("injuries"):
                lines.append("   ⚕ " + "; ".join(g["injuries"][:2]))
        if doc.get("crazy"):
            lines.append(f"\n🎲 CRAZY DAZY (koef. ≥ {doc['crazy_min_odds']:g})")
            for c in doc["crazy"]:
                lines.append(f"• {c['game']}: {c['bet']} — {c['confidence']}/100 (~{c['est_odds']:.2f})")
    else:
        s = doc["summary"]
        lines.append(f"📊 {L.title} REZULTĀTI — nakts uz {_night_label(us_date)}: izgāja {s['hits']}/{s['settled']}")
        for g in doc["games"]:
            lines.append(f"\n{g.get('score') or g['away'] + ' @ ' + g['home'] + ' (vēl nav beigusies)'}")
            for p in g["picks"]:
                lines.append(f"{RES_ICON.get(p.get('result'), '⏳')} {p['bet']} ({p['confidence']}/100)")
        if doc.get("crazy"):
            cs = doc["crazy_summary"]
            lines.append(f"\n🎲 CRAZY DAZY: izgāja {cs['hits']}/{cs['settled']}")
            for c in doc["crazy"]:
                lines.append(f"{RES_ICON.get(c.get('result'), '⏳')} {c['game']}: {c['bet']} (~{c['est_odds']:.2f})")
        lines.append(f"\nKopā: {_overall_line(stats['overall'])}")
        if stats.get("crazy_overall", {}).get("settled"):
            lines.append(f"Crazy Dazy kopā: {_overall_line(stats['crazy_overall'])}")
    L.export.mkdir(parents=True, exist_ok=True)
    L.message.write_text("\n".join(lines) + "\n")


def write_markdown(stats: dict) -> None:
    """PROGNOZES.md: readable on a phone in the GitHub app."""
    nights = sorted(L.export.glob("nights/*.json"))
    now = pd.Timestamp.now(tz="UTC").tz_convert(RIGA)
    out = [f"# {L.emoji} {L.title} prognozes", "", f"_Atjaunināts {now:%d.%m.%Y %H:%M} (Rīgas laiks)_", ""]
    if nights:
        doc = json.loads(nights[-1].read_text())
        s = doc["summary"]
        head = f"## Nakts uz {_night_label(doc['date'])} — {len(doc['games'])} spēles, koef. ≥ {doc['min_odds']:g}"
        if s["settled"]:
            head += f" — izgāja {s['hits']}/{s['settled']}"
        out += [head, "",
                "Drošība = robota aprēķinātā varbūtība, ka likme izies. ~Optibet = aptuvenais koeficients, "
                "godīgais = koeficients, pie kura likme ir tieši \"vērta savu cenu\".", ""]
        for g in doc["games"]:
            t = (g.get("riga_time") or " ").split(" ")[-1]
            out.append(f"### {g['away']} @ {g['home']} · {t}")
            info = []
            def _tot(v):
                return v if isinstance(v, str) else f"kopā {v}"
            if g.get("robot_line"):
                info.append(f"Robots: **{g['robot_line']}**, {_tot(g.get('robot_total'))}")
            if g.get("market_line") and g["market_line"] != "—":
                info.append(f"bukmeikeri: {g['market_line']}, {_tot(g.get('market_total'))}")
            if g.get("score"):
                info.append(f"gala rezultāts: **{g['score']}**")
            if info:
                out += [" · ".join(info), ""]
            out += ["| # | Likme | Drošība | ~Optibet | Godīgais | Rezultāts |", "|---|---|---|---|---|---|"]
            for i, p in enumerate(g["picks"], 1):
                res = f"{RES_ICON[p['result']]} {p['result']}" if p.get("result") in RES_ICON else "⏳ gaida"
                star = " ★" if p.get("top") else ""
                out.append(f"| {i}{star} | **{p['bet']}** | {p['confidence']}/100 | {p['est_odds']:.2f} | "
                           f"{p['fair_odds']:.2f} | {res} |")
            if g.get("injuries"):
                out += ["", "Savainojumi/sastāvs: " + "; ".join(g["injuries"])]
            out.append("")
        if doc.get("crazy"):
            out += [f"## 🎲 Crazy Dazy — koef. ≥ {doc['crazy_min_odds']:g}", "",
                    "3 likmes no visām nakts spēlēm, kurām ar koeficientu 4+ ir vislielākā iespēja iziet.", "",
                    "| Spēle | Likme | Drošība | ~Optibet | Godīgais | Rezultāts |", "|---|---|---|---|---|---|"]
            for c in doc["crazy"]:
                res = f"{RES_ICON[c['result']]} {c['result']}" if c.get("result") in RES_ICON else "⏳ gaida"
                out.append(f"| {c['game']} | **{c['bet']}** | {c['confidence']}/100 | {c['est_odds']:.2f} | "
                           f"{c['fair_odds']:.2f} | {res} |")
            out.append("")
    t = stats.get("top_overall", {})
    top_line = (f"uzvarēts {t['hits']}/{t['settled']}, vienības {t['profit_units']:+.2f}"
                if t.get("settled") else "vēl nav noslēgtu likmju")
    out += ["## Statistika", "", f"★ Visticamākās (1 katrā spēlē): {top_line}", "",
            f"Visas likmes kopā: {_overall_line(stats['overall'])}"]
    if stats.get("crazy_overall", {}).get("picks"):
        out.append(f"Crazy Dazy kopā: {_overall_line(stats['crazy_overall'])}")
    out.append("")
    if stats.get("months"):
        out += ["### Mēneša statistika", "",
                "| Mēnesis | Visas: uzvarēts | Vien. | ★ Visticamākās | Vien. | Crazy Dazy | Vien. |",
                "|---|---|---|---|---|---|---|"]
        for m in stats["months"]:
            cell = lambda s: (f"{s['hits']}/{s['settled']} | {s['profit_units']:+.2f}" if s["settled"] else "– | –")
            tag = " — simulācija" if m.get("simulated") else ""
            out.append(f"| {m['label']} ({m['days']}){tag} | {cell(m['all'])} | {cell(m['top'])} | {cell(m['crazy'])} |")
        out += ["", "_Simulācija = arhīvs: ko robots būtu ieteicis pirms šīs sistēmas palaišanas, izmantojot tikai "
                "pirms spēles pieejamos datus (bez tā laika savainojumu saraksta)._"]
        out.append("")
    rows = [n for n in reversed(stats["nights"]) if n["settled"] and not n.get("simulated")][:14]
    if rows:
        out += ["| Nakts | Izgāja | Robots gaidīja | Vienības |", "|---|---|---|---|"]
        for n in rows:
            out.append(f"| {_night_label(n['date'])} | {n['hits']}/{n['settled']} | "
                       f"{n['expected_hits']:.1f} | {n['profit_units']:+.2f} |")
        out.append("")
    if stats.get("calibration"):
        out += ["### Vai drošība atbilst realitātei?", "", "| Drošība | Likmes | Izgāja | Gaidīts |", "|---|---|---|---|"]
        for c in stats["calibration"]:
            out.append(f"| {c['range']} | {c['n']} | {100 * c['hit_rate']:.0f}% | {100 * c['expected']:.0f}% |")
        out.append("")
    L.report.write_text("\n".join(out))


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
        print(f"Eksportēts: {L.export.relative_to(ROOT)}/")
        return
    (cmd_likmes if args.komanda == "likmes" else cmd_rezultati)(args)


if __name__ == "__main__":
    sys.exit(main())
