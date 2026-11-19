"""UEFA Champions League prognozes — the same rules (3 bets >= 2.1, ★, Drošās >= 1.7, Crazy Dazy >= 4.0).

Data: ESPN (schedule, results, DraftKings odds: home / draw / away and total goals).
Model: the goal expectations of both teams are solved from the bookmakers' odds (the simulation on
NBA/NHL showed the bookmakers' probabilities are the most reliable), then every bet is priced from a
Poisson score grid with a calibrated boost for draws. All bets are settled on the 90-minute result.

  python cl_bot.py likmes | rezultati | eksports | kalibret
History: cl/history/, page data: cl/export/, report: CL.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import bot as core
import espn
import nhl_bets as HB
import nhl_data as ND
import nhl_model as N

RIGA = ZoneInfo("Europe/Riga")
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions"
PARAMS_FILE = core.ROOT / "cl_params.json"
DEFAULTS = dict(tie_infl=0.10)
SEASONS = {2025: (date(2025, 7, 1), date(2026, 6, 15)), 2026: (date(2026, 7, 1), date(2027, 6, 15))}
PAYOUT = core.PAYOUT_SAFE


def load_params() -> dict:
    p = dict(DEFAULTS)
    if PARAMS_FILE.exists():
        p.update(json.loads(PARAMS_FILE.read_text()))
    return p


def season_for(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1


# --------------------------------------------------------------------------- data

def _odds(o: dict | None) -> dict:
    out = {"ml_home": None, "ml_away": None, "ml_draw": None, "total_line": None, "over_odds": None, "under_odds": None}
    if not o:
        return out
    dec = ND.american_to_decimal
    ml = o.get("moneyline") or {}
    out["ml_home"] = dec((ml.get("home", {}).get("close") or {}).get("odds")) or dec((o.get("homeTeamOdds") or {}).get("moneyLine"))
    out["ml_away"] = dec((ml.get("away", {}).get("close") or {}).get("odds")) or dec((o.get("awayTeamOdds") or {}).get("moneyLine"))
    out["ml_draw"] = dec((o.get("drawOdds") or {}).get("moneyLine"))
    out["total_line"] = o.get("overUnder")
    tot = o.get("total") or {}
    out["over_odds"] = dec((tot.get("over", {}).get("close") or {}).get("odds")) or dec(o.get("overOdds"))
    out["under_odds"] = dec((tot.get("under", {}).get("close") or {}).get("odds")) or dec(o.get("underOdds"))
    return out


def _parse(e: dict) -> dict | None:
    c = e["competitions"][0]
    t = {x["homeAway"]: x for x in c["competitors"]}
    if "home" not in t or "away" not in t:
        return None
    h, a = t["home"], t["away"]
    st = e["status"]["type"]
    done = bool(st.get("completed"))
    detail = (st.get("detail") or st.get("shortDetail") or "").upper()
    utc = pd.Timestamp(e["date"])
    row = {
        "game_id": str(e["id"]), "date": utc.tz_convert(RIGA).date(), "start_utc": utc,
        "home_id": str(h["team"]["id"]), "away_id": str(a["team"]["id"]),
        "home": h["team"].get("abbreviation", h["team"].get("shortDisplayName", "")),
        "away": a["team"].get("abbreviation", a["team"].get("shortDisplayName", "")),
        "home_name": h["team"].get("displayName", ""), "away_name": a["team"].get("displayName", ""),
        "home_short": h["team"].get("shortDisplayName") or h["team"].get("displayName", ""),
        "away_short": a["team"].get("shortDisplayName") or a["team"].get("displayName", ""),
        "completed": done, "detail": detail,
        "home_goals": float(h["score"]) if done else None, "away_goals": float(a["score"]) if done else None,
        "extra_time": done and ("AET" in detail or "PEN" in detail or "ET" == detail),
        "home_record": None, "away_record": None,
    }
    row.update(_odds((c.get("odds") or [None])[0]))
    return row


def load_games(start: date, end: date) -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    with ThreadPoolExecutor(4) as ex:
        boards = list(ex.map(lambda d: espn.scoreboard(d, base=BASE), days))
    rows = [r for sb in boards for e in sb.get("events", []) if (r := _parse(e))]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("game_id").sort_values("start_utc").reset_index(drop=True)
    df["home_reg"], df["away_reg"] = df["home_goals"], df["away_goals"]
    for i in df.index[df["extra_time"].fillna(False)]:  # 90-minute score of games that went to extra time
        hr, ar = regulation_score(df.loc[i])
        df.loc[i, "home_reg"], df.loc[i, "away_reg"] = hr, ar
    return df


def regulation_score(g) -> tuple[float | None, float | None]:
    """Goals scored up to 90'+stoppage, from the match summary (only needed after extra time)."""
    # separate cache entry from the trimmed summary (that one has no key events)
    s = espn.get_json("/summary", {"event": g["game_id"], "lang": "en"}, base=BASE)
    h = a = 0
    for k in s.get("keyEvents") or []:
        if not k.get("scoringPlay"):
            continue
        clock = (k.get("clock") or {}).get("displayValue", "")
        m = re.match(r"(\d+)", clock)
        if not m or int(m.group(1)) > 90:
            continue
        tid = str((k.get("team") or {}).get("id", ""))
        text = (k.get("text") or "") + " " + (k.get("shortText") or "")
        if tid == g["home_id"] or (not tid and g["home_name"] and g["home_name"] in text):
            h += 1
        elif tid == g["away_id"] or (not tid and g["away_name"] and g["away_name"] in text):
            a += 1
    return float(h), float(a)


def closing_odds(game_ids: list[str]) -> pd.DataFrame:
    def one(gid):
        try:
            s = espn.summary(gid, base=BASE)
        except (LookupError, RuntimeError):
            return {"game_id": gid}
        pc = s.get("pickcenter") or []
        return {"game_id": gid, **_odds(pc[0] if pc else None)}
    with ThreadPoolExecutor(4) as ex:
        return pd.DataFrame(list(ex.map(one, game_ids)))


# --------------------------------------------------------------------------- model

def devig3(h, d, a):
    if not (h and d and a):
        return None
    x = np.array([1 / h, 1 / d, 1 / a])
    return x / x.sum()


def probs3(lh, la, tie):
    reg = N.reg_matrix(lh, la, tie)
    return np.array([reg[N._HW].sum(), reg[N._DIAG].sum(), reg[N._H < N._A].sum()])


def p_over(lh, la, tie, line):
    reg = N.reg_matrix(lh, la, tie)
    return float(reg[N._TOT_REG > line].sum())


def solve_lambdas(p3, line, q_over, tie) -> tuple[float, float]:
    """Goal expectations that best reproduce the bookmakers' 1X2 and over/under prices."""
    best, bl = None, (1.3, 1.1)
    grid = np.arange(0.2, 4.01, 0.1)
    for lh in grid:
        for la in grid:
            pr = probs3(lh, la, tie)
            err = (pr[0] - p3[0]) ** 2 + (pr[2] - p3[2]) ** 2
            if line is not None and q_over is not None:
                err += (p_over(lh, la, tie, line) - q_over) ** 2
            if best is None or err < best:
                best, bl = err, (lh, la)
    # refine
    lh0, la0 = bl
    for lh in np.arange(lh0 - 0.1, lh0 + 0.101, 0.02):
        for la in np.arange(la0 - 0.1, la0 + 0.101, 0.02):
            if lh <= 0.05 or la <= 0.05:
                continue
            pr = probs3(lh, la, tie)
            err = (pr[0] - p3[0]) ** 2 + (pr[2] - p3[2]) ** 2
            if line is not None and q_over is not None:
                err += (p_over(lh, la, tie, line) - q_over) ** 2
            if err < best:
                best, bl = err, (lh, la)
    return bl


# --------------------------------------------------------------------------- bets (90 minutes)

def btts(yes: bool) -> HB.Bet:
    def fn(g):
        both = (g.RH > 0) & (g.RA > 0)
        return (both if yes else ~both), np.zeros_like(g.RH, bool)
    return HB.Bet("abas komandas gūst vārtus" if yes else "abas komandas negūst vārtus", fn)


def double_chance(team: str, home: bool) -> HB.Bet:
    def fn(g):
        return ((g.RH >= g.RA) if home else (g.RA >= g.RH)), np.zeros_like(g.RH, bool)
    return HB.Bet(f"{team} vai neizšķirts", fn)


def parse_bet(text: str, game) -> HB.Bet:
    parts = [x.strip() for x in text.split("&")]
    if len(parts) > 1:
        return HB.combo([parse_bet(x, game) for x in parts])
    low = text.strip().lower()
    if low == "abas komandas gūst vārtus":
        return btts(True)
    if low == "abas komandas negūst vārtus":
        return btts(False)
    m = re.fullmatch(r"(\S+) vai neizšķirts", text.strip())
    if m:
        t = HB._team(m.group(1), game)
        if t is None:
            raise ValueError(f"nepazīstu komandu '{m.group(1)}'")
        return double_chance(*t)
    return HB.parse_bet(text, game)


def settle(text: str, g) -> tuple[bool, bool, str]:
    if g.home_reg is None or pd.isna(g.home_reg):
        raise ValueError("nav zināms 90 minūšu rezultāts")
    bet = parse_bet(text, g)
    w, push = bet.fn(HB.Actual(g))
    score = f"{g.away} {int(g.away_reg)} - {int(g.home_reg)} {g.home}"
    if g.extra_time:
        score += f" (90 min.; pēc pagarinājuma {int(g.away_goals)} - {int(g.home_goals)})"
    return bool(w[0]), bool(push[0]), score


CL = core.League("cl", "Čempionu līga", "⚽", core.ROOT / "cl", core.ROOT / "CL.md", load_games, settle, day_offset=0)


def candidates(r, g, pmin: float, pmax: float, crazy: bool) -> list:
    """(bet, p, category) for every common 90-minute market."""
    H, A = r.home, r.away
    out = []

    def add(bet, cat):
        pw, push = HB.prob(g, bet)
        if push == 0 and pmin <= pw <= pmax:
            out.append((bet, pw, cat))

    for team, home in ((H, True), (A, False)):
        add(HB.reg_win(team, home), "uzvarētājs / handikaps")
        add(double_chance(team, home), "uzvarētājs / handikaps")
        for line in (-2.5, -1.5, 1.5, 2.5):
            add(HB.handicap(team, home, line), "uzvarētājs / handikaps")
        for line in (0.5, 1.5, 2.5, 3.5):
            add(HB.team_total(team, home, True, line), "komandas vārti")
            add(HB.team_total(team, home, False, line), "komandas vārti")
        if crazy:
            for line in (1.5, 2.5, 3.5):
                for over in (True, False):
                    add(HB.combo([HB.reg_win(team, home), HB.total(over, line)]), "kombinācija")
            add(HB.combo([HB.reg_win(team, home), btts(True)]), "kombinācija")
    add(HB.reg_draw(), "uzvarētājs / handikaps")
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        add(HB.total(True, line), "kopā")
        add(HB.total(False, line), "kopā")
    add(btts(True), "kopā")
    add(btts(False), "kopā")
    if crazy:
        for line in (2.5, 3.5):
            add(HB.combo([btts(True), HB.total(True, line)]), "kombinācija")
        add(HB.combo([HB.reg_draw(), btts(True)]), "kombinācija")
    return out


def pick_for_game(r, p, min_odds: float) -> list[dict]:
    g = N.HockeyGrid(r.lh, r.la, p["tie_infl"], 0.0)
    cands = candidates(r, g, 0.15, PAYOUT / min_odds, crazy=False)
    groups = ("uzvarētājs / handikaps", "kopā", "komandas vārti")
    ranked = sorted(cands, key=lambda c: -c[1])
    chosen, used = [], {}
    for cat in groups:
        cs = [c for c in ranked if c[2] == cat]
        if cs and cs[0][1] >= 0.33:
            chosen.append(cs[0]); used[cat] = 1
    for c in ranked:
        if len(chosen) >= 3:
            break
        if c in chosen or used.get(c[2], 0) >= 2:
            continue
        if any(HB.prob(g, HB.combo([c[0], x[0]]))[0] < 0.02 for x in chosen):
            continue
        chosen.append(c); used[c[2]] = used.get(c[2], 0) + 1
    picks = [{"category": c[2], "bet": c[0].label, "p": c[1], "_bet": c[0]} for c in chosen]
    picks.sort(key=lambda x: -x["p"])
    core.add_data_view(picks, g, g, prob=lambda gr, b: HB.prob(gr, b)[0])  # ★ = bookmakers' most likely
    for x in picks:
        x.update(fair_odds=round(1 / x["p"], 2), est_odds=round(PAYOUT / x["p"], 2), confidence=int(round(100 * x["p"])))
    return picks[:3]


def pick_crazy(df, p, min_odds=4.0, n=3) -> list[dict]:
    rows = []
    for r in df.itertuples():
        g = N.HockeyGrid(r.lh, r.la, p["tie_infl"], 0.0)
        cands = candidates(r, g, 0.08, PAYOUT / min_odds, crazy=True)
        if cands:
            b, pw, cat = max(cands, key=lambda c: c[1])
            rows.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                         "riga_time": r.riga_time, "category": cat, "bet": b.label, "p": pw})
    rows.sort(key=lambda x: -x["p"])
    for x in rows[:n]:
        x.update(fair_odds=round(1 / x["p"], 2), est_odds=round(PAYOUT / x["p"], 2),
                 confidence=int(round(100 * x["p"])), result=None)
    return rows[:n]


def pick_safe(df, p) -> list[dict]:
    per_game = []
    for r in df.itertuples():
        g = N.HockeyGrid(r.lh, r.la, p["tie_infl"], 0.0)
        cands = candidates(r, g, 0.35, PAYOUT / core.SAFE_MIN_ODDS, crazy=False)
        per_game.append((r, [(b, pw, pw, cat, pw) for b, pw, cat in cands]))
    return core.choose_safe(per_game, "market")


def predict_day(day: date, p) -> pd.DataFrame:
    games = load_games(day, day)
    if games.empty:
        return games
    todo = games[~games["completed"]]
    rows = []
    for g in todo.itertuples():
        p3 = devig3(g.ml_home, g.ml_draw, g.ml_away)
        if p3 is None:
            continue  # no odds yet -> no picks (the model needs the bookmakers' prices)
        q = N.devig2(g.over_odds, g.under_odds) if g.over_odds and g.under_odds else (0.5 if g.total_line else None)
        lh, la = solve_lambdas(p3, g.total_line, q, p["tie_infl"])
        rows.append({
            "date": g.date, "game_id": g.game_id, "start_utc": g.start_utc,
            "riga_time": g.start_utc.tz_convert(RIGA).strftime("%d.%m %H:%M"),
            "home": g.home, "away": g.away, "home_name": g.home_name, "away_name": g.away_name,
            "home_short": g.home_short, "away_short": g.away_short,
            "lh": lh, "la": la, "p3": p3, "total_line": g.total_line,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- commands

def cmd_likmes(args) -> None:
    p = load_params()
    CL.message.unlink(missing_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    day = pd.Timestamp.now(tz=RIGA).date()
    df = predict_day(day, p)
    if not df.empty:
        df = df[df["start_utc"] > now]
    if df.empty:
        print("Šodien nav Čempionu līgas spēļu ar bukmeikeru koeficientiem.")
        return
    print(f"ČEMPIONU LĪGA — {day:%d.%m.%Y}, min. koeficients {args.min_odds:g} (visas likmes pēc 90 minūtēm)")
    record, games_info = [], {}
    for r in df.itertuples():
        picks = pick_for_game(r, p, args.min_odds)
        ph, pd_, pa = r.p3
        print(f"\n{r.away} @ {r.home}  {r.riga_time}  |  bukmeikeri: {r.home} {ph:.0%} · X {pd_:.0%} · {r.away} {pa:.0%}, "
              f"vārti {r.lh:.2f}–{r.la:.2f}")
        top = max(range(len(picks)), key=lambda i: picks[i].get("data_score", picks[i]["p"])) if picks else -1
        for i, x in enumerate(picks, 1):
            print(f"  {i}.{'★' if i - 1 == top else ' '}{x['bet']:<34} drošība {x['confidence']:>3}/100   "
                  f"godīgais koef. {x['fair_odds']:.2f}  (Optibet ~{x['est_odds']:.2f})   [{x['category']}]")
            record.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                           **x, "result": None})
        games_info[r.game_id] = {
            "game_id": r.game_id, "away": r.away, "home": r.home, "away_name": r.away_name, "home_name": r.home_name,
            "away_short": r.away_short, "home_short": r.home_short,
            "start_utc": str(r.start_utc), "riga_time": r.riga_time,
            "robot_line": f"{r.home} {ph:.0%} · X {pd_:.0%} · {r.away} {pa:.0%}", "robot_total": f"vārti {r.lh:.1f}–{r.la:.1f}",
            "market_line": "—", "market_total": (f"kopā {r.total_line:g}" if r.total_line else None), "injuries": [],
        }
    crazy = pick_crazy(df, p)
    safe = pick_safe(df, p)
    core.print_safe(safe, 34)
    print("\nCRAZY DAZY: " + "; ".join(f"{x['game']} {x['bet']} (~{x['est_odds']:.2f})" for x in crazy))
    f = core.save_night(day, record, games_info, crazy, args.min_odds, now, safe=safe)
    print(f"\nSaglabāts: {f.relative_to(core.ROOT)}")


def cmd_kalibret(args) -> None:
    """Draw boost so the Poisson grid reproduces the bookmakers' draw prices (last season), then check vs results."""
    start, end = SEASONS[season_for(espn.today_et()) - 1]
    games = load_games(start, min(end, espn.today_et()))
    games = games[games["completed"] & ~games["extra_time"].fillna(False)]
    odds = closing_odds(games["game_id"].tolist())
    g = games.drop(columns=[c for c in odds.columns if c != "game_id"]).merge(odds, on="game_id")
    g = g[g["ml_home"].notna() & g["ml_draw"].notna() & g["ml_away"].notna()]
    print(f"Sezona {start.year}-{end.year % 100}: {len(g)} spēles ar noslēguma koeficientiem")
    best = None
    for tie in np.arange(0.0, 0.31, 0.05):
        err, pdraw = [], []
        for x in g.itertuples():
            p3 = devig3(x.ml_home, x.ml_draw, x.ml_away)
            q = N.devig2(x.over_odds, x.under_odds) if x.over_odds and x.under_odds else None
            lh, la = solve_lambdas(p3, x.total_line, q, tie)
            pr = probs3(lh, la, tie)
            err.append((pr[1] - p3[1]) ** 2)
            pdraw.append(pr[1])
        e = float(np.mean(err))
        print(f"  tie_infl={tie:.2f}: neizšķirta kļūda {e:.5f}, vidēji neizšķirts {np.mean(pdraw):.1%}")
        if best is None or e < best[0]:
            best = (e, tie, np.mean(pdraw))
    _, tie, mean_draw = best
    actual = float((g["home_goals"] == g["away_goals"]).mean())
    print(f"  -> tie_infl={tie:.2f}; modelis neizšķirts {mean_draw:.1%}, realitātē {actual:.1%}")
    PARAMS_FILE.write_text(json.dumps({"tie_infl": round(float(tie), 2)}, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Čempionu līgas likmju robots")
    ap.add_argument("komanda", type=lambda s: s.lower().replace("ā", "a").replace("ū", "u"),
                    choices=["likmes", "rezultati", "eksports", "kalibret"])
    ap.add_argument("--min-odds", type=float, default=2.1)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    espn.FORCE_REFRESH = args.refresh
    core.configure(CL)
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
