"""NHL prognozes: same commands as bot.py, for hockey.

  python nhl_bot.py likmes      # 3 bets per game tonight (odds >= 2.1) + Crazy Dazy (>= 4.0)
  python nhl_bot.py rezultati   # settle last night's bets + statistics
  python nhl_bot.py eksports

History: nhl/history/, page data: nhl/export/, report: NHL.md
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import bot as core
import espn
import model as M
import nhl_bets as HB
import nhl_data as D
import nhl_model as N

RIGA = ZoneInfo("Europe/Riga")
NHL = core.League("nhl", "NHL", "🏒", core.ROOT / "nhl", core.ROOT / "NHL.md", D.load_games, HB.settle)
PAYOUT = 0.93
TOTAL_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
TEAM_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
LEAN_PER_GOAL = 0.01   # tie-break towards the side our own model likes more than the bookmakers


# --------------------------------------------------------------------------- predictions

def predict_day(day, p: dict) -> pd.DataFrame:
    season = D.season_for(day)
    games = D.load_season(season, until=day)
    rf = N.rating_frame(games)
    todo = rf[(rf["date"] == day) & ~rf["completed"]]
    if todo.empty:
        return pd.DataFrame()
    prior = N.season_prior(D.load_season(season - 1), p) if season - 1 in D.SEASONS else None
    as_of = todo["start_utc"].min()
    r = N.fit(rf, as_of=as_of, prior=prior, p=p)

    skaters, goalies, _ = D.load_boxscores(games[games["completed"]])
    starts = N.starters(goalies) if not goalies.empty else goalies
    gv, _ = N.goalie_values(goalies, p=p) if not goalies.empty else (pd.DataFrame(columns=["gsaa", "name"]), 0.9)
    sv = N.skater_values(skaters, p=p) if not skaters.empty else pd.DataFrame(columns=["value", "name"])
    sk_logs = N.skater_logs_for_availability(skaters) if not skaters.empty else skaters
    inj = D.load_injuries()
    rosters = D.load_rosters(sorted(set(todo["home_id"]) | set(todo["away_id"])))

    rows = []
    for g in todo.itertuples():
        m_lh, m_la = N.lambdas(r, g.home_id, g.away_id, g.neutral, g.home_b2b, g.away_b2b)
        notes, adj = {}, {}
        for side, tid, gid, gname, gstat in (("h", g.home_id, g.home_goalie_id, g.home_goalie, g.home_goalie_status),
                                             ("a", g.away_id, g.away_goalie_id, g.away_goalie, g.away_goalie_status)):
            gd, _ = N.goalie_delta(tid, gid, starts, gv, None, p) if not starts.empty else (0.0, None)
            ti = inj[inj["team_id"] == tid] if not inj.empty else inj
            probs = {x.player_id: M.status_to_prob(x.status, x.return_date, g.date) for x in ti.itertuples()}
            stats = {x.player_id: x.status for x in ti.itertuples()}
            sa = (M.availability_adjustment(tid, sk_logs, sv, probs, stats, rosters.get(tid) or None, None,
                                            p["recent_games"]) if not sk_logs.empty else M.Availability())
            adj[side] = (gd, sa.adj)
            n = []
            if gname:
                st = {"confirmed": "apstiprināts", "expected": "gaidāms"}.get(gstat or "", gstat or "")
                n.append(("vārtsargs " + gname, st, gd))
            n += [(nm, s, d * p["skater_coef"] / max(p["skater_coef"], 1e-9)) for nm, s, d in sa.notes]
            notes[side] = n
        (gh, sh), (ga, sa_) = adj["h"], adj["a"]
        lh = max(0.3, m_lh - ga + (sh - sa_) / 2)
        la = max(0.3, m_la - gh - (sh - sa_) / 2)
        model_l = (lh, la)
        p_mk = N.devig2(g.ml_home, g.ml_away)
        market_l = None
        if p_mk is not None:
            market_l = N.market_lambdas(p_mk, g.total_line, N.devig2(g.over_odds, g.under_odds),
                                        p["tie_infl"], p["ot_k"], lh + la)
        fl = N.blend(model_l, market_l, p["market_weight"])
        rows.append({
            "date": g.date, "game_id": g.game_id, "start_utc": g.start_utc,
            "riga_time": g.start_utc.tz_convert(RIGA).strftime("%d.%m %H:%M"),
            "home": g.home, "away": g.away, "home_name": g.home_name, "away_name": g.away_name,
            "home_record": g.home_record, "away_record": g.away_record,
            "lh": fl[0], "la": fl[1], "model_lh": model_l[0], "model_la": model_l[1],
            "mk_lh": market_l[0] if market_l else None, "mk_la": market_l[1] if market_l else None,
            "p_market": p_mk, "total_line": g.total_line, "has_market": market_l is not None,
            "notes_home": notes["h"], "notes_away": notes["a"],
        })
    return pd.DataFrame(rows)


def grid(r, p) -> N.HockeyGrid:
    return N.HockeyGrid(r.lh, r.la, p["tie_infl"], p["ot_k"])


def win_line(r, p) -> tuple[str, float]:
    """Regulation 1X2 probabilities, e.g. 'NJ 47% · X 25% · MTL 28%'."""
    reg = N.reg_matrix(r.lh, r.la, p["tie_infl"])
    ph, pdraw = float(reg[N._HW].sum()), float(reg[N._DIAG].sum())
    return f"{r.home} {ph:.0%} · X {pdraw:.0%} · {r.away} {1 - ph - pdraw:.0%}", ph


def market_text(r) -> str | None:
    if r.p_market is None or pd.isna(r.p_market):
        return None
    t = f"{r.home} {r.p_market:.0%}" if r.p_market >= 0.5 else f"{r.away} {1 - r.p_market:.0%}"
    return t + " ar papildl."


def header(r, p) -> str:
    rl, _ = win_line(r, p)
    mk = market_text(r) or "—"
    if r.total_line is not None and pd.notna(r.total_line):
        mk += f", kopā {r.total_line:g}"
    return (f"{r.away} @ {r.home}  {r.riga_time}  |  robots (pamatlaiks): {rl}, vārti {r.lh:.1f}–{r.la:.1f}  "
            f"|  bukmeikeri: {mk}")


def _note_text(team, notes, limit=3):
    return [f"{team}: {n}{' (' + s + ')' if s else ''} {d:+.2f}" for n, s, d in notes[:limit] if n]


# --------------------------------------------------------------------------- picks

def _leans(r) -> tuple[float, float]:
    """+ = our model likes home (goal margin) / more goals, compared with the final blend."""
    return (r.model_lh - r.model_la) - (r.lh - r.la), (r.model_lh + r.model_la) - (r.lh + r.la)


def _candidates(r, g, pmin, pmax, crazy: bool):
    lean_m, lean_t = _leans(r)
    H, A = r.home, r.away
    out = []

    def add(bet, lean, cat):
        pw, push = HB.prob(g, bet)
        if push == 0 and pmin <= pw <= pmax:
            out.append((bet, pw, pw + LEAN_PER_GOAL * lean, cat))

    for team, home in ((H, True), (A, False)):
        s = 1 if home else -1
        add(HB.ml(team, home), s * lean_m, "uzvarētājs")
        add(HB.reg_win(team, home), s * lean_m, "uzvarētājs")
        for line in (-2.5, -1.5, 1.5, 2.5):
            add(HB.handicap(team, home, line), s * lean_m, "handikaps")
        for line in TEAM_LINES:
            add(HB.team_total(team, home, True, line), (lean_t + s * lean_m) / 2, "komandas vārti")
            add(HB.team_total(team, home, False, line), -(lean_t + s * lean_m) / 2, "komandas vārti")
        if crazy:
            main = r.total_line if r.total_line is not None and pd.notna(r.total_line) else 5.5
            for over in (True, False):
                for line in (main - 1, main):
                    add(HB.combo([HB.reg_win(team, home), HB.total(over, line)]),
                        s * lean_m + (1 if over else -1) * lean_t, "kombinācija")
    add(HB.reg_draw(), -abs(lean_m), "neizšķirts")
    for line in TOTAL_LINES:
        add(HB.total(True, line), lean_t, "kopā")
        add(HB.total(False, line), -lean_t, "kopā")
    return out


def pick_for_game(r, p, min_odds: float) -> list[dict]:
    g = grid(r, p)
    cands = _candidates(r, g, 0.15, PAYOUT / min_odds, crazy=False)
    groups = {"uzvarētājs / handikaps": ("uzvarētājs", "handikaps", "neizšķirts"),
              "kopā": ("kopā",), "komandas vārti": ("komandas vārti",)}
    group_of = {k: cat for cat, kinds in groups.items() for k in kinds}
    ranked = sorted(cands, key=lambda c: -c[2])
    chosen, used = [], {}
    # 1) the best bet of each category, unless it is a weak one (hockey lines are sparse)
    for cat in groups:
        cs = [c for c in ranked if group_of[c[3]] == cat]
        if cs and cs[0][1] >= 0.33:
            chosen.append(cs[0])
            used[cat] = 1
    # 2) fill up to 3 with the next most likely bets, at most 2 per category
    for c in ranked:
        if len(chosen) >= 3:
            break
        cat = group_of[c[3]]
        if c in chosen or used.get(cat, 0) >= 2:
            continue
        # never pair bets that cannot both win (e.g. NJ over 3.5 and NJ under 2.5)
        if any(HB.prob(g, HB.combo([c[0], x[0]]))[0] < 0.02 for x in chosen):
            continue
        chosen.append(c)
        used[cat] = used.get(cat, 0) + 1
    picks = [{"category": group_of[c[3]], "bet": c[0].label, "p": c[1], "_bet": c[0]} for c in chosen]
    picks.sort(key=lambda x: -x["p"])
    own = N.HockeyGrid(r.model_lh, r.model_la, p["tie_infl"], p["ot_k"])
    mkt = (N.HockeyGrid(r.mk_lh, r.mk_la, p["tie_infl"], p["ot_k"])
           if r.mk_lh is not None and pd.notna(r.mk_lh) else None)
    core.add_data_view(picks, own, mkt, prob=lambda gr, b: HB.prob(gr, b)[0])
    for x in picks:
        x["fair_odds"] = round(1 / x["p"], 2)
        x["est_odds"] = round(PAYOUT / x["p"], 2)
        x["confidence"] = int(round(100 * x["p"]))
    return picks[:3]


def pick_crazy(df: pd.DataFrame, p, min_odds: float = 4.0, n: int = 3) -> list[dict]:
    rows = []
    for r in df.itertuples():
        cands = _candidates(r, grid(r, p), 0.08, PAYOUT / min_odds, crazy=True)
        if cands:
            b, pw, score, cat = max(cands, key=lambda c: c[2])
            rows.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                         "riga_time": r.riga_time, "category": cat, "bet": b.label, "p": pw, "_s": score})
    rows.sort(key=lambda x: -x["_s"])
    out = []
    for x in rows[:n]:
        x.pop("_s")
        x.update(fair_odds=round(1 / x["p"], 2), est_odds=round(PAYOUT / x["p"], 2),
                 confidence=int(round(100 * x["p"])), result=None)
        out.append(x)
    return out


def pick_safe(df: pd.DataFrame, p, rank_by: str = "market", all_games: bool = False) -> list[dict]:
    """Drošās likmes: odds >= 1.7, ranked by the bookmakers' probability (see core.choose_safe)."""
    per_game = []
    for r in df.itertuples():
        g = grid(r, p)
        own = N.HockeyGrid(r.model_lh, r.model_la, p["tie_infl"], p["ot_k"])
        mkt = (N.HockeyGrid(r.mk_lh, r.mk_la, p["tie_infl"], p["ot_k"])
               if r.mk_lh is not None and pd.notna(r.mk_lh) else None)
        cands = _candidates(r, g, 0.35, core.PAYOUT_SAFE / core.SAFE_MIN_ODDS, crazy=False)
        per_game.append((r, [(b, pw, HB.prob(own, b)[0], cat, HB.prob(mkt, b)[0] if mkt is not None else None)
                             for b, pw, _, cat in cands]))
    out = core.choose_safe(per_game, rank_by, all_games)
    return core.with_names(out, df) if all_games else out


def pick_robot(df: pd.DataFrame, p) -> list[dict]:
    """Robota lapa: chosen only by the robot's own goal model; the bookmakers' line only gives the price."""
    return core.choose_robot(robot_per_game(df, p), df)


def robot_per_game(df: pd.DataFrame, p) -> list:
    per_game = []
    for r in df.itertuples():
        own = N.HockeyGrid(r.model_lh, r.model_la, p["tie_infl"], p["ot_k"])
        mkt = (N.HockeyGrid(r.mk_lh, r.mk_la, p["tie_infl"], p["ot_k"])
               if r.mk_lh is not None and pd.notna(r.mk_lh) else None)
        cands = _candidates(r, own, 0.0, 1.0, crazy=False)
        per_game.append((r, [(b, pw, HB.prob(mkt, b)[0] if mkt is not None else None, cat) for b, pw, _, cat in cands]))
    return per_game


# --------------------------------------------------------------------------- commands

def cmd_likmes(args) -> None:
    p = N.load_params()
    NHL.message.unlink(missing_ok=True)
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
        print("Tuvākajā laikā nav neizspēlētu NHL spēļu.")
        return
    print(f"NHL LIKMES — nakts uz {(day + timedelta(days=1)):%d.%m.%Y} (ASV datums {day:%Y-%m-%d}), "
          f"min. koeficients {args.min_odds:g}")
    print("Visas likmes vērtē pēc pamatlaika (60 min), ja vien nav rakstīts '(ieskaitot papildlaiku)'.")
    record, games_info = [], {}
    for r in df.itertuples():
        picks = pick_for_game(r, p, args.min_odds)
        print("\n" + header(r, p))
        top = max(range(len(picks)), key=lambda i: picks[i].get("data_score", picks[i]["p"])) if picks else -1
        for i, x in enumerate(picks, 1):
            print(f"  {i}.{'★' if i - 1 == top else ' '}{x['bet']:<24} drošība {x['confidence']:>3}/100   "
                  f"godīgais koef. {x['fair_odds']:.2f}  (Optibet ~{x['est_odds']:.2f})   [{x['category']}]")
        notes = _note_text(r.home, r.notes_home) + _note_text(r.away, r.notes_away)
        if notes:
            print("  sastāvs: " + " | ".join(notes))
        for x in picks:
            record.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                           **x, "result": None})
        rl, _ = win_line(r, p)
        mk = market_text(r)
        games_info[r.game_id] = {
            "game_id": r.game_id, "away": r.away, "home": r.home, "away_name": r.away_name,
            "home_name": r.home_name, "away_record": r.away_record, "home_record": r.home_record,
            "start_utc": str(r.start_utc), "riga_time": r.riga_time,
            "robot_line": rl, "robot_total": f"vārti {r.lh:.1f}–{r.la:.1f}",
            "market_line": mk or "—",
            "market_total": (f"kopā {r.total_line:g}" if r.total_line is not None and pd.notna(r.total_line) else None),
            "injuries": notes,
        }
    crazy = pick_crazy(df, p, NHL.crazy_min_odds)
    print(f"\nCRAZY DAZY — 3 reālākās likmes ar koeficientu >= {NHL.crazy_min_odds:g}:")
    for i, x in enumerate(crazy, 1):
        print(f"  {i}. {x['game']:<10} {x['bet']:<34} drošība {x['confidence']:>3}/100   "
              f"godīgais koef. {x['fair_odds']:.2f}  (Optibet ~{x['est_odds']:.2f})   [{x['category']}]")
    safe = pick_safe(df, p)
    core.print_safe(safe, 34)
    f = core.save_night(day, record, games_info, crazy, args.min_odds, now, safe=safe,
                        safe_all=pick_safe(df, p, all_games=True), robot=pick_robot(df, p),
                        p15=core.choose_15(robot_per_game(df, p), df, "nhl"), extra=core.extra_bets(df, "nhl"))
    print(f"\nSaglabāts: {f.relative_to(core.ROOT)}  — rīt palaid: python nhl_bot.py rezultati")


def main():
    ap = argparse.ArgumentParser(description="NHL likmju robots")
    ap.add_argument("komanda", type=lambda s: s.lower().replace("ā", "a").replace("ū", "u"),
                    choices=["likmes", "rezultati", "eksports"])
    ap.add_argument("--min-odds", type=float, default=2.1)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    espn.FORCE_REFRESH = args.refresh
    core.configure(NHL)
    if args.komanda == "eksports":
        core.export_all()
        print(f"Eksportēts: {NHL.export.relative_to(core.ROOT)}/")
    elif args.komanda == "likmes":
        cmd_likmes(args)
    else:
        core.cmd_rezultati(args)


if __name__ == "__main__":
    sys.exit(main())
