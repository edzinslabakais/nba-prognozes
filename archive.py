"""Archive: what the robot WOULD have recommended on every past day of the season, and how it ended.

Each day is simulated with only what was known before the games:
  * team ratings from earlier games only (walk-forward), rest / back-to-backs,
  * the bookmakers' closing lines (they already include the injury news of that day),
  * NHL: the probable/confirmed starting goalies.
Historical injury reports are not available, so the robot's own injury adjustment is left out.
The picks use exactly the same rules as live (3 bets >= 2.1, ★ by data, Crazy Dazy >= 4.0)
and are settled on the regulation result unless a bet says "(ieskaitot papildlaiku)".

Output: archive/<date>.json (NBA) and nhl/archive/<date>.json, marked "simulated": true.

  python archive.py              # both leagues, from season start up to the first live night
  python archive.py --league nhl
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import backtest
import bot as core
import data
import espn
import model as M
import nhl_backtest as NB
import nhl_bets as HB
import nhl_bot
import nhl_data as D
import nhl_model as N
import predict as P

RIGA = ZoneInfo("Europe/Riga")


def first_live_date(league: core.League) -> date:
    files = sorted(league.hist.glob("*.json"))
    return date.fromisoformat(files[0].stem) if files else espn.today_et()


def write_night(league: core.League, day, games_info: dict, picks: list, crazy: list) -> None:
    league.archive.mkdir(parents=True, exist_ok=True)
    rec = {"date": str(day), "min_odds": 2.1, "simulated": True,
           "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
           "games": games_info, "picks": picks, "crazy": crazy}
    (league.archive / f"{day}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))


def settle_list(items: list, by_id: dict, settle_fn) -> None:
    for p in items:
        g = by_id.get(p["game_id"])
        if g is None or not g.completed:
            continue
        win, push, score = settle_fn(p["bet"], g)
        p["result"] = "atgriezts" if push else ("izgāja" if win else "neizgāja")
        p["score"] = score


# --------------------------------------------------------------------------- NBA

def archive_nba(until: date) -> int:
    core.configure(core.NBA)
    params = P.load_params()
    mw = params.get("market_weight_suggested", 0.9)
    season = data.season_for(until)
    games = data.load_season(season, until=until - timedelta(days=1))
    games = games[games["completed"] & (games["date"] < until)]
    prior = None
    if season - 1 in data.SEASONS:
        prev = M.add_rest(data.load_season(season - 1))
        prior = M.fit_ratings(prev, half_life=60, ridge=2.0).shrunk(params["prior_shrink"])
    wf = backtest.walk_forward(games, prior, half_life=params["half_life"], ridge=params["ridge"])
    _, lines = data.load_player_logs(games)
    g = M.add_rest(games).merge(wf[["game_id", "pred", "pred_total"]], on="game_id").merge(lines, on="game_id", how="left")
    n = 0
    for day, todays in g.groupby("date"):
        rows = []
        for x in todays.itertuples():
            has_m = pd.notna(x.close_margin)
            has_t = pd.notna(x.close_total)
            margin = (1 - mw) * x.pred + mw * x.close_margin if has_m else x.pred
            rows.append({
                "game_id": x.game_id, "start_utc": x.start_utc, "date": day,
                "riga_time": x.start_utc.tz_convert(RIGA).strftime("%d.%m %H:%M"),
                "home": x.home, "away": x.away, "home_name": x.home_name, "away_name": x.away_name,
                "home_record": None, "away_record": None,
                "margin_home": margin, "model_margin_home": x.pred,
                "market_margin_home": x.close_margin if has_m else None,
                "total": float(x.close_total) if has_t else x.pred_total, "model_total": x.pred_total,
                "market_total": float(x.close_total) if has_t else None, "has_market": bool(has_m),
            })
        df = pd.DataFrame(rows)
        picks, games_info = [], {}
        for r in df.itertuples():
            for pk in core.pick_for_game(r, 2.1):
                picks.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                              **pk, "result": None})
            games_info[r.game_id] = {
                "game_id": r.game_id, "away": r.away, "home": r.home, "away_name": r.away_name,
                "home_name": r.home_name, "start_utc": str(r.start_utc), "riga_time": r.riga_time,
                "robot_line": (f"{r.home} {-r.margin_home:+.1f}" if r.margin_home >= 0 else f"{r.away} {r.margin_home:+.1f}"),
                "robot_total": round(float(r.total), 1),
                "market_line": P._fmt_line(r.market_margin_home, r.home, r.away),
                "market_total": r.market_total, "injuries": [],
            }
        crazy = core.pick_crazy(df)
        by_id = {x.game_id: x for x in todays.itertuples()}
        settle_list(picks, by_id, core._nba_settle)
        settle_list(crazy, by_id, core._nba_settle)
        write_night(core.NBA, day, games_info, picks, crazy)
        n += 1
    return n


# --------------------------------------------------------------------------- NHL

def archive_nhl(until: date) -> int:
    core.configure(nhl_bot.NHL)
    p = N.load_params()
    season = D.season_for(until)
    games = D.load_season(season, until=until - timedelta(days=1))
    games = games[games["completed"] & games["home_reg"].notna() & (games["date"] < until)]
    prior = N.season_prior(D.load_season(season - 1), p) if season - 1 in D.SEASONS else None
    wf = NB.walk_forward(games, prior, p)
    skaters, goalies, odds = D.load_boxscores(games)
    starts = N.starters(goalies)
    g = games.merge(wf[["game_id", "lh", "la"]], on="game_id").merge(odds, on="game_id", how="left")
    n = 0
    for day, todays in g.groupby("date"):
        as_of = todays["start_utc"].min()
        gv, _ = N.goalie_values(goalies, as_of=as_of, p=p)
        rows = []
        for x in todays.itertuples():
            gh, _ = N.goalie_delta(x.home_id, x.home_goalie_id, starts, gv, x.start_utc, p)
            ga, _ = N.goalie_delta(x.away_id, x.away_goalie_id, starts, gv, x.start_utc, p)
            mlh, mla = max(0.3, x.lh - ga), max(0.3, x.la - gh)
            p_mk = N.devig2(x.close_ml_home, x.close_ml_away) if pd.notna(x.close_ml_home) else None
            mk = None
            if p_mk is not None:
                tl = x.close_total if pd.notna(x.close_total) else None
                q = N.devig2(x.close_over, x.close_under) if pd.notna(x.close_over) else None
                mk = N.market_lambdas(p_mk, tl, q, p["tie_infl"], p["ot_k"], mlh + mla)
            fl = N.blend((mlh, mla), mk, p["market_weight"])
            notes_h = [("vārtsargs " + x.home_goalie, "", gh)] if isinstance(x.home_goalie, str) else []
            notes_a = [("vārtsargs " + x.away_goalie, "", ga)] if isinstance(x.away_goalie, str) else []
            rows.append({
                "date": day, "game_id": x.game_id, "start_utc": x.start_utc,
                "riga_time": x.start_utc.tz_convert(RIGA).strftime("%d.%m %H:%M"),
                "home": x.home, "away": x.away, "home_name": x.home_name, "away_name": x.away_name,
                "home_record": None, "away_record": None,
                "lh": fl[0], "la": fl[1], "model_lh": mlh, "model_la": mla,
                "mk_lh": mk[0] if mk else None, "mk_la": mk[1] if mk else None,
                "p_market": p_mk, "total_line": x.close_total if pd.notna(x.close_total) else None,
                "has_market": mk is not None, "notes_home": notes_h, "notes_away": notes_a,
            })
        df = pd.DataFrame(rows)
        picks, games_info = [], {}
        for r in df.itertuples():
            for pk in nhl_bot.pick_for_game(r, p, 2.1):
                picks.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc),
                              **pk, "result": None})
            rl, _ = nhl_bot.win_line(r, p)
            games_info[r.game_id] = {
                "game_id": r.game_id, "away": r.away, "home": r.home, "away_name": r.away_name,
                "home_name": r.home_name, "start_utc": str(r.start_utc), "riga_time": r.riga_time,
                "robot_line": rl, "robot_total": f"vārti {r.lh:.1f}–{r.la:.1f}",
                "market_line": nhl_bot.market_text(r) or "—",
                "market_total": f"kopā {r.total_line:g}" if r.total_line is not None and pd.notna(r.total_line) else None,
                "injuries": nhl_bot._note_text(r.home, r.notes_home) + nhl_bot._note_text(r.away, r.notes_away),
            }
        crazy = nhl_bot.pick_crazy(df, p, nhl_bot.NHL.crazy_min_odds)
        by_id = {x.game_id: x for x in todays.itertuples()}
        settle_list(picks, by_id, HB.settle)
        settle_list(crazy, by_id, HB.settle)
        write_night(nhl_bot.NHL, day, games_info, picks, crazy)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["nba", "nhl", "all"], default="all")
    args = ap.parse_args()
    if args.league in ("nba", "all"):
        until = first_live_date(core.NBA)
        print(f"NBA: arhīvs līdz {until - timedelta(days=1)} — {archive_nba(until)} dienas")
        core.configure(core.NBA)
        core.export_all()
    if args.league in ("nhl", "all"):
        until = first_live_date(nhl_bot.NHL)
        print(f"NHL: arhīvs līdz {until - timedelta(days=1)} — {archive_nhl(until)} dienas")
        core.configure(nhl_bot.NHL)
        core.export_all()


if __name__ == "__main__":
    main()
