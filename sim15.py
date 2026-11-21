"""1.5+ page: season-long test of every bet at odds 1.5+ and the choice of rules for the page.

For every finished game of last season and this season (NBA, NHL, Champions League, Championship) the robot
is fitted only on earlier games (walk-forward), the bookmakers' closing line gives the price, and every common
bet with estimated odds 1.5-2.2 is settled on the real (regulation / 90-minute) result. Then simple rules are
compared on LAST season (which bet types, does the robot have to agree with the bookmakers, how many bets a
day) and the chosen rule is checked on THIS season, which it has not seen.

  python sim15.py dati      # build the bet table (sim15/<league>.csv.gz), takes a while the first time
  python sim15.py analize   # choose the rules -> p15_rules.json and print the report
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import bets as B
import bot as core
import espn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sim15"
RULES = ROOT / "p15_rules.json"
PAYOUT = B.PAYOUT
P_MIN, P_MAX = PAYOUT / 2.2, PAYOUT / 1.5   # estimated odds 1.5 ... 2.2


def _keep(p_mkt) -> bool:
    return p_mkt is not None and P_MIN <= p_mkt <= P_MAX


# --------------------------------------------------------------------------- NBA

def nba_rows(season: int, prior_season_fit: bool = True) -> list[dict]:
    import backtest as BT
    import data
    import model
    prm = json.loads((ROOT / "params.json").read_text())
    games = data.load_season(season, until=espn.today_et())
    games = games[games["completed"]]
    prior = BT.season_prior(season, prm["prior_shrink"])
    wf = BT.walk_forward(games, prior, half_life=prm["half_life"], ridge=prm["ridge"])
    logs, lines = data.load_player_logs(games)
    wf["adj"] = BT.availability_raw(wf, logs)
    wf = wf.merge(lines, on="game_id", how="left").merge(
        games[["game_id", "home", "away", "home_reg", "away_reg", "home_pts", "away_pts"]], on="game_id")
    wf = wf.dropna(subset=["close_margin", "close_total"])
    out = []
    for g in wf.itertuples():
        own_m = g.pred + prm["injury_coef"] * g.adj
        own = B.Grid(own_m, g.pred_total, B.SIGMA["margin_model"], B.SIGMA["total_model"])
        mkt = B.Grid(g.close_margin, g.close_total, B.SIGMA["margin_market"], B.SIGMA["total_market"])
        act = core.Actual(g.home_reg, g.away_reg, g.home_pts, g.away_pts)
        cands = []
        for team, home in ((g.home, True), (g.away, False)):
            cands.append((B.bet_ml(team, home), "uzvarētājs"))
            for x in range(-25, 25):
                cands.append((B.bet_handicap(team, home, x + 0.5), "handikaps"))
            exp = (g.close_total + (g.close_margin if home else -g.close_margin)) / 2
            for x in range(-10, 10):
                line = round(exp) + x + 0.5
                cands.append((B.bet_team_total(team, home, True, line), "komandas punkti"))
                cands.append((B.bet_team_total(team, home, False, line), "komandas punkti"))
        for x in range(-15, 15):
            line = round(g.close_total) + x + 0.5
            cands.append((B.bet_total(True, line), "kopā"))
            cands.append((B.bet_total(False, line), "kopā"))
        for bet, cat in cands:
            pk, push = mkt.prob(bet)
            if push > 0 or not _keep(pk):
                continue
            w, ps = bet.fn(act)
            out.append({"league": "nba", "season": season, "date": str(g.date), "game_id": g.game_id,
                        "cat": cat, "bet": bet.label, "p_mkt": pk, "p_own": own.prob(bet)[0],
                        "won": bool(w[0]), "push": bool(ps[0])})
    return out


# --------------------------------------------------------------------------- NHL

def nhl_rows(season: int) -> list[dict]:
    import nhl_backtest as NB
    import nhl_bets as HB
    import nhl_bot as NBOT
    import nhl_data as D
    import nhl_model as N
    p = N.load_params()
    prev = D.load_season(season - 1)
    games = D.load_season(season, until=espn.today_et())
    games = games[games["completed"] & games["home_reg"].notna()]
    wf = NB.walk_forward(games, N.season_prior(prev, p), p)
    sk, gl, odds = D.load_boxscores(games)
    wf = NB.add_adjustments(wf, sk, gl, p).merge(odds, on="game_id", how="left")
    lh, la = NB.adjusted(wf, p["goalie_coef"], p["skater_coef"])
    wf = wf.merge(games[["game_id", "home", "away", "so"]], on="game_id")
    out = []
    for i, g in enumerate(wf.itertuples()):
        if pd.isna(g.close_ml_home) or pd.isna(g.close_ml_away) or pd.isna(g.close_total):
            continue
        q = N.devig2(g.close_over, g.close_under) if not (pd.isna(g.close_over) or pd.isna(g.close_under)) else 0.5
        ml = N.market_lambdas(N.devig2(g.close_ml_home, g.close_ml_away), g.close_total, q,
                              p["tie_infl"], p["ot_k"], lh[i] + la[i])
        own = N.HockeyGrid(lh[i], la[i], p["tie_infl"], p["ot_k"])
        mkt = N.HockeyGrid(ml[0], ml[1], p["tie_infl"], p["ot_k"])
        r = SimpleNamespace(home=g.home, away=g.away, total_line=g.close_total, model_lh=lh[i], model_la=la[i],
                            lh=ml[0], la=ml[1])
        act = HB.Actual(SimpleNamespace(home_reg=g.rh, away_reg=g.ra, home_goals=g.fh, away_goals=g.fa))
        for bet, pk, _, cat in NBOT._candidates(r, mkt, P_MIN, P_MAX, crazy=False):
            w, ps = bet.fn(act)
            out.append({"league": "nhl", "season": season, "date": str(g.date), "game_id": g.game_id,
                        "cat": cat, "bet": bet.label, "p_mkt": pk, "p_own": HB.prob(own, bet)[0],
                        "won": bool(w[0]), "push": bool(ps[0])})
    return out


# --------------------------------------------------------------------------- football

def football_rows(base: str, league: str, season: int) -> list[dict]:
    import cl_bot as F
    import nhl_bets as HB
    import nhl_model as N
    F.BASE = base
    p = F.load_params() if league == "cl" else {"tie_infl": 0.10}
    start, end = F.SEASONS[season]
    hist_start = F.SEASONS[season - 1][0] if season - 1 in F.SEASONS else start
    games = F.load_games(hist_start, min(end, espn.today_et()))
    target = games[(games["date"] >= start) & games["completed"] & games["home_reg"].notna()]
    odds = F.closing_odds(target["game_id"].tolist())
    target = target.drop(columns=[c for c in odds.columns if c != "game_id"]).merge(odds, on="game_id")
    out = []
    for day, todays in target.groupby("date"):
        model = F.fit_football(games, todays["start_utc"].min())
        if model is None:
            continue
        for g in todays.itertuples():
            p3 = F.devig3(g.ml_home, g.ml_draw, g.ml_away)
            if p3 is None or g.total_line is None or pd.isna(g.total_line):
                continue
            q = N.devig2(g.over_odds, g.under_odds) if g.over_odds and g.under_odds else 0.5
            mlh, mla = F.solve_lambdas(p3, g.total_line, q, p["tie_infl"])
            olh, ola = model(g.home_id, g.away_id)
            own = N.HockeyGrid(olh, ola, p["tie_infl"], 0.0)
            mkt = N.HockeyGrid(mlh, mla, p["tie_infl"], 0.0)
            act = HB.Actual(SimpleNamespace(home_reg=g.home_reg, away_reg=g.away_reg,
                                            home_goals=g.home_goals, away_goals=g.away_goals))
            for bet, pk, cat in F.candidates(g, mkt, P_MIN, P_MAX, crazy=False):
                w, ps = bet.fn(act)
                out.append({"league": league, "season": season, "date": str(day), "game_id": g.game_id,
                            "cat": cat, "bet": bet.label, "p_mkt": pk, "p_own": HB.prob(own, bet)[0],
                            "won": bool(w[0]), "push": bool(ps[0])})
    return out


LEAGUES = {
    "nba": (lambda s: nba_rows(s), (2026, 2027)),
    "nhl": (lambda s: nhl_rows(s), (2026, 2027)),
    "cl": (lambda s: football_rows("https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions", "cl", s), (2025, 2026)),
    "ch": (lambda s: football_rows("https://site.api.espn.com/apis/site/v2/sports/soccer/eng.2", "ch", s), (2025, 2026)),
}


def cmd_dati(args):
    OUT.mkdir(exist_ok=True)
    for lg in args.liga or LEAGUES:
        fn, seasons = LEAGUES[lg]
        rows = []
        for s in seasons:
            r = fn(s)
            print(f"{lg} {s}: {len(r)} likmes no {len({x['game_id'] for x in r})} spēlēm", flush=True)
            rows += r
        pd.DataFrame(rows).to_csv(OUT / f"{lg}.csv.gz", index=False)


# --------------------------------------------------------------------------- choosing the rules

def units(df: pd.DataFrame) -> float:
    odds = PAYOUT / df["p_mkt"]
    return float(np.where(df["push"], 0.0, np.where(df["won"], odds - 1, -1.0)).sum())


def apply_rule(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """One bet per game: allowed bet types, the robot's own model must agree (p_own - p_mkt >= gap),
    the safest such bet (bookmakers), then at most `per_day` bets a day (biggest agreement first)."""
    d = df[df["cat"].isin(rule["cats"]) & (df["p_mkt"] >= rule["p_min"])]
    if rule["gap"] is not None:
        d = d[d["p_own"] - d["p_mkt"] >= rule["gap"]]
    if d.empty:
        return d
    d = d.sort_values(["game_id", "p_mkt"], ascending=[True, False]).drop_duplicates("game_id")
    if rule["per_day"]:
        d = d.assign(_e=d["p_own"] - d["p_mkt"]).sort_values(["date", "_e"], ascending=[True, False])
        d = d.groupby("date").head(rule["per_day"]).drop(columns="_e")
    return d


def describe(d: pd.DataFrame) -> str:
    n = int((~d["push"]).sum())
    if not n:
        return "0 likmes"
    hits = int((d["won"] & ~d["push"]).sum())
    u = units(d)
    return f"{hits}/{n} ({hits / n:.0%}, gaidīts {d.loc[~d['push'], 'p_mkt'].mean():.0%}), vienības {u:+.1f} ({u / n:+.1%})"


def cmd_analize(args):
    rules, report = {}, []
    for lg in LEAGUES:
        f = OUT / f"{lg}.csv.gz"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        train_s, test_s = LEAGUES[lg][1]
        tr, te = df[df["season"] == train_s], df[df["season"] == test_s]
        print(f"\n===== {lg.upper()}: pārbaude {train_s} ({tr['game_id'].nunique()} sp.), "
              f"kontrole {test_s} ({te['game_id'].nunique()} sp.)")
        print("Pa kategorijām (visas likmes 1.5-2.2, pagājušā sezona):")
        for cat, g in tr.groupby("cat"):
            print(f"   {cat:<24} {describe(g)}")
        cats = sorted(df["cat"].unique())
        cat_sets = [cats] + [[c] for c in cats] + [list(x) for x in itertools.combinations(cats, 2)]
        min_n = {"nba": 150, "nhl": 150, "cl": 40, "ch": 120}[lg]
        best = None
        for cs, gap, pmin, per_day in itertools.product(cat_sets, [None, 0.0, 0.03, 0.06, 0.10],
                                                        [P_MIN, 0.50, 0.55], [0, 5, 3]):
            rule = {"cats": cs, "gap": gap, "p_min": pmin, "per_day": per_day}
            d = apply_rule(tr, rule)
            n = int((~d["push"]).sum()) if len(d) else 0
            if n < min_n:
                continue
            roi = units(d) / n
            if best is None or roi > best[0]:
                best = (roi, rule, d)
        if best is None:
            continue
        roi, rule, d = best
        dt = apply_rule(te, rule)
        base = apply_rule(te, {"cats": cats, "gap": None, "p_min": P_MIN, "per_day": 0})
        gap_txt = "nav vajadzīga" if rule["gap"] is None else f"robots ≥ bukmeikeri {rule['gap']:+.0%}"
        day_txt = "visas" if not rule["per_day"] else f"max {rule['per_day']} dienā"
        print(f"Labākais noteikums pagājušajā sezonā: kategorijas {rule['cats']}, robota piekrišana: {gap_txt}, "
              f"bukm. varbūtība ≥ {rule['p_min']:.0%}, {day_txt}")
        print(f"   pagājušā sezona:      {describe(d)}")
        print(f"   šī sezona (kontrole): {describe(dt)}")
        print(f"   salīdzinājumam, bez filtriem šajā sezonā: {describe(base)}")
        test_n = int((~dt['push']).sum()) if len(dt) else 0
        keep = bool(test_n and units(dt) / test_n > units(base) / max(int((~base['push']).sum()), 1))
        if not keep:
            # the filter only fitted last season: use the plain rule (safest bet of every game at 1.5+)
            rule = {"cats": cats, "gap": None, "p_min": P_MIN, "per_day": 0}
            d, dt = apply_rule(tr, rule), base
            print("   -> filtrs šajā sezonā nepalīdzēja, tiek izmantots vienkāršais noteikums (drošākā likme katrā spēlē)")
        rules[lg] = {**rule, "train": describe(d), "test": describe(dt), "baseline_test": describe(base),
                     "enabled": True, "filters_help": keep}
    RULES.write_text(json.dumps(rules, indent=2, ensure_ascii=False))
    print(f"\nSaglabāts: {RULES.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("komanda", choices=["dati", "analize"])
    ap.add_argument("--liga", nargs="*", choices=list(LEAGUES))
    args = ap.parse_args()
    {"dati": cmd_dati, "analize": cmd_analize}[args.komanda](args)


if __name__ == "__main__":
    sys.exit(main())
