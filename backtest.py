"""Walk-forward backtest + parameter tuning.

For every day, the model is fitted ONLY on games played before that day and then
predicts that day's games — exactly how it is used for real.

  python backtest.py            # tune on last season, evaluate this season, save params.json
  python backtest.py --quick    # small grid, skips injury calibration (~1 min)
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
import espn
import model

PARAMS_FILE = Path(__file__).resolve().parent / "params.json"


def walk_forward(games: pd.DataFrame, prior: model.Ratings | None, **kw) -> pd.DataFrame:
    games = model.add_rest(games)
    done = games[games["completed"]]
    out = []
    for day, todays in done.groupby("date"):
        as_of = todays["start_utc"].min()
        r = model.fit_ratings(games, as_of=as_of, prior=prior, **kw)
        for g in todays.itertuples():
            out.append({"game_id": g.game_id, "date": day, "home_id": g.home_id, "away_id": g.away_id,
                        "start_utc": g.start_utc,
                        "pred": r.margin(g.home_id, g.away_id, g.neutral, g.home_b2b, g.away_b2b),
                        "pred_total": r.total(g.home_id, g.away_id),
                        "actual": g.home_pts - g.away_pts, "actual_total": g.home_pts + g.away_pts})
    return pd.DataFrame(out)


def availability_raw(wf: pd.DataFrame, logs: pd.DataFrame) -> np.ndarray:
    """Home-minus-away availability adjustment (coef=1) using who actually played.

    Player values only use games before each game, so nothing leaks from the future.
    """
    out, cache = [], {}
    by_game = {k: v for k, v in logs.groupby(["game_id", "team_id"])}
    for g in wf.itertuples():
        if g.date not in cache:
            cache[g.date] = model.player_values(logs, as_of=g.start_utc, coef=1.0)
        vals, adj = cache[g.date], {}
        for side, tid in (("h", g.home_id), ("a", g.away_id)):
            box = by_game.get((g.game_id, tid))
            if box is None:
                adj[side] = 0.0
                continue
            played = {p: float(m > 0) for p, m in zip(box["player_id"], box["min"])}
            adj[side] = model.availability_adjustment(tid, logs, vals, played, {}, set(played), g.start_utc).adj
        out.append(adj["h"] - adj["a"])
    return np.array(out)


def logloss(pred, actual, sigma):
    p = np.clip([model.win_prob(m, sigma) for m in pred], 1e-6, 1 - 1e-6)
    y = (np.asarray(actual) > 0).astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_sigma(pred, actual) -> float:
    return float(min(np.arange(8, 20, 0.25), key=lambda s: logloss(pred, actual, s)))


def metrics(pred, actual, sigma) -> dict:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    return {"n": len(pred), "accuracy": float(np.mean((pred > 0) == (actual > 0))),
            "mae_margin": float(np.mean(np.abs(pred - actual))),
            "logloss": logloss(pred, actual, sigma)}


def fmt(name, m):
    return (f"  {name:<36} n={m['n']:>4}  uzminēti uzvarētāji {100*m['accuracy']:5.1f}%  "
            f"vid. kļūda {m['mae_margin']:5.2f} p.  logloss {m['logloss']:.4f}")


def season_prior(season: int, shrink: float) -> model.Ratings | None:
    if season - 1 not in data.SEASONS:
        return None
    prev = model.add_rest(data.load_season(season - 1))
    return model.fit_ratings(prev, half_life=60, ridge=2.0).shrunk(shrink)


def compare(wf: pd.DataFrame, sigma: float, title: str) -> float | None:
    """Print model vs. closing betting line; return the best blend weight."""
    print(f"\n{title}")
    print(fmt("Tikai komandu reitingi", metrics(wf["pred"], wf["actual"], sigma)))
    print(fmt("+ sastāvi/savainojumi", metrics(wf["pred_inj"], wf["actual"], sigma)))
    mk = wf.dropna(subset=["close_margin"])
    if mk.empty:
        return None
    print(f"  -- {len(mk)} spēles ar bukmeikeru noslēguma līniju --")
    print(fmt("Bukmeikeru līnija", metrics(mk["close_margin"], mk["actual"], sigma)))
    print(fmt("Robots", metrics(mk["pred_inj"], mk["actual"], sigma)))
    best = min(np.arange(0, 1.01, 0.1), key=lambda w: float(np.mean(
        ((1 - w) * mk["pred_inj"] + w * mk["close_margin"] - mk["actual"]) ** 2)))
    blend = (1 - best) * mk["pred_inj"] + best * mk["close_margin"]
    print(fmt(f"Robots + tirgus ({best:.0%} tirgus)", metrics(blend, mk["actual"], sigma)))
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-season", type=int, default=2026)
    ap.add_argument("--eval-season", type=int, default=data.season_for(espn.today_et()))
    ap.add_argument("--quick", action="store_true", help="small grid, no injury calibration")
    args = ap.parse_args()

    print(f"Ielādē sezonas {args.tune_season - 1}, {args.tune_season}, {args.eval_season} ...")
    tune_games = data.load_season(args.tune_season)
    tune_games = tune_games[tune_games["completed"]]
    prev_fit = model.fit_ratings(model.add_rest(data.load_season(args.tune_season - 1)),
                                 half_life=60, ridge=2.0)

    # ---------------------------------------------------------------- 1. team-rating params
    grid = {"half_life": [25, 40, 70, 1e9], "ridge": [3, 6, 12], "prior_shrink": [0.4, 0.6, 0.8]}
    if args.quick:
        grid = {"half_life": [40], "ridge": [6], "prior_shrink": [0.6]}
    print("\n1) Komandu reitingu parametru meklēšana (sezona %d):" % args.tune_season)
    results = []
    for hl, rg, sh in itertools.product(*grid.values()):
        wf = walk_forward(tune_games, prev_fit.shrunk(sh), half_life=hl, ridge=rg)
        mse = float(np.mean((wf["pred"] - wf["actual"]) ** 2))
        results.append((mse, hl, rg, sh, wf))
        print(f"   half_life={hl:>5g} ridge={rg:>3} prior_shrink={sh}: vid. kv. kļūda {mse:.2f}")
    _, hl, rg, sh, wf_tune = min(results, key=lambda x: x[0])
    params = {"half_life": hl, "ridge": rg, "prior_shrink": sh}
    print(f"   -> labākie: {params}")

    # ---------------------------------------------------------------- 2. injury coefficient
    logs_t, lines_t = data.load_player_logs(tune_games)
    wf_tune = wf_tune.merge(lines_t, on="game_id", how="left")
    if args.quick:
        coef = model.DEFAULTS["injury_coef"]
        wf_tune["raw_adj"] = 0.0
    else:
        print("\n2) Savainojumu/sastāvu ietekmes kalibrēšana (dažas minūtes) ...")
        wf_tune["raw_adj"] = availability_raw(wf_tune, logs_t)
        resid = (wf_tune["actual"] - wf_tune["pred"]).to_numpy()
        x = wf_tune["raw_adj"].to_numpy()
        coef = float(np.clip(np.dot(x, resid) / max(np.dot(x, x), 1e-9), 0.05, 0.8))
        print(f"   koeficients = {coef:.3f}")
    params["injury_coef"] = round(coef, 3)
    wf_tune["pred_inj"] = wf_tune["pred"] + coef * wf_tune["raw_adj"]
    sigma = fit_sigma(wf_tune["pred_inj"].to_numpy(), wf_tune["actual"].to_numpy())
    params["sigma"] = sigma
    w_tune = compare(wf_tune, sigma, f"Sezona {args.tune_season} (sigma={sigma}):")

    # ---------------------------------------------------------------- 3. current season
    cur = data.load_season(args.eval_season, until=espn.today_et())
    cur = cur[cur["completed"]]
    if len(cur) >= 30:
        wf = walk_forward(cur, season_prior(args.eval_season, sh), half_life=hl, ridge=rg)
        logs, lines = data.load_player_logs(cur)
        wf = wf.merge(lines, on="game_id", how="left")
        wf["pred_inj"] = wf["pred"] + coef * availability_raw(wf, logs)
        compare(wf, sigma, f"Sezona {args.eval_season} līdz šim:")
    if w_tune is not None:
        params["market_weight_suggested"] = round(w_tune, 2)

    PARAMS_FILE.write_text(json.dumps(params, indent=2))
    print(f"\nParametri saglabāti: {PARAMS_FILE}")
    print("Piezīme: vēsturē izmantots faktiskais sastāvs (kas spēlēja), tāpēc reālā "
          "savainojumu informācija pirms spēles būs nedaudz mazāk precīza.")


if __name__ == "__main__":
    main()
