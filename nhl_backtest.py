"""NHL walk-forward backtest + calibration -> nhl_params.json

Every day is predicted only from games before it. Steps:
 1. team-rating parameters (half-life, ridge, prior shrink): Poisson likelihood of regulation goals
 2. tie inflation + overtime strength: regulation 3-way outcome / OT winner likelihood
 3. goalie and skater coefficients (using who actually started / played)
 4. blend weight with bookmaker closing lines; compare robot vs bookmakers

  python nhl_backtest.py           # ~10 min
  python nhl_backtest.py --quick
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np
import pandas as pd

import espn
import model as M
import nhl_data as D
import nhl_model as N


def poisson_ll(lam: np.ndarray, k: np.ndarray) -> np.ndarray:
    lam = np.clip(lam, 0.05, None)
    return k * np.log(lam) - lam - np.array([math.lgamma(x + 1) for x in k])


def walk_forward(games: pd.DataFrame, prior: M.Ratings, p: dict) -> pd.DataFrame:
    rf = N.rating_frame(games)
    done = rf[rf["completed"]]
    out = []
    for day, todays in done.groupby("date"):
        r = N.fit(rf, as_of=todays["start_utc"].min(), prior=prior, p=p)
        for g in todays.itertuples():
            lh, la = N.lambdas(r, g.home_id, g.away_id, g.neutral, g.home_b2b, g.away_b2b)
            out.append({"game_id": g.game_id, "date": day, "start_utc": g.start_utc,
                        "home_id": g.home_id, "away_id": g.away_id, "lh": lh, "la": la,
                        "rh": g.home_reg, "ra": g.away_reg, "fh": g.home_goals, "fa": g.away_goals,
                        "ot": bool(g.ot)})
    return pd.DataFrame(out)


def add_adjustments(wf: pd.DataFrame, skaters: pd.DataFrame, goalies: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Goalie deltas and skater availability (coef 1) from who actually played."""
    starts = N.starters(goalies)
    st_by = {(r.game_id, r.team_id): r.player_id for r in starts.itertuples()}
    sk_logs = N.skater_logs_for_availability(skaters)
    by_game = {k: v for k, v in sk_logs.groupby(["game_id", "team_id"])}
    cache = {}
    gh, ga, sh, sa = [], [], [], []
    for g in wf.itertuples():
        if g.date not in cache:
            gv, _ = N.goalie_values(goalies, as_of=g.start_utc, p=p)
            sv = N.skater_values(skaters, as_of=g.start_utc, p=p, coef=1.0)
            cache = {g.date: (gv, sv)}
        gv, sv = cache[g.date]
        vals = []
        for tid in (g.home_id, g.away_id):
            pg = {**p, "goalie_coef": 1.0}
            d, _ = N.goalie_delta(tid, st_by.get((g.game_id, tid)), starts, gv, g.start_utc, pg)
            box = by_game.get((g.game_id, tid))
            if box is None:
                s_adj = 0.0
            else:
                played = {pid: float(t > 0) for pid, t in zip(box["player_id"], box["min"])}
                s_adj = M.availability_adjustment(tid, sk_logs, sv, played, {}, set(played), g.start_utc,
                                                  p["recent_games"]).adj
            vals.append((d, s_adj))
        gh.append(vals[0][0]); ga.append(vals[1][0]); sh.append(vals[0][1]); sa.append(vals[1][1])
    return wf.assign(gh=gh, ga=ga, sh=sh, sa=sa)


def adjusted(wf: pd.DataFrame, cg: float, cs: float) -> tuple[np.ndarray, np.ndarray]:
    lh = wf["lh"] - cg * wf["ga"] + cs * (wf["sh"] - wf["sa"]) / 2
    la = wf["la"] - cg * wf["gh"] - cs * (wf["sh"] - wf["sa"]) / 2
    return np.clip(lh.to_numpy(), 0.3, None), np.clip(la.to_numpy(), 0.3, None)


def reg_ll(lh, la, wf) -> float:
    return float(np.mean(poisson_ll(lh, wf["rh"].to_numpy()) + poisson_ll(la, wf["ra"].to_numpy())))


def three_way_ll(lh, la, wf, tie) -> float:
    ll = []
    for a, b, rh, ra in zip(lh, la, wf["rh"], wf["ra"]):
        reg = N.reg_matrix(a, b, tie)
        p = reg[N._HW].sum() if rh > ra else (reg[N._DIAG].sum() if rh == ra else reg[N._H < N._A].sum())
        ll.append(math.log(max(p, 1e-9)))
    return float(np.mean(ll))


def ml_probs(lh, la, p) -> np.ndarray:
    return np.array([N.fast_p_home(a, b, p["tie_infl"], p["ot_k"]) for a, b in zip(lh, la)])


def logloss(prob, y) -> float:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    p = dict(N.DEFAULTS)

    print("Ielādē NHL sezonas 2024-25, 2025-26 un šo sezonu ...")
    prev, tune = D.load_season(2025), D.load_season(2026)
    tune = tune[tune["completed"] & tune["home_reg"].notna()]

    # 1. team ratings
    grid = {"half_life": [30, 60, 120, 1e9], "ridge": [4, 8, 16], "prior_shrink": [0.3, 0.5, 0.7]}
    if args.quick:
        grid = {"half_life": [60], "ridge": [8], "prior_shrink": [0.5]}
    print("\n1) Komandu reitingi (2025-26, pamatlaika vārti, vid. log-ticamība — lielāka ir labāka):")
    best = None
    for hl, rg, sh in itertools.product(*grid.values()):
        q = {**p, "half_life": hl, "ridge": rg, "prior_shrink": sh}
        wf = walk_forward(tune, N.season_prior(prev, q), q)
        ll = reg_ll(wf["lh"].to_numpy(), wf["la"].to_numpy(), wf)
        print(f"   half_life={hl:>5g} ridge={rg:>3} prior_shrink={sh}: {ll:.4f}")
        if best is None or ll > best[0]:
            best = (ll, q, wf)
    _, p, wf = best
    print(f"   -> {dict((k, p[k]) for k in grid)}")

    # 2. ties and overtime
    lh, la = wf["lh"].to_numpy(), wf["la"].to_numpy()
    tie = max(np.arange(0, 1.01, 0.05), key=lambda t: three_way_ll(lh, la, wf, t))
    p["tie_infl"] = round(float(tie), 3)
    otg = wf[wf["ot"]]
    y_ot = (otg["fh"] > otg["fa"]).astype(float).to_numpy()

    def ot_ll(k):
        pr = np.array([N.p_ot_home(a, b, k) for a, b in zip(otg["lh"], otg["la"])])
        return -logloss(pr, y_ot)
    p["ot_k"] = round(float(max(np.arange(0, 1.01, 0.1), key=ot_ll)), 2)
    pred_ot = np.mean([N.reg_matrix(a, b, p["tie_infl"])[N._DIAG].sum() for a, b in zip(lh, la)])
    print(f"\n2) Neizšķirts pamatlaikā: faktiski {wf['ot'].mean():.1%}, modelis {pred_ot:.1%} "
          f"(tie_infl={p['tie_infl']}); papildlaika spēks ot_k={p['ot_k']}")

    # 3. goalies + skaters
    print("\n3) Vārtsargi un spēlētāju sastāvs (dažas minūtes) ...")
    skaters, goalies, odds = D.load_boxscores(tune)
    wf = add_adjustments(wf, skaters, goalies, p)
    base_ll = reg_ll(*adjusted(wf, 0, 0), wf)
    cg, cs = max(itertools.product(np.arange(0, 1.61, 0.2), np.arange(0, 1.61, 0.2)),
                 key=lambda c: reg_ll(*adjusted(wf, *c), wf))
    p["goalie_coef"], p["skater_coef"] = round(float(cg), 2), round(float(cs), 2)
    print(f"   vārtsargu koef. {cg:.1f}, spēlētāju koef. {cs:.1f}: log-ticamība {base_ll:.4f} -> "
          f"{reg_ll(*adjusted(wf, cg, cs), wf):.4f}")

    # 4. bookmakers
    wf = wf.merge(odds, on="game_id", how="left")
    lh, la = adjusted(wf, cg, cs)
    y = (wf["fh"] > wf["fa"]).astype(float).to_numpy()
    pr_model = ml_probs(lh, la, p)
    mk = wf["close_ml_home"].notna() & wf["close_ml_away"].notna()
    print(f"\n4) Uzvarētājs (ar papildlaiku), sezona 2025-26, {mk.sum()} spēles ar bukmeikeru koeficientiem:")
    p_mk = np.array([N.devig2(a, b) for a, b in zip(wf.loc[mk, "close_ml_home"], wf.loc[mk, "close_ml_away"])])
    print(f"   robots:     logloss {logloss(pr_model[mk], y[mk]):.4f}, uzminēti {np.mean((pr_model[mk] > .5) == y[mk]):.1%}")
    print(f"   bukmeikeri: logloss {logloss(p_mk, y[mk]):.4f}, uzminēti {np.mean((p_mk > .5) == y[mk]):.1%}")
    m_l = []
    for (i, g), pm in zip(wf[mk].iterrows(), p_mk):
        q_over = N.devig2(g["close_over"], g["close_under"])
        m_l.append(N.market_lambdas(pm, g["close_total"], q_over, p["tie_infl"], p["ot_k"], lh[i] + la[i]))
    idx = np.where(mk)[0]
    best_w, best_ll = 0.0, 9.0
    for w in np.arange(0, 1.01, 0.1):
        prs = [N.fast_p_home(*N.blend((lh[i], la[i]), ml, w), p["tie_infl"], p["ot_k"]) for i, ml in zip(idx, m_l)]
        ll = logloss(np.array(prs), y[mk])
        if ll < best_ll:
            best_w, best_ll = float(w), ll
    p["market_weight"] = round(best_w, 1)
    print(f"   robots + bukmeikeri ({best_w:.0%} bukm.): logloss {best_ll:.4f}")
    # totals calibration at the closing line
    tot_rows = wf[mk & wf["close_total"].notna()]
    if len(tot_rows):
        ov = []
        for (i, g), ml in zip(wf[mk].iterrows(), m_l):
            if pd.isna(g["close_total"]):
                continue
            bl = N.blend((lh[i], la[i]), ml, best_w)
            ov.append((N.fast_p_over(*bl, p["tie_infl"], g["close_total"]), float(g["fh"] + g["fa"] > g["close_total"])))
        ov = np.array(ov)
        print(f"   virs/zem pie bukm. līnijas: prognozēts virs {ov[:, 0].mean():.1%}, faktiski {ov[:, 1].mean():.1%}")

    # 5. this season so far (fixed parameters)
    cur_season = D.season_for(espn.today_et())
    cur = D.load_season(cur_season, until=espn.today_et())
    cur = cur[cur["completed"] & cur["home_reg"].notna()]
    if len(cur) >= 50:
        wf2 = walk_forward(cur, N.season_prior(tune, p), p)
        sk2, gl2, od2 = D.load_boxscores(cur)
        wf2 = add_adjustments(wf2, sk2, gl2, p).merge(od2, on="game_id", how="left")
        lh2, la2 = adjusted(wf2, cg, cs)
        y2 = (wf2["fh"] > wf2["fa"]).astype(float).to_numpy()
        mk2 = (wf2["close_ml_home"].notna() & wf2["close_ml_away"].notna()).to_numpy()
        pm2 = np.array([N.devig2(a, b) for a, b in zip(wf2.loc[mk2, "close_ml_home"], wf2.loc[mk2, "close_ml_away"])])
        pr2 = ml_probs(lh2, la2, p)
        bl2 = []
        for i, (_, g) in zip(np.where(mk2)[0], wf2[mk2].iterrows()):
            ml = N.market_lambdas(N.devig2(g["close_ml_home"], g["close_ml_away"]), g["close_total"],
                                  N.devig2(g["close_over"], g["close_under"]), p["tie_infl"], p["ot_k"], lh2[i] + la2[i])
            bl2.append(N.fast_p_home(*N.blend((lh2[i], la2[i]), ml, p["market_weight"]), p["tie_infl"], p["ot_k"]))
        bl2 = np.array(bl2)
        print(f"\n5) Šī sezona ({mk2.sum()} spēles): robots logloss {logloss(pr2[mk2], y2[mk2]):.4f} "
              f"({np.mean((pr2[mk2] > .5) == y2[mk2]):.1%}), bukmeikeri {logloss(pm2, y2[mk2]):.4f} "
              f"({np.mean((pm2 > .5) == y2[mk2]):.1%}), sajaukums {logloss(bl2, y2[mk2]):.4f} "
              f"({np.mean((bl2 > .5) == y2[mk2]):.1%})")

    N.PARAMS_FILE.write_text(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                         for k, v in p.items()}, indent=2))
    print(f"\nSaglabāts: {N.PARAMS_FILE.name}")


if __name__ == "__main__":
    main()
