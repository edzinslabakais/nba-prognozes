"""NBA spēļu prognozes.

  python predict.py                     # šodienas (ASV laiks) spēles
  python predict.py --date 2026-11-20   # konkrēta diena
  python predict.py --days 3            # šodiena + nākamās 2 dienas
  python predict.py --market-weight 0.5 # sajaukt ar bukmeikeru līniju
  python predict.py --refresh           # ignorēt kešu (svaigi savainojumi)
  python predict.py --detail            # parādīt katras spēles faktorus
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import data
import espn
import model

ROOT = Path(__file__).resolve().parent
RIGA = ZoneInfo("Europe/Riga")


def load_params() -> dict:
    p = dict(model.DEFAULTS)
    f = ROOT / "params.json"
    if f.exists():
        p.update(json.loads(f.read_text()))
    return p


def build_context(target_dates: list[date], params: dict):
    season = data.season_for(target_dates[0])
    games = data.load_season(season, until=max(target_dates))
    games = model.add_rest(games)

    prior = None
    if season - 1 in data.SEASONS:
        prev = model.add_rest(data.load_season(season - 1))
        prior = model.fit_ratings(prev, half_life=60, ridge=2.0).shrunk(params["prior_shrink"])

    logs, _ = data.load_player_logs(games[games["completed"]])
    return games, prior, logs


def predict(target_dates: list[date], market_weight: float, params: dict) -> pd.DataFrame:
    games, prior, logs = build_context(target_dates, params)
    todo = games[games["date"].isin(target_dates) & ~games["completed"]]
    if todo.empty:
        return pd.DataFrame()

    as_of = todo["start_utc"].min()
    ratings = model.fit_ratings(games, as_of=as_of, prior=prior,
                                half_life=params["half_life"], ridge=params["ridge"])
    values = model.player_values(logs, coef=params["injury_coef"], replacement=params["replacement"])
    inj = data.load_injuries()
    team_ids = sorted(set(todo["home_id"]) | set(todo["away_id"]))
    rosters = data.load_rosters(team_ids)

    rows = []
    for g in todo.itertuples():
        side = {}
        for tid in (g.home_id, g.away_id):
            ti = inj[inj["team_id"] == tid] if not inj.empty else inj
            probs = {r.player_id: model.status_to_prob(r.status, r.return_date, g.date) for r in ti.itertuples()}
            stats = {r.player_id: r.status for r in ti.itertuples()}
            side[tid] = model.availability_adjustment(tid, logs, values, probs, stats,
                                                      rosters.get(tid), None, params["recent_games"])
        base = ratings.margin(g.home_id, g.away_id, g.neutral, g.home_b2b, g.away_b2b)
        inj_diff = side[g.home_id].adj - side[g.away_id].adj
        model_margin = base + inj_diff
        margin = model_margin
        if g.market_margin is not None and pd.notna(g.market_margin) and market_weight > 0:
            margin = (1 - market_weight) * model_margin + market_weight * g.market_margin
        model_total = ratings.total(g.home_id, g.away_id)
        has_mt = g.market_total is not None and pd.notna(g.market_total)
        # backtest: the betting total is clearly more accurate than ours -> use it when available
        total = float(g.market_total) if has_mt else model_total
        p_home = model.win_prob(margin, params["sigma"])
        winner_home = margin >= 0
        rows.append({
            "date": g.date,
            "game_id": g.game_id,
            "start_utc": g.start_utc,
            "riga_time": g.start_utc.tz_convert(RIGA).strftime("%d.%m %H:%M"),
            "away": g.away, "home": g.home,
            "away_name": g.away_name, "home_name": g.home_name,
            "away_record": g.away_record, "home_record": g.home_record,
            "pick": g.home if winner_home else g.away,
            "win_prob": p_home if winner_home else 1 - p_home,
            "home_pts": round((total + margin) / 2),
            "away_pts": round((total - margin) / 2),
            "margin_home": margin,
            "total": total, "model_total": model_total,
            "market_total": float(g.market_total) if has_mt else None,
            "has_market": g.market_margin is not None and pd.notna(g.market_margin),
            "model_margin_home": model_margin,
            "market_margin_home": g.market_margin,
            "home_net": ratings.net(g.home_id), "away_net": ratings.net(g.away_id),
            "hca": 0 if g.neutral else ratings.hca,
            "rest": ratings.b2b * (g.away_b2b - g.home_b2b),
            "home_b2b": g.home_b2b, "away_b2b": g.away_b2b,
            "inj_home": side[g.home_id], "inj_away": side[g.away_id],
        })
    return pd.DataFrame(rows)


def _fmt_line(m: float | None, home: str, away: str) -> str:
    if m is None or pd.isna(m):
        return "—"
    return f"{home} -{m:.1f}" if m >= 0 else f"{away} -{-m:.1f}"


def _inj_text(a: model.Availability, limit=3) -> str:
    parts = [f"{n}{' (' + s + ')' if s else ''} {d:+.1f}" for n, s, d in a.notes[:limit]]
    return "; ".join(parts)


def render(df: pd.DataFrame, detail: bool) -> str:
    lines = []
    for day, dg in df.groupby("date"):
        lines.append(f"\n=== {day:%Y-%m-%d} (ASV datums) — {len(dg)} spēles ===\n")
        hdr = f"{'Laiks (Rīga)':<12} {'Spēle':<13} {'Prognoze':<9} {'Varb.':>6}  {'Rezultāts':<15} {'Robota līnija':<14} {'Bukmeikeri':<12}"
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for r in dg.itertuples():
            score = f"{r.away} {r.away_pts}–{r.home_pts} {r.home}"
            lines.append(f"{r.riga_time:<12} {r.away + ' @ ' + r.home:<13} {r.pick:<9} {100*r.win_prob:5.0f}%  "
                         f"{score:<15} {_fmt_line(r.model_margin_home, r.home, r.away):<14} "
                         f"{_fmt_line(r.market_margin_home, r.home, r.away):<12}")
            if detail:
                lines.append(f"    reitings: {r.home} {r.home_net:+.1f}, {r.away} {r.away_net:+.1f} | "
                             f"mājas laukums {r.hca:+.1f} | atpūta {r.rest:+.1f}"
                             + (f" ({r.home} spēlē 2. dienu pēc kārtas)" if r.home_b2b else "")
                             + (f" ({r.away} spēlē 2. dienu pēc kārtas)" if r.away_b2b else ""))
                for team, a in ((r.home, r.inj_home), (r.away, r.inj_away)):
                    if a.notes:
                        lines.append(f"    sastāvs {team} {a.adj:+.1f}: {_inj_text(a, 5)}")
    return "\n".join(lines)


def save(df: pd.DataFrame) -> Path:
    out = ROOT / "predictions"
    out.mkdir(exist_ok=True)
    flat = df.copy()
    flat["inj_home"] = flat["inj_home"].map(lambda a: _inj_text(a, 5))
    flat["inj_away"] = flat["inj_away"].map(lambda a: _inj_text(a, 5))
    f = out / f"{df['date'].min():%Y-%m-%d}.csv"
    flat.round(3).to_csv(f, index=False)
    return f


def main():
    ap = argparse.ArgumentParser(description="NBA spēļu prognožu robots")
    ap.add_argument("--date", type=date.fromisoformat, default=None, help="YYYY-MM-DD (ASV datums)")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--market-weight", type=float, default=0.0,
                    help="0 = tikai robots, 1 = tikai bukmeikeru līnija")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    espn.FORCE_REFRESH = args.refresh
    start = args.date or espn.today_et()
    dates = [start + timedelta(days=i) for i in range(args.days)]
    params = load_params()

    df = predict(dates, args.market_weight, params)
    if df.empty:
        print("Šajās dienās nav neizspēlētu spēļu.")
        return
    print(render(df, args.detail))
    print(f"\nSaglabāts: {save(df)}")
    print("Varb. = uzvaras varbūtība izvēlētajai komandai. Līnija 'BOS -4.5' = BOS uzvar ar ~4.5 p.")


if __name__ == "__main__":
    main()
