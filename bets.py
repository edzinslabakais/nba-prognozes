"""Likmju analīze (Optibet tipa spēlei: 3 likmes uz spēli, koeficients >= 2.1).

1) Cenu lapa: robots katrai spēlei aprēķina likmju varbūtības un "godīgo"
   koeficientu (= 1 / varbūtība). Ja Optibet piedāvā VAIRĀK par godīgo, likme
   ilgtermiņā ir izdevīga.
      python bets.py                  # šodienas spēles
      python bets.py --date 2026-11-20

   Tiek izveidots arī fails odds/YYYY-MM-DD.csv ar tukšu kolonnu "koef".

2) Ieraksti tajā failā Optibet koeficientus (vai pievieno savas likmes) un palaid:
      python bets.py --odds odds/2026-11-18.csv
   Robots katrai spēlei sakārto likmes pēc sagaidāmā ieguvuma un iesaka 3 labākās
   ar koeficientu >= --min-odds.

Likmju pieraksts (komandu saīsinājumi kā ESPN: BOS, GS, NY, SA, NO, UTAH, WSH ...):
   BOS uzvar            uzvarētājs (ieskaitot papildlaiku)
   BOS -5.5 / DEN +5.5  handikaps
   virs 228.5 / zem 228.5        kopējais punktu skaits
   BOS virs 115.5 / BOS zem 115.5  komandas punkti
   BOS 1-5 / BOS 6-10 / BOS 21+    uzvaras starpība
   BOS uzvar & virs 228.5          kombinācija (visām daļām jāiziet)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import espn
import predict as P

ROOT = Path(__file__).resolve().parent

# Spread of actual results around the prediction, fitted on 2025-26 across all
# handicap/total lines (see README). Without a betting line the model is less sure.
SIGMA = {"margin_market": 13.75, "total_market": 18.25, "margin_model": 14.5, "total_model": 19.0}
PAYOUT = 0.93  # typical bookmaker payout on such markets (estimate, used only for hints)

ALIASES = {"GSW": "GS", "NYK": "NY", "SAS": "SA", "NOP": "NO", "UTA": "UTAH", "WAS": "WSH",
           "PHO": "PHX", "BRK": "BKN", "BRO": "BKN", "CHO": "CHA", "NOLA": "NO"}


def _phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))


# --------------------------------------------------------------------------- outcome grid

class Grid:
    """Joint distribution of final (home margin, total points) on integer scores."""

    def __init__(self, margin: float, total: float, s_m: float, s_t: float):
        M = np.arange(-70, 71)
        M = M[M != 0]                      # no ties in the NBA (overtime)
        T = np.arange(130, 331)
        pm = _phi((M + 0.5 - margin) / s_m) - _phi((M - 0.5 - margin) / s_m)
        pt = _phi((T + 0.5 - total) / s_t) - _phi((T - 0.5 - total) / s_t)
        MM, TT = np.meshgrid(M, T, indexing="ij")
        W = np.outer(pm, pt) * ((MM + TT) % 2 == 0)   # team scores must be integers
        self.W = W / W.sum()
        self.M, self.T = MM, TT
        self.H, self.A = (TT + MM) / 2, (TT - MM) / 2

    def prob(self, bet: "Bet") -> tuple[float, float]:
        win, push = bet.fn(self)
        return float((self.W * win).sum()), float((self.W * push).sum())


@dataclass
class Bet:
    label: str
    fn: Callable[[Grid], tuple[np.ndarray, np.ndarray]]


def _cmp(x, line, over: bool):
    return ((x > line) if over else (x < line)), (x == line)


def bet_ml(team: str, home: bool) -> Bet:
    return Bet(f"{team} uzvar", lambda g: ((g.M > 0) if home else (g.M < 0), np.zeros_like(g.M, bool)))


def bet_handicap(team: str, home: bool, line: float) -> Bet:
    def fn(g):
        x = (g.M if home else -g.M) + line
        return x > 0, x == 0
    return Bet(f"{team} {line:+g}", fn)


def bet_total(over: bool, line: float) -> Bet:
    return Bet(f"{'virs' if over else 'zem'} {line:g}", lambda g: _cmp(g.T, line, over))


def bet_team_total(team: str, home: bool, over: bool, line: float) -> Bet:
    return Bet(f"{team} {'virs' if over else 'zem'} {line:g}",
               lambda g: _cmp(g.H if home else g.A, line, over))


def bet_band(team: str, home: bool, lo: int, hi: float) -> Bet:
    def fn(g):
        x = g.M if home else -g.M
        return (x >= lo) & (x <= hi), np.zeros_like(x, bool)
    return Bet(f"{team} {lo}-{int(hi)}" if hi < 999 else f"{team} {lo}+", fn)


def bet_combo(parts: list[Bet]) -> Bet:
    def fn(g):
        win = np.ones_like(g.M, bool)
        for p in parts:
            w, _ = p.fn(g)
            win &= w
        return win, np.zeros_like(win)
    return Bet(" & ".join(p.label for p in parts), fn)


# --------------------------------------------------------------------------- parsing user bets

def _team(tok: str, game) -> tuple[str, bool] | None:
    t = ALIASES.get(tok.upper(), tok.upper())
    for abbr, name, home in ((game.home, game.home_name, True), (game.away, game.away_name, False)):
        if t == abbr.upper() or tok.lower() == name.split()[-1].lower():
            return abbr, home
    return None


NUM = r"(\d+(?:[.,]\d+)?)"


def parse_bet(text: str, game) -> Bet:
    parts = [p.strip() for p in text.split("&")]
    if len(parts) > 1:
        return bet_combo([parse_bet(p, game) for p in parts])
    s = re.sub(r"\s+", " ", text.strip()).replace(",", ".")
    toks = s.split(" ")
    low = s.lower()

    m = re.fullmatch(rf"(virs|over|o|zem|under|u) {NUM}", low)
    if m:
        return bet_total(m.group(1) in ("virs", "over", "o"), float(m.group(2)))

    team = _team(toks[0], game)
    if team is None:
        raise ValueError(f"nepazīstu komandu '{toks[0]}' spēlē {game.away}@{game.home}")
    abbr, home = team
    rest = " ".join(toks[1:]).lower()
    if re.fullmatch(r"(uzv\w*|win|ml|1|2)", rest):
        return bet_ml(abbr, home)
    m = re.fullmatch(rf"([+-]{NUM})", rest)
    if m:
        return bet_handicap(abbr, home, float(m.group(1)))
    m = re.fullmatch(rf"(virs|over|zem|under) {NUM}", rest)
    if m:
        return bet_team_total(abbr, home, m.group(1) in ("virs", "over"), float(m.group(2)))
    m = re.fullmatch(r"(?:ar |by )?(\d+) ?- ?(\d+)", rest)
    if m:
        return bet_band(abbr, home, int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(?:ar |by )?(\d+) ?\+", rest)
    if m:
        return bet_band(abbr, home, int(m.group(1)), 999)
    raise ValueError(f"nesaprotu likmi '{text}'")


# --------------------------------------------------------------------------- price sheet

def half(x: float) -> float:
    return math.floor(x) + 0.5


def candidate_bets(r) -> list[Bet]:
    """A compact set of bets around fair odds ~1.8-4 that bookmakers typically offer."""
    H, A = r.home, r.away
    fav_home = r.margin_home >= 0
    fav, dog = (H, A) if fav_home else (A, H)
    m_abs = abs(r.margin_home)
    bets = [bet_ml(H, True), bet_ml(A, False)]
    # handicaps: main line, favourite with bigger handicap, underdog with smaller cushion
    main = half(m_abs)
    for line in (main, main + 3, main + 6):
        bets.append(bet_handicap(fav, fav_home, -line))
    for line in (main - 3, main - 6):
        bets.append(bet_handicap(dog, not fav_home, line))
    # totals
    t = half(r.total)
    for d in (0, 5, 10):
        bets.append(bet_total(True, t + d))
        bets.append(bet_total(False, t - d))
    # team totals
    for team, home in ((H, True), (A, False)):
        exp = (r.total + (r.margin_home if home else -r.margin_home)) / 2
        bets.append(bet_team_total(team, home, True, half(exp) + 3))
        bets.append(bet_team_total(team, home, False, half(exp) - 3))
    # winning margin bands
    for team, home in ((fav, fav_home), (dog, not fav_home)):
        for lo, hi in ((1, 5), (6, 10), (11, 15), (16, 999)):
            bets.append(bet_band(team, home, lo, hi))
    # combos: winner & total
    for team, home in ((fav, fav_home), (dog, not fav_home)):
        for over in (True, False):
            bets.append(bet_combo([bet_ml(team, home), bet_total(over, t)]))
    return bets


def grid_for(r) -> Grid:
    s_m = SIGMA["margin_market"] if r.has_market else SIGMA["margin_model"]
    s_t = SIGMA["total_market"] if r.market_total is not None and pd.notna(r.market_total) else SIGMA["total_model"]
    return Grid(r.margin_home, r.total, s_m, s_t)


def header(r) -> str:
    line = f"{r.home} {-r.margin_home:+.1f}" if r.margin_home >= 0 else f"{r.away} {r.margin_home:+.1f}"
    mk = P._fmt_line(r.market_margin_home, r.home, r.away)
    mt = f"{r.market_total:g}" if r.market_total is not None and pd.notna(r.market_total) else "—"
    return (f"{r.away} @ {r.home}  {r.riga_time}  |  robots: {line}, kopā {r.total:.1f}  "
            f"|  bukmeikeri: {mk}, kopā {mt}")


def price_sheet(df: pd.DataFrame, min_odds: float) -> tuple[str, list[dict]]:
    out, template = [], []
    for r in df.itertuples():
        g = grid_for(r)
        out.append("\n" + header(r))
        if not r.has_market:
            out.append("  (nav bukmeikeru līnijas — tikai robota vērtējums, mazāk precīzs)")
        rows = []
        for b in candidate_bets(r):
            p, push = g.prob(b)
            if p <= 0.03:
                continue
            fair = (1 - push) / p
            rows.append((b.label, p, fair))
        rows.sort(key=lambda x: -x[1])
        out.append(f"  {'Likme':<28} {'Varb.':>6} {'Godīgais koef.':>15}")
        for label, p, fair in rows:
            hint = ""
            if fair * PAYOUT >= min_odds * 0.97 and p >= 0.30:
                hint = f"  <- Optibet varētu dot >= {min_odds:g}"
            out.append(f"  {label:<28} {100*p:5.0f}% {fair:15.2f}{hint}")
            template.append({"spēle": f"{r.away}@{r.home}", "likme": label, "koef": ""})
    return "\n".join(out), template


# --------------------------------------------------------------------------- evaluate real odds

def evaluate(df: pd.DataFrame, odds_file: Path, min_odds: float, top: int) -> str:
    games = {frozenset((r.home, r.away)): r for r in df.itertuples()}
    rows = []
    with open(odds_file, newline="", encoding="utf-8-sig") as fh:
        for i, rec in enumerate(csv.DictReader(fh), start=2):
            koef = (rec.get("koef") or "").strip().replace(",", ".")
            if not koef:
                continue
            teams = re.split(r"[@\-\s/]+", rec["spēle"].strip().upper())
            teams = frozenset(ALIASES.get(t, t) for t in teams if t)
            r = games.get(teams)
            if r is None:
                rows.append({"err": f"rinda {i}: spēle '{rec['spēle']}' nav atrasta šajā datumā"})
                continue
            try:
                b = parse_bet(rec["likme"], r)
            except ValueError as e:
                rows.append({"err": f"rinda {i}: {e}"})
                continue
            p, push = grid_for(r).prob(b)
            o = float(koef)
            rows.append({"game": f"{r.away}@{r.home}", "r": r, "label": b.label, "odds": o,
                         "p": p, "push": push, "ev": p * o + push, "fair": (1 - push) / max(p, 1e-9)})
    out = [x["err"] for x in rows if "err" in x]
    ok = pd.DataFrame([x for x in rows if "err" not in x])
    if ok.empty:
        return "\n".join(out + ["Failā nav nevienas likmes ar aizpildītu koeficientu."])
    for game, gdf in ok.groupby("game", sort=False):
        out.append("\n" + header(gdf.iloc[0]["r"]))
        out.append(f"  {'Likme':<28} {'Koef.':>6} {'Varb.':>6} {'Godīgais':>9} {'Sagaid. ieguvums':>17}")
        gdf = gdf.sort_values("ev", ascending=False)
        eligible = gdf[gdf["odds"] >= min_odds].head(top)
        for x in gdf.itertuples():
            mark = " ★" if x.Index in eligible.index else ("  (koef. < %g)" % min_odds if x.odds < min_odds else "")
            out.append(f"  {x.label:<28} {x.odds:6.2f} {100*x.p:5.0f}% {x.fair:9.2f} "
                       f"{100*(x.ev-1):+16.1f}%{mark}")
    out.append("\n★ = ieteiktās likmes (augstākais sagaidāmais ieguvums starp koef. >= %g)." % min_odds)
    out.append("Sagaid. ieguvums = varbūtība × koeficients − 1. Pozitīvs => Optibet maksā vairāk nekā godīgi.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Likmju analīze")
    ap.add_argument("--date", type=date.fromisoformat, default=None)
    ap.add_argument("--odds", type=Path, default=None, help="CSV ar Optibet koeficientiem")
    ap.add_argument("--min-odds", type=float, default=2.1)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--market-weight", type=float, default=None,
                    help="cik lielā mērā uzticēties bukmeikeru līnijai (noklusējums no backtest)")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    espn.FORCE_REFRESH = args.refresh
    params = P.load_params()
    mw = args.market_weight if args.market_weight is not None else params.get("market_weight_suggested", 0.9)
    day = args.date or espn.today_et()
    df = P.predict([day], mw, params)
    if df.empty:
        print("Šajā dienā nav neizspēlētu spēļu.")
        return

    if args.odds:
        print(evaluate(df, args.odds, args.min_odds, args.top))
        return

    text, template = price_sheet(df, args.min_odds)
    print(text)
    out_dir = ROOT / "odds"
    out_dir.mkdir(exist_ok=True)
    f = out_dir / f"{day:%Y-%m-%d}.csv"
    if not f.exists():
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["spēle", "likme", "koef"])
            w.writeheader()
            w.writerows(template)
        print(f"\nIzveidots {f} — ieraksti Optibet koeficientus kolonnā 'koef' un palaid:")
    else:
        print(f"\n{f} jau eksistē (netiek pārrakstīts). Kad koeficienti ierakstīti:")
    print(f"  python bets.py --odds {f.relative_to(ROOT)}" + (f" --date {day}" if args.date else ""))
    print("\nGodīgais koef. = 1 / varbūtība. Ja Optibet dod vairāk, likme ilgtermiņā ir izdevīga.")


if __name__ == "__main__":
    main()
