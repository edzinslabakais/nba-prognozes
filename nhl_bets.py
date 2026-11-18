"""NHL bets on the hockey score grid.

Rule (as agreed for the friends' game): every bet is settled on the REGULATION result (60 min),
unless the bet says "(ieskaitot papildlaiku)" - then overtime/shootout counts too
(the shootout winner gets +1 goal).

Bet notation (team abbreviations as ESPN: BOS, NJ, TB, LA, SJ, UTA, VGK ...):
   BOS uzvar                               regulation win
   neizšķirts                              regulation draw
   BOS uzvar (ieskaitot papildlaiku)       winner incl. overtime/shootout
   BOS -1.5 / DET +1.5                     handicap, regulation
   virs 5.5 / zem 6.5                      total goals, regulation
   BOS virs 2.5                            team goals, regulation
   BOS uzvar & virs 5.5                    combination (all parts must win)
Add " (ieskaitot papildlaiku)" to a handicap/total to count overtime as well.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

ALIASES = {"NJD": "NJ", "TBL": "TB", "LAK": "LA", "SJS": "SJ", "UTAH": "UTA", "VEG": "VGK", "WAS": "WSH",
           "MON": "MTL", "CLB": "CBJ", "NAS": "NSH", "WIN": "WPG", "CAL": "CGY", "ARI": "UTA"}
OT_TAG = " (ieskaitot papildlaiku)"


@dataclass
class Bet:
    label: str
    fn: Callable  # grid -> (win mask, push mask)


def _none(g):
    return np.zeros_like(g.FH, bool)


def _scores(g, ot: bool):
    return (g.FH, g.FA) if ot else (g.RH, g.RA)


def _tag(ot: bool) -> str:
    return OT_TAG if ot else ""


def win(team, home, ot=False):
    def fn(g):
        h, a = _scores(g, ot)
        return ((h > a) if home else (a > h)), _none(g)
    return Bet(f"{team} uzvar{_tag(ot)}", fn)


def ml(team, home):
    """Winner including overtime/shootout."""
    return win(team, home, ot=True)


def reg_win(team, home):
    return win(team, home, ot=False)


def reg_draw():
    return Bet("neizšķirts", lambda g: (g.RH == g.RA, _none(g)))


def handicap(team, home, line, ot=False):
    def fn(g):
        h, a = _scores(g, ot)
        x = ((h - a) if home else (a - h)) + line
        return x > 0, x == 0
    return Bet(f"{team} {line:+g}{_tag(ot)}", fn)


def total(over, line, ot=False):
    def fn(g):
        h, a = _scores(g, ot)
        t = h + a
        return ((t > line) if over else (t < line)), t == line
    return Bet(f"{'virs' if over else 'zem'} {line:g}{_tag(ot)}", fn)


def team_total(team, home, over, line, ot=False):
    def fn(g):
        h, a = _scores(g, ot)
        t = h if home else a
        return ((t > line) if over else (t < line)), t == line
    return Bet(f"{team} {'virs' if over else 'zem'} {line:g}{_tag(ot)}", fn)


def combo(parts):
    def fn(g):
        w = np.ones_like(g.FH, bool)
        for p in parts:
            pw, _ = p.fn(g)
            w &= pw
        return w, _none(g)
    return Bet(" & ".join(p.label for p in parts), fn)


def prob(g, bet: Bet) -> tuple[float, float]:
    win, push = bet.fn(g)
    return float(g.W[win].sum()), float(g.W[push].sum())


# --------------------------------------------------------------------------- parsing

def _team(tok: str, game):
    t = ALIASES.get(tok.upper(), tok.upper())
    for abbr, name, home in ((game.home, game.home_name, True), (game.away, game.away_name, False)):
        if t == abbr.upper() or tok.lower() == name.split()[-1].lower():
            return abbr, home
    return None


NUM = r"(\d+(?:[.,]\d+)?)"


def parse_bet(text: str, game) -> Bet:
    parts = [p.strip() for p in text.split("&")]
    if len(parts) > 1:
        return combo([parse_bet(p, game) for p in parts])
    s = re.sub(r"\s+", " ", text.strip()).replace(",", ".")
    low = s.lower()
    ot = "ieskaitot papildlaiku" in low or "ar papildlaiku" in low
    low = re.sub(r"\(?\s*(ieskaitot|ar) papildlaiku\s*\)?", "", low).replace("pamatlaikā", "").strip()
    low = re.sub(r"\s+", " ", low)
    if low in ("neizšķirts", "x"):
        return reg_draw()
    m = re.fullmatch(rf"(virs|over|zem|under) {NUM}", low)
    if m:
        return total(m.group(1) in ("virs", "over"), float(m.group(2)), ot)
    toks = low.split(" ")
    team = _team(toks[0], game)
    if team is None:
        raise ValueError(f"nepazīstu komandu '{toks[0]}' spēlē {game.away}@{game.home}")
    abbr, home = team
    rest = " ".join(toks[1:])
    if re.fullmatch(r"(uzv\w*|win|ml)", rest):
        return win(abbr, home, ot)
    m = re.fullmatch(rf"([+-]{NUM})", rest)
    if m:
        return handicap(abbr, home, float(m.group(1)), ot)
    m = re.fullmatch(rf"(virs|over|zem|under) {NUM}", rest)
    if m:
        return team_total(abbr, home, m.group(1) in ("virs", "over"), float(m.group(2)), ot)
    raise ValueError(f"nesaprotu likmi '{text}'")


class Actual:
    """A finished game as a one-scenario grid, so bets can be settled with the same code."""

    def __init__(self, g):
        self.RH, self.RA = np.array([g.home_reg], float), np.array([g.away_reg], float)
        self.FH, self.FA = np.array([g.home_goals], float), np.array([g.away_goals], float)
        self.W = np.array([1.0])


def settle(text: str, g) -> tuple[bool, bool, str]:
    bet = parse_bet(text, g)
    w, push = bet.fn(Actual(g))
    score = f"{g.away} {int(g.away_reg)} - {int(g.home_reg)} {g.home}"
    if g.ot:
        how = "bullīšos" if g.so else "papildlaikā"
        score += f" (pamatlaikā; {how} {int(g.away_goals)} - {int(g.home_goals)})"
    return bool(w[0]), bool(push[0]), score
