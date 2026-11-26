"""Bookmaker lines from The Odds API (the-odds-api.com) for leagues whose own data has no odds: SHL, EuroLeague.

Needs the environment variable ODDS_API_KEY (GitHub: repository secret ODDS_API_KEY). Without it, or when the
monthly free credits run low, everything returns None and the robots keep using their own model, as before.
One call per league and day (cached for 2 hours); a call costs (number of markets x regions) credits.

The consensus line: Pinnacle when it has the game (the sharpest European bookmaker), otherwise the median of
the other European bookmakers, with the bookmaker margin removed.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from statistics import median

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORTS = {"shl": "icehockey_sweden_hockey_league", "el": "basketball_euroleague"}
MIN_CREDITS_LEFT = 40          # keep a reserve: below this no more calls this month
STATE = CACHE / "odds_api_state.json"


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    drop = {"hc", "if", "bk", "ik", "hk", "ifk", "fc", "bc", "kk", "sk", "ab", "hockey", "basket", "basketball"}
    return " ".join(w for w in s.split() if w not in drop)


def _similar(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.95
    return difflib.SequenceMatcher(None, a, b).ratio()


def fetch(league: str, markets: str) -> list | None:
    """Raw events with bookmaker odds, or None (no key, no credits left, error)."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key or league not in SPORTS:
        return None
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"odds_api_{league}.json"
    if f.exists() and time.time() - f.stat().st_mtime < 7200:
        return json.loads(f.read_text())
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get("remaining") is not None and int(state["remaining"]) < MIN_CREDITS_LEFT \
            and time.time() - state.get("at", 0) < 3 * 86400:
        print(f"(The Odds API: atlikuši tikai {state['remaining']} kredīti — koeficientus neprasu)")
        return None
    try:
        r = requests.get(API.format(sport=SPORTS[league]), timeout=30,
                         params={"apiKey": key, "regions": "eu", "markets": markets, "oddsFormat": "decimal"})
    except requests.RequestException as e:
        print(f"(The Odds API nav pieejams: {e})")
        return None
    left, used = r.headers.get("x-requests-remaining"), r.headers.get("x-requests-used")
    if left is not None:
        STATE.write_text(json.dumps({"remaining": left, "used": used, "at": time.time()}))
        print(f"(The Odds API: izlietoti {used}, atlikuši {left} kredīti šajā mēnesī)")
    if r.status_code != 200:
        print(f"(The Odds API kļūda {r.status_code}: {r.text[:200]})")
        return None
    data = r.json()
    f.write_text(json.dumps(data))
    return data


def _consensus(event: dict, market: str) -> list[dict]:
    """Outcome lists of this market: Pinnacle first if present, otherwise every bookmaker."""
    books = event.get("bookmakers") or []
    out = []
    for b in books:
        for m in b.get("markets") or []:
            if m.get("key") == market and m.get("outcomes"):
                out.append((b.get("key") == "pinnacle", m["outcomes"]))
    pin = [o for is_pin, o in out if is_pin]
    return pin or [o for _, o in out]


def _devig(prices: list[float]) -> list[float] | None:
    if not prices or any(not p or p <= 1 for p in prices):
        return None
    inv = [1 / p for p in prices]
    s = sum(inv)
    return [x / s for x in inv]


def lines_for(event: dict, home: str, away: str) -> dict:
    """{'p_home': win prob incl. OT (2-way), 'p3': (home, draw, away) if 3-way, 'total': line, 'q_over': prob,
    'spread_home': home handicap (e.g. -4.5)}"""
    res = {}
    h2h = _consensus(event, "h2h")
    p2, p3 = [], []
    for outs in h2h:
        by = {o["name"]: o["price"] for o in outs}
        hp = next((v for k, v in by.items() if k == event["home_team"]), None)
        ap = next((v for k, v in by.items() if k == event["away_team"]), None)
        dp = by.get("Draw")
        if dp:
            d = _devig([hp, dp, ap])
            if d:
                p3.append(d)
        else:
            d = _devig([hp, ap])
            if d:
                p2.append(d[0])
    if p3:
        res["p3"] = tuple(median(x[i] for x in p3) for i in range(3))
    if p2:
        res["p_home"] = median(p2)
    tot = []
    for outs in _consensus(event, "totals"):
        ov = next((o for o in outs if o["name"] == "Over"), None)
        un = next((o for o in outs if o["name"] == "Under" and ov and o.get("point") == ov.get("point")), None)
        if ov and un:
            d = _devig([ov["price"], un["price"]])
            if d:
                tot.append((ov["point"], d[0]))
    if tot:
        main = median(t[0] for t in tot)
        near = [t for t in tot if t[0] == main] or tot
        res["total"], res["q_over"] = float(main), float(median(t[1] for t in near))
    sp = []
    for outs in _consensus(event, "spreads"):
        h = next((o for o in outs if o["name"] == event["home_team"]), None)
        a = next((o for o in outs if o["name"] == event["away_team"]), None)
        if h and a and h.get("point") is not None:
            d = _devig([h["price"], a["price"]])
            if d:
                sp.append((h["point"], d[0]))
    if sp:
        # shift the median line by the price difference (~0.05 probability per point is close enough here)
        pt = median(s[0] for s in sp)
        res["spread_home"] = float(pt)
    return res


def match(events: list, home_name: str, away_name: str, start_utc) -> dict | None:
    """The API event for this game (same day, best name match on both teams)."""
    best, score = None, 0.0
    t0 = pd.Timestamp(start_utc)
    for e in events or []:
        t = pd.Timestamp(e["commence_time"])
        if abs((t - t0).total_seconds()) > 6 * 3600:
            continue
        s = min(_similar(e["home_team"], home_name), _similar(e["away_team"], away_name))
        if s > score:
            best, score = e, s
    return best if score >= 0.6 else None
