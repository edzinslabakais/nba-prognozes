"""Thin client for ESPN's public NBA JSON endpoints, with an on-disk cache.

Finished days / finished games are cached forever; "live" data (today's
scoreboard, injury report, rosters) uses a short TTL so it stays fresh.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
ET = ZoneInfo("America/New_York")

_session = requests.Session()

# Set to True (via predict.py --refresh) to ignore TTL caches for live data.
FORCE_REFRESH = False


def today_et() -> date:
    """NBA schedules are organised by US Eastern date."""
    return datetime.now(ET).date()


NHL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"


def _cache_path(path: str, params: dict | None, base: str = BASE) -> Path:
    prefix = "" if base == BASE else base  # NBA keeps its original cache keys
    key = hashlib.sha1((prefix + path + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def get_json(path: str, params: dict | None = None, ttl: float | None = None,
             trim=None, base: str = BASE) -> dict:
    """GET BASE+path. ttl=None caches forever, ttl=0 never caches.
    trim: optional function that drops unused parts before caching."""
    f = _cache_path(path, params, base)
    if f.exists() and ttl != 0 and not (FORCE_REFRESH and ttl is not None):
        if ttl is None or time.time() - f.stat().st_mtime < ttl:
            with open(f) as fh:
                return json.load(fh)

    last_err: Exception | None = None
    for attempt in range(4):
        try:
            r = _session.get(base + path, params=params, timeout=30)
            if r.status_code == 404:
                raise LookupError(f"ESPN 404: {path} {params}")
            r.raise_for_status()
            data = r.json()
            if trim:
                data = trim(data)
            break
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(1.5 * 2**attempt)
    else:
        raise RuntimeError(f"ESPN request failed: {path} {params}: {last_err}")

    if ttl != 0:
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, f)
    return data


def scoreboard(d: date, base: str = BASE) -> dict:
    # Days at least two days in the past are final -> cache forever.
    ttl = None if d <= today_et() - timedelta(days=2) else 600
    return get_json("/scoreboard", {"dates": d.strftime("%Y%m%d")}, ttl=ttl, trim=trim_scoreboard, base=base)


def summary(event_id: str, base: str = BASE) -> dict:
    """Box score + closing odds. Only call this for finished games."""
    return get_json("/summary", {"event": str(event_id)}, ttl=None, trim=trim_summary, base=base)


_HEAVY_COMP_KEYS = ("broadcasts", "broadcast", "geoBroadcasts", "headlines", "highlights",
                    "tickets", "venue", "notes", "leaders", "details", "situation")


def trim_scoreboard(sb: dict) -> dict:
    events = []
    for e in sb.get("events", []):
        e = {k: v for k, v in e.items() if k not in ("links", "weather")}
        comps = []
        for c in e.get("competitions", []):
            c = {k: v for k, v in c.items() if k not in _HEAVY_COMP_KEYS}
            c["competitors"] = [{k: v for k, v in t.items() if k not in ("leaders", "statistics")}
                                for t in c.get("competitors", [])]
            comps.append(c)
        e["competitions"] = comps
        events.append(e)
    return {"day": sb.get("day"), "events": events}


def trim_summary(s: dict) -> dict:
    return {"boxscore": {"players": s.get("boxscore", {}).get("players", [])},
            "pickcenter": s.get("pickcenter", [])}


def injuries(base: str = BASE) -> dict:
    return get_json("/injuries", ttl=900, base=base)


def roster(team_id: str, base: str = BASE) -> dict:
    return get_json(f"/teams/{team_id}/roster", ttl=6 * 3600, base=base)


_ID_RE = re.compile(r"/id/(\d+)")


def athlete_id_from_links(athlete: dict) -> str | None:
    if athlete.get("id"):
        return str(athlete["id"])
    for link in athlete.get("links", []):
        m = _ID_RE.search(link.get("href", ""))
        if m:
            return m.group(1)
    return None
