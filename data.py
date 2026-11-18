"""Turn raw ESPN JSON into tidy pandas tables: games, player game logs, injuries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd

import espn

# Regular season start dates (ESPN "season year" = year the season ends).
SEASONS = {
    2025: (date(2024, 10, 22), date(2025, 6, 25)),
    2026: (date(2025, 10, 21), date(2026, 6, 25)),
    2027: (date(2026, 10, 20), date(2027, 6, 25)),
}
NON_PRESEASON = {2, 3, 5}  # regular season, playoffs, play-in


def season_for(d: date) -> int:
    for yr, (start, end) in SEASONS.items():
        if start - timedelta(days=30) <= d <= end:
            return yr
    return d.year + 1 if d.month >= 8 else d.year


def _parse_event(e: dict, day: date) -> dict | None:
    comp = e["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    if "home" not in teams or "away" not in teams:
        return None
    h, a = teams["home"], teams["away"]
    status = e["status"]["type"]
    row = {
        "game_id": str(e["id"]),
        "date": day,
        "start_utc": pd.Timestamp(e["date"]),
        "season": e["season"]["year"],
        "season_type": e["season"]["type"],
        "neutral": bool(comp.get("neutralSite", False)),
        "home_id": str(h["team"]["id"]),
        "away_id": str(a["team"]["id"]),
        "home": h["team"]["abbreviation"],
        "away": a["team"]["abbreviation"],
        "home_name": h["team"].get("displayName", h["team"]["abbreviation"]),
        "away_name": a["team"].get("displayName", a["team"]["abbreviation"]),
        "completed": bool(status.get("completed", False)),
        "status": status.get("name", ""),
        "home_pts": float(h["score"]) if status.get("completed") else None,
        "away_pts": float(a["score"]) if status.get("completed") else None,
        "home_record": next((r["summary"] for r in h.get("records", []) if r.get("type") == "total"), None),
        "away_record": next((r["summary"] for r in a.get("records", []) if r.get("type") == "total"), None),
        "market_margin": None,
        "market_total": None,
    }
    odds = comp.get("odds") or []
    if odds and odds[0].get("spread") is not None:
        # "spread" is the home team's line: -4.5 => home expected to win by 4.5
        row["market_margin"] = -float(odds[0]["spread"])
        row["market_total"] = odds[0].get("overUnder")
    return row


def load_games(start: date, end: date, workers: int = 4) -> pd.DataFrame:
    """All non-preseason games between start and end (inclusive, ET dates)."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    with ThreadPoolExecutor(workers) as ex:
        boards = list(ex.map(espn.scoreboard, days))
    rows = []
    for day, sb in zip(days, boards):
        for e in sb.get("events", []):
            r = _parse_event(e, day)
            if r and r["season_type"] in NON_PRESEASON:
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates("game_id").sort_values("start_utc").reset_index(drop=True)


def load_season(season: int, until: date | None = None) -> pd.DataFrame:
    start, end = SEASONS[season]
    end = min(end, until or espn.today_et() + timedelta(days=7))
    return load_games(start, end)


# --------------------------------------------------------------------------- box scores

def _minutes(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _split(v: str) -> tuple[float, float]:
    try:
        m, a = v.split("-")
        return float(m), float(a)
    except (AttributeError, ValueError):
        return 0.0, 0.0


def parse_boxscore(game_id: str, s: dict) -> tuple[list[dict], float | None, float | None]:
    """Player rows + closing market margin/total (home perspective)."""
    rows = []
    for team in s.get("boxscore", {}).get("players", []):
        tid = str(team["team"]["id"])
        for block in team.get("statistics", []):
            keys = block.get("labels") or block.get("names")
            for a in block.get("athletes", []):
                ath = a["athlete"]
                base = {"game_id": game_id, "team_id": tid, "player_id": espn.athlete_id_from_links(ath) or ath.get("displayName", ""),
                        "name": ath.get("displayName", ""), "starter": bool(a.get("starter"))}
                if a.get("didNotPlay") or not a.get("stats"):
                    rows.append({**base, "min": 0.0, "pts": 0.0, "gmsc": 0.0, "pm": 0.0,
                                 "dnp_reason": a.get("reason", "")})
                    continue
                st = dict(zip(keys, a["stats"]))
                fgm, fga = _split(st.get("FG"))
                ftm, fta = _split(st.get("FT"))
                pts = _minutes(st.get("PTS"))
                oreb, dreb = _minutes(st.get("OREB")), _minutes(st.get("DREB"))
                ast, stl, blk = _minutes(st.get("AST")), _minutes(st.get("STL")), _minutes(st.get("BLK"))
                pf, tov = _minutes(st.get("PF")), _minutes(st.get("TO"))
                # Hollinger Game Score: a one-number box-score summary of a game.
                gmsc = (pts + 0.4 * fgm - 0.7 * fga - 0.4 * (fta - ftm) + 0.7 * oreb + 0.3 * dreb
                        + stl + 0.7 * ast + 0.7 * blk - 0.4 * pf - tov)
                pm = st.get("+/-", "0").replace("+", "")
                rows.append({**base, "min": _minutes(st.get("MIN")), "pts": pts, "gmsc": gmsc,
                             "pm": _minutes(pm), "dnp_reason": ""})
    mm = mt = None
    pc = s.get("pickcenter") or []
    if pc and pc[0].get("spread") is not None:
        mm = -float(pc[0]["spread"])
        mt = pc[0].get("overUnder")
    return rows, mm, mt


def load_player_logs(games: pd.DataFrame, workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player game logs for completed games + closing lines per game."""
    done = games[games["completed"]]
    ids = done["game_id"].tolist()
    def fetch(gid):
        try:
            return espn.summary(gid)
        except (LookupError, RuntimeError):
            return {}  # e.g. All-Star game: no regular box score

    with ThreadPoolExecutor(workers) as ex:
        sums = list(ex.map(fetch, ids))
    rows, lines = [], []
    for gid, s in zip(ids, sums):
        r, mm, mt = parse_boxscore(gid, s)
        rows.extend(r)
        lines.append({"game_id": gid, "close_margin": mm, "close_total": mt})
    logs = pd.DataFrame(rows)
    if not logs.empty:
        logs = logs.merge(done[["game_id", "date", "start_utc"]], on="game_id")
    return logs, pd.DataFrame(lines)


# --------------------------------------------------------------------------- injuries / rosters

def load_injuries() -> pd.DataFrame:
    raw = espn.injuries()
    rows = []
    for team in raw.get("injuries", []):
        for inj in team.get("injuries", []):
            ath = inj.get("athlete", {})
            t = ath.get("team") or {}
            rows.append({
                "team_id": str(t.get("id", team.get("id", ""))),
                "player_id": espn.athlete_id_from_links(ath),
                "name": ath.get("displayName", ""),
                "status": inj.get("status", ""),
                "comment": inj.get("shortComment", ""),
                "return_date": (inj.get("details") or {}).get("returnDate"),
                "updated": inj.get("date"),
            })
    return pd.DataFrame(rows)


def load_rosters(team_ids: list[str], workers: int = 4) -> dict[str, set[str]]:
    with ThreadPoolExecutor(workers) as ex:
        rs = list(ex.map(espn.roster, team_ids))
    return {tid: {str(a["id"]) for a in r.get("athletes", [])} for tid, r in zip(team_ids, rs)}
