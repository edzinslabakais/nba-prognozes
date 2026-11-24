"""NHL data from ESPN: games (with regulation scores, odds, probable goalies), box scores, injuries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

import espn

BASE = espn.NHL
SEASONS = {  # ESPN season year = year the season ends
    2025: (date(2024, 10, 1), date(2025, 6, 30)),
    2026: (date(2025, 10, 1), date(2026, 6, 30)),
    2027: (date(2026, 10, 1), date(2027, 6, 30)),
}
NON_PRESEASON = {2, 3}


def season_for(d: date) -> int:
    for yr, (start, end) in SEASONS.items():
        if start - timedelta(days=30) <= d <= end:
            return yr
    return d.year + 1 if d.month >= 8 else d.year


def american_to_decimal(v) -> float | None:
    try:
        x = float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if x == 0:
        return None
    return round(1 + (x / 100 if x > 0 else 100 / -x), 4)


def _ml_from_odds(o: dict) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """(home dec, away dec, total line, over dec, under dec) from a scoreboard odds entry."""
    home = away = line = over = under = None
    ml = o.get("moneyline") or {}
    if ml:
        home = american_to_decimal((ml.get("home", {}).get("close") or {}).get("odds"))
        away = american_to_decimal((ml.get("away", {}).get("close") or {}).get("odds"))
    if home is None:
        home = american_to_decimal((o.get("homeTeamOdds") or {}).get("moneyLine"))
        away = american_to_decimal((o.get("awayTeamOdds") or {}).get("moneyLine"))
    line = o.get("overUnder")
    tot = o.get("total") or {}
    if tot:
        over = american_to_decimal((tot.get("over", {}).get("close") or {}).get("odds"))
        under = american_to_decimal((tot.get("under", {}).get("close") or {}).get("odds"))
    if over is None and o.get("overOdds") is not None:
        over = american_to_decimal(o.get("overOdds"))
        under = american_to_decimal(o.get("underOdds"))
    return home, away, (float(line) if line is not None else None), over, under


def _goalie(comp: dict) -> tuple[str | None, str | None, str | None]:
    for p in comp.get("probables") or []:
        if p.get("name") == "probableStartingGoalie":
            a = p.get("athlete") or {}
            return str(a.get("id") or p.get("playerId")), a.get("displayName"), (p.get("status") or {}).get("type")
    return None, None, None


def _parse_event(e: dict, day: date) -> dict | None:
    comp = e["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    if "home" not in teams or "away" not in teams:
        return None
    h, a = teams["home"], teams["away"]
    st = e["status"]["type"]
    done = bool(st.get("completed"))
    detail = st.get("detail", "") or ""

    def p1(c):
        ls = c.get("linescores") or []
        return float(ls[0].get("value", 0) or 0) if ls else None

    def reg(c):
        ls = c.get("linescores") or []
        return float(sum(float(x.get("value", 0) or 0) for x in ls[:3])) if ls else None

    row = {
        "game_id": str(e["id"]), "date": day, "start_utc": pd.Timestamp(e["date"]),
        "season": e["season"]["year"], "season_type": e["season"]["type"],
        "neutral": bool(comp.get("neutralSite", False)),
        "home_id": str(h["team"]["id"]), "away_id": str(a["team"]["id"]),
        "home": h["team"]["abbreviation"], "away": a["team"]["abbreviation"],
        "home_name": h["team"].get("displayName", h["team"]["abbreviation"]),
        "away_name": a["team"].get("displayName", a["team"]["abbreviation"]),
        "completed": done, "status": st.get("name", ""), "detail": detail,
        "home_goals": float(h["score"]) if done else None, "away_goals": float(a["score"]) if done else None,
        "home_reg": reg(h) if done else None, "away_reg": reg(a) if done else None,
        "home_p1": p1(h) if done else None, "away_p1": p1(a) if done else None,
        "ot": done and ("OT" in detail or "SO" in detail), "so": done and "SO" in detail,
        "home_record": next((r["summary"] for r in h.get("records", []) if r.get("type") == "total"), None),
        "away_record": next((r["summary"] for r in a.get("records", []) if r.get("type") == "total"), None),
    }
    row["home_goalie_id"], row["home_goalie"], row["home_goalie_status"] = _goalie(h)
    row["away_goalie_id"], row["away_goalie"], row["away_goalie_status"] = _goalie(a)
    odds = comp.get("odds") or []
    hm, am, line, ov, un = _ml_from_odds(odds[0]) if odds else (None,) * 5
    row.update({"ml_home": hm, "ml_away": am, "total_line": line, "over_odds": ov, "under_odds": un})
    return row


def load_games(start: date, end: date, workers: int = 4) -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    with ThreadPoolExecutor(workers) as ex:
        boards = list(ex.map(lambda d: espn.scoreboard(d, base=BASE), days))
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

def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _toi(v) -> float:
    try:
        m, s = str(v).split(":")
        return int(m) + int(s) / 60
    except ValueError:
        return 0.0


def parse_boxscore(game_id: str, s: dict) -> tuple[list[dict], list[dict], dict]:
    """Skater rows, goalie rows and closing odds of one finished game."""
    skaters, goalies = [], []
    for team in s.get("boxscore", {}).get("players", []):
        tid = str(team["team"]["id"])
        for block in team.get("statistics", []):
            labels = block.get("labels") or []
            kind = block.get("name", "")
            for a in block.get("athletes", []):
                ath = a.get("athlete", {})
                pid = espn.athlete_id_from_links(ath) or ath.get("displayName", "")
                st = dict(zip(labels, a.get("stats") or []))
                base = {"game_id": game_id, "team_id": tid, "player_id": str(pid), "name": ath.get("displayName", "")}
                if kind == "goalies":
                    goalies.append({**base, "toi": _toi(st.get("TOI")), "sa": _num(st.get("SA")),
                                    "ga": _num(st.get("GA")), "sv": _num(st.get("SV"))})
                elif kind in ("forwards", "defenses", "skaters"):
                    toi = _toi(st.get("TOI"))
                    g, ast, sog = _num(st.get("G")), _num(st.get("A")), _num(st.get("SOG"))
                    # simplified hockey "game score"
                    gs = (0.75 * g + 0.6 * ast + 0.075 * sog + 0.05 * _num(st.get("BS"))
                          + 0.01 * (_num(st.get("FW")) - _num(st.get("FL")))
                          + 0.02 * (_num(st.get("TK")) - _num(st.get("GV"))) - 0.03 * _num(st.get("PIM")))
                    skaters.append({**base, "pos": "D" if kind == "defenses" else "F", "toi": toi,
                                    "g": g, "a": ast, "sog": sog, "gs": gs})
    odds = {}
    pc = s.get("pickcenter") or []
    if pc:
        hm, am, line, ov, un = _ml_from_odds(pc[0])
        odds = {"close_ml_home": hm, "close_ml_away": am, "close_total": line,
                "close_over": ov, "close_under": un}
    return skaters, goalies, odds


def load_boxscores(games: pd.DataFrame, workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    done = games[games["completed"]]
    ids = done["game_id"].tolist()

    def fetch(gid):
        try:
            return espn.summary(gid, base=BASE)
        except (LookupError, RuntimeError):
            return {}

    with ThreadPoolExecutor(workers) as ex:
        sums = list(ex.map(fetch, ids))
    sk, gl, od = [], [], []
    for gid, s in zip(ids, sums):
        a, b, c = parse_boxscore(gid, s)
        sk += a
        gl += b
        od.append({"game_id": gid, **c})
    meta = done[["game_id", "date", "start_utc"]]
    skaters = pd.DataFrame(sk)
    goalies = pd.DataFrame(gl)
    if not skaters.empty:
        skaters = skaters.merge(meta, on="game_id")
    if not goalies.empty:
        goalies = goalies.merge(meta, on="game_id")
    return skaters, goalies, pd.DataFrame(od)


# --------------------------------------------------------------------------- injuries / rosters

def load_injuries() -> pd.DataFrame:
    raw = espn.injuries(base=BASE)
    rows = []
    for team in raw.get("injuries", []):
        for inj in team.get("injuries", []):
            ath = inj.get("athlete", {})
            t = ath.get("team") or {}
            rows.append({"team_id": str(t.get("id", team.get("id", ""))),
                         "player_id": espn.athlete_id_from_links(ath), "name": ath.get("displayName", ""),
                         "status": inj.get("status", ""), "return_date": (inj.get("details") or {}).get("returnDate")})
    return pd.DataFrame(rows)


def load_rosters(team_ids: list[str], workers: int = 4) -> dict[str, set[str]]:
    with ThreadPoolExecutor(workers) as ex:
        rs = list(ex.map(lambda t: espn.roster(t, base=BASE), team_ids))
    out = {}
    for tid, r in zip(team_ids, rs):
        ids = set()
        for grp in r.get("athletes", []):
            # NHL rosters are grouped by position
            for a in grp.get("items", [grp]) if isinstance(grp, dict) else []:
                if a.get("id"):
                    ids.add(str(a["id"]))
        out[tid] = ids
    return out
