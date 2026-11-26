"""European hockey: Czech Extraliga (hokej.cz), German DEL (penny-del.org), Swedish SHL (shl.se).

The leagues' own sites give the schedule and the results by period; there are no bookmaker odds in them, so
(as for the EuroLeague) the probabilities come from the robot's own model: a Poisson goal model of every team's
attack and defence from this season's regulation results (for SHL also last season, older games count less)
plus home ice. A regulation tie is a little more likely than independent goals suggest; that boost is measured
on the league's own results. All bets are settled on regulation time (60 minutes) unless a bet says
"(ieskaitot papildlaiku)", exactly like NHL. Picks are made at ~12:30 Riga time with the other European leagues.

  python cz_bot.py | del_bot.py | shl_bot.py  likmes | rezultati | eksports | kalibret
"""
from __future__ import annotations

import argparse
import hashlib
import html as H
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import bot as core
import nhl_bets as HB
import nhl_bot as NB
import nhl_model as N
import odds

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RIGA = ZoneInfo("Europe/Riga")
UA = {"User-Agent": "Mozilla/5.0 (prognozes)"}
FORCE_REFRESH = False
DEFAULTS = {"tie_infl": 0.25, "ot_k": 0.0, "half_life": 120.0, "ridge": 3.0, "market_weight": 0.8}


def fetch(url: str, ttl: float | None = 600) -> str:
    """GET with a small on-disk cache (ttl=None: keep forever)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / ("hk_" + hashlib.sha1(url.encode()).hexdigest() + ".txt")
    if f.exists() and not FORCE_REFRESH and (ttl is None or time.time() - f.stat().st_mtime < ttl):
        return f.read_text()
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=40)
            r.raise_for_status()
            f.write_text(r.text)
            return r.text
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * 2 ** attempt)
    raise RuntimeError(f"Neizdevās ielādēt {url}: {last}")


def _txt(s: str) -> str:
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def _row(gid, start_utc, home, away, home_name, away_name, home_short, away_short, final=None, reg=None, ot=False, so=False):
    done = final is not None
    return {"game_id": str(gid), "date": start_utc.tz_convert(RIGA).date(), "start_utc": start_utc,
            "home": home, "away": away, "home_id": home, "away_id": away, "home_name": home_name, "away_name": away_name,
            "home_short": home_short, "away_short": away_short, "completed": done,
            "home_goals": float(final[0]) if done else None, "away_goals": float(final[1]) if done else None,
            "home_reg": float(reg[0]) if done else None, "away_reg": float(reg[1]) if done else None,
            "ot": bool(done and ot), "so": bool(done and so)}


def _season_start(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1


# --------------------------------------------------------------------------- Czech Extraliga (hokej.cz)

def load_cz(today: date) -> pd.DataFrame:
    h = fetch("https://www.hokej.cz/tipsport-extraliga/zapasy?do=matchList-view-all")
    y0 = _season_start(today)
    tz = ZoneInfo("Europe/Prague")
    rows = []
    for gid, body in re.findall(r'<tr data-href="/zapas/(\d+)"(.*?)</tr>', h, re.S):
        names = re.findall(r'preview__name--long">([^<]*)<.*?preview__name--medium">([^<]*)<.*?preview__name--short">([^<]*)<',
                           body, re.S)
        if len(names) < 2:
            continue
        (hl, hm, hs), (al, am, as_) = [tuple(H.unescape(x).strip() for x in n) for n in names[:2]]
        when = re.search(r"(\d{2})\. (\d{2})\.(?: (\d{1,2}):(\d{2}))?", _txt(body))
        if not when:
            continue
        dd, mm = int(when.group(1)), int(when.group(2))
        hh, mi = (int(when.group(3)), int(when.group(4))) if when.group(3) else (17, 30)
        yy = y0 if mm >= 7 else y0 + 1
        start = pd.Timestamp(datetime(yy, mm, dd, hh, mi, tzinfo=tz)).tz_convert("UTC")
        scores = re.findall(r'preview__score[^"]*"><span[^>]*>(\d+)</span>', body)
        per = re.search(r"\(([\d:, \-]+)\)", _txt(body))
        if len(scores) >= 2 and per:
            parts = [p.strip() for p in re.split(r"[,\-]", per.group(1)) if ":" in p]
            reg = [tuple(int(x) for x in p.split(":")) for p in parts[:3]]
            rh, ra = sum(x[0] for x in reg), sum(x[1] for x in reg)
            fh, fa = int(scores[0]), int(scores[1])
            ot = len(parts) > 3 or (fh, fa) != (rh, ra)
            so = " SN" in _txt(body)
            rows.append(_row(gid, start, hs, as_, hl, al, hm, am, (fh, fa), (rh, ra), ot, so))
        else:
            rows.append(_row(gid, start, hs, as_, hl, al, hm, am))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- DEL (penny-del.org)

DEL_MONTHS = [("september", 9), ("oktober", 10), ("november", 11), ("dezember", 12), ("januar", 1),
              ("februar", 2), ("maerz", 3), ("april", 4)]
DEL_CODES = {"eisbaeren-berlin": ("EBB", "Berlin"), "adler-mannheim": ("MAN", "Mannheim"),
             "ehc-red-bull-muenchen": ("RBM", "München"), "erc-ingolstadt": ("ING", "Ingolstadt"),
             "koelner-haie": ("KEC", "Köln"), "grizzlys-wolfsburg": ("WOB", "Wolfsburg"),
             "straubing-tigers": ("STR", "Straubing"), "nuernberg-ice-tigers": ("NIT", "Nürnberg"),
             "augsburger-panther": ("AEV", "Augsburg"), "schwenninger-wild-wings": ("SWW", "Schwenningen"),
             "pinguins-bremerhaven": ("BHV", "Bremerhaven"), "duesseldorfer-eg": ("DEG", "Düsseldorf"),
             "iserlohn-roosters": ("IEC", "Iserlohn"), "loewen-frankfurt": ("FRA", "Frankfurt"),
             "dresdner-eisloewen": ("DRE", "Dresden"), "krefeld-pinguine": ("KEV", "Krefeld")}


def _del_code(slug: str, name: str) -> tuple[str, str]:
    return DEL_CODES.get(slug) or (re.sub(r"[^A-Z]", "", slug.split("-")[-1].upper())[:3], name.split()[-1])


def load_del(today: date) -> pd.DataFrame:
    y0 = _season_start(today)
    tz = ZoneInfo("Europe/Berlin")
    rows = []
    for name, mm in DEL_MONTHS:
        yy = y0 if mm >= 7 else y0 + 1
        if date(yy, mm, 1) > today + timedelta(days=40):
            continue
        past = date(yy, mm, 1) + timedelta(days=40) < today
        h = fetch(f"https://www.penny-del.org/spiele/monat/{name}", ttl=None if past else 600)
        for body in re.findall(r"<tr>(.*?)</tr>", h, re.S):
            teams = re.findall(r'href="/teams/([a-z0-9-]+)/uebersicht">\s*([^<]+?)\s*</a>', body)
            when = re.search(r"(\d{2})\.(\d{2})\.(\d{4}).*?(\d{1,2}):(\d{2})", _txt(body))
            if len(teams) < 2 or not when:
                continue
            (hslug, hname), (aslug, aname) = teams[0], teams[1]
            hname, aname = H.unescape(hname), H.unescape(aname)
            start = pd.Timestamp(datetime(int(when.group(3)), int(when.group(2)), int(when.group(1)),
                                          int(when.group(4)), int(when.group(5)), tzinfo=tz)).tz_convert("UTC")
            link = re.search(r"spieldetails/([^\"]+)\"", body)
            gid = link.group(1).rsplit("_", 1)[-1] if link else f"{start:%Y%m%d}_{hslug}_{aslug}"
            res = re.search(r"(\d+) : (\d+)\s*(\((?:OT|SO)\))?.*?\(([\d:, ]+)\)", _txt(body))
            (hc, hsh), (ac, ash) = _del_code(hslug, hname), _del_code(aslug, aname)
            if res:
                per = [tuple(int(x) for x in p.strip().split(":")) for p in res.group(4).split(",") if ":" in p]
                rh, ra = sum(x[0] for x in per[:3]), sum(x[1] for x in per[:3])
                tag = res.group(3) or ""
                rows.append(_row(gid, start, hc, ac, hname, aname, hsh, ash,
                                 (int(res.group(1)), int(res.group(2))), (rh, ra), bool(tag) or len(per) > 3, "SO" in tag))
            else:
                rows.append(_row(gid, start, hc, ac, hname, aname, hsh, ash))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- SHL (shl.se)

SHL_API = "https://www.shl.se/api/sports-v2"
SHL_SERIES = "qQ9-bb0bzEWUk"
SHL_TYPES = ("qQ9-af37Ti40B", "qQ9-7debq38kX")  # regular season, playoffs


def _shl_seasons() -> dict:
    d = json.loads(fetch(f"{SHL_API}/season-series-game-types-filter", ttl=86400))
    return {int(s["code"]): s["uuid"] for s in d.get("season", [])}


def load_shl(today: date, seasons: int = 2) -> pd.DataFrame:
    ids = _shl_seasons()
    y0 = _season_start(today)
    rows = []
    for y in range(y0 - seasons + 1, y0 + 1):
        if y not in ids:
            continue
        for gt in SHL_TYPES:
            url = f"{SHL_API}/game-schedule?seasonUuid={ids[y]}&seriesUuid={SHL_SERIES}&gameTypeUuid={gt}&gamePlace=all&played=all"
            try:
                d = json.loads(fetch(url, ttl=None if y < y0 else 600))
            except (RuntimeError, ValueError):
                continue  # e.g. no playoff schedule yet
            for g in d.get("gameInfo", []):
                ht, at = g.get("homeTeamInfo") or {}, g.get("awayTeamInfo") or {}
                if not ht.get("code") or not at.get("code"):
                    continue  # playoff slot without teams yet
                start = pd.Timestamp(g["rawStartDateTime"])
                hn, an = ht.get("names") or {}, at.get("names") or {}
                args = (g["uuid"], start, ht["code"], at["code"], hn.get("long", ht["code"]), an.get("long", at["code"]),
                        hn.get("short", ht["code"]), an.get("short", at["code"]))
                if g.get("state") == "post-game" and ht.get("score") is not None:
                    fh, fa = int(ht["score"]), int(at["score"])
                    ot, so = bool(g.get("overtime")), bool(g.get("shootout"))
                    reg = (min(fh, fa), min(fh, fa)) if (ot or so) else (fh, fa)
                    rows.append(_row(*args, (fh, fa), reg, ot or so, so))
                else:
                    rows.append(_row(*args))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- model

def fit_poisson(games: pd.DataFrame, as_of: pd.Timestamp, half_life: float, ridge: float):
    """Penalised Poisson model of regulation goals: log(goals) = base + home + attack(team) - defence(opponent)."""
    g = games[games["completed"] & games["home_reg"].notna() & (games["start_utc"] < as_of)]
    if len(g) < 20:
        return None
    teams = sorted(set(g["home"]) | set(g["away"]))
    ix = {t: i for i, t in enumerate(teams)}
    n, k = len(teams), 2 + 2 * len(teams)
    age = (as_of - g["start_utc"]).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / half_life)
    X = np.zeros((2 * len(g), k))
    y = np.zeros(2 * len(g))
    wv = np.repeat(w, 2)
    for j, (h, a, hg, ag) in enumerate(zip(g["home"], g["away"], g["home_reg"], g["away_reg"])):
        X[2 * j, [0, 1, 2 + ix[h]]] = 1; X[2 * j, 2 + n + ix[a]] = -1; y[2 * j] = hg
        X[2 * j + 1, [0, 2 + ix[a]]] = 1; X[2 * j + 1, 2 + n + ix[h]] = -1; y[2 * j + 1] = ag
    pen = np.full(k, ridge); pen[:2] = 1e-6
    th = np.zeros(k); th[0] = math.log(max(y.mean(), 0.3))
    for _ in range(50):
        mu = np.exp(X @ th)
        step = np.linalg.solve((X * (wv * mu)[:, None]).T @ X + np.diag(pen), X.T @ (wv * (y - mu)) - pen * th)
        th += step
        if np.abs(step).max() < 1e-7:
            break

    def predict(home: str, away: str) -> tuple[float, float]:
        ah, aa = (th[2 + ix[t]] if t in ix else 0.0 for t in (home, away))
        dh, da = (th[2 + n + ix[t]] if t in ix else 0.0 for t in (home, away))
        return float(math.exp(th[0] + th[1] + ah - da)), float(math.exp(th[0] + aa - dh))
    return predict


# --------------------------------------------------------------------------- leagues

@dataclass
class Spec:
    key: str
    title: str
    loader: object
    report: str


SPECS = {"cz": Spec("cz", "Extraliga", load_cz, "CZ.md"), "del": Spec("del", "DEL", load_del, "DEL.md"),
         "shl": Spec("shl", "SHL", load_shl, "SHL.md")}


def params_file(key: str) -> Path:
    return ROOT / f"{key}_params.json"


def load_params(key: str) -> dict:
    p = dict(DEFAULTS)
    f = params_file(key)
    if f.exists():
        p.update(json.loads(f.read_text()))
    return p


_GAMES: dict = {}


def all_games(spec: Spec) -> pd.DataFrame:
    if spec.key not in _GAMES:
        df = spec.loader(datetime.now(RIGA).date())
        _GAMES[spec.key] = (df.drop_duplicates("game_id").sort_values("start_utc").reset_index(drop=True)
                            if not df.empty else df)
    return _GAMES[spec.key]


def make_league(spec: Spec) -> core.League:
    def load_games(start: date, end: date) -> pd.DataFrame:
        g = all_games(spec)
        return g[(g["date"] >= start) & (g["date"] <= end)] if not g.empty else g
    return core.League(spec.key, spec.title, "🏒", ROOT / spec.key, ROOT / spec.report, load_games, HB.settle,
                       day_offset=0)


def predict_day(spec: Spec, day: date, p: dict) -> pd.DataFrame:
    games = all_games(spec)
    if games.empty:
        return games
    todo = games[(games["date"] == day) & ~games["completed"]]
    if todo.empty:
        return pd.DataFrame()
    model = fit_poisson(games, todo["start_utc"].min(), p["half_life"], p["ridge"])
    if model is None:
        return pd.DataFrame()
    # bookmakers' lines where The Odds API has the league (SHL): then the analysis is the same as NHL —
    # 80 % bookmakers / 20 % robot model, ★ and 1.7+ picks by the bookmakers' probability
    events = odds.fetch(spec.key, "h2h,totals") if spec.key in odds.SPORTS else None
    rows = []
    for g in todo.itertuples():
        olh, ola = model(g.home, g.away)
        lh, la, mk, total, p_mk = olh, ola, None, None, None
        ev = odds.match(events, g.home_name, g.away_name, g.start_utc) if events else None
        if ev:
            ln = odds.lines_for(ev, g.home_name, g.away_name)
            total = ln.get("total")
            if ln.get("p_home") is not None:
                mk = N.market_lambdas(ln["p_home"], total, ln.get("q_over"), p["tie_infl"], p["ot_k"], olh + ola)
            elif ln.get("p3"):
                import cl_bot
                mk = cl_bot.solve_lambdas(ln["p3"], total, ln.get("q_over"), p["tie_infl"])
            if mk:
                lh, la = N.blend((olh, ola), mk, p.get("market_weight", 0.8))
                p_mk = N.fast_p_home(mk[0], mk[1], p["tie_infl"], p["ot_k"])
        rows.append({**g._asdict(), "riga_time": g.start_utc.tz_convert(RIGA).strftime("%d.%m %H:%M"),
                     "lh": lh, "la": la, "model_lh": olh, "model_la": ola,
                     "mk_lh": mk[0] if mk else None, "mk_la": mk[1] if mk else None,
                     "total_line": total, "p_market": p_mk, "home_record": None, "away_record": None})
    return pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")


def cmd_likmes(spec: Spec, L: core.League, args) -> None:
    p = load_params(spec.key)
    L.message.unlink(missing_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    day = datetime.now(RIGA).date()
    df = predict_day(spec, day, p)
    if not df.empty:
        df = df[df["start_utc"] > now]
    if df.empty:
        print(f"Šodien nav {spec.title} spēļu.")
        return
    src = ("bukmeikeru līnijas (The Odds API) 80 % + robota modelis 20 %, kā NHL" if df["mk_lh"].notna().any()
           else "bukmeikeru koeficientu nav: varbūtības no robota modeļa")
    print(f"{spec.title.upper()} — {day:%d.%m.%Y}, min. koeficients {args.min_odds:g} ({src}; visas likmes pēc pamatlaika)")
    record, games_info = [], {}
    for r in df.itertuples():
        picks = NB.pick_for_game(r, p, args.min_odds)
        rl, _ = NB.win_line(r, p)
        mk_txt = NB.market_text(r)
        print(f"\n{r.away} @ {r.home}  {r.riga_time}  |  robots (pamatlaiks): {rl}, vārti {r.lh:.1f}–{r.la:.1f}"
              f"  |  bukmeikeri: {(mk_txt or '—') + (f', kopā {r.total_line:g}' if r.total_line else '')}")
        top = max(range(len(picks)), key=lambda i: picks[i].get("data_score", picks[i]["p"])) if picks else -1
        for i, x in enumerate(picks, 1):
            print(f"  {i}.{'★' if i - 1 == top else ' '}{x['bet']:<34} drošība {x['confidence']:>3}/100   "
                  f"godīgais koef. {x['fair_odds']:.2f}  (Optibet ~{x['est_odds']:.2f})   [{x['category']}]")
            record.append({"game_id": r.game_id, "game": f"{r.away}@{r.home}", "start_utc": str(r.start_utc), **x,
                           "result": None})
        games_info[r.game_id] = {
            "game_id": r.game_id, "away": r.away, "home": r.home, "away_name": r.away_name, "home_name": r.home_name,
            "away_short": r.away_short, "home_short": r.home_short, "start_utc": str(r.start_utc),
            "riga_time": r.riga_time, "robot_line": rl, "robot_total": f"vārti {r.lh:.1f}–{r.la:.1f}",
            "market_line": NB.market_text(r) or "—",
            "market_total": f"kopā {r.total_line:g}" if r.total_line else None, "injuries": [],
        }
    crazy = NB.pick_crazy(df, p)
    safe = NB.pick_safe(df, p)
    core.print_safe(safe, 34)
    print("\nCRAZY DAZY: " + "; ".join(f"{x['game']} {x['bet']} (~{x['est_odds']:.2f})" for x in crazy))
    f = core.save_night(day, record, games_info, crazy, args.min_odds, now, safe=safe,
                        safe_all=NB.pick_safe(df, p, all_games=True))
    print(f"\nSaglabāts: {f.relative_to(ROOT)}")


def cmd_kalibret(spec: Spec, args) -> None:
    """Regulation-tie boost and the model's memory, walk-forward on the season's finished games."""
    games = all_games(spec)
    done = games[games["completed"] & games["home_reg"].notna()]
    print(f"{spec.title}: {len(done)} noslēgtas spēles, neizšķirts pamatlaikā {np.mean(done['home_reg'] == done['away_reg']):.1%}")
    best = None
    for hl, rg in ((120, 16), (120, 30), (120, 60), (120, 120), (60, 30), (1e9, 30)):
        preds = []
        for day, todays in done.groupby("date"):
            m = fit_poisson(games, todays["start_utc"].min(), hl, rg)
            if m is None:
                continue
            for g in todays.itertuples():
                preds.append((*m(g.home, g.away), g.home_reg, g.away_reg))
        if not preds:
            continue
        a = np.array(preds)
        ll = float(np.mean([math.log(max(N.reg_matrix(lh, la, 0.0)[int(min(h, 15)), int(min(aw, 15))], 1e-9))
                            for lh, la, h, aw in a]))
        print(f"  half_life={hl:g} ridge={rg}: {len(a)} spēles, log-ticamība {ll:.4f}")
        if best is None or ll > best[0]:
            best = (ll, hl, rg, a)
    if best is None:
        print("Par maz datu.")
        return
    _, hl, rg, a = best
    real_tie = float(np.mean(a[:, 2] == a[:, 3]))
    tie = min(np.arange(0, 1.01, 0.05),
              key=lambda t: abs(np.mean([N.reg_matrix(lh, la, t)[N._DIAG].sum() for lh, la, _, _ in a]) - real_tie))
    p = {**DEFAULTS, "half_life": hl, "ridge": rg, "tie_infl": round(float(tie), 2)}
    params_file(spec.key).write_text(json.dumps(p, indent=2))
    print(f"  -> half_life={hl:g}, ridge={rg}, tie_infl={p['tie_infl']} (neizšķirts realitātē {real_tie:.1%})")


def main(key: str) -> int:
    global FORCE_REFRESH
    spec = SPECS[key]
    ap = argparse.ArgumentParser(description=f"{spec.title} likmju robots")
    ap.add_argument("komanda", type=lambda s: s.lower().replace("ā", "a").replace("ū", "u"),
                    choices=["likmes", "rezultati", "eksports", "kalibret"])
    ap.add_argument("--min-odds", type=float, default=2.1)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    FORCE_REFRESH = args.refresh
    L = make_league(spec)
    core.configure(L)
    if args.komanda == "kalibret":
        cmd_kalibret(spec, args)
    elif args.komanda == "eksports":
        core.export_all()
    elif args.komanda == "likmes":
        cmd_likmes(spec, L, args)
    else:
        core.cmd_rezultati(args)
    return 0
