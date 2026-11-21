"""Build the public web pages (GitHub Pages) from the design in web/nba-prognozes.html.

The claude.ai version loads data from a Claude-only database; these public pages have the
data built in, so they need no login. Output: site/index.html (NBA) and site/nhl/index.html (NHL).

  python site.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "web" / "nba-prognozes.html"
OUT = ROOT / "site"

LEAGUES = [
    # key, nav label, page title, export folder, output folder, link from the site root, day mode, picks time
    ("nba", "NBA", "NBA", ROOT / "export", OUT, "", False, (21, 30)),
    ("nhl", "NHL", "NHL", ROOT / "nhl" / "export", OUT / "nhl", "nhl/", False, (21, 30)),
    ("el", "EL", "EuroLeague", ROOT / "el" / "export", OUT / "el", "el/", True, (12, 30)),
    ("cl", "ČL", "Čempionu līgas", ROOT / "cl" / "export", OUT / "cl", "cl/", True, (12, 30)),
    ("ch", "CH", "Championship", ROOT / "ch" / "export", OUT / "ch", "ch/", True, (12, 30)),
]
SIMPLE = {"ch"}  # only the day's best bets + Crazy Dazy, and kept out of the combined statistics

NHL_NOTE = ('<p data-i18n-html="noteNhl"><b>Hokejā</b> visas likmes vērtē pēc <b>pamatlaika</b> (60 minūtes), tāpēc iespējams arī '
            'neizšķirts. Tikai likmes ar norādi "ieskaitot papildlaiku" ietver papildlaiku un bullīšus '
            '(bullīšu uzvarētājam pieskaita 1 vārtu).</p>')


NBA_NOTE = ('<p data-i18n-html="noteNba"><b>Basketbolā</b> visas likmes vērtē pēc <b>pamatlaika</b> (48 minūtes). Tikai likmes ar norādi '
            '"ieskaitot papildlaiku" ietver papildlaiku.</p>')

STATIC_LOADER = """function start() {
  let data = { nights: [], stats: null };
  try { data = JSON.parse(document.getElementById("nba-data").textContent); } catch (e) { /* keep empty */ }
  state.nights = data.nights || [];
  state.stats = data.stats || null;
  state.combined = data.combined || [];
  render();
}
start();"""


def load_data(export: Path, max_nights: int = 60) -> dict:
    nights = [json.loads(f.read_text()) for f in sorted((export / "nights").glob("*.json"), reverse=True)[:max_nights]]
    stats_file = export / "stats.json"
    return {"nights": nights, "stats": json.loads(stats_file.read_text()) if stats_file.exists() else None}


def nav(current: str, depth: int) -> str:
    up = "../" * depth
    links = []
    for key, label, href in [(x[0], x[1], x[5]) for x in LEAGUES] + [("drosas", "1.7+", "drosas/"),
                                                                       ("stats", "Statistika", "stats/")]:
        cur = ' aria-current="page"' if key == current else ""
        links.append(f'<a href="{up}{href or "./"}" data-nav="{key}"{cur}>{label}</a>')
    return '<nav class="league" aria-label="Līga">' + "".join(links) + "</nav>"


EL_NOTE = ('<p data-i18n-html="noteEl"><b>EuroLeague</b>: bukmeikeru koeficientu šai līgai nav, tāpēc varbūtības '
           'nāk no robota modeļa (komandu reitingi, mājas laukums, atpūta), kas kalibrēts uz 2025-26 sezonu '
           '(uzvarētāju uzmin ~62 % gadījumu). Visas likmes vērtē pēc <b>pamatlaika</b> (40 minūtes).</p>')
CL_NOTE = ('<p data-i18n-html="noteCl"><b>Futbolā</b> visas likmes vērtē pēc <b>90 minūtēm</b> (bez pagarinājuma un '
           'pēcspēles sitieniem). Sagaidāmie vārti ir aprēķināti no bukmeikeru koeficientiem, un no tiem tiek '
           'novērtētas visas likmes, arī "abas komandas gūst vārtus" un "uzvara vai neizšķirts".</p>')
CH_NOTE = ('<p data-i18n-html="noteCh"><b>Championship</b> (Anglijas otrā līga): no katras spēles drošākā likme ar koeficientu '
           '2.1+, parādītas līdz <b>10 visdrošākajām</b>, kā arī Crazy Dazy. Sagaidāmie vārti ir aprēķināti no bukmeikeru '
           'koeficientiem, likmes vērtē pēc <b>90 minūtēm</b>. Championship netiek ieskaitīts kopējā statistikā.</p>')
NOTES_BY_LEAGUE = {"nba": "NBA_NOTE", "nhl": "NHL_NOTE", "el": "EL_NOTE", "cl": "CL_NOTE", "ch": "CH_NOTE"}


def combined_data() -> dict:
    """Every league's per-night summaries for the overall statistics page (grouped by week/month/year there)."""
    rows, updated = [], []
    for key, label, _, export, _, _, _, _ in LEAGUES:
        if key in SIMPLE:
            continue
        f = export / "stats.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        updated.append(s.get("updated_at") or "")
        for n in s.get("nights", []):
            rows.append({"league": key, "title": label, "date": n["date"],
                         "all": {k: n.get(k, 0) for k in ("picks", "settled", "hits", "pushes", "expected_hits", "profit_units")},
                         "top": n.get("top"), "safe": n.get("safe"), "crazy": n.get("crazy")})
    return {"nights": [], "stats": {"updated_at": max(updated) if updated else None}, "combined": rows}


def safe_page_data(max_days: int = 60) -> dict:
    """Drošās likmes page: the safest bet (odds >= 1.7) of every game in every league, per day."""
    days: dict[str, list] = {}
    updated = []
    for key, label, title, export, *_ in LEAGUES:
        sf = export / "stats.json"
        if sf.exists():
            updated.append(json.loads(sf.read_text()).get("updated_at") or "")
        for f in sorted((export / "nights").glob("*.json")) if (export / "nights").exists() else []:
            n = json.loads(f.read_text())
            for b in n.get("safe_all") or []:
                days.setdefault(n["date"], []).append({**b, "league": key, "league_title": title if key != "cl" else "Čempionu līga"})
    nights = [{"date": d, "bets": sorted(v, key=lambda b: (b.get("start_utc") or "", b["league"]))}
              for d, v in sorted(days.items(), reverse=True)[:max_days]]
    return {"nights": nights, "stats": {"updated_at": max(updated) if updated else None}}


def build_one(key: str, title: str, export: Path | None, out_dir: Path, depth: int, day_mode: bool = False,
              picks_time=(21, 30), data_override: dict | None = None) -> Path:
    src = TEMPLATE.read_text()
    head, body = src.split('<div class="wrap">', 1)
    head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">', "")
    head = head.replace("<title>NBA prognozes</title>", f"<title>{title} prognozes</title>")
    body = '<div class="wrap">' + body
    body = body.replace("<!--LEAGUE-NAV-->", nav(key, depth))
    if key == "drosas":
        head = head.replace("<title>1.7+ likmes prognozes</title>", "<title>1.7+ likmes</title>")
        body = body.replace('<h1>NBA <span data-i18n="h1">prognozes</span></h1>', '<h1>1.7+ <span>likmes</span></h1>')
    if key == "stats":
        head = head.replace("<title>Statistika prognozes</title>", "<title>Kopējā statistika</title>")
        body = body.replace('<h1>NBA <span data-i18n="h1">prognozes</span></h1>', '<h1>Kopējā <span>statistika</span></h1>')
    body = body.replace('<h1>NBA <span data-i18n="h1">', f'<h1>{title} <span data-i18n="h1">')
    body = body.replace('const LEAGUE_KEY = "nba"; const DAY_MODE = false; const LEAGUE_TITLE = "NBA";',
                        f'const LEAGUE_KEY = "{key}"; const DAY_MODE = {"true" if day_mode else "false"}; '
                        f'const LEAGUE_TITLE = "{title}";')
    body = body.replace('{ h: 21, m: 30, name: "slotPicks" }', f'{{ h: {picks_time[0]}, m: {picks_time[1]}, name: "slotPicks" }}')
    if key in NOTES_BY_LEAGUE:
        body = body.replace('<p id="updated"></p>', globals()[NOTES_BY_LEAGUE[key]] + '\n    <p id="updated"></p>')
    if key in SIMPLE:
        body = body.replace("const SIMPLE = false;", "const SIMPLE = true;")
    body, n = re.subn(r"async function start\(\) \{.*?\n\}\nstart\(\);", lambda _: STATIC_LOADER, body, flags=re.S)
    if n != 1:
        raise SystemExit("site.py: neizdevās atrast start() funkciju veidnē")
    data = json.dumps(data_override if data_override is not None else load_data(export),
                      ensure_ascii=False).replace("</", "<\\/")
    html = (
        "<!doctype html>\n<html lang=\"lv\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        f"<meta name=\"description\" content=\"Katras nakts {title} likmes ar koeficientu 2.1+, drošība un rezultāti.\">\n"
        f"{head.strip()}\n"
        "<style>body { margin: 0; -webkit-text-size-adjust: 100%; } [hidden] { display: none !important; }</style>\n"
        "</head>\n<body>\n"
        f"<script id=\"nba-data\" type=\"application/json\">{data}</script>\n"
        f"{body.strip()}\n</body>\n</html>\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(html)
    return out


def build() -> list[Path]:
    OUT.mkdir(exist_ok=True)
    (OUT / ".nojekyll").write_text("")
    pages = [build_one(key, title, exp, out, 0 if out == OUT else 1, day_mode, picks_time)
             for key, _, title, exp, out, _, day_mode, picks_time in LEAGUES]
    pages.append(build_one("drosas", "1.7+ likmes", None, OUT / "drosas", 1, True, (12, 30), data_override=safe_page_data()))
    pages.append(build_one("stats", "Statistika", None, OUT / "stats", 1, data_override=combined_data()))
    return pages


if __name__ == "__main__":
    for f in build():
        print(f"Lapa uzbūvēta: {f.relative_to(ROOT)}")
