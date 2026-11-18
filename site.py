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
    # key, title, export folder, output folder, link from the site root
    ("nba", "NBA", ROOT / "export", OUT, ""),
    ("nhl", "NHL", ROOT / "nhl" / "export", OUT / "nhl", "nhl/"),
]

NHL_NOTE = ('<p><b>Hokejā</b> visas likmes vērtē pēc <b>pamatlaika</b> (60 minūtes), tāpēc iespējams arī '
            'neizšķirts. Tikai likmes ar norādi "ieskaitot papildlaiku" ietver papildlaiku un bullīšus '
            '(bullīšu uzvarētājam pieskaita 1 vārtu).</p>')


NBA_NOTE = ('<p><b>Basketbolā</b> visas likmes vērtē pēc <b>pamatlaika</b> (48 minūtes). Tikai likmes ar norādi '
            '"ieskaitot papildlaiku" ietver papildlaiku.</p>')

STATIC_LOADER = """function start() {
  let data = { nights: [], stats: null };
  try { data = JSON.parse(document.getElementById("nba-data").textContent); } catch (e) { /* keep empty */ }
  state.nights = data.nights || [];
  state.stats = data.stats || null;
  render();
}
start();"""


def load_data(export: Path, max_nights: int = 30) -> dict:
    nights = [json.loads(f.read_text()) for f in sorted((export / "nights").glob("*.json"), reverse=True)[:max_nights]]
    stats_file = export / "stats.json"
    return {"nights": nights, "stats": json.loads(stats_file.read_text()) if stats_file.exists() else None}


def nav(current: str, depth: int) -> str:
    up = "../" * depth
    links = []
    for key, title, _, _, href in LEAGUES:
        cur = ' aria-current="page"' if key == current else ""
        links.append(f'<a href="{up}{href or "./"}"{cur}>{title}</a>')
    return '<nav class="league" aria-label="Līga">' + "".join(links) + "</nav>"


def build_one(key: str, title: str, export: Path, out_dir: Path, depth: int) -> Path:
    src = TEMPLATE.read_text()
    head, body = src.split('<div class="wrap">', 1)
    head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">', "")
    head = head.replace("<title>NBA prognozes</title>", f"<title>{title} prognozes</title>")
    body = '<div class="wrap">' + body
    body = body.replace("<!--LEAGUE-NAV-->", nav(key, depth))
    body = body.replace("<h1>NBA <span>prognozes</span></h1>", f"<h1>{title} <span>prognozes</span></h1>")
    if key == "nba":
        body = body.replace('<p id="updated"></p>', NBA_NOTE + '\n    <p id="updated"></p>')
    if key == "nhl":
        body = body.replace('<p id="updated"></p>', NHL_NOTE + '\n    <p id="updated"></p>')
        body = body.replace('"kopā": "Kopējie punkti"', '"kopā": "Kopējie vārti"')
    body, n = re.subn(r"async function start\(\) \{.*?\n\}\nstart\(\);", lambda _: STATIC_LOADER, body, flags=re.S)
    if n != 1:
        raise SystemExit("site.py: neizdevās atrast start() funkciju veidnē")
    data = json.dumps(load_data(export), ensure_ascii=False).replace("</", "<\\/")
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
    return [build_one(key, title, exp, out, depth=0 if out == OUT else 1)
            for key, title, exp, out, _ in LEAGUES]


if __name__ == "__main__":
    for f in build():
        print(f"Lapa uzbūvēta: {f.relative_to(ROOT)}")
