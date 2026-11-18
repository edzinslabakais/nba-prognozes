"""Build the public web page (GitHub Pages) from the same design as web/nba-prognozes.html.

The claude.ai version loads data from a Claude-only database; this public version has
the data built into the page, so it needs no login. Output: site/index.html

  python site.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "web" / "nba-prognozes.html"
EXPORT = ROOT / "export"
OUT = ROOT / "site"

STATIC_LOADER = """function start() {
  let data = { nights: [], stats: null };
  try { data = JSON.parse(document.getElementById("nba-data").textContent); } catch (e) { /* keep empty */ }
  state.nights = data.nights || [];
  state.stats = data.stats || null;
  render();
}
start();"""


def load_data(max_nights: int = 30) -> dict:
    nights = []
    for f in sorted((EXPORT / "nights").glob("*.json"), reverse=True)[:max_nights]:
        nights.append(json.loads(f.read_text()))
    stats_file = EXPORT / "stats.json"
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else None
    return {"nights": nights, "stats": stats}


def build() -> Path:
    src = TEMPLATE.read_text()
    head, body = src.split('<div class="wrap">', 1)
    head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">', "")
    body = '<div class="wrap">' + body
    body, n = re.subn(r"async function start\(\) \{.*?\n\}\nstart\(\);", lambda _: STATIC_LOADER,
                      body, flags=re.S)
    if n != 1:
        raise SystemExit("site.py: neizdevās atrast start() funkciju veidnē")
    data = json.dumps(load_data(), ensure_ascii=False).replace("</", "<\\/")
    html = (
        "<!doctype html>\n<html lang=\"lv\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        "<meta name=\"description\" content=\"Katras nakts NBA likmes ar koeficientu 2.1+, drošība un rezultāti.\">\n"
        f"{head.strip()}\n"
        "<style>body { margin: 0; -webkit-text-size-adjust: 100%; } [hidden] { display: none !important; }</style>\n"
        "</head>\n<body>\n"
        f"<script id=\"nba-data\" type=\"application/json\">{data}</script>\n"
        f"{body.strip()}\n</body>\n</html>\n"
    )
    OUT.mkdir(exist_ok=True)
    out = OUT / "index.html"
    out.write_text(html)
    (OUT / ".nojekyll").write_text("")
    return out


if __name__ == "__main__":
    print(f"Lapa uzbūvēta: {build().relative_to(ROOT)}")
