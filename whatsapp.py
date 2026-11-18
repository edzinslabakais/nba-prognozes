"""Morning WhatsApp message with the statistics only (sent ~07:30 Riga time by GitHub Actions).

Uses CallMeBot (free WhatsApp API for personal notifications). One-time setup:
  1. Add the CallMeBot WhatsApp number shown on
     https://www.callmebot.com/blog/free-api-whatsapp-messages/ to your phone contacts.
  2. Send it this WhatsApp message:  I allow callmebot to send me messages
  3. You get a reply with your apikey.
  4. GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository secret:
       WHATSAPP_PHONE   your number with country code, e.g. +37120000000
       WHATSAPP_APIKEY  the apikey from step 3
Without these secrets nothing is sent.

  python whatsapp.py            # send
  python whatsapp.py --dry-run  # only print the message
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
LEAGUES = [("NBA", ROOT / "export"), ("NHL", ROOT / "nhl" / "export")]


def _units(x: float) -> str:
    return f"{x:+.2f}".replace(".", ",") + " v."


def _cell(s: dict | None) -> str:
    if not s or not s.get("settled"):
        return "–"
    return f"{s['hits']}/{s['settled']} ({_units(s['profit_units'])})"


def _night_label(us_date: str) -> str:
    from datetime import date, timedelta
    d = date.fromisoformat(us_date) + timedelta(days=1)
    return f"{d:%d.%m.}"


def build_message() -> str:
    lines = []
    for title, export in LEAGUES:
        stats_f = export / "stats.json"
        if not stats_f.exists():
            continue
        stats = json.loads(stats_f.read_text())
        settled_nights = [n for n in stats.get("nights", []) if n.get("settled")]
        lines.append(f"*{title}*")
        if settled_nights:
            n = settled_nights[-1]
            waiting = n["picks"] - n["settled"]
            extra = f", {waiting} gaida" if waiting else ""
            lines.append(f"Nakts uz {_night_label(n['date'])}: visas {_cell(n)} · ★ {_cell(n.get('top'))} · "
                         f"drošās {_cell(n.get('safe'))} · Crazy {_cell(n.get('crazy'))}{extra}")
        m = (stats.get("months") or [None])[0]
        if m:
            lines.append(f"{m['label']}: visas {_cell(m['all'])} · ★ {_cell(m['top'])} · drošās {_cell(m.get('safe'))} · "
                         f"Crazy {_cell(m['crazy'])}")
        lines.append(f"Kopā: visas {_cell(stats.get('overall'))} · ★ {_cell(stats.get('top_overall'))} · "
                     f"drošās {_cell(stats.get('safe_overall'))} · Crazy {_cell(stats.get('crazy_overall'))}")
        lines.append("")
    if not lines:
        return ""
    head = "📊 Prognožu statistika (uzvarēts/notikušas, vienības pie 1 v. likmes)"
    return head + "\n\n" + "\n".join(lines).strip() + "\nhttps://edzinslabakais.github.io/nba-prognozes/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    text = build_message()
    if not text:
        print("Nav statistikas, ko sūtīt.")
        return 0
    print(text)
    phone = os.environ.get("WHATSAPP_PHONE", "").strip()
    key = os.environ.get("WHATSAPP_APIKEY", "").strip()
    if args.dry_run:
        return 0
    if not phone or not key:
        print("\nWhatsApp nav iestatīts (WHATSAPP_PHONE / WHATSAPP_APIKEY) — ziņa netiek sūtīta.")
        return 0
    r = requests.get("https://api.callmebot.com/whatsapp.php",
                     params={"phone": phone, "text": text, "apikey": key}, timeout=60)
    ok = r.ok and "error" not in r.text.lower()
    print("\nZiņa nosūtīta." if ok else f"\nWhatsApp kļūda: {r.status_code} {r.text[:300]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
