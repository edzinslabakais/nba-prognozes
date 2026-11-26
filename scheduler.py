"""Decide which command a GitHub Actions run should execute.

GitHub's cron runs in UTC and does not know about Latvian summer time, so the
workflow is scheduled at both possible UTC times and this script keeps only the
one that matches a Riga time below on that day:
  ~21:30 LIKMES · ~07:00 REZULTĀTI (+ catch-up) · ~07:30 WhatsApp statistics · ~08:30 REZULTĀTI again
  + every 5 minutes: results for leagues whose games should be over (rezultati_live)

Usage: python scheduler.py "<github.event.schedule>" "<manual input>"
Prints `komanda=<likmes|rezultati|whatsapp>` (or `komanda=` to skip) for $GITHUB_OUTPUT.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

RIGA = ZoneInfo("Europe/Riga")
# Slots sit a little before the nominal times and off the full/half hour, because GitHub delays
# scheduled runs the most at :00 and :30 (sometimes by up to an hour). Results can run several
# times safely: each pass only settles games that have finished.
TARGETS = {
    (12, 17): "likmes_eu",    # nominal 12:30 (EuroLeague, Champions League: evening games in Europe)
    (21, 17): "likmes",       # nominal 21:30 (NBA, NHL)
    (6, 47): "rezultati",     # nominal 07:00
    (7, 17): "rezultati",     # catch-up pass
    (7, 33): "whatsapp",      # nominal 07:30
    (8, 17): "rezultati",     # nominal 08:30 (late west-coast games)
}


LIVE_CRON = "*/5 * * * *"
ROOT = Path(__file__).resolve().parent
# history folder, robot, how long after the start a game is surely over (results are checked from then on)
LIVE = [("history", "bot.py", 2.25), ("nhl/history", "nhl_bot.py", 2.25), ("el/history", "el_bot.py", 1.9),
        ("cl/history", "cl_bot.py", 1.9), ("cz/history", "cz_bot.py", 2.5), ("del/history", "del_bot.py", 2.5),
        ("shl/history", "shl_bot.py", 2.5)]
LISTS = ("picks", "crazy", "safe", "safe_all")


def live_bots(now: datetime | None = None) -> list[str]:
    """Robots with a bet whose game should be over by now but has no result yet (only the last 36 h)."""
    now = now or datetime.now(timezone.utc)
    out = []
    for folder, bot, hours in LIVE:
        files = sorted((ROOT / folder).glob("*.json"))[-3:]
        due = False
        for f in files:
            try:
                rec = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            for k in LISTS:
                for b in rec.get(k) or []:
                    if b.get("result") or not b.get("start_utc"):
                        continue
                    start = datetime.fromisoformat(str(b["start_utc"]).replace(" ", "T"))
                    if start.tzinfo is None:
                        start = start.replace(tzinfo=timezone.utc)
                    if start + timedelta(hours=hours) <= now <= start + timedelta(hours=36):
                        due = True
        if due:
            out.append(bot)
    return out


def decide(cron: str, manual: str, now: datetime | None = None) -> str:
    if manual:
        return manual
    if not cron:
        return ""
    if cron == LIVE_CRON:  # every 5 minutes: results as soon as games are over
        return "rezultati_live" if live_bots(now) else ""
    minute, hour = (int(x) for x in cron.split()[:2])
    now = now or datetime.now(timezone.utc)
    slot = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    local = slot.astimezone(RIGA)
    return TARGETS.get((local.hour, local.minute), "")


if __name__ == "__main__":
    cron = sys.argv[1] if len(sys.argv) > 1 else ""
    manual = sys.argv[2] if len(sys.argv) > 2 else ""
    cmd = decide(cron.strip(), manual.strip())
    print(f"komanda={cmd}")
    print(f"bots={' '.join(live_bots()) if cmd == 'rezultati_live' else ''}")
