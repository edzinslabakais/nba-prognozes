"""Decide which command a GitHub Actions run should execute.

GitHub's cron runs in UTC and does not know about Latvian summer time, so the
workflow is scheduled at both possible UTC times and this script keeps only the
one that matches a Riga time below on that day:
  21:30 LIKMES · 07:00 REZULTĀTI · 07:30 WhatsApp statistics · 08:30 REZULTĀTI again (late games)

Usage: python scheduler.py "<github.event.schedule>" "<manual input>"
Prints `komanda=<likmes|rezultati|whatsapp>` (or `komanda=` to skip) for $GITHUB_OUTPUT.
"""
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

RIGA = ZoneInfo("Europe/Riga")
TARGETS = {(21, 30): "likmes", (7, 0): "rezultati", (7, 30): "whatsapp", (8, 30): "rezultati"}


def decide(cron: str, manual: str, now: datetime | None = None) -> str:
    if manual:
        return manual
    if not cron:
        return ""
    minute, hour = (int(x) for x in cron.split()[:2])
    now = now or datetime.now(timezone.utc)
    slot = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    local = slot.astimezone(RIGA)
    return TARGETS.get((local.hour, local.minute), "")


if __name__ == "__main__":
    cron = sys.argv[1] if len(sys.argv) > 1 else ""
    manual = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"komanda={decide(cron.strip(), manual.strip())}")
