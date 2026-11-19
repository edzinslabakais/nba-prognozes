"""Decide which command a GitHub Actions run should execute.

GitHub's cron runs in UTC and does not know about Latvian summer time, so the
workflow is scheduled at both possible UTC times and this script keeps only the
one that matches a Riga time below on that day:
  ~21:30 LIKMES · ~07:00 REZULTĀTI (+ catch-up) · ~07:30 WhatsApp statistics · ~08:30 REZULTĀTI again

Usage: python scheduler.py "<github.event.schedule>" "<manual input>"
Prints `komanda=<likmes|rezultati|whatsapp>` (or `komanda=` to skip) for $GITHUB_OUTPUT.
"""
import sys
from datetime import datetime, timezone
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


LEAGUE_WORDS = {"nba": "nba", "nhl": "nhl", "el": "el", "euroleague": "el", "eirolīga": "el",
                "cl": "cl", "čl": "cl", "champions": "cl", "čempionu": "cl"}


def from_issue(title: str, author: str, owner: str) -> str:
    """A "refresh" request opened from the web page as a GitHub issue, e.g. "atjaunot nba".
    Only the repository owner may trigger it."""
    if not title or not author or author.lower() != (owner or "").lower():
        return ""
    words = title.strip().lower().split()
    if not words or words[0] not in ("atjaunot", "refresh", "atjauno"):
        return ""
    league = LEAGUE_WORDS.get(words[1]) if len(words) > 1 else None
    return f"rezultati_{league}" if league else "rezultati"


if __name__ == "__main__":
    import os
    cron = sys.argv[1] if len(sys.argv) > 1 else ""
    manual = sys.argv[2] if len(sys.argv) > 2 else ""
    if os.environ.get("ISSUE_TITLE"):
        cmd = from_issue(os.environ["ISSUE_TITLE"], os.environ.get("ISSUE_AUTHOR", ""), os.environ.get("REPO_OWNER", ""))
    else:
        cmd = decide(cron.strip(), manual.strip())
    print(f"komanda={cmd}")
