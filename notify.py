"""Send export/message.txt to Telegram, if the bot is configured.

Needs two environment variables (GitHub: Settings -> Secrets and variables -> Actions):
  TELEGRAM_BOT_TOKEN  token from @BotFather
  TELEGRAM_CHAT_ID    chat id(s), comma-separated (your own chat and/or the friends' group)
Without them it does nothing, so the workflow still works.
"""
import os
import sys
from pathlib import Path

import requests

MESSAGE = Path(__file__).resolve().parent / "export" / "message.txt"
LIMIT = 3900  # Telegram max is 4096 characters per message


def chunks(text: str):
    part = ""
    for line in text.splitlines(keepends=True):
        if len(part) + len(line) > LIMIT:
            yield part
            part = ""
        part += line
    if part.strip():
        yield part


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chats = [c.strip() for c in os.environ.get("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
    if not MESSAGE.exists():
        print("Nav jaunas ziņas, ko sūtīt.")
        return 0
    if not token or not chats:
        print("Telegram nav iestatīts (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — ziņa netiek sūtīta.")
        return 0
    text = MESSAGE.read_text()
    failed = False
    for chat in chats:
        for part in chunks(text):
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": part, "disable_web_page_preview": True},
                              timeout=30)
            if not r.ok:
                print(f"Telegram kļūda čatam {chat}: {r.status_code} {r.text[:200]}")
                failed = True
                break
    print("Ziņa nosūtīta." if not failed else "Daļa ziņu netika nosūtītas.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
