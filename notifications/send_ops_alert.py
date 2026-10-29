"""
CLI + importable helper for sending a one-off Telegram INFRASTRUCTURE
alert -- task failures, watchdog restarts, daily health-check summaries.
Reuses notifications/notification_engine.py's TelegramChannel/Alert
exactly as the live session's own NotificationEngine does, not a second
Telegram integration -- same bot token/chat ID, same delivery code.

Distinct from that module's usage in run_live_session.py: THAT is for
TRADE signals/entries/exits, this is for OPS events about the system
itself (a scheduled task failed, a process got restarted, is everything
healthy this morning). Both funnel through the SAME Telegram bot/chat,
just from different callers with different messages.

Usable two ways:
  - As a library: `from notifications.send_ops_alert import send_ops_alert`
    (examples/run_with_watchdog.py, examples/run_backend_watchdog.py,
    examples/health_check.py all do this).
  - As a CLI, from a .bat wrapper that can't easily import Python code
    inline: `python -m notifications.send_ops_alert --title X --message Y`
    (run_daily_universe_refresh.bat, run_daily_archive.bat, run_auto_start.bat
    all call this on a nonzero exit code from the task they just ran).

Silently returns False (never raises, CLI always exits 0) if
TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID aren't configured, or if the send
itself fails (no internet, bad token, Telegram API down) -- an ops alert
failing to send must never make the CALLING task/script look like it
failed too, and must never crash a watchdog loop that's already dealing
with a real problem.
"""
from __future__ import annotations

import argparse
import os
import sys

from config.settings import PROJECT_ROOT  # noqa: F401 -- import alone triggers load_dotenv()
from notifications.notification_engine import Alert, TelegramChannel


def send_ops_alert(title: str, message: str, urgency: str = "normal") -> bool:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return False
    channel = TelegramChannel(bot_token, chat_id)
    try:
        return channel.send(Alert(title=title, message=message, reasoning=[], urgency=urgency))
    except Exception:
        # TelegramChannel.send() already catches requests.RequestException
        # internally and returns False -- this is an extra belt-and-braces
        # catch for the CLI/watchdog callers specifically, since THIS
        # function must never be the thing that crashes a watchdog loop
        # or makes a .bat wrapper report a false failure.
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a one-off Telegram ops/infrastructure alert.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--urgency", default="normal", choices=["normal", "high"])
    args = parser.parse_args()

    ok = send_ops_alert(args.title, args.message, args.urgency)
    print(f"[send_ops_alert] {'sent' if ok else 'NOT sent (Telegram not configured, or the send itself failed)'}")
    # Always exit 0 (see module docstring) -- a .bat wrapper calling this
    # after its own task already failed must not layer a SECOND failure
    # signal on top from the alert-sending step itself.
    sys.exit(0)


if __name__ == "__main__":
    main()
