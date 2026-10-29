"""
Notification Engine: dispatches alerts when the scanner finds a
qualifying setup, or when a paper/live trade enters or exits. Pluggable
channels -- add more without touching the dispatch logic.

Channels included:
  - ConsoleChannel: prints to terminal. Always works, zero setup, good
    default while testing.
  - FileLogChannel: appends to a local log file -- an audit trail
    independent of the trade journal (this logs NOTIFICATIONS sent, not
    trade outcomes; the two can diverge, e.g. if a channel fails to
    deliver, which is exactly why logging the attempt separately matters).
  - TelegramChannel: real push notifications to your phone via a
    Telegram bot -- free, no paid service, and what most retail algo
    traders actually use for this. Needs a bot token + chat ID (see
    setup instructions in the class docstring below) -- this project
    can't create those for you since they're tied to your own Telegram
    account, but the integration itself is fully functional once you
    have them.

Design: NotificationEngine holds a list of channels and broadcasts to
all of them. A channel failing to send (e.g. no internet, bad token)
does NOT stop the others from receiving the alert, and the failure
itself gets logged rather than silently swallowed.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

from config.settings import PROJECT_ROOT

DEFAULT_LOG_PATH = PROJECT_ROOT / "notifications" / "alerts.log"


@dataclass
class Alert:
    title: str
    message: str
    reasoning: list[str]
    urgency: str = "normal"  # "normal" | "high" -- channels can use this to
    # decide formatting (e.g. Telegram could bold/emoji high-urgency alerts)


class NotificationChannel(ABC):
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Returns True if delivery succeeded, False otherwise. Must not
        raise -- a channel failure should never crash the caller; catch
        and return False instead."""
        raise NotImplementedError


class ConsoleChannel(NotificationChannel):
    def send(self, alert: Alert) -> bool:
        marker = "🔴" if alert.urgency == "high" else "🔔"
        lines = [f"\n{marker} {alert.title}", f"   {alert.message}"]
        lines += [f"   - {reason}" for reason in alert.reasoning]
        for line in lines:
            try:
                print(line)
            except UnicodeEncodeError:
                # Windows cmd.exe/PowerShell default to a legacy codepage
                # (cp1252) that can't encode the emoji marker -- without
                # this, the whole send() raises and NotificationEngine
                # reports a false "delivery failure" over one cosmetic
                # character, even though every other channel succeeded.
                encoding = sys.stdout.encoding or "ascii"
                print(line.encode(encoding, "replace").decode(encoding))
        return True


class FileLogChannel(NotificationChannel):
    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_LOG_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> bool:
        try:
            with open(self.path, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] {alert.title}: {alert.message}\n")
                for reason in alert.reasoning:
                    f.write(f"    - {reason}\n")
            return True
        except OSError:
            return False


class TelegramChannel(NotificationChannel):
    """
    Setup (one-time, free, ~5 minutes):
      1. Open Telegram, search for "@BotFather", send /newbot, follow the
         prompts -- it gives you a bot token like "123456:ABC-DEF...".
      2. Send any message to your new bot (search its username and message it).
      3. Visit https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates in a
         browser -- find "chat":{"id": ...} in the response, that's your chat_id.
      4. Pass both below.
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(self, alert: Alert) -> bool:
        emoji = "🔴" if alert.urgency == "high" else "🔔"
        text = f"{emoji} *{alert.title}*\n{alert.message}"
        if alert.reasoning:
            text += "\n\n" + "\n".join(f"• {r}" for r in alert.reasoning)

        try:
            resp = requests.post(
                self.api_url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False


class NotificationEngine:
    def __init__(self, channels: list[NotificationChannel] | None = None):
        self.channels = channels or [ConsoleChannel(), FileLogChannel()]
        self._log = FileLogChannel()  # always log delivery status, even if
        # the caller didn't include FileLogChannel in their active channels

    def notify(self, alert: Alert) -> dict[str, bool]:
        """Broadcasts to all channels. Returns {channel_class_name: success}
        so the caller can see which channels actually delivered."""
        results = {}
        for channel in self.channels:
            channel_name = type(channel).__name__
            try:
                success = channel.send(alert)
            except Exception:
                success = False  # a channel must never crash the engine
            results[channel_name] = success
            if not success:
                self._log.send(Alert(
                    title="DELIVERY FAILED",
                    message=f"{channel_name} failed to deliver: {alert.title}",
                    reasoning=[],
                ))
        return results

    def notify_scan_result(self, scan_result) -> dict[str, bool]:
        """Convenience: build and send an Alert directly from a
        detectors.scanner.ScanResult."""
        urgency = "high" if scan_result.score >= 0.75 else "normal"
        alert = Alert(
            title=f"{scan_result.symbol}: {scan_result.setup_type} ({scan_result.direction})",
            message=f"Score {scan_result.score:.2f}",
            reasoning=scan_result.reasoning,
            urgency=urgency,
        )
        return self.notify(alert)
