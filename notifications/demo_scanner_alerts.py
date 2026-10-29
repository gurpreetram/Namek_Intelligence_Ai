"""
Demo: runs the scanner across a synthetic universe and sends a
notification for every qualifying result -- this is how you'd wire it up
for real use once Upstox data is flowing: run the scanner periodically
(e.g. every few minutes during market hours), and notify_scan_result()
for each hit.

To actually use Telegram for real alerts, replace the channels list in
main() with:
    NotificationEngine([ConsoleChannel(), TelegramChannel(bot_token, chat_id)])
using your own bot token and chat ID (see TelegramChannel's docstring in
notifications/notification_engine.py for the one-time setup).

Run: python -m notifications.demo_scanner_alerts
"""
from __future__ import annotations

import pandas as pd

from detectors.scanner import scan_universe
from examples.synthetic_pullback_test import generate_trend_with_pullback_day
from notifications.notification_engine import ConsoleChannel, FileLogChannel, NotificationEngine


def main() -> None:
    days = [f"2026-06-{d:02d}" for d in range(1, 6)]
    universe = {
        "TRENDER": pd.concat([generate_trend_with_pullback_day(d, seed=1) for d in days]),
    }

    engine = NotificationEngine([ConsoleChannel(), FileLogChannel()])

    scan_times = pd.date_range("2026-06-05 09:40", "2026-06-05 15:00", freq="10min")
    notified_already = set()

    for as_of in scan_times:
        results = scan_universe(universe, as_of, min_score=0.5)
        for r in results:
            key = f"{r.symbol}_{r.setup_type}_{as_of.date()}"
            if key in notified_already:
                continue  # don't spam the same setup every 10 minutes
            engine.notify_scan_result(r)
            notified_already.add(key)

    print(f"\n{len(notified_already)} unique alert(s) sent this session.")


if __name__ == "__main__":
    main()
