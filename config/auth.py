"""
Upstox OAuth2 login flow.

Upstox access tokens expire daily (~3:30am IST invalidation). This module
handles the login-code-for-token exchange and caches the token locally for
the rest of the trading day. There is no way to skip the human login step
on the free tier -- this module makes it a 2-command process instead of a
manual dance through the API docs each morning.

Usage (once per trading day, before running fetcher.py):
    python -m config.auth login       # prints the URL to open in a browser
    python -m config.auth exchange <code>   # after you're redirected, paste the code
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config.settings import (
    TOKEN_CACHE_PATH,
    UPSTOX_ANALYTICS_TOKEN,
    UPSTOX_AUTH_URL,
    UPSTOX_CLIENT_ID,
    UPSTOX_CLIENT_SECRET,
    UPSTOX_REDIRECT_URI,
    UPSTOX_TOKEN_URL,
)


class AuthError(RuntimeError):
    pass


@dataclass
class CachedToken:
    access_token: str
    obtained_at: float  # unix timestamp
    expires_at_market_open: float  # unix timestamp for next 3:30am IST invalidation

    def is_valid(self) -> bool:
        return time.time() < self.expires_at_market_open


def build_login_url() -> str:
    if not UPSTOX_CLIENT_ID:
        raise AuthError(
            "UPSTOX_CLIENT_ID is not set. Register an app at "
            "https://developer.upstox.com/ and add credentials to .env"
        )
    return (
        f"{UPSTOX_AUTH_URL}?client_id={UPSTOX_CLIENT_ID}"
        f"&redirect_uri={UPSTOX_REDIRECT_URI}&response_type=code"
    )


def exchange_code_for_token(auth_code: str) -> CachedToken:
    """
    Exchange the authorization code (obtained after the browser login
    redirect) for an access token, and cache it locally.
    """
    if not UPSTOX_CLIENT_SECRET:
        raise AuthError("UPSTOX_CLIENT_SECRET is not set. Check your .env file.")

    resp = requests.post(
        UPSTOX_TOKEN_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        data={
            "code": auth_code,
            "client_id": UPSTOX_CLIENT_ID,
            "client_secret": UPSTOX_CLIENT_SECRET,
            "redirect_uri": UPSTOX_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    if resp.status_code != 200:
        raise AuthError(f"Token exchange failed ({resp.status_code}): {resp.text}")

    payload = resp.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise AuthError(f"No access_token in response: {payload}")

    # Upstox tokens are valid until ~3:30am IST the next day. We store a
    # conservative expiry of 20 hours from issuance so the engine refuses
    # to use a stale token rather than silently failing mid-backtest.
    token = CachedToken(
        access_token=access_token,
        obtained_at=time.time(),
        expires_at_market_open=time.time() + 20 * 3600,
    )
    _save_token(token)
    return token


def _save_token(token: CachedToken) -> None:
    TOKEN_CACHE_PATH.write_text(json.dumps(token.__dict__))


def load_cached_token() -> Optional[CachedToken]:
    if not TOKEN_CACHE_PATH.exists():
        return None
    data = json.loads(TOKEN_CACHE_PATH.read_text())
    token = CachedToken(**data)
    return token if token.is_valid() else None


def get_valid_token() -> str:
    """
    Returns a valid access token or raises AuthError telling the user
    exactly what command to run.

    Checks in order:
      1. UPSTOX_ANALYTICS_TOKEN from .env -- the long-lived (1-year),
         read-only token Upstox now offers, generated once from the
         Developer Apps page with no OAuth dance required. Preferred
         path for market/historical data.
      2. Cached daily OAuth token (the old flow) -- fallback for anyone
         who hasn't switched to the Analytics Token yet, or who needs
         account/portfolio endpoints the Analytics Token doesn't cover
         without a static IP.
    """
    if UPSTOX_ANALYTICS_TOKEN:
        return UPSTOX_ANALYTICS_TOKEN

    cached = load_cached_token()
    if cached is not None:
        return cached.access_token
    raise AuthError(
        "No Upstox token available. Either set UPSTOX_ANALYTICS_TOKEN in "
        ".env (generate one from the Analytics tab on the Upstox Developer "
        "Apps page -- no login flow needed, valid for 1 year), or use the "
        "daily OAuth flow:\n"
        "  python -m config.auth login\n"
        "then open the URL, log in, copy the 'code' param from the redirect "
        "URL, and run:\n"
        "  python -m config.auth exchange <code>"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "login":
        print("Open this URL, log in, and note the 'code' parameter in the redirect:")
        print(build_login_url())
    elif cmd == "exchange":
        if len(sys.argv) < 3:
            print("Usage: python -m config.auth exchange <code>")
            sys.exit(1)
        tok = exchange_code_for_token(sys.argv[2])
        print(f"Token cached, valid until ~{time.ctime(tok.expires_at_market_open)}")
    else:
        print(__doc__)
