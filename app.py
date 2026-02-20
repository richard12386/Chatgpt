from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import csv
import hmac
import io
import math
import os
import random
import re
import secrets
import smtplib
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import wraps
from threading import Event, Lock, Thread
from typing import Optional
from urllib.parse import urlparse

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from flask import Flask, jsonify, make_response, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from yfinance.exceptions import YFRateLimitError

app = Flask(__name__)
app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "0") == "1"
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("FLASK_SECURE_COOKIE", "0") == "1"
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024
app.permanent_session_lifetime = timedelta(hours=12)
DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")

PERIOD_OPTIONS = [
    ("1d", "1D"),
    ("5d", "5D"),
    ("1mo", "1M"),
    ("6mo", "6M"),
    ("ytd", "YTD"),
    ("1y", "1Y"),
    ("5y", "5Y"),
    ("max", "All"),
]
DAY_THRESHOLD_OPTIONS = [3, 5, 8, 13, 21, 34, 55]
WEEK_THRESHOLD_OPTIONS = [3, 5, 8, 13, 21, 34, 55]
THRESHOLD_OPTIONS = WEEK_THRESHOLD_OPTIONS
GRANULARITY_OPTIONS = [("week", "Week End"), ("day", "Every Day")]
CACHE_TTL_SECONDS = 900
STALE_CACHE_MAX_AGE_SECONDS = 21600
SEARCH_CACHE_TTL_SECONDS = 43200
AUTO_REFRESH_INTERVAL_SECONDS = 60
MAX_RECENT_SEARCHES = 4
MAX_SEARCH_HISTORY = 200
LISTINGS_CACHE_TTL_SECONDS = 21600
TWO_FA_CODE_TTL_SECONDS = 600
TWO_FA_MAX_ATTEMPTS = 5
TWO_FA_FORCE_EMAIL = os.getenv("TWO_FA_FORCE_EMAIL", "1") == "1"
TWO_FA_DEV_SHOW_CODE = os.getenv("TWO_FA_DEV_SHOW_CODE", "1") == "1"

ANALYSIS_CACHE: dict[tuple[str, str, int, str], tuple[float, "AnalysisResult"]] = {}
SEARCH_CACHE: dict[str, tuple[float, str]] = {}
LISTINGS_CACHE: dict[str, tuple[float, list[dict[str, str]]]] = {}
RECENT_SEARCHES: deque[dict[str, str]] = deque(maxlen=MAX_RECENT_SEARCHES)
SEARCH_HISTORY: deque[dict[str, str]] = deque(maxlen=MAX_SEARCH_HISTORY)
LAST_MARKET_OVERVIEW: Optional[dict] = None
AUTO_REFRESH_STOP = Event()
AUTO_REFRESH_LOCK = Lock()
AUTO_REFRESH_THREAD: Optional[Thread] = None
AUTO_REFRESH_LAST_RUN_AT: Optional[float] = None
REQUEST_RATE_BUCKETS: dict[str, deque[float]] = {}
REQUEST_RATE_LOCK = Lock()

# Global-ish liquid tickers (US + EU + Asia) in Yahoo symbol format.
MARKET_URLS = {
    "most_active": "https://finance.yahoo.com/markets/stocks/most-active/",
    "gainers": "https://finance.yahoo.com/markets/stocks/gainers/",
    "losers": "https://finance.yahoo.com/markets/stocks/losers/",
}
FINVIZ_URLS = {
    "most_active": "https://finviz.com/screener.ashx?v=111&s=ta_mostactive",
    "gainers": "https://finviz.com/screener.ashx?v=111&s=ta_topgainers",
    "losers": "https://finviz.com/screener.ashx?v=111&s=ta_toplosers",
}
TRADINGVIEW_URLS = {
    "most_active": "https://www.tradingview.com/markets/stocks-usa/market-movers-active/",
    "gainers": "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/",
    "losers": "https://www.tradingview.com/markets/stocks-usa/market-movers-losers/",
}
GOOGLE_FINANCE_URLS = {
    "most_active": "https://www.google.com/finance/markets/most-active",
    "gainers": "https://www.google.com/finance/markets/gainers",
    "losers": "https://www.google.com/finance/markets/losers",
}
CMC_SLUG_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "bnb",
    "SOL": "solana",
    "XRP": "xrp",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche",
    "DOT": "polkadot-new",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "TRX": "tron",
    "MATIC": "polygon",
}

COUNTRY_CODE_MAP = {
    "USA": "United States",
    "US": "United States",
    "GB": "United Kingdom",
    "DE": "Germany",
    "FR": "France",
    "ES": "Spain",
    "IT": "Italy",
    "NL": "Netherlands",
    "SE": "Sweden",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "CH": "Switzerland",
    "AT": "Austria",
    "BE": "Belgium",
    "IE": "Ireland",
    "CA": "Canada",
    "AU": "Australia",
    "NZ": "New Zealand",
    "JP": "Japan",
    "CN": "China",
    "HK": "Hong Kong",
    "SG": "Singapore",
    "IN": "India",
    "KR": "South Korea",
    "TW": "Taiwan",
    "BR": "Brazil",
    "MX": "Mexico",
    "ZA": "South Africa",
}
SYMBOL_COUNTRY_SUFFIX_MAP = {
    "PA": "France",
    "DE": "Germany",
    "AS": "Netherlands",
    "MI": "Italy",
    "MC": "Spain",
    "L": "United Kingdom",
    "TO": "Canada",
    "V": "Canada",
    "AX": "Australia",
    "HK": "Hong Kong",
    "T": "Japan",
    "KS": "South Korea",
    "KQ": "South Korea",
    "SS": "China",
    "SZ": "China",
    "TW": "Taiwan",
    "NS": "India",
    "BO": "India",
    "SI": "Singapore",
}


@dataclass
class AnalysisResult:
    ticker: str
    company_name: Optional[str]
    company_state: Optional[str]
    currency: Optional[str]
    exchange: Optional[str]
    current_price: Optional[float]
    last_dividend: Optional[float]
    last_dividend_date: Optional[str]
    threshold_pct: int
    volatile_weeks: list[dict[str, str | float]]
    weekly_price_points: list[dict[str, str | float | bool]]
    volatility_week_count: int
    max_weekly_gain_pct: Optional[float]
    max_weekly_loss_pct: Optional[float]
    granularity: str


@dataclass
class CryptoPriceResult:
    symbol: str
    display_symbol: str
    price_usd: Optional[float]
    source_prices: dict[str, Optional[float]]
    source_symbols: dict[str, str]
    as_of: str


@dataclass
class CryptoChartResult:
    symbol: str
    display_symbol: str
    name: str
    selected_range: str
    current_price_usd: Optional[float]
    change_pct: Optional[float]
    points: list[dict[str, float | str]]
    as_of: str


CRYPTO_RANGE_OPTIONS = [
    ("24h", "24h"),
    ("1w", "1W"),
    ("1m", "1M"),
    ("1y", "1Y"),
    ("all", "All"),
]


def _parse_price_text(raw: str) -> Optional[float]:
    if not raw:
        return None
    match = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", raw.replace("$", ""))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _normalize_crypto_symbol(query: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]", "", (query or "").upper())
    if clean.endswith("USDT"):
        clean = clean[:-4]
    if clean.endswith("USD"):
        clean = clean[:-3]
    return clean


def _fetch_crypto_price_binance(base_symbol: str) -> tuple[Optional[float], str]:
    pair = f"{base_symbol}USDT"
    try:
        payload = _request_json_with_retry(
            "https://api.binance.com/api/v3/ticker/price",
            {"symbol": pair},
            retries=2,
            timeout=8,
        )
        price = float(payload.get("price")) if isinstance(payload, dict) and payload.get("price") else None
        return price, pair
    except Exception:
        return None, pair


def _normalize_crypto_range(raw_range: str) -> str:
    allowed = {x[0] for x in CRYPTO_RANGE_OPTIONS}
    return raw_range if raw_range in allowed else "24h"


def _resolve_crypto_input(query: str) -> tuple[str, str]:
    raw = (query or "").strip()
    if not raw:
        raise ValueError("Please enter a crypto symbol or coin name.")

    symbol_guess = _normalize_crypto_symbol(raw)
    raw_upper = raw.upper()
    # Treat as direct symbol mostly when user enters short uppercase ticker-like input.
    is_symbol_like = bool(re.fullmatch(r"[A-Z0-9]{2,10}", raw_upper))
    if is_symbol_like and (raw == raw_upper or raw_upper.endswith("USD") or raw_upper.endswith("USDT")):
        return symbol_guess, symbol_guess

    try:
        payload = _request_json_with_retry(
            "https://api.coingecko.com/api/v3/search",
            {"query": raw},
            retries=2,
            timeout=10,
        )
        coins = payload.get("coins", []) if isinstance(payload, dict) else []
        if coins:
            exact = None
            raw_l = raw.lower()
            for coin in coins:
                if str(coin.get("name", "")).lower() == raw_l or str(coin.get("symbol", "")).lower() == raw_l:
                    exact = coin
                    break
            if exact:
                chosen = exact
            else:
                # Prefer ranked projects if available.
                chosen = sorted(
                    coins,
                    key=lambda c: (
                        10_000 if c.get("market_cap_rank") in (None, "", 0) else int(c.get("market_cap_rank"))
                    ),
                )[0]
            sym = _normalize_crypto_symbol(str(chosen.get("symbol", "")))
            nm = str(chosen.get("name") or sym)
            if sym:
                return sym, nm
    except Exception:
        pass

    raise ValueError("Crypto symbol/name not found. Try e.g. BTC, ETH, Solana, Cardano.")


def _fetch_crypto_chart_yahoo(base_symbol: str, selected_range: str) -> tuple[list[dict[str, float | str]], Optional[float], Optional[float]]:
    yahoo_symbol = f"{base_symbol}-USD"
    range_map = {
        "24h": ("1d", "5m"),
        "1w": ("7d", "30m"),
        "1m": ("1mo", "1h"),
        "1y": ("1y", "1d"),
        "all": ("max", "1d"),
    }
    req_range, req_interval = range_map.get(selected_range, ("1d", "5m"))

    payload = _request_json_with_retry(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}",
        {"range": req_range, "interval": req_interval},
        retries=2,
        timeout=12,
    )

    rows = ((payload.get("chart") or {}).get("result")) or []
    if not rows:
        return [], None, None

    row = rows[0]
    timestamps = row.get("timestamp") or []
    quote0 = (((row.get("indicators") or {}).get("quote") or [{}])[0]) or {}
    closes = quote0.get("close") or []

    points: list[dict[str, float | str]] = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        label = dt.strftime("%m-%d %H:%M") if selected_range in {"24h", "1w"} else dt.strftime("%Y-%m-%d")
        points.append({"label": label, "price": float(close)})

    if not points:
        return [], None, None

    first_price = float(points[0]["price"])
    last_price = float(points[-1]["price"])
    change_pct = ((last_price - first_price) / first_price * 100.0) if first_price else None
    return points, last_price, change_pct


def _fetch_crypto_price_coingecko(base_symbol: str) -> tuple[Optional[float], str]:
    try:
        search_payload = _request_json_with_retry(
            "https://api.coingecko.com/api/v3/search",
            {"query": base_symbol},
            retries=2,
            timeout=10,
        )
        coins = search_payload.get("coins", []) if isinstance(search_payload, dict) else []
        coin_id = None
        for coin in coins:
            if str(coin.get("symbol", "")).upper() == base_symbol:
                coin_id = coin.get("id")
                break
        if not coin_id and coins:
            coin_id = coins[0].get("id")
        if not coin_id:
            return None, base_symbol

        price_payload = _request_json_with_retry(
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids": coin_id, "vs_currencies": "usd"},
            retries=2,
            timeout=10,
        )
        usd = (price_payload.get(coin_id) or {}).get("usd") if isinstance(price_payload, dict) else None
        return (float(usd) if usd is not None else None), str(coin_id)
    except Exception:
        return None, base_symbol


def _fetch_crypto_price_yahoo(base_symbol: str) -> tuple[Optional[float], str]:
    yahoo_symbol = f"{base_symbol}-USD"
    quote_rows = _fetch_quote_batch([yahoo_symbol], retries=2)
    if quote_rows:
        price = quote_rows[0].get("regularMarketPrice")
        try:
            return (float(price) if price is not None else None), yahoo_symbol
        except (TypeError, ValueError):
            pass
    # Fallback via Yahoo chart endpoint.
    try:
        payload = _request_json_with_retry(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}",
            {"range": "1d", "interval": "1m"},
            retries=2,
            timeout=8,
        )
        rows = ((payload.get("chart") or {}).get("result")) or []
        meta = (rows[0].get("meta") if rows else {}) or {}
        price = meta.get("regularMarketPrice")
        return (float(price) if price is not None else None), yahoo_symbol
    except Exception:
        pass
    return None, yahoo_symbol


def _fetch_crypto_price_cmc(base_symbol: str) -> tuple[Optional[float], str]:
    slug = CMC_SLUG_MAP.get(base_symbol, base_symbol.lower())
    url = f"https://coinmarketcap.com/currencies/{slug}/"
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for selector in [
            '[data-test="text-cdp-price-display"]',
            'span.sc-65e7f566-0',
            'span[data-test="text-cdp-price-display"]',
        ]:
            node = soup.select_one(selector)
            if node:
                parsed = _parse_price_text(node.get_text(" ", strip=True))
                if parsed is not None:
                    return parsed, slug
        # Fallback: first $-like number in page.
        parsed = _parse_price_text(soup.get_text(" ", strip=True))
        return parsed, slug
    except Exception:
        return None, slug


def get_crypto_price_summary(query: str) -> CryptoPriceResult:
    base = _normalize_crypto_symbol(query)
    if not base:
        raise ValueError("Please enter a valid crypto symbol (e.g. BTC, ETH, SOL).")

    cmc_price, cmc_ref = _fetch_crypto_price_cmc(base)
    yahoo_price, yahoo_ref = _fetch_crypto_price_yahoo(base)
    gecko_price, gecko_ref = _fetch_crypto_price_coingecko(base)
    binance_price, binance_ref = _fetch_crypto_price_binance(base)

    source_prices = {
        "CoinMarketCap": cmc_price,
        "Yahoo Finance": yahoo_price,
        "CoinGecko API": gecko_price,
        "Binance API": binance_price,
    }
    source_symbols = {
        "CoinMarketCap": cmc_ref,
        "Yahoo Finance": yahoo_ref,
        "CoinGecko API": gecko_ref,
        "Binance API": binance_ref,
    }

    available = [p for p in source_prices.values() if p is not None]
    final_price = sum(available) / len(available) if available else None

    return CryptoPriceResult(
        symbol=base,
        display_symbol=f"{base}/USD",
        price_usd=final_price,
        source_prices=source_prices,
        source_symbols=source_symbols,
        as_of=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS login_otp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                code_hash TEXT NOT NULL,
                expires_ts REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                consumed_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_login_otp_user_created
                ON login_otp(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS registration_otp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                expires_ts REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                consumed_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_registration_otp_email_created
                ON registration_otp(email, created_at DESC);

            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, symbol),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS symbol_cache (
                query_key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                updated_ts REAL NOT NULL
            );
            """
        )

        user_columns = {
            str(row["name"]).lower()
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
            if row["name"]
        }
        if "email" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        conn.commit()


def _get_current_user() -> Optional[dict[str, str | int]]:
    user_id = session.get("user_id")
    if not user_id:
        return None
    with _db_connect() as conn:
        row = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        session.clear()
        return None
    return {"id": int(row["id"]), "username": str(row["username"])}


def _mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    if not local or not domain:
        return email
    prefix = local[:2] if len(local) > 2 else local[:1]
    return f"{prefix}{'*' * max(1, len(local) - len(prefix))}@{domain}"


def _otp_hash(code: str) -> str:
    secret = str(app.config.get("SECRET_KEY", "")).encode("utf-8")
    return hmac.new(secret, code.encode("utf-8"), "sha256").hexdigest()


def _clear_pending_2fa():
    for key in (
        "pending_2fa_user_id",
        "pending_2fa_next",
        "pending_2fa_email",
        "pending_2fa_created_at",
        "pending_2fa_username",
        "pending_2fa_dev_code",
    ):
        session.pop(key, None)


def _clear_pending_registration():
    for key in (
        "pending_register_username",
        "pending_register_email",
        "pending_register_password_hash",
        "pending_register_created_at",
        "pending_register_dev_code",
    ):
        session.pop(key, None)


def _send_2fa_email(email: str, code: str) -> bool:
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user or "noreply@localhost")
    use_tls = os.getenv("SMTP_STARTTLS", "1") == "1"

    if not smtp_host:
        app.logger.warning("2FA email not sent: SMTP_HOST is not configured.")
        return False

    msg = EmailMessage()
    msg["Subject"] = "Your security code"
    msg["From"] = smtp_from
    msg["To"] = email
    msg.set_content(
        f"Your verification code is: {code}\n"
        f"It expires in {TWO_FA_CODE_TTL_SECONDS // 60} minutes.\n"
        "If this was not you, change your password immediately."
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as smtp:
            if use_tls:
                smtp.starttls()
            if smtp_user:
                smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
        return True
    except Exception:
        app.logger.exception("Failed to send 2FA email to %s", email)
        return False


def _send_register_email(email: str, code: str) -> bool:
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user or "noreply@localhost")
    use_tls = os.getenv("SMTP_STARTTLS", "1") == "1"

    if not smtp_host:
        app.logger.warning("Registration email not sent: SMTP_HOST is not configured.")
        return False

    msg = EmailMessage()
    msg["Subject"] = "Confirm your account"
    msg["From"] = smtp_from
    msg["To"] = email
    msg.set_content(
        f"Your registration verification code is: {code}\n"
        f"It expires in {TWO_FA_CODE_TTL_SECONDS // 60} minutes."
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as smtp:
            if use_tls:
                smtp.starttls()
            if smtp_user:
                smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
        return True
    except Exception:
        app.logger.exception("Failed to send registration email to %s", email)
        return False


def _issue_login_otp(user_id: int) -> str:
    code = f"{random.randint(0, 999999):06d}"
    code_hash = _otp_hash(code)
    expires_ts = time.time() + TWO_FA_CODE_TTL_SECONDS
    now_ts = time.time()

    with _db_connect() as conn:
        conn.execute("DELETE FROM login_otp WHERE user_id = ?", (user_id,))
        conn.execute(
            """
            INSERT INTO login_otp(user_id, code_hash, expires_ts, attempts, consumed_at)
            VALUES(?, ?, ?, 0, NULL)
            """,
            (user_id, code_hash, expires_ts),
        )
        conn.execute(
            "DELETE FROM login_otp WHERE expires_ts < ? OR consumed_at IS NOT NULL",
            (now_ts - 300,),
        )
        conn.commit()
    return code


def _issue_registration_otp(username: str, email: str, password_hash: str) -> str:
    code = f"{random.randint(0, 999999):06d}"
    code_hash = _otp_hash(code)
    expires_ts = time.time() + TWO_FA_CODE_TTL_SECONDS
    now_ts = time.time()

    with _db_connect() as conn:
        conn.execute(
            "DELETE FROM registration_otp WHERE username = ? OR email = ?",
            (username, email),
        )
        conn.execute(
            """
            INSERT INTO registration_otp(username, email, password_hash, code_hash, expires_ts, attempts, consumed_at)
            VALUES(?, ?, ?, ?, ?, 0, NULL)
            """,
            (username, email, password_hash, code_hash, expires_ts),
        )
        conn.execute(
            "DELETE FROM registration_otp WHERE expires_ts < ? OR consumed_at IS NOT NULL",
            (now_ts - 300,),
        )
        conn.commit()
    return code


def _verify_registration_otp(username: str, email: str, code: str) -> tuple[bool, str]:
    now_ts = time.time()
    with _db_connect() as conn:
        row = conn.execute(
            """
            SELECT id, code_hash, expires_ts, attempts
            FROM registration_otp
            WHERE username = ? AND email = ? AND consumed_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (username, email),
        ).fetchone()
        if not row:
            return False, "Verification session expired. Please register again."

        otp_id = int(row["id"])
        attempts = int(row["attempts"])
        expires_ts = float(row["expires_ts"])
        stored_hash = str(row["code_hash"])

        if now_ts > expires_ts:
            conn.execute("DELETE FROM registration_otp WHERE id = ?", (otp_id,))
            conn.commit()
            return False, "Code expired. Please register again."

        if attempts >= TWO_FA_MAX_ATTEMPTS:
            conn.execute("DELETE FROM registration_otp WHERE id = ?", (otp_id,))
            conn.commit()
            return False, "Too many invalid attempts. Please register again."

        if not hmac.compare_digest(_otp_hash(code), stored_hash):
            conn.execute("UPDATE registration_otp SET attempts = attempts + 1 WHERE id = ?", (otp_id,))
            conn.commit()
            return False, "Invalid code."

        conn.execute(
            "UPDATE registration_otp SET consumed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (otp_id,),
        )
        conn.commit()
    return True, ""


def _verify_login_otp(user_id: int, code: str) -> tuple[bool, str]:
    now_ts = time.time()
    with _db_connect() as conn:
        row = conn.execute(
            """
            SELECT id, code_hash, expires_ts, attempts
            FROM login_otp
            WHERE user_id = ? AND consumed_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if not row:
            return False, "Verification session expired. Please login again."

        otp_id = int(row["id"])
        attempts = int(row["attempts"])
        expires_ts = float(row["expires_ts"])
        stored_hash = str(row["code_hash"])

        if now_ts > expires_ts:
            conn.execute("DELETE FROM login_otp WHERE id = ?", (otp_id,))
            conn.commit()
            return False, "Code expired. Please login again."

        if attempts >= TWO_FA_MAX_ATTEMPTS:
            conn.execute("DELETE FROM login_otp WHERE id = ?", (otp_id,))
            conn.commit()
            return False, "Too many invalid attempts. Please login again."

        if not hmac.compare_digest(_otp_hash(code), stored_hash):
            conn.execute("UPDATE login_otp SET attempts = attempts + 1 WHERE id = ?", (otp_id,))
            conn.commit()
            return False, "Invalid code."

        conn.execute(
            "UPDATE login_otp SET consumed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (otp_id,),
        )
        conn.commit()
    return True, ""


def _watchlist_symbols(user_id: int) -> list[str]:
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT symbol FROM watchlist WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [str(r["symbol"]) for r in rows]


def _watchlist_add(user_id: int, symbol: str):
    clean_symbol = _normalize_symbol(symbol)
    if not clean_symbol:
        return
    with _db_connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist(user_id, symbol) VALUES(?, ?)",
            (user_id, clean_symbol),
        )
        conn.commit()


def _watchlist_remove(user_id: int, symbol: str):
    clean_symbol = _normalize_symbol(symbol)
    if not clean_symbol:
        return
    with _db_connect() as conn:
        conn.execute("DELETE FROM watchlist WHERE user_id = ? AND symbol = ?", (user_id, clean_symbol))
        conn.commit()


def _known_symbols() -> list[str]:
    symbols: set[str] = set()

    with _db_connect() as conn:
        rows = conn.execute("SELECT DISTINCT symbol FROM watchlist").fetchall()
        symbols.update(_normalize_symbol(str(r["symbol"])) for r in rows if r["symbol"])
        rows = conn.execute("SELECT DISTINCT symbol FROM symbol_cache").fetchall()
        symbols.update(_normalize_symbol(str(r["symbol"])) for r in rows if r["symbol"])

    for (ticker, _, _, _), _cached in ANALYSIS_CACHE.items():
        symbols.add(_normalize_symbol(str(ticker)))

    return [s for s in symbols if s]


def refresh_all_known_companies_once():
    global AUTO_REFRESH_LAST_RUN_AT

    symbols = _known_symbols()
    if not symbols:
        AUTO_REFRESH_LAST_RUN_AT = time.time()
        return

    now = time.time()
    refreshed_any = False

    for symbol in symbols:
        keys_for_symbol = [k for k in ANALYSIS_CACHE.keys() if k[0] == symbol]
        if not keys_for_symbol:
            keys_for_symbol = [(symbol, "1y", 5, "week")]

        for _ticker, period, threshold_pct, granularity in keys_for_symbol:
            try:
                fresh = get_stock_analysis_http(
                    ticker_symbol=symbol,
                    period=period,
                    threshold_pct=threshold_pct,
                    granularity=granularity,
                )
                ANALYSIS_CACHE[(symbol, period, threshold_pct, granularity)] = (now, fresh)
                refreshed_any = True
            except Exception:
                # Keep old cache entry if refresh fails for one symbol.
                continue

    if refreshed_any:
        app.logger.info("Auto refresh completed for %s symbols.", len(symbols))
    AUTO_REFRESH_LAST_RUN_AT = time.time()


def _auto_refresh_worker():
    while not AUTO_REFRESH_STOP.is_set():
        # Wait first, then refresh, to avoid blocking app startup.
        if AUTO_REFRESH_STOP.wait(AUTO_REFRESH_INTERVAL_SECONDS):
            break
        if not AUTO_REFRESH_LOCK.acquire(blocking=False):
            continue
        try:
            refresh_all_known_companies_once()
        finally:
            AUTO_REFRESH_LOCK.release()


def start_auto_refresh():
    global AUTO_REFRESH_THREAD
    if AUTO_REFRESH_THREAD and AUTO_REFRESH_THREAD.is_alive():
        return
    AUTO_REFRESH_STOP.clear()
    AUTO_REFRESH_THREAD = Thread(target=_auto_refresh_worker, daemon=True, name="auto-refresh")
    AUTO_REFRESH_THREAD.start()


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_limit(scope: str, limit: int, window_seconds: int) -> bool:
    now = time.time()
    key = f"{scope}:{_client_ip()}"
    with REQUEST_RATE_LOCK:
        bucket = REQUEST_RATE_BUCKETS.setdefault(key, deque())
        while bucket and now - bucket[0] > window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
    return True


def _is_safe_next_url(next_url: str) -> bool:
    if not next_url:
        return False
    parsed = urlparse(next_url)
    return not parsed.scheme and not parsed.netloc and next_url.startswith("/")


def _csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


@app.context_processor
def _inject_template_security():
    return {"csrf_token": _csrf_token}


@app.before_request
def _security_checks():
    if request.method != "POST":
        return None

    if not _rate_limit(scope="post", limit=80, window_seconds=60):
        return ("Too many requests. Please slow down and try again.", 429)

    if request.path in {"/login", "/register"} and not _rate_limit(
        scope="auth", limit=15, window_seconds=60
    ):
        return ("Too many login/register attempts. Try again in a minute.", 429)

    if app.config.get("TESTING"):
        return None

    token = request.form.get("_csrf_token") or request.headers.get("X-CSRF-Token")
    expected = session.get("_csrf_token")
    if not token or not expected or not hmac.compare_digest(token, expected):
        return ("Security check failed (invalid CSRF token). Refresh page and try again.", 400)
    return None


@app.after_request
def _set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    if request.is_secure:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


def _login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return func(*args, **kwargs)

    return wrapper


def _fetch_screen_live(query_name: str, size: int = 25, retries: int = 3) -> list[dict]:
    delay = 0.8
    for attempt in range(retries):
        try:
            data = yf.screen(query_name, size=size)
            return data.get("quotes", []) if isinstance(data, dict) else []
        except YFRateLimitError:
            if attempt == retries - 1:
                return []
            time.sleep(delay + random.uniform(0.0, 0.3))
            delay *= 1.8
        except Exception:
            return []
    return []


def _fetch_quote_batch(symbols: list[str], retries: int = 3) -> list[dict]:
    if not symbols:
        return []

    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    delay = 0.7
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params={"symbols": ",".join(symbols)},
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("quoteResponse", {}).get("result", [])
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(delay + random.uniform(0.0, 0.25))
            delay *= 1.8
    return []


def _request_json_with_retry(url: str, params: dict, retries: int = 4, timeout: int = 12) -> dict:
    delay = 0.8
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            time.sleep(delay + random.uniform(0.0, 0.3))
            delay *= 1.8
    raise RuntimeError("Yahoo endpoint request failed.") from last_exc


def _search_symbol_by_name_http(query: str) -> str:
    payload = _request_json_with_retry(
        "https://query2.finance.yahoo.com/v1/finance/search",
        {"q": query, "quotesCount": 10, "newsCount": 0},
    )
    quotes = payload.get("quotes", []) if isinstance(payload, dict) else []
    for quote in quotes:
        symbol = quote.get("symbol")
        if symbol:
            return str(symbol).upper()
    raise ValueError(f"No ticker found for company name '{query}'.")


def _is_probable_ticker(query: str) -> bool:
    # Fast heuristic: ticker-like inputs are short and use market symbol characters.
    clean = query.strip().upper()
    if not clean or len(clean) > 10:
        return False
    return bool(re.fullmatch(r"[A-Z0-9.\-:=]+", clean))


def _get_search_cache(query: str) -> Optional[str]:
    key = query.strip().lower()
    if not key:
        return None

    now = time.time()
    cached = SEARCH_CACHE.get(key)
    if cached:
        cached_at, symbol = cached
        if now - cached_at <= SEARCH_CACHE_TTL_SECONDS:
            return symbol
        SEARCH_CACHE.pop(key, None)

    # Persistent cache fallback (survives app restarts).
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT symbol, updated_ts FROM symbol_cache WHERE query_key = ?",
            (key,),
        ).fetchone()
    if not row:
        return None

    updated_ts = float(row["updated_ts"])
    if now - updated_ts > SEARCH_CACHE_TTL_SECONDS:
        with _db_connect() as conn:
            conn.execute("DELETE FROM symbol_cache WHERE query_key = ?", (key,))
            conn.commit()
        return None

    symbol = str(row["symbol"]).upper()
    SEARCH_CACHE[key] = (updated_ts, symbol)
    return symbol


def _set_search_cache(query: str, symbol: str):
    key = query.strip().lower()
    if not key:
        return
    now = time.time()
    normalized_symbol = symbol.upper()
    SEARCH_CACHE[key] = (now, normalized_symbol)
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO symbol_cache(query_key, symbol, updated_ts)
            VALUES(?, ?, ?)
            ON CONFLICT(query_key) DO UPDATE SET
                symbol = excluded.symbol,
                updated_ts = excluded.updated_ts
            """,
            (key, normalized_symbol, now),
        )
        conn.commit()


def _to_number_with_suffix(raw_value: str) -> float:
    raw = (raw_value or "").replace(",", "").strip()
    match = re.match(r"^([+-]?\d+(?:\.\d+)?)([KMBT]?)$", raw, re.IGNORECASE)
    if not match:
        return 0.0
    value = float(match.group(1))
    suffix = match.group(2).upper()
    if suffix == "K":
        return value * 1_000
    if suffix == "M":
        return value * 1_000_000
    if suffix == "B":
        return value * 1_000_000_000
    if suffix == "T":
        return value * 1_000_000_000_000
    return value


def _extract_first_float(raw_value: str) -> Optional[float]:
    match = re.search(r"[+-]?\d+(?:\.\d+)?", raw_value or "")
    return float(match.group(0)) if match else None


def _extract_change_pct(raw_value: str) -> Optional[float]:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", raw_value or "")
    return float(match.group(1)) if match else None


def _normalize_symbol(raw_symbol: str) -> str:
    symbol = (raw_symbol or "").strip().upper()
    symbol = re.sub(r"[^A-Z0-9\.\-:]", "", symbol)
    return symbol


def _normalize_country(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    code = raw.upper()
    return COUNTRY_CODE_MAP.get(code, raw)


def _format_location(state: Optional[str], country: Optional[str], region: Optional[str] = None) -> Optional[str]:
    state_clean = str(state).strip() if state else None
    country_clean = _normalize_country(country) or _normalize_country(region)
    if state_clean and country_clean:
        return f"{state_clean}, {country_clean}"
    if country_clean:
        return country_clean
    if state_clean:
        return state_clean
    return None


def _country_from_symbol(symbol: str) -> Optional[str]:
    clean = _normalize_symbol(symbol)
    if not clean:
        return None
    if "." in clean:
        suffix = clean.rsplit(".", 1)[1].upper()
        mapped = SYMBOL_COUNTRY_SUFFIX_MAP.get(suffix)
        if mapped:
            return mapped
    # Default heuristic for common plain tickers.
    return "United States"


def _country_suffix_groups() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for suffix, country in SYMBOL_COUNTRY_SUFFIX_MAP.items():
        grouped.setdefault(country, []).append(f".{suffix}")
    for country in grouped:
        grouped[country] = sorted(grouped[country])
    return dict(sorted(grouped.items(), key=lambda x: x[0]))


def _listing_from_quote(quote: dict) -> Optional[dict[str, str]]:
    symbol = quote.get("symbol")
    if not symbol:
        return None
    symbol_text = _normalize_symbol(str(symbol))
    if not symbol_text:
        return None

    exchange = quote.get("exchange") or quote.get("fullExchangeName") or quote.get("exchDisp")
    country = _format_location(None, quote.get("country"), quote.get("region")) or _country_from_symbol(symbol_text)
    return {
        "symbol": symbol_text,
        "name": str(quote.get("longName") or quote.get("shortName") or symbol_text),
        "exchange": str(exchange) if exchange else "N/A",
        "currency": str(quote.get("currency") or "N/A"),
        "country": str(country or "N/A"),
    }


def _company_keywords(company_name: Optional[str], fallback_query: str) -> list[str]:
    base = (company_name or fallback_query or "").lower()
    words = re.findall(r"[a-z0-9]+", base)
    keywords = [w for w in words if len(w) >= 3]
    return keywords[:4]


def _get_exchange_listings(
    ticker_symbol: str, company_name: Optional[str], original_query: str
) -> list[dict[str, str]]:
    cache_key = ticker_symbol.upper()
    now = time.time()
    cached = LISTINGS_CACHE.get(cache_key)
    if cached and now - cached[0] < LISTINGS_CACHE_TTL_SECONDS:
        return cached[1]

    keywords = _company_keywords(company_name, original_query)
    candidates: dict[str, dict[str, str]] = {}
    queries = [company_name or "", original_query or "", ticker_symbol]
    seen_queries: set[str] = set()

    for raw_query in queries:
        query = raw_query.strip()
        if not query:
            continue
        query_l = query.lower()
        if query_l in seen_queries:
            continue
        seen_queries.add(query_l)

        try:
            search_result = yf.Search(query, max_results=30)
            quotes = getattr(search_result, "quotes", []) or []
        except Exception:
            quotes = []

        for quote in quotes:
            quote_type = str(quote.get("quoteType") or "").upper()
            if quote_type and quote_type not in {"EQUITY", "ETF"}:
                continue

            listing = _listing_from_quote(quote)
            if not listing:
                continue

            text = f"{listing['name']} {listing['symbol']}".lower()
            if listing["symbol"] != cache_key and keywords and not any(k in text for k in keywords):
                continue
            candidates[listing["symbol"]] = listing

    # Ensure the currently analyzed ticker is always present.
    if cache_key not in candidates:
        quote_rows = _fetch_quote_batch([cache_key], retries=1)
        fallback_listing = _listing_from_quote(quote_rows[0]) if quote_rows else None
        if fallback_listing:
            candidates[cache_key] = fallback_listing
        else:
            candidates[cache_key] = {
                "symbol": cache_key,
                "name": company_name or cache_key,
                "exchange": "N/A",
                "currency": "N/A",
                "country": _country_from_symbol(cache_key) or "N/A",
            }

    # Try alternate exchange suffixes for the same base symbol.
    base_symbol = cache_key.split(".", 1)[0]
    alt_symbols = [f"{base_symbol}.{suffix}" for suffix in sorted(SYMBOL_COUNTRY_SUFFIX_MAP.keys())]
    if base_symbol and base_symbol != cache_key:
        alt_symbols.append(base_symbol)
    alt_symbols = [s for s in alt_symbols if s not in candidates]
    if alt_symbols:
        alt_quotes = _fetch_quote_batch(alt_symbols[:60], retries=1)
        for quote in alt_quotes:
            listing = _listing_from_quote(quote)
            if listing:
                candidates[listing["symbol"]] = listing

    # Enrich missing fields from quote batch.
    enrich_symbols = list(candidates.keys())[:30]
    enrich_rows = _fetch_quote_batch(enrich_symbols, retries=1)
    by_symbol = {
        _normalize_symbol(str(r.get("symbol", ""))): r for r in enrich_rows if r.get("symbol")
    }
    for symbol, listing in candidates.items():
        row = by_symbol.get(symbol)
        if not row:
            continue
        row_exchange = row.get("exchange") or row.get("fullExchangeName") or row.get("exchDisp")
        row_country = _format_location(None, row.get("country"), row.get("region"))
        listing["exchange"] = listing["exchange"] if listing["exchange"] != "N/A" else str(row_exchange or "N/A")
        listing["currency"] = listing["currency"] if listing["currency"] != "N/A" else str(row.get("currency") or "N/A")
        listing["country"] = listing["country"] if listing["country"] != "N/A" else str(row_country or "N/A")

    listings = sorted(
        candidates.values(),
        key=lambda x: (0 if x["symbol"] == cache_key else 1, x["symbol"]),
    )[:20]
    LISTINGS_CACHE[cache_key] = (now, listings)
    return listings


def _scrape_market_table(url: str, retries: int = 1) -> list[dict]:
    delay = 0.8
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if not table:
                return []
            rows = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(cells) < 9:
                    continue
                symbol = cells[0]
                change_pct = _extract_change_pct(cells[5])
                row = {
                    "symbol": symbol,
                    "name": cells[1] or symbol,
                    "price": _extract_first_float(cells[3]),
                    "change_pct": change_pct,
                    "volume": int(_to_number_with_suffix(cells[6])),
                    "exchange": None,
                    "country": None,
                    "currency": None,
                    "market_cap": _to_number_with_suffix(cells[8]),
                    "source": "yahoo",
                }
                rows.append(row)
            return rows
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(delay + random.uniform(0.0, 0.3))
            delay *= 1.8
    return []


def _scrape_finviz_table(url: str, retries: int = 1) -> list[dict]:
    delay = 0.8
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.select_one("table.screener_table")
            if not table:
                return []

            rows: list[dict] = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(cells) < 11:
                    continue
                symbol = _normalize_symbol(cells[1])
                if not symbol:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "name": cells[2] or symbol,
                        "price": _extract_first_float(cells[8]),
                        "change_pct": _extract_change_pct(cells[9]),
                        "volume": int(_to_number_with_suffix(cells[10])),
                        "exchange": None,
                        "country": cells[5] or None,
                        "currency": None,
                        "market_cap": _to_number_with_suffix(cells[6]),
                        "source": "finviz",
                    }
                )
            return rows
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(delay + random.uniform(0.0, 0.3))
            delay *= 1.8
    return []


def _scrape_tradingview_table(url: str, retries: int = 1) -> list[dict]:
    delay = 0.8
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if not table:
                return []

            rows: list[dict] = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(cells) < 6:
                    continue

                symbol_and_name = cells[0].split(" ", 1)
                symbol = _normalize_symbol(symbol_and_name[0])
                if not symbol:
                    continue
                name = symbol_and_name[1].strip() if len(symbol_and_name) > 1 else symbol

                rows.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "price": _extract_first_float(cells[2] if len(cells) > 2 else ""),
                        "change_pct": _extract_change_pct(cells[1] if len(cells) > 1 else ""),
                        "volume": int(_to_number_with_suffix(cells[3] if len(cells) > 3 else "0")),
                        "exchange": None,
                        "country": None,
                        "currency": "USD",
                        "market_cap": _to_number_with_suffix(cells[5] if len(cells) > 5 else "0"),
                        "source": "tradingview",
                    }
                )
            return rows
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(delay + random.uniform(0.0, 0.3))
            delay *= 1.8
    return []


def _scrape_google_finance_table(url: str, retries: int = 1) -> list[dict]:
    delay = 0.8
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else ""
            if "Before you continue" in title:
                return []

            table = soup.find("table")
            if not table:
                return []

            rows: list[dict] = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(cells) < 4:
                    continue
                symbol = _normalize_symbol(cells[0].split(" ", 1)[0])
                if not symbol:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "name": cells[0],
                        "price": _extract_first_float(cells[1] if len(cells) > 1 else ""),
                        "change_pct": _extract_change_pct(cells[2] if len(cells) > 2 else ""),
                        "volume": int(_to_number_with_suffix(cells[3] if len(cells) > 3 else "0")),
                        "exchange": None,
                        "country": None,
                        "currency": None,
                        "market_cap": 0.0,
                        "source": "google_finance",
                    }
                )
            return rows
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(delay + random.uniform(0.0, 0.3))
            delay *= 1.8
    return []


def _merge_market_rows(*sources: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for rows in sources:
        for row in rows:
            symbol = _normalize_symbol(str(row.get("symbol", "")))
            if not symbol:
                continue
            existing = merged.get(symbol)
            if existing is None:
                merged[symbol] = dict(row, symbol=symbol)
                continue

            if not existing.get("name") and row.get("name"):
                existing["name"] = row.get("name")
            if existing.get("price") is None and row.get("price") is not None:
                existing["price"] = row.get("price")
            if existing.get("change_pct") is None and row.get("change_pct") is not None:
                existing["change_pct"] = row.get("change_pct")
            if (row.get("volume") or 0) > (existing.get("volume") or 0):
                existing["volume"] = row.get("volume")
            if (row.get("market_cap") or 0) > (existing.get("market_cap") or 0):
                existing["market_cap"] = row.get("market_cap")
            if not existing.get("exchange") and row.get("exchange"):
                existing["exchange"] = row.get("exchange")
            if not existing.get("country") and row.get("country"):
                existing["country"] = row.get("country")
            if not existing.get("currency") and row.get("currency"):
                existing["currency"] = row.get("currency")
            if existing.get("source") and row.get("source") and row["source"] not in existing["source"]:
                existing["source"] = f"{existing['source']},{row['source']}"
    return list(merged.values())


def _to_market_row(quote: dict) -> Optional[dict]:
    symbol = quote.get("symbol")
    if not symbol:
        return None
    return {
        "symbol": str(symbol),
        "name": quote.get("longName") or quote.get("shortName") or str(symbol),
        "price": quote.get("regularMarketPrice"),
        "change_pct": quote.get("regularMarketChangePercent"),
        "volume": quote.get("regularMarketVolume") or 0,
        "exchange": quote.get("exchange") or quote.get("fullExchangeName"),
        "country": _normalize_country(quote.get("country") or quote.get("region")),
        "currency": quote.get("currency"),
        "market_cap": quote.get("marketCap") or 0,
        "source": "yahoo",
    }


def _build_recommendations(buys: list[dict], gainers: list[dict], count: int = 5) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in buys + gainers:
        symbol = row["symbol"]
        existing = merged.get(symbol)
        if existing is None or (row.get("change_pct") or -999) > (existing.get("change_pct") or -999):
            merged[symbol] = row

    scored: list[dict] = []
    for row in merged.values():
        change_pct = float(row.get("change_pct") or 0.0)
        if change_pct <= 0:
            continue
        volume = float(row.get("volume") or 0.0)
        market_cap = float(row.get("market_cap") or 0.0)
        score = (change_pct * 0.65) + (math.log10(volume + 1.0) * 0.25) + (math.log10(market_cap + 1.0) * 0.10)
        enriched = dict(row)
        enriched["score"] = round(score, 3)
        scored.append(enriched)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:count]


def _to_crypto_market_row(row: dict) -> Optional[dict]:
    symbol = _normalize_symbol(str(row.get("symbol", "")).upper())
    if not symbol:
        return None
    return {
        "symbol": symbol,
        "name": str(row.get("name") or symbol),
        "price": row.get("current_price"),
        "change_pct": row.get("price_change_percentage_24h"),
        "volume": int(float(row.get("total_volume") or 0)),
        "exchange": "Multi-source",
        "country": "N/A",
        "currency": "USD",
        "market_cap": float(row.get("market_cap") or 0),
        "source": "coingecko",
    }


def _fetch_crypto_markets_coingecko() -> list[dict]:
    try:
        payload = _request_json_with_retry(
            "https://api.coingecko.com/api/v3/coins/markets",
            {
                "vs_currency": "usd",
                "order": "volume_desc",
                "per_page": 250,
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "24h",
            },
            retries=2,
            timeout=12,
        )
        if not isinstance(payload, list):
            return []
        rows = []
        for item in payload:
            mapped = _to_crypto_market_row(item)
            if mapped:
                rows.append(mapped)
        return rows
    except Exception:
        return []


def _fetch_crypto_markets_binance() -> list[dict]:
    try:
        payload = _request_json_with_retry(
            "https://api.binance.com/api/v3/ticker/24hr",
            {},
            retries=2,
            timeout=10,
        )
        if not isinstance(payload, list):
            return []
        rows = []
        for item in payload:
            symbol_text = str(item.get("symbol") or "")
            if not symbol_text.endswith("USDT"):
                continue
            base = symbol_text[:-4]
            mapped = {
                "symbol": _normalize_symbol(base),
                "name": _normalize_symbol(base),
                "price": _extract_first_float(str(item.get("lastPrice", ""))),
                "change_pct": _extract_first_float(str(item.get("priceChangePercent", ""))),
                "volume": int(float(item.get("quoteVolume") or 0)),
                "exchange": "Binance",
                "country": "N/A",
                "currency": "USD",
                "market_cap": 0.0,
                "source": "binance",
            }
            if mapped["symbol"]:
                rows.append(mapped)
        rows.sort(key=lambda x: x["volume"], reverse=True)
        return rows[:250]
    except Exception:
        return []


def get_crypto_market_overview_live() -> dict:
    if app.config.get("TESTING"):
        return {
            "active_buys": [],
            "active_sells": [],
            "gainers": [],
            "losers": [],
            "recommendations": [],
        }

    gecko_rows = _fetch_crypto_markets_coingecko()
    binance_rows = _fetch_crypto_markets_binance()
    merged_rows = _merge_market_rows(gecko_rows, binance_rows)

    if not merged_rows:
        return {
            "active_buys": [],
            "active_sells": [],
            "gainers": [],
            "losers": [],
            "recommendations": [],
        }

    active_sorted = sorted(merged_rows, key=lambda x: x.get("volume", 0), reverse=True)
    active_buys = [r for r in active_sorted if (r.get("change_pct") or 0) > 0][:10]
    active_sells = [r for r in active_sorted if (r.get("change_pct") or 0) < 0][:10]
    gainers = sorted(merged_rows, key=lambda x: x.get("change_pct") or -999, reverse=True)[:10]
    losers = sorted(merged_rows, key=lambda x: x.get("change_pct") or 999)[:10]
    recommendations = _build_recommendations(active_buys, gainers, count=5)

    return {
        "active_buys": active_buys,
        "active_sells": active_sells,
        "gainers": gainers,
        "losers": losers,
        "recommendations": recommendations,
    }


def get_market_overview_live() -> dict:
    global LAST_MARKET_OVERVIEW

    if app.config.get("TESTING"):
        return {
            "active_buys": [],
            "active_sells": [],
            "gainers": [],
            "losers": [],
            "recommendations": [],
        }

    # Live on every load from multiple providers:
    # Yahoo Finance + Finviz + TradingView + Google Finance.
    yahoo_actives = _scrape_market_table(MARKET_URLS["most_active"])
    yahoo_gainers = _scrape_market_table(MARKET_URLS["gainers"])
    yahoo_losers = _scrape_market_table(MARKET_URLS["losers"])

    finviz_actives = _scrape_finviz_table(FINVIZ_URLS["most_active"])
    finviz_gainers = _scrape_finviz_table(FINVIZ_URLS["gainers"])
    finviz_losers = _scrape_finviz_table(FINVIZ_URLS["losers"])

    tv_actives = _scrape_tradingview_table(TRADINGVIEW_URLS["most_active"])
    tv_gainers = _scrape_tradingview_table(TRADINGVIEW_URLS["gainers"])
    tv_losers = _scrape_tradingview_table(TRADINGVIEW_URLS["losers"])

    gf_actives = _scrape_google_finance_table(GOOGLE_FINANCE_URLS["most_active"])
    gf_gainers = _scrape_google_finance_table(GOOGLE_FINANCE_URLS["gainers"])
    gf_losers = _scrape_google_finance_table(GOOGLE_FINANCE_URLS["losers"])

    actives_rows = _merge_market_rows(yahoo_actives, finviz_actives, tv_actives, gf_actives)
    gainers_rows = _merge_market_rows(yahoo_gainers, finviz_gainers, tv_gainers, gf_gainers)
    losers_rows = _merge_market_rows(yahoo_losers, finviz_losers, tv_losers, gf_losers)

    if not actives_rows and not gainers_rows and not losers_rows:
        # Last-resort fallback to yfinance screen.
        actives_raw = _fetch_screen_live("most_actives", size=40)
        gainers_raw = _fetch_screen_live("day_gainers", size=25)
        losers_raw = _fetch_screen_live("day_losers", size=25)
        actives_rows = [row for row in (_to_market_row(q) for q in actives_raw) if row is not None]
        gainers_rows = [row for row in (_to_market_row(q) for q in gainers_raw) if row is not None]
        losers_rows = [row for row in (_to_market_row(q) for q in losers_raw) if row is not None]

    active_buys = [r for r in actives_rows if (r.get("change_pct") or 0) > 0]
    active_sells = [r for r in actives_rows if (r.get("change_pct") or 0) < 0]

    active_buys.sort(key=lambda x: x.get("volume", 0), reverse=True)
    active_sells.sort(key=lambda x: x.get("volume", 0), reverse=True)
    gainers_rows.sort(key=lambda x: x.get("change_pct") or -999, reverse=True)
    losers_rows.sort(key=lambda x: x.get("change_pct") or 999)

    top_active_buys = active_buys[:10]
    top_active_sells = active_sells[:10]
    top_gainers = gainers_rows[:10]
    top_losers = losers_rows[:10]
    recommendations = _build_recommendations(top_active_buys, top_gainers, count=5)

    overview = {
        "active_buys": top_active_buys,
        "active_sells": top_active_sells,
        "gainers": top_gainers,
        "losers": top_losers,
        "recommendations": recommendations,
    }

    has_any = any(overview[k] for k in ("active_buys", "active_sells", "gainers", "losers", "recommendations"))
    if has_any:
        LAST_MARKET_OVERVIEW = overview
        return overview
    if LAST_MARKET_OVERVIEW is not None:
        return LAST_MARKET_OVERVIEW
    return overview


def _history_with_retry(
    ticker: yf.Ticker,
    period: str,
    interval: str = "1d",
    retries: int = 5,
    initial_delay: float = 1.2,
):
    delay = initial_delay
    for attempt in range(retries):
        try:
            return ticker.history(period=period, interval=interval, auto_adjust=False)
        except YFRateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(delay + random.uniform(0.0, 0.4))
            delay *= 2


def _search_symbol_by_name(query: str, retries: int = 5, initial_delay: float = 1.2) -> str:
    cached_symbol = _get_search_cache(query)
    if cached_symbol:
        return cached_symbol

    delay = initial_delay
    for attempt in range(retries):
        try:
            search_result = yf.Search(query, max_results=10)
            quotes = getattr(search_result, "quotes", []) or []
            for quote in quotes:
                symbol = quote.get("symbol")
                if symbol:
                    resolved = str(symbol).upper()
                    _set_search_cache(query, resolved)
                    return resolved
            raise ValueError(f"No ticker found for company name '{query}'.")
        except YFRateLimitError:
            if attempt == retries - 1:
                resolved = _search_symbol_by_name_http(query)
                _set_search_cache(query, resolved)
                return resolved
            time.sleep(delay + random.uniform(0.0, 0.4))
            delay *= 2


def _resolve_input_symbol(input_value: str) -> str:
    raw = input_value.strip()
    if _is_probable_ticker(raw):
        return raw.upper()

    cached_symbol = _get_search_cache(raw)
    if cached_symbol:
        return cached_symbol

    # Prefer lightweight HTTP search first for company names.
    try:
        resolved = _search_symbol_by_name_http(raw)
        _set_search_cache(raw, resolved)
        return resolved
    except Exception:
        resolved = _search_symbol_by_name(raw)
        _set_search_cache(raw, resolved)
        return resolved


def _normalize_period(requested_period: str) -> str:
    # Keep backward compatibility for old links that might still contain 2y.
    valid_values = {value for value, _ in PERIOD_OPTIONS} | {"2y"}
    return requested_period if requested_period in valid_values else "1y"


def _threshold_options_for_granularity(granularity: str) -> list[int]:
    return WEEK_THRESHOLD_OPTIONS if granularity == "week" else DAY_THRESHOLD_OPTIONS


def _normalize_threshold(requested_threshold: str, granularity: str = "week") -> int:
    valid_thresholds = _threshold_options_for_granularity(granularity)
    try:
        threshold = int(requested_threshold)
    except (TypeError, ValueError):
        return 5
    return threshold if threshold in valid_thresholds else 5


def _normalize_granularity(requested_granularity: str) -> str:
    valid_values = {value for value, _ in GRANULARITY_OPTIONS}
    return requested_granularity if requested_granularity in valid_values else "week"


def _history_interval(period: str, granularity: str) -> str:
    if granularity == "day" and period == "1d":
        return "60m"
    return "1d"


def _record_recent_search(query: str, ticker: str, period: str, threshold_pct: int, granularity: str):
    entry = {
        "query": query,
        "ticker": ticker,
        "period": period,
        "threshold": str(threshold_pct),
        "granularity": granularity,
        "timestamp": time.strftime("%H:%M:%S"),
    }

    deduped_history = deque(
        (x for x in SEARCH_HISTORY if not (
            x["query"].lower() == query.lower()
            and x["ticker"] == ticker
            and x["period"] == period
            and x["threshold"] == str(threshold_pct)
            and x.get("granularity", "week") == granularity
        )),
        maxlen=MAX_SEARCH_HISTORY,
    )
    deduped_history.appendleft(entry)
    SEARCH_HISTORY.clear()
    SEARCH_HISTORY.extend(deduped_history)

    RECENT_SEARCHES.clear()
    for item in list(SEARCH_HISTORY)[:MAX_RECENT_SEARCHES]:
        RECENT_SEARCHES.append(item)


def _clear_search_history():
    SEARCH_HISTORY.clear()
    RECENT_SEARCHES.clear()


def _latest_cached_result() -> Optional["AnalysisResult"]:
    if not ANALYSIS_CACHE:
        return None
    latest_timestamp = max(ts for ts, _ in ANALYSIS_CACHE.values())
    for ts, result in ANALYSIS_CACHE.values():
        if ts == latest_timestamp:
            return result
    return None


def _week_end_friday(day: datetime) -> datetime:
    offset = (4 - day.weekday()) % 7
    return day + timedelta(days=offset)


def get_stock_analysis_http(
    ticker_symbol: str, period: str = "1y", threshold_pct: int = 5, granularity: str = "week"
) -> AnalysisResult:
    interval = _history_interval(period, granularity)
    quote_row: dict = {}
    try:
        quote_payload = _request_json_with_retry(
            "https://query1.finance.yahoo.com/v7/finance/quote",
            {"symbols": ticker_symbol},
        )
        quote_rows = ((quote_payload.get("quoteResponse") or {}).get("result")) or []
        quote_row = quote_rows[0] if quote_rows else {}
    except Exception:
        quote_row = {}

    chart: list[dict] = []
    try:
        chart_payload = _request_json_with_retry(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}",
            {
                "range": period,
                "interval": interval,
                "events": "div,splits",
                "includePrePost": "false",
            },
        )
        chart = (chart_payload.get("chart") or {}).get("result") or []
    except Exception:
        chart = []

    result0 = chart[0] if chart else {}
    timestamps = result0.get("timestamp") or []
    quote0 = (((result0.get("indicators") or {}).get("quote") or [{}])[0]) or {}
    closes = quote0.get("close") or []

    raw_points: list[tuple[datetime, float]] = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        day = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        raw_points.append((day, float(close)))
    raw_points.sort(key=lambda x: x[0])

    grouped_map: dict[str, tuple[datetime, float]] = {}
    for point_dt, close in raw_points:
        if granularity == "week":
            period_end = _week_end_friday(point_dt).date().isoformat()
        elif interval == "60m":
            period_end = point_dt.strftime("%Y-%m-%d %H:%M")
        else:
            period_end = point_dt.date().isoformat()

        previous = grouped_map.get(period_end)
        if previous is None or point_dt > previous[0]:
            grouped_map[period_end] = (point_dt, close)

    close_items = sorted((k, v[1]) for k, v in grouped_map.items())

    threshold_ratio = threshold_pct / 100.0
    volatile_weeks: list[dict[str, str | float]] = []
    weekly_price_points: list[dict[str, str | float | bool]] = []

    changes: list[float] = []
    volatile_dates: set[str] = set()
    if close_items:
        prev_close = close_items[0][1]
        for period_end, close in close_items[1:]:
            change = (close - prev_close) / prev_close if prev_close else 0.0
            changes.append(change * 100.0)
            if abs(change) > threshold_ratio:
                change_pct = change * 100.0
                volatile_dates.add(period_end)
                volatile_weeks.append(
                    {
                        "period_end": period_end,
                        "week_end": period_end,
                        "change_pct": f"{change_pct:.2f}%",
                        "change_value": change_pct,
                    }
                )
            prev_close = close

    for period_end, close in close_items:
        weekly_price_points.append(
            {"date": period_end, "close": float(close), "is_volatile": period_end in volatile_dates}
        )

    meta = result0.get("meta") or {}
    current_price = quote_row.get("regularMarketPrice") or meta.get("regularMarketPrice")
    if current_price is None:
        current_price = close_items[-1][1] if close_items else None
    current_price = float(current_price) if current_price is not None else None

    events = result0.get("events") or {}
    dividend_events = events.get("dividends") or {}
    last_dividend = None
    last_dividend_date = None
    if isinstance(dividend_events, dict) and dividend_events:
        latest_ts = max(int(ts) for ts in dividend_events.keys())
        latest = dividend_events.get(str(latest_ts), {}) or dividend_events.get(latest_ts, {}) or {}
        amount = latest.get("amount")
        if amount is not None:
            last_dividend = float(amount)
            last_dividend_date = datetime.fromtimestamp(latest_ts, tz=timezone.utc).strftime("%Y-%m-%d")

    asset_profile: dict = {}
    try:
        profile_payload = _request_json_with_retry(
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker_symbol}",
            {"modules": "assetProfile"},
        )
        summary_rows = ((profile_payload.get("quoteSummary") or {}).get("result")) or []
        asset_profile = (summary_rows[0].get("assetProfile") if summary_rows else {}) or {}
    except Exception:
        asset_profile = {}

    company_name = quote_row.get("longName") or quote_row.get("shortName")
    company_state = _format_location(
        asset_profile.get("state"),
        asset_profile.get("country") or quote_row.get("country"),
        quote_row.get("region"),
    )
    company_state = company_state or _country_from_symbol(ticker_symbol)
    currency = quote_row.get("currency") or meta.get("currency")
    exchange = quote_row.get("exchange") or quote_row.get("fullExchangeName") or meta.get("exchangeName")

    if current_price is None and not company_name:
        raise ValueError("No price data found for this ticker.")

    return AnalysisResult(
        ticker=ticker_symbol.upper(),
        company_name=company_name,
        company_state=company_state,
        currency=currency,
        exchange=exchange,
        current_price=current_price,
        last_dividend=last_dividend,
        last_dividend_date=last_dividend_date,
        threshold_pct=threshold_pct,
        volatile_weeks=volatile_weeks,
        weekly_price_points=weekly_price_points,
        volatility_week_count=len(volatile_weeks),
        max_weekly_gain_pct=max(changes) if changes else None,
        max_weekly_loss_pct=min(changes) if changes else None,
        granularity=granularity,
    )


def get_stock_analysis(
    ticker_symbol: str,
    period: str = "1y",
    threshold_pct: int = 5,
    granularity: str = "week",
) -> AnalysisResult:
    ticker = yf.Ticker(ticker_symbol)
    interval = _history_interval(period, granularity)
    price_history = _history_with_retry(ticker, period=period, interval=interval)
    if price_history.empty or "Close" not in price_history.columns:
        raise ValueError("No price data found for this ticker.")

    close_series = price_history["Close"].dropna()
    if close_series.empty:
        raise ValueError("No price data found for this ticker.")

    if granularity == "week":
        grouped_close = close_series.resample("W-FRI").last().dropna()
        date_formatter = "%Y-%m-%d"
    elif interval == "60m":
        grouped_close = close_series.resample("60min").last().dropna()
        date_formatter = "%Y-%m-%d %H:%M"
    else:
        grouped_close = close_series.resample("D").last().dropna()
        date_formatter = "%Y-%m-%d"

    period_change = grouped_close.pct_change().dropna()
    threshold_ratio = threshold_pct / 100.0
    significant_weeks = period_change[period_change.abs() > threshold_ratio]

    volatile_weeks: list[dict[str, str | float]] = []
    for week_end, change in significant_weeks.items():
        change_pct = float(change * 100)
        period_end = week_end.strftime(date_formatter)
        volatile_weeks.append(
            {
                "period_end": period_end,
                "week_end": period_end,
                "change_pct": f"{change_pct:.2f}%",
                "change_value": change_pct,
            }
        )

    all_weekly_changes_pct = period_change * 100
    max_weekly_gain_pct = (
        float(all_weekly_changes_pct.max()) if not all_weekly_changes_pct.empty else None
    )
    max_weekly_loss_pct = (
        float(all_weekly_changes_pct.min()) if not all_weekly_changes_pct.empty else None
    )

    volatile_dates = {d.strftime(date_formatter) for d in significant_weeks.index}
    weekly_price_points: list[dict[str, str | float | bool]] = []
    for point_date, close_price in grouped_close.items():
        date_str = point_date.strftime(date_formatter)
        weekly_price_points.append(
            {
                "date": date_str,
                "close": float(close_price),
                "is_volatile": date_str in volatile_dates,
            }
        )

    # Prefer latest value already available in the selected period to reduce extra API calls.
    current_price: Optional[float] = float(grouped_close.iloc[-1]) if not grouped_close.empty else None
    try:
        recent_price = _history_with_retry(ticker, period="5d", interval="1d")
        if not recent_price.empty and "Close" in recent_price.columns:
            current_price = float(recent_price["Close"].dropna().iloc[-1])
    except YFRateLimitError:
        # Keep fallback current price from weekly series.
        pass

    last_dividend: Optional[float] = None
    last_dividend_date: Optional[str] = None
    try:
        dividends = ticker.dividends
        if dividends is not None and not dividends.empty:
            last_dividend = float(dividends.iloc[-1])
            last_dividend_date = dividends.index[-1].strftime("%Y-%m-%d")
    except Exception:
        pass

    company_name: Optional[str] = None
    company_state: Optional[str] = None
    currency: Optional[str] = None
    exchange: Optional[str] = None
    try:
        info = ticker.info or {}
        company_name = info.get("longName") or info.get("shortName")
        company_state = _format_location(info.get("state"), info.get("country"))
        currency = info.get("currency")
        exchange = info.get("exchange")
    except Exception:
        company_name = None
        company_state = None
        currency = None
        exchange = None

    # Fallback profile from lightweight quote endpoint when yfinance.info is empty/unavailable.
    if not company_name or not company_state or not currency or not exchange:
        try:
            quote_payload = _request_json_with_retry(
                "https://query1.finance.yahoo.com/v7/finance/quote",
                {"symbols": ticker_symbol},
            )
            quote_rows = ((quote_payload.get("quoteResponse") or {}).get("result")) or []
            quote_row = quote_rows[0] if quote_rows else {}
            company_name = company_name or quote_row.get("longName") or quote_row.get("shortName")
            company_state = company_state or _format_location(
                None, quote_row.get("country"), quote_row.get("region")
            )
            currency = currency or quote_row.get("currency")
            exchange = exchange or quote_row.get("exchange") or quote_row.get("fullExchangeName")
        except Exception:
            pass
    company_state = company_state or _country_from_symbol(ticker_symbol)

    return AnalysisResult(
        ticker=ticker_symbol.upper(),
        company_name=company_name,
        company_state=company_state,
        currency=currency,
        exchange=exchange,
        current_price=current_price,
        last_dividend=last_dividend,
        last_dividend_date=last_dividend_date,
        threshold_pct=threshold_pct,
        volatile_weeks=volatile_weeks,
        weekly_price_points=weekly_price_points,
        volatility_week_count=len(volatile_weeks),
        max_weekly_gain_pct=max_weekly_gain_pct,
        max_weekly_loss_pct=max_weekly_loss_pct,
        granularity=granularity,
    )


def get_stock_analysis_cached(
    ticker_symbol: str, period: str = "1y", threshold_pct: int = 5, granularity: str = "week"
) -> AnalysisResult:
    cache_key = (ticker_symbol.upper(), period, threshold_pct, granularity)
    now = time.time()
    cached = ANALYSIS_CACHE.get(cache_key)
    if cached:
        cached_at, cached_result = cached
        if now - cached_at < CACHE_TTL_SECONDS:
            return cached_result

    try:
        result = get_stock_analysis(
            ticker_symbol=ticker_symbol,
            period=period,
            threshold_pct=threshold_pct,
            granularity=granularity,
        )
    except (YFRateLimitError, ValueError) as exc:
        try:
            result = get_stock_analysis_http(
                ticker_symbol=ticker_symbol,
                period=period,
                threshold_pct=threshold_pct,
                granularity=granularity,
            )
            ANALYSIS_CACHE[cache_key] = (now, result)
            return result
        except Exception:
            pass
        if cached:
            cached_at, cached_result = cached
            if now - cached_at < STALE_CACHE_MAX_AGE_SECONDS:
                return cached_result
        raise exc

    ANALYSIS_CACHE[cache_key] = (now, result)
    return result


def _analyze_input(
    input_value: str,
    period: str,
    threshold_pct: int,
    granularity: str = "week",
    record_history: bool = True,
) -> AnalysisResult:
    resolved_ticker = _resolve_input_symbol(input_value)
    result = get_stock_analysis_cached(
        resolved_ticker,
        period=period,
        threshold_pct=threshold_pct,
        granularity=granularity,
    )

    if record_history:
        _record_recent_search(input_value, result.ticker, period, threshold_pct, granularity)
    return result


def _export_csv_response(result: AnalysisResult, query: str) -> tuple[str, int]:
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)

    writer.writerow(["query", query])
    writer.writerow(["ticker", result.ticker])
    writer.writerow(["company_name", result.company_name or ""])
    writer.writerow(["period", "custom"])
    writer.writerow(["granularity", result.granularity])
    writer.writerow(["threshold_pct", result.threshold_pct])
    writer.writerow([])
    writer.writerow(["period_end", "change_pct"])
    for week in result.volatile_weeks:
        writer.writerow([week.get("period_end") or week.get("week_end"), week["change_pct"]])

    response = make_response(csv_buffer.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers[
        "Content-Disposition"
    ] = f"attachment; filename=volatility_{result.ticker}_{result.threshold_pct}pct.csv"
    return response


init_db()


@app.route("/register", methods=["GET", "POST"])
def register():
    error: Optional[str] = None
    username = ""
    email = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if len(username) < 3:
            error = "Username must have at least 3 characters."
        elif TWO_FA_FORCE_EMAIL and (
            not email
            or len(email) > 254
            or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email)
        ):
            error = "Please provide a valid email for 2FA."
        elif len(password) < 6:
            error = "Password must have at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            with _db_connect() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
            if exists:
                error = "Username already exists."
            else:
                password_hash = generate_password_hash(password)
                code = _issue_registration_otp(username, email, password_hash)
                sent = _send_register_email(email, code)
                if not sent and not (app.config.get("DEBUG") and TWO_FA_DEV_SHOW_CODE):
                    error = "Could not send verification email right now. Try again later."
                else:
                    _clear_pending_registration()
                    session["pending_register_username"] = username
                    session["pending_register_email"] = email
                    session["pending_register_password_hash"] = password_hash
                    session["pending_register_created_at"] = int(time.time())
                    if (not sent) and app.config.get("DEBUG") and TWO_FA_DEV_SHOW_CODE:
                        session["pending_register_dev_code"] = code
                    return redirect(url_for("register_verify"))

    return render_template("register.html", error=error, username=username, email=email)


@app.route("/register/verify", methods=["GET", "POST"])
def register_verify():
    username = str(session.get("pending_register_username") or "")
    email = str(session.get("pending_register_email") or "")
    password_hash = str(session.get("pending_register_password_hash") or "")
    if not username or not email or not password_hash:
        return redirect(url_for("register"))

    error: Optional[str] = None
    created_at = int(session.get("pending_register_created_at", 0))
    if created_at and time.time() - created_at > TWO_FA_CODE_TTL_SECONDS:
        _clear_pending_registration()
        return redirect(url_for("register"))

    if request.method == "POST":
        if not _rate_limit(scope="register-2fa", limit=20, window_seconds=60):
            error = "Too many attempts. Wait a minute and try again."
        else:
            code = re.sub(r"\D", "", request.form.get("code", ""))
            if len(code) != 6:
                error = "Enter the 6-digit code."
            else:
                ok, message = _verify_registration_otp(username, email, code)
                if not ok:
                    error = message
                else:
                    try:
                        with _db_connect() as conn:
                            conn.execute(
                                "INSERT INTO users(username, email, password_hash) VALUES(?, ?, ?)",
                                (username, email, password_hash),
                            )
                            conn.commit()
                            row = conn.execute(
                                "SELECT id FROM users WHERE username = ?",
                                (username,),
                            ).fetchone()
                        session["user_id"] = int(row["id"])
                        _clear_pending_registration()
                        return redirect(url_for("index"))
                    except sqlite3.IntegrityError:
                        _clear_pending_registration()
                        return redirect(url_for("login"))

    return render_template(
        "register_verify.html",
        error=error,
        username=username,
        masked_email=_mask_email(email),
        dev_code=session.get("pending_register_dev_code"),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    error: Optional[str] = None
    username = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        next_candidate = request.args.get("next", "")
        next_url = next_candidate if _is_safe_next_url(next_candidate) else url_for("index")

        with _db_connect() as conn:
            row = conn.execute(
                "SELECT id, email, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()

        if not row or not check_password_hash(str(row["password_hash"]), password):
            error = "Invalid username or password."
        else:
            user_id = int(row["id"])
            email = str(row["email"] or "").strip()
            if not email and TWO_FA_FORCE_EMAIL:
                error = "Your account has no email for 2FA. Create a new account with email."
            else:
                code = _issue_login_otp(user_id)
                sent = _send_2fa_email(email, code) if email else False

                _clear_pending_2fa()
                session["pending_2fa_user_id"] = user_id
                session["pending_2fa_next"] = next_url
                session["pending_2fa_email"] = _mask_email(email) if email else "N/A"
                session["pending_2fa_username"] = username
                session["pending_2fa_created_at"] = int(time.time())

                if (not sent) and app.config.get("DEBUG") and TWO_FA_DEV_SHOW_CODE:
                    session["pending_2fa_dev_code"] = code

                return redirect(url_for("login_verify"))

    return render_template("login.html", error=error, username=username)


@app.route("/login/verify", methods=["GET", "POST"])
def login_verify():
    pending_user_id = session.get("pending_2fa_user_id")
    if not pending_user_id:
        return redirect(url_for("login"))

    error: Optional[str] = None
    created_at = int(session.get("pending_2fa_created_at", 0))
    if created_at and time.time() - created_at > TWO_FA_CODE_TTL_SECONDS:
        _clear_pending_2fa()
        return redirect(url_for("login"))

    if request.method == "POST":
        if not _rate_limit(scope="2fa", limit=20, window_seconds=60):
            error = "Too many attempts. Wait a minute and try again."
        else:
            code = re.sub(r"\D", "", request.form.get("code", ""))
            if len(code) != 6:
                error = "Enter the 6-digit code."
            else:
                ok, message = _verify_login_otp(int(pending_user_id), code)
                if not ok:
                    error = message
                else:
                    next_url = str(session.get("pending_2fa_next") or url_for("index"))
                    session["user_id"] = int(pending_user_id)
                    _clear_pending_2fa()
                    return redirect(next_url if _is_safe_next_url(next_url) else url_for("index"))

    return render_template(
        "login_verify.html",
        error=error,
        username=session.get("pending_2fa_username", ""),
        masked_email=session.get("pending_2fa_email", ""),
        dev_code=session.get("pending_2fa_dev_code"),
    )


@app.post("/logout")
def logout():
    _clear_pending_2fa()
    _clear_pending_registration()
    session.clear()
    return redirect(url_for("index"))


@app.post("/recent/clear")
def recent_clear():
    _clear_search_history()
    return redirect(request.referrer or url_for("index"))


@app.post("/watchlist/add")
@_login_required
def watchlist_add():
    user = _get_current_user()
    if user:
        _watchlist_add(int(user["id"]), request.form.get("symbol", ""))
    return redirect(request.referrer or url_for("index"))


@app.post("/watchlist/remove")
@_login_required
def watchlist_remove():
    user = _get_current_user()
    if user:
        _watchlist_remove(int(user["id"]), request.form.get("symbol", ""))
    return redirect(request.referrer or url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    current_user = _get_current_user()
    watchlist: list[str] = _watchlist_symbols(int(current_user["id"])) if current_user else []
    exchange_listings: list[dict[str, str]] = []

    input_value = ""
    selected_period = "1y"
    selected_threshold = 5
    selected_granularity = "week"

    if request.method == "POST":
        input_value = request.form.get("ticker", "").strip()
        selected_period = _normalize_period(request.form.get("period", "1y"))
        selected_granularity = _normalize_granularity(request.form.get("granularity", "week"))
        selected_threshold = _normalize_threshold(
            request.form.get("threshold", "5"), selected_granularity
        )
    elif request.args.get("query"):
        input_value = request.args.get("query", "").strip()
        selected_period = _normalize_period(request.args.get("period", "1y"))
        selected_granularity = _normalize_granularity(request.args.get("granularity", "week"))
        selected_threshold = _normalize_threshold(
            request.args.get("threshold", "5"), selected_granularity
        )

    if input_value:
        try:
            result = _analyze_input(
                input_value,
                period=selected_period,
                threshold_pct=selected_threshold,
                granularity=selected_granularity,
            )
        except YFRateLimitError:
            # Extra fallback path when yfinance is throttled.
            try:
                result = get_stock_analysis_http(
                    ticker_symbol=input_value.upper(),
                    period=selected_period,
                    threshold_pct=selected_threshold,
                    granularity=selected_granularity,
                )
            except Exception:
                try:
                    resolved = _search_symbol_by_name_http(input_value)
                    result = get_stock_analysis_http(
                        ticker_symbol=resolved,
                        period=selected_period,
                        threshold_pct=selected_threshold,
                        granularity=selected_granularity,
                    )
                except Exception:
                    fallback_result = _latest_cached_result()
                    if fallback_result is not None:
                        result = fallback_result
                    else:
                        error = "Could not refresh data right now. Try another symbol or range."
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Unexpected error while analyzing input: %s", input_value)
            error = "Could not analyze this input right now. Please try again."
    elif request.method == "POST":
        error = "Please enter a ticker symbol or company name."

    if result:
        exchange_listings = _get_exchange_listings(
            ticker_symbol=result.ticker,
            company_name=result.company_name,
            original_query=input_value or result.ticker,
        )

    return render_template(
        "index.html",
        result=result,
        error=error,
        current_user=current_user,
        watchlist=watchlist,
        period_options=PERIOD_OPTIONS,
        selected_period=selected_period,
        threshold_options=THRESHOLD_OPTIONS,
        day_threshold_options=DAY_THRESHOLD_OPTIONS,
        week_threshold_options=WEEK_THRESHOLD_OPTIONS,
        selected_threshold=selected_threshold,
        granularity_options=GRANULARITY_OPTIONS,
        selected_granularity=selected_granularity,
        input_value=input_value,
        recent_searches=list(RECENT_SEARCHES),
        search_history=list(SEARCH_HISTORY),
        exchange_listings=exchange_listings,
    )


@app.route("/crypto", methods=["GET", "POST"])
def crypto():
    current_user = _get_current_user()
    input_value = ""
    result: Optional[CryptoChartResult] = None
    error: Optional[str] = None
    selected_range = "24h"
    log_scale = "0"

    if request.method == "POST":
        input_value = request.form.get("symbol", "").strip()
        selected_range = _normalize_crypto_range(request.form.get("range", "24h"))
        log_scale = "1" if request.form.get("log_scale") == "1" else "0"
    elif request.args.get("symbol"):
        input_value = request.args.get("symbol", "").strip()
        selected_range = _normalize_crypto_range(request.args.get("range", "24h"))
        log_scale = "1" if request.args.get("log_scale") == "1" else "0"

    if input_value:
        try:
            base_symbol, coin_name = _resolve_crypto_input(input_value)
            points, current_price, change_pct = _fetch_crypto_chart_yahoo(base_symbol, selected_range)
            if not points:
                raise ValueError("Could not load chart data for this coin right now.")
            summary = get_crypto_price_summary(base_symbol)
            display_price = summary.price_usd if summary.price_usd is not None else current_price
            result = CryptoChartResult(
                symbol=base_symbol,
                display_symbol=f"{base_symbol}/USD",
                name=coin_name,
                selected_range=selected_range,
                current_price_usd=display_price,
                change_pct=change_pct,
                points=points,
                as_of=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Unexpected crypto lookup error: %s", input_value)
            error = "Could not load crypto chart right now. Please try again."
    elif request.method == "POST":
        error = "Please enter a crypto symbol."

    return render_template(
        "crypto.html",
        current_user=current_user,
        input_value=input_value,
        selected_range=selected_range,
        range_options=CRYPTO_RANGE_OPTIONS,
        log_scale=log_scale,
        result=result,
        error=error,
    )


@app.get("/crypto/markets")
def crypto_markets():
    market = get_crypto_market_overview_live()
    current_user = _get_current_user()
    return render_template("crypto_markets.html", market=market, current_user=current_user)


@app.get("/markets")
def markets():
    market = get_market_overview_live()
    return render_template("markets.html", market=market)


@app.get("/export.csv")
def export_csv():
    query = request.args.get("query", "").strip()
    if not query:
        return ("Missing query parameter.", 400)

    period = _normalize_period(request.args.get("period", "1y"))
    granularity = _normalize_granularity(request.args.get("granularity", "week"))
    threshold = _normalize_threshold(request.args.get("threshold", "5"), granularity)

    try:
        result = _analyze_input(
            query,
            period=period,
            threshold_pct=threshold,
            granularity=granularity,
            record_history=False,
        )
    except YFRateLimitError:
        return ("Export is temporarily unavailable. Please try again shortly.", 429)
    except ValueError as exc:
        return (str(exc), 400)
    except Exception:
        app.logger.exception("Unexpected error during CSV export: %s", query)
        return ("Could not export CSV right now.", 500)

    return _export_csv_response(result, query)


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        timestamp=int(time.time()),
        cache_entries=len(ANALYSIS_CACHE),
        recent_searches=len(RECENT_SEARCHES),
        search_history=len(SEARCH_HISTORY),
        auto_refresh_running=bool(AUTO_REFRESH_THREAD and AUTO_REFRESH_THREAD.is_alive()),
        auto_refresh_interval_seconds=AUTO_REFRESH_INTERVAL_SECONDS,
        auto_refresh_last_run_at=int(AUTO_REFRESH_LAST_RUN_AT) if AUTO_REFRESH_LAST_RUN_AT else None,
    )


if __name__ == "__main__":
    start_auto_refresh()
    app.run(debug=app.config["DEBUG"])
