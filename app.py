from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import csv
import io
import math
import os
import random
import re
import time
from typing import Optional

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from flask import Flask, jsonify, make_response, render_template, request
from yfinance.exceptions import YFRateLimitError

app = Flask(__name__)
app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "0") == "1"

PERIOD_OPTIONS = [("6mo", "6 Months"), ("1y", "1 Year"), ("2y", "2 Years"), ("5y", "5 Years")]
THRESHOLD_OPTIONS = [3, 5, 10]
CACHE_TTL_SECONDS = 900
STALE_CACHE_MAX_AGE_SECONDS = 21600
MAX_RECENT_SEARCHES = 8

ANALYSIS_CACHE: dict[tuple[str, str, int], tuple[float, "AnalysisResult"]] = {}
RECENT_SEARCHES: deque[dict[str, str]] = deque(maxlen=MAX_RECENT_SEARCHES)
LAST_MARKET_OVERVIEW: Optional[dict] = None

# Global-ish liquid tickers (US + EU + Asia) in Yahoo symbol format.
MARKET_URLS = {
    "most_active": "https://finance.yahoo.com/markets/stocks/most-active/",
    "gainers": "https://finance.yahoo.com/markets/stocks/gainers/",
    "losers": "https://finance.yahoo.com/markets/stocks/losers/",
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


def _scrape_market_table(url: str, retries: int = 3) -> list[dict]:
    delay = 0.8
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if not table:
                return []
            rows = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(cells) < 8:
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
                    "currency": None,
                    "market_cap": _to_number_with_suffix(cells[8]),
                }
                rows.append(row)
            return rows
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(delay + random.uniform(0.0, 0.3))
            delay *= 1.8
    return []


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
        "currency": quote.get("currency"),
        "market_cap": quote.get("marketCap") or 0,
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

    # Live on every load: scrape Yahoo market pages directly (works even when quote/screener APIs are limited).
    actives_rows = _scrape_market_table(MARKET_URLS["most_active"])
    gainers_rows = _scrape_market_table(MARKET_URLS["gainers"])
    losers_rows = _scrape_market_table(MARKET_URLS["losers"])

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
    ticker: yf.Ticker, period: str, retries: int = 5, initial_delay: float = 1.2
):
    delay = initial_delay
    for attempt in range(retries):
        try:
            return ticker.history(period=period, auto_adjust=False)
        except YFRateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(delay + random.uniform(0.0, 0.4))
            delay *= 2


def _search_symbol_by_name(query: str, retries: int = 5, initial_delay: float = 1.2) -> str:
    delay = initial_delay
    for attempt in range(retries):
        try:
            search_result = yf.Search(query, max_results=10)
            quotes = getattr(search_result, "quotes", []) or []
            for quote in quotes:
                symbol = quote.get("symbol")
                if symbol:
                    return str(symbol).upper()
            raise ValueError(f"No ticker found for company name '{query}'.")
        except YFRateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(delay + random.uniform(0.0, 0.4))
            delay *= 2


def _normalize_period(requested_period: str) -> str:
    return requested_period if requested_period in dict(PERIOD_OPTIONS) else "2y"


def _normalize_threshold(requested_threshold: str) -> int:
    try:
        threshold = int(requested_threshold)
    except (TypeError, ValueError):
        return 5
    return threshold if threshold in THRESHOLD_OPTIONS else 5


def _record_recent_search(query: str, ticker: str, period: str, threshold_pct: int):
    entry = {
        "query": query,
        "ticker": ticker,
        "period": period,
        "threshold": str(threshold_pct),
        "timestamp": time.strftime("%H:%M:%S"),
    }

    deduped = deque(
        (x for x in RECENT_SEARCHES if not (
            x["query"].lower() == query.lower()
            and x["ticker"] == ticker
            and x["period"] == period
            and x["threshold"] == str(threshold_pct)
        )),
        maxlen=MAX_RECENT_SEARCHES,
    )
    deduped.appendleft(entry)
    RECENT_SEARCHES.clear()
    RECENT_SEARCHES.extend(deduped)


def _latest_cached_result() -> Optional["AnalysisResult"]:
    if not ANALYSIS_CACHE:
        return None
    latest_timestamp = max(ts for ts, _ in ANALYSIS_CACHE.values())
    for ts, result in ANALYSIS_CACHE.values():
        if ts == latest_timestamp:
            return result
    return None


def get_stock_analysis(ticker_symbol: str, period: str = "2y", threshold_pct: int = 5) -> AnalysisResult:
    ticker = yf.Ticker(ticker_symbol)

    price_history = _history_with_retry(ticker, period=period)
    if price_history.empty or "Close" not in price_history.columns:
        raise ValueError("No price data found for this ticker.")

    weekly_close = price_history["Close"].resample("W-FRI").last().dropna()
    weekly_change = weekly_close.pct_change().dropna()
    threshold_ratio = threshold_pct / 100.0
    significant_weeks = weekly_change[weekly_change.abs() > threshold_ratio]

    volatile_weeks: list[dict[str, str | float]] = []
    for week_end, change in significant_weeks.items():
        change_pct = float(change * 100)
        volatile_weeks.append(
            {
                "week_end": week_end.strftime("%Y-%m-%d"),
                "change_pct": f"{change_pct:.2f}%",
                "change_value": change_pct,
            }
        )

    all_weekly_changes_pct = weekly_change * 100
    max_weekly_gain_pct = (
        float(all_weekly_changes_pct.max()) if not all_weekly_changes_pct.empty else None
    )
    max_weekly_loss_pct = (
        float(all_weekly_changes_pct.min()) if not all_weekly_changes_pct.empty else None
    )

    volatile_dates = {d.strftime("%Y-%m-%d") for d in significant_weeks.index}
    weekly_price_points: list[dict[str, str | float | bool]] = []
    for point_date, close_price in weekly_close.items():
        date_str = point_date.strftime("%Y-%m-%d")
        weekly_price_points.append(
            {
                "date": date_str,
                "close": float(close_price),
                "is_volatile": date_str in volatile_dates,
            }
        )

    # Prefer latest value already available in the selected period to reduce extra API calls.
    current_price: Optional[float] = float(weekly_close.iloc[-1]) if not weekly_close.empty else None
    try:
        recent_price = _history_with_retry(ticker, period="5d")
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
        company_state = info.get("state")
        currency = info.get("currency")
        exchange = info.get("exchange")
    except Exception:
        company_name = None
        company_state = None
        currency = None
        exchange = None

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
    )


def get_stock_analysis_cached(
    ticker_symbol: str, period: str = "2y", threshold_pct: int = 5
) -> AnalysisResult:
    cache_key = (ticker_symbol.upper(), period, threshold_pct)
    now = time.time()
    cached = ANALYSIS_CACHE.get(cache_key)
    if cached:
        cached_at, cached_result = cached
        if now - cached_at < CACHE_TTL_SECONDS:
            return cached_result

    try:
        result = get_stock_analysis(
            ticker_symbol=ticker_symbol, period=period, threshold_pct=threshold_pct
        )
    except YFRateLimitError:
        if cached:
            cached_at, cached_result = cached
            if now - cached_at < STALE_CACHE_MAX_AGE_SECONDS:
                return cached_result
        raise

    ANALYSIS_CACHE[cache_key] = (now, result)
    return result


def _analyze_input(
    input_value: str, period: str, threshold_pct: int, record_history: bool = True
) -> AnalysisResult:
    try:
        result = get_stock_analysis_cached(
            input_value.upper(), period=period, threshold_pct=threshold_pct
        )
    except ValueError:
        resolved_ticker = _search_symbol_by_name(input_value)
        result = get_stock_analysis_cached(
            resolved_ticker, period=period, threshold_pct=threshold_pct
        )

    if record_history:
        _record_recent_search(input_value, result.ticker, period, threshold_pct)
    return result


def _export_csv_response(result: AnalysisResult, query: str) -> tuple[str, int]:
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)

    writer.writerow(["query", query])
    writer.writerow(["ticker", result.ticker])
    writer.writerow(["company_name", result.company_name or ""])
    writer.writerow(["period", "custom"])
    writer.writerow(["threshold_pct", result.threshold_pct])
    writer.writerow([])
    writer.writerow(["week_end", "change_pct"])
    for week in result.volatile_weeks:
        writer.writerow([week["week_end"], week["change_pct"]])

    response = make_response(csv_buffer.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers[
        "Content-Disposition"
    ] = f"attachment; filename=volatility_{result.ticker}_{result.threshold_pct}pct.csv"
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None

    input_value = ""
    selected_period = "2y"
    selected_threshold = 5

    if request.method == "POST":
        input_value = request.form.get("ticker", "").strip()
        selected_period = _normalize_period(request.form.get("period", "2y"))
        selected_threshold = _normalize_threshold(request.form.get("threshold", "5"))
    elif request.args.get("query"):
        input_value = request.args.get("query", "").strip()
        selected_period = _normalize_period(request.args.get("period", "2y"))
        selected_threshold = _normalize_threshold(request.args.get("threshold", "5"))

    if input_value:
        try:
            result = _analyze_input(
                input_value, period=selected_period, threshold_pct=selected_threshold
            )
        except YFRateLimitError:
            # Avoid noisy user-facing rate-limit popups: fall back to latest cached snapshot.
            fallback_result = _latest_cached_result()
            if fallback_result is not None:
                result = fallback_result
            else:
                error = "Data could not be loaded right now. Please try again shortly."
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Unexpected error while analyzing input: %s", input_value)
            error = "Could not analyze this input right now. Please try again."
    elif request.method == "POST":
        error = "Please enter a ticker symbol or company name."

    return render_template(
        "index.html",
        result=result,
        error=error,
        period_options=PERIOD_OPTIONS,
        selected_period=selected_period,
        threshold_options=THRESHOLD_OPTIONS,
        selected_threshold=selected_threshold,
        input_value=input_value,
        recent_searches=list(RECENT_SEARCHES),
    )


@app.get("/markets")
def markets():
    market = get_market_overview_live()
    return render_template("markets.html", market=market)


@app.get("/export.csv")
def export_csv():
    query = request.args.get("query", "").strip()
    if not query:
        return ("Missing query parameter.", 400)

    period = _normalize_period(request.args.get("period", "2y"))
    threshold = _normalize_threshold(request.args.get("threshold", "5"))

    try:
        result = _analyze_input(query, period=period, threshold_pct=threshold, record_history=False)
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
    )


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
