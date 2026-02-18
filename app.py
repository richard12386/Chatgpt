from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import csv
import io
import os
import random
import time
from typing import Optional

import yfinance as yf
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
            error = "Data provider is busy right now. Please try again shortly."
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
        return ("Data provider is busy right now. Please try export again shortly.", 429)
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
