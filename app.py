from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Optional

import yfinance as yf
from flask import Flask, render_template, request
from yfinance.exceptions import YFRateLimitError

app = Flask(__name__)
app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "0") == "1"
PERIOD_OPTIONS = [("6mo", "6 Months"), ("1y", "1 Year"), ("2y", "2 Years"), ("5y", "5 Years")]
CACHE_TTL_SECONDS = 300
ANALYSIS_CACHE: dict[tuple[str, str], tuple[float, "AnalysisResult"]] = {}


@dataclass
class AnalysisResult:
    ticker: str
    company_name: Optional[str]
    current_price: Optional[float]
    last_dividend: Optional[float]
    last_dividend_date: Optional[str]
    company_state: Optional[str]
    volatile_weeks: list[dict[str, str]]


def _history_with_retry(
    ticker: yf.Ticker, period: str, retries: int = 3, initial_delay: float = 1.0
):
    delay = initial_delay
    for attempt in range(retries):
        try:
            return ticker.history(period=period, auto_adjust=False)
        except YFRateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _search_symbol_by_name(query: str, retries: int = 3, initial_delay: float = 1.0) -> str:
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
            time.sleep(delay)
            delay *= 2


def get_stock_analysis(ticker_symbol: str, period: str = "2y") -> AnalysisResult:
    ticker = yf.Ticker(ticker_symbol)

    price_history = _history_with_retry(ticker, period=period)
    if price_history.empty or "Close" not in price_history.columns:
        raise ValueError("No price data found for this ticker.")

    weekly_close = price_history["Close"].resample("W-FRI").last().dropna()
    weekly_change = weekly_close.pct_change().dropna()
    significant_weeks = weekly_change[weekly_change.abs() > 0.05]

    volatile_weeks: list[dict[str, str]] = []
    for week_end, change in significant_weeks.items():
        volatile_weeks.append(
            {
                "week_end": week_end.strftime("%Y-%m-%d"),
                "change_pct": f"{change * 100:.2f}%",
            }
        )

    recent_price = _history_with_retry(ticker, period="5d")
    current_price: Optional[float] = None
    if not recent_price.empty and "Close" in recent_price.columns:
        current_price = float(recent_price["Close"].dropna().iloc[-1])

    dividends = ticker.dividends
    last_dividend: Optional[float] = None
    last_dividend_date: Optional[str] = None
    if dividends is not None and not dividends.empty:
        last_dividend = float(dividends.iloc[-1])
        last_dividend_date = dividends.index[-1].strftime("%Y-%m-%d")

    company_name: Optional[str] = None
    company_state: Optional[str] = None
    try:
        info = ticker.info or {}
        company_name = info.get("longName") or info.get("shortName")
        company_state = info.get("state")
    except Exception:
        # Some tickers do not expose profile data consistently.
        company_name = None
        company_state = None

    return AnalysisResult(
        ticker=ticker_symbol.upper(),
        company_name=company_name,
        current_price=current_price,
        last_dividend=last_dividend,
        last_dividend_date=last_dividend_date,
        company_state=company_state,
        volatile_weeks=volatile_weeks,
    )


def get_stock_analysis_cached(ticker_symbol: str, period: str = "2y") -> AnalysisResult:
    cache_key = (ticker_symbol.upper(), period)
    now = time.time()
    cached = ANALYSIS_CACHE.get(cache_key)
    if cached:
        cached_at, cached_result = cached
        if now - cached_at < CACHE_TTL_SECONDS:
            return cached_result

    result = get_stock_analysis(ticker_symbol=ticker_symbol, period=period)
    ANALYSIS_CACHE[cache_key] = (now, result)
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    selected_period = "2y"

    if request.method == "POST":
        input_value = request.form.get("ticker", "").strip()
        requested_period = request.form.get("period", "2y")
        selected_period = requested_period if requested_period in dict(PERIOD_OPTIONS) else "2y"
        if not input_value:
            error = "Please enter a ticker symbol or company name."
        else:
            try:
                # First try as direct ticker input.
                try:
                    result = get_stock_analysis_cached(input_value.upper(), period=selected_period)
                except ValueError:
                    # If no data for direct ticker, try to resolve by company name.
                    resolved_ticker = _search_symbol_by_name(input_value)
                    result = get_stock_analysis_cached(resolved_ticker, period=selected_period)
            except YFRateLimitError:
                error = (
                    "Yahoo Finance is temporarily rate-limiting requests. "
                    "Please wait a minute and try again."
                )
            except ValueError as exc:
                error = str(exc)
            except Exception:
                app.logger.exception("Unexpected error while analyzing input: %s", input_value)
                error = "Could not analyze this input right now. Please try again."

    return render_template(
        "index.html",
        result=result,
        error=error,
        period_options=PERIOD_OPTIONS,
        selected_period=selected_period,
    )


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
