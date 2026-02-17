from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request

app = Flask(__name__)


@dataclass
class AnalysisResult:
    ticker: str
    current_price: Optional[float]
    last_dividend: Optional[float]
    last_dividend_date: Optional[str]
    volatile_weeks: list[dict[str, str]]


def get_stock_analysis(ticker_symbol: str, period: str = "2y") -> AnalysisResult:
    ticker = yf.Ticker(ticker_symbol)

    price_history = ticker.history(period=period, auto_adjust=False)
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

    recent_price = ticker.history(period="5d", auto_adjust=False)
    current_price: Optional[float] = None
    if not recent_price.empty and "Close" in recent_price.columns:
        current_price = float(recent_price["Close"].dropna().iloc[-1])

    dividends = ticker.dividends
    last_dividend: Optional[float] = None
    last_dividend_date: Optional[str] = None
    if dividends is not None and not dividends.empty:
        last_dividend = float(dividends.iloc[-1])
        last_dividend_date = dividends.index[-1].strftime("%Y-%m-%d")

    return AnalysisResult(
        ticker=ticker_symbol.upper(),
        current_price=current_price,
        last_dividend=last_dividend,
        last_dividend_date=last_dividend_date,
        volatile_weeks=volatile_weeks,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()
        if not ticker:
            error = "Please enter a ticker symbol."
        else:
            try:
                result = get_stock_analysis(ticker)
            except Exception as exc:
                error = f"Could not analyze ticker '{ticker}': {exc}"

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
