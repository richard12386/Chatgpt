import unittest
from unittest.mock import Mock, patch

from yfinance.exceptions import YFRateLimitError

import app


class HistoryRetryTests(unittest.TestCase):
    @patch("app.time.sleep")
    def test_history_with_retry_retries_then_returns(self, sleep_mock: Mock):
        ticker = Mock()
        expected = object()
        ticker.history.side_effect = [YFRateLimitError(), YFRateLimitError(), expected]

        result = app._history_with_retry(ticker, period="1y", retries=3, initial_delay=1.0)

        self.assertIs(result, expected)
        self.assertEqual(ticker.history.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)
        first_sleep = sleep_mock.call_args_list[0].args[0]
        second_sleep = sleep_mock.call_args_list[1].args[0]
        self.assertGreaterEqual(first_sleep, 1.0)
        self.assertLess(first_sleep, 1.5)
        self.assertGreater(second_sleep, first_sleep)

    @patch("app.time.sleep")
    def test_history_with_retry_raises_after_last_retry(self, sleep_mock: Mock):
        ticker = Mock()
        ticker.history.side_effect = [YFRateLimitError(), YFRateLimitError(), YFRateLimitError()]

        with self.assertRaises(YFRateLimitError):
            app._history_with_retry(ticker, period="1y", retries=3, initial_delay=1.0)

        self.assertEqual(ticker.history.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)


class RouteTests(unittest.TestCase):
    def setUp(self):
        app.app.config["TESTING"] = True
        self.client = app.app.test_client()
        with app._db_connect() as conn:
            conn.execute("DELETE FROM login_otp")
            conn.execute("DELETE FROM watchlist")
            conn.execute("DELETE FROM users")
            conn.commit()

    def test_post_empty_input_returns_validation_error(self):
        response = self.client.post("/", data={"ticker": "", "period": "2y"})
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Please enter a ticker symbol or company name.", body)

    @patch("app.get_stock_analysis_cached")
    def test_post_valid_ticker_renders_analysis(self, analysis_mock: Mock):
        analysis_mock.return_value = app.AnalysisResult(
            ticker="AAPL",
            company_name="Apple Inc.",
            company_state="CA",
            currency="USD",
            exchange="NMS",
            current_price=123.45,
            last_dividend=0.25,
            last_dividend_date="2026-01-15",
            threshold_pct=5,
            volatile_weeks=[{"week_end": "2026-01-09", "change_pct": "6.10%", "change_value": 6.1}],
            weekly_price_points=[{"date": "2026-01-09", "close": 123.45, "is_volatile": True}],
            volatility_week_count=1,
            max_weekly_gain_pct=6.1,
            max_weekly_loss_pct=-3.2,
            granularity="week",
        )

        response = self.client.post("/", data={"ticker": "AAPL", "period": "1y", "threshold": "5"})
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Apple Inc.", body)
        self.assertIn("AAPL", body)
        self.assertIn("6.10%", body)
        analysis_mock.assert_called_once_with("AAPL", period="1y", threshold_pct=5, granularity="week")

    @patch("app._resolve_input_symbol")
    @patch("app.get_stock_analysis_cached")
    def test_post_company_name_uses_search_fallback(
        self, analysis_mock: Mock, resolve_mock: Mock
    ):
        resolve_mock.return_value = "HO.PA"
        analysis_mock.return_value = app.AnalysisResult(
            ticker="HO.PA",
            company_name="Thales S.A.",
            company_state=None,
            currency="EUR",
            exchange="PAR",
            current_price=150.0,
            last_dividend=1.0,
            last_dividend_date="2026-01-01",
            threshold_pct=5,
            volatile_weeks=[],
            weekly_price_points=[],
            volatility_week_count=0,
            max_weekly_gain_pct=None,
            max_weekly_loss_pct=None,
            granularity="week",
        )

        response = self.client.post("/", data={"ticker": "Thales", "period": "2y", "threshold": "5"})
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Thales S.A.", body)
        resolve_mock.assert_called_once_with("Thales")
        analysis_mock.assert_called_once()
        self.assertEqual(analysis_mock.call_args.args[0], "HO.PA")
        self.assertEqual(analysis_mock.call_args.kwargs["threshold_pct"], 5)
        self.assertEqual(analysis_mock.call_args.kwargs["granularity"], "week")

    @patch("app._analyze_input")
    def test_export_csv_returns_attachment(self, analyze_mock: Mock):
        analyze_mock.return_value = app.AnalysisResult(
            ticker="AAPL",
            company_name="Apple Inc.",
            company_state="CA",
            currency="USD",
            exchange="NMS",
            current_price=123.45,
            last_dividend=0.25,
            last_dividend_date="2026-01-15",
            threshold_pct=5,
            volatile_weeks=[{"week_end": "2026-01-09", "change_pct": "6.10%", "change_value": 6.1}],
            weekly_price_points=[],
            volatility_week_count=1,
            max_weekly_gain_pct=6.1,
            max_weekly_loss_pct=-3.2,
            granularity="week",
        )

        response = self.client.get("/export.csv?query=AAPL&period=1y&threshold=5")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.content_type)
        self.assertIn("period_end,change_pct", body)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")

    @patch("app._send_2fa_email", return_value=True)
    @patch("app.random.randint", return_value=123456)
    def test_login_redirects_to_2fa_verify(self, _rand_mock: Mock, _send_mock: Mock):
        with app._db_connect() as conn:
            conn.execute(
                "INSERT INTO users(username, email, password_hash) VALUES(?, ?, ?)",
                ("alice", "alice@example.com", app.generate_password_hash("secret123")),
            )
            conn.commit()

        response = self.client.post(
            "/login",
            data={"username": "alice", "password": "secret123"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/verify", response.headers.get("Location", ""))

    @patch("app._send_2fa_email", return_value=True)
    @patch("app.random.randint", return_value=123456)
    def test_login_2fa_verify_sets_user_session(self, _rand_mock: Mock, _send_mock: Mock):
        with app._db_connect() as conn:
            conn.execute(
                "INSERT INTO users(username, email, password_hash) VALUES(?, ?, ?)",
                ("alice", "alice@example.com", app.generate_password_hash("secret123")),
            )
            conn.commit()

        step1 = self.client.post(
            "/login",
            data={"username": "alice", "password": "secret123"},
            follow_redirects=False,
        )
        self.assertEqual(step1.status_code, 302)
        self.assertIn("/login/verify", step1.headers.get("Location", ""))

        step2 = self.client.post(
            "/login/verify",
            data={"code": "123456"},
            follow_redirects=False,
        )
        self.assertEqual(step2.status_code, 302)
        self.assertEqual(step2.headers.get("Location"), "/")

    def test_crypto_get_renders_page(self):
        response = self.client.get("/crypto")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Crypto Hub", body)
        self.assertIn("Load Chart", body)

    def test_crypto_post_empty_input_validation(self):
        response = self.client.post("/crypto", data={"symbol": ""})
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Please enter a crypto symbol.", body)

    def test_crypto_markets_page(self):
        response = self.client.get("/crypto/markets")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Crypto Markets", body)
        self.assertIn("TOP 10 Gainers", body)

    @patch("app.get_crypto_price_summary")
    @patch("app._fetch_crypto_chart_yahoo")
    @patch("app._resolve_crypto_input")
    def test_crypto_post_valid_renders_chart(
        self,
        resolve_mock: Mock,
        fetch_chart_mock: Mock,
        summary_mock: Mock,
    ):
        resolve_mock.return_value = ("BTC", "Bitcoin")
        fetch_chart_mock.return_value = (
            [
                {"label": "02-19 10:00", "price": 65000.0},
                {"label": "02-19 10:05", "price": 65100.0},
            ],
            65100.0,
            0.15,
        )
        summary_mock.return_value = app.CryptoPriceResult(
            symbol="BTC",
            display_symbol="BTC/USD",
            price_usd=65050.0,
            source_prices={
                "CoinMarketCap": 65020.0,
                "Yahoo Finance": 65010.0,
                "CoinGecko API": 65100.0,
                "Binance API": 65070.0,
            },
            source_symbols={
                "CoinMarketCap": "bitcoin",
                "Yahoo Finance": "BTC-USD",
                "CoinGecko API": "bitcoin",
                "Binance API": "BTCUSDT",
            },
            as_of="2026-02-19 10:00:00",
        )

        response = self.client.post("/crypto", data={"symbol": "bitcoin", "range": "24h", "log_scale": "0"})
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Bitcoin (BTC/USD)", body)
        self.assertIn("cryptoChart", body)
        resolve_mock.assert_called_once_with("bitcoin")
        fetch_chart_mock.assert_called_once_with("BTC", "24h")
        summary_mock.assert_called_once_with("BTC")

    @patch("app._analyze_input")
    @patch("app._latest_cached_result")
    @patch("app.get_stock_analysis_http")
    @patch("app._search_symbol_by_name_http")
    def test_rate_limit_without_fallback_shows_error(
        self,
        search_http_mock: Mock,
        analysis_http_mock: Mock,
        latest_cached_mock: Mock,
        analyze_mock: Mock,
    ):
        analyze_mock.side_effect = YFRateLimitError()
        latest_cached_mock.return_value = None
        analysis_http_mock.side_effect = RuntimeError("fallback failed")
        search_http_mock.side_effect = RuntimeError("search failed")

        response = self.client.post("/", data={"ticker": "AAPL", "period": "2y", "threshold": "5"})
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Could not refresh data right now. Try another symbol or range.", body)


if __name__ == "__main__":
    unittest.main()
