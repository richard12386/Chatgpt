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
        sleep_mock.assert_any_call(1.0)
        sleep_mock.assert_any_call(2.0)

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
        self.client = app.app.test_client()

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
            current_price=123.45,
            last_dividend=0.25,
            last_dividend_date="2026-01-15",
            company_state="CA",
            volatile_weeks=[{"week_end": "2026-01-09", "change_pct": "6.10%"}],
        )

        response = self.client.post("/", data={"ticker": "AAPL", "period": "1y"})
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Apple Inc.", body)
        self.assertIn("AAPL", body)
        self.assertIn("6.10%", body)
        analysis_mock.assert_called_once_with("AAPL", period="1y")

    @patch("app._search_symbol_by_name")
    @patch("app.get_stock_analysis_cached")
    def test_post_company_name_uses_search_fallback(
        self, analysis_mock: Mock, search_mock: Mock
    ):
        search_mock.return_value = "HO.PA"
        analysis_mock.side_effect = [
            ValueError("No price data found for this ticker."),
            app.AnalysisResult(
                ticker="HO.PA",
                company_name="Thales S.A.",
                current_price=150.0,
                last_dividend=1.0,
                last_dividend_date="2026-01-01",
                company_state=None,
                volatile_weeks=[],
            ),
        ]

        response = self.client.post("/", data={"ticker": "Thales", "period": "2y"})
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Thales S.A.", body)
        search_mock.assert_called_once_with("Thales")
        self.assertEqual(analysis_mock.call_count, 2)
        self.assertEqual(analysis_mock.call_args_list[0].args[0], "THALES")
        self.assertEqual(analysis_mock.call_args_list[1].args[0], "HO.PA")


if __name__ == "__main__":
    unittest.main()
