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


if __name__ == "__main__":
    unittest.main()
