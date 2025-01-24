"""
Unit tests for the GuaxinimBot class
"""

import unittest
from unittest.mock import patch, MagicMock
from guaxinim_bot import GuaxinimBot
from coffee_data import CoffeePreparationData


class TestGuaxinimBot(unittest.TestCase):
    """Test cases for the GuaxinimBot class."""

    def setUp(self):
        """Set up test cases"""
        with patch('guaxinim_bot.OpenAI') as mock_openai:
            self.bot = GuaxinimBot()
            self.mock_openai = mock_openai

    def test_initialization(self):
        """Test bot initialization"""
        self.assertIsNotNone(self.bot)
        self.assertIsNotNone(self.bot.client)

    @patch('os.getenv')
    def test_initialization_without_api_key(self, mock_getenv):
        """Test bot initialization without API key"""
        mock_getenv.return_value = None
        with self.assertRaises(ValueError):
            GuaxinimBot()

    def test_get_coffee_guide(self):
        """Test getting coffee brewing guide"""
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test brewing guide"
        self.bot.client.chat.completions.create.return_value = mock_response

        guide = self.bot.get_coffee_guide("V60")
        self.assertEqual(guide, "Test brewing guide")
        self.bot.client.chat.completions.create.assert_called_once()

    def test_improve_coffee(self):
        """Test getting improvement suggestions for coffee preparation"""
        # Create test data
        coffee_data = CoffeePreparationData(
            issue_encountered="Too bitter",
            amount_of_coffee=15.0,
            amount_of_water=250.0
        )

        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test suggestions"
        self.bot.client.chat.completions.create.return_value = mock_response

        suggestions = self.bot.improve_coffee(coffee_data)
        self.assertEqual(suggestions, "Test suggestions")
        self.bot.client.chat.completions.create.assert_called_once()

    def test_ask_guaxinim(self):
        """Test getting answers to coffee-related questions"""
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test answer"
        self.bot.client.chat.completions.create.return_value = mock_response

        answer = self.bot.ask_guaxinim("What is coffee bloom?")
        self.assertEqual(answer, "Test answer")
        self.bot.client.chat.completions.create.assert_called_once()


if __name__ == '__main__':
    unittest.main()
