"""
Unit tests for the Streamlit home page functionality
"""

import unittest
from unittest.mock import patch
from home import get_coffee_preparation_data, BREWING_METHODS


# pylint: disable=duplicate-code
class TestHome(unittest.TestCase):
    """Test cases for the Streamlit home page functionality."""

    @patch("streamlit.selectbox")
    @patch("streamlit.number_input")
    def test_get_coffee_preparation_data_minimal(
        self, mock_number_input, mock_selectbox
    ):
        """Test getting minimal coffee preparation data"""
        # Mock streamlit inputs
        mock_selectbox.return_value = "Too bitter"
        mock_number_input.side_effect = [15.0, 250.0]  # coffee amount, water amount
        data = get_coffee_preparation_data(show_all_fields=False)
        self.assertEqual(data.issue_encountered, "Too bitter")
        self.assertEqual(data.amount_of_coffee, 15.0)
        self.assertEqual(data.amount_of_water, 250.0)

    def test_brewing_methods(self):
        """Test brewing methods constant"""
        expected_methods = [
            "V60",
            "French Press",
            "Espresso",
            "Aeropress",
            "Cold Brew",
            "Moka Pot",
            "Chemex",
        ]
        self.assertEqual(BREWING_METHODS, expected_methods)
        self.assertEqual(len(BREWING_METHODS), 7)


if __name__ == "__main__":
    unittest.main()
