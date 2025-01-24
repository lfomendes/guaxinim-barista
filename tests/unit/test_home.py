"""
Unit tests for the Streamlit home page functionality
"""

import unittest
from unittest.mock import patch
from guaxinim.ui.home import get_coffee_preparation_data, BREWING_METHODS


# pylint: disable=duplicate-code
class TestHome(unittest.TestCase):
    """Test class for home.py functions."""

    @patch("streamlit.selectbox")
    @patch("streamlit.number_input")
    def test_coffee_data_input(self, mock_number_input, mock_selectbox):
        """Test coffee data input values"""
        mock_selectbox.return_value = "V60"
        # Mock values for all number inputs in the form
        mock_number_input.side_effect = [
            15.0,  # amount_of_coffee
            250.0,  # amount_of_water
            0,     # total_extraction_time (optional)
            0,     # water_temperature (optional)
            0,     # bloom_time (optional)
            0,     # number_of_pours (optional)
            0,     # amount_in_each_pour (optional)
        ]

        data = get_coffee_preparation_data()
        self.assertEqual(data.amount_of_coffee, 15.0)
        self.assertEqual(data.amount_of_water, 250.0)
        self.assertIsNone(data.total_extraction_time)  # Optional field should be None when 0

    def test_brewing_methods(self):
        """Test brewing methods constant"""
        expected = [
            "V60",
            "French Press",
            "Aeropress",
            "Chemex",
            "Moka Pot",
            "Espresso",
        ]
        self.assertEqual(BREWING_METHODS, expected)


if __name__ == "__main__":
    unittest.main()
