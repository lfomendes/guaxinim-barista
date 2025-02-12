"""
Unit tests for the CoffeePreparationData class
"""

import unittest
from guaxinim.core.coffee_data import CoffeePreparationData


class TestCoffeePreparationData(unittest.TestCase):
    """Test cases for the CoffeePreparationData class."""

    def test_create_minimal_coffee_data(self):
        """Test creating CoffeePreparationData with only required fields"""
        data = CoffeePreparationData(issue_encountered="Too bitter", brewing_method="V60")
        self.assertEqual(data.issue_encountered, "Too bitter")
        self.assertIsNone(data.amount_of_coffee)
        self.assertIsNone(data.amount_of_water)

    def test_create_complete_coffee_data(self):
        """Test creating CoffeePreparationData with all fields"""
        data = CoffeePreparationData(
            issue_encountered="Too acidic",
            brewing_method="V60",
            amount_of_coffee=15.0,
            amount_of_water=250.0,
            type_of_bean="Colombian",
            total_extraction_time=180,
            water_temperature=93.5,
            grinder_granularity="Medium",
            bloom_time=30,
            number_of_pours=3,
            amount_in_each_pour=83.33,
            notes="First try with new beans",
        )

        self.assertEqual(data.issue_encountered, "Too acidic")
        self.assertEqual(data.amount_of_coffee, 15.0)
        self.assertEqual(data.amount_of_water, 250.0)
        self.assertEqual(data.type_of_bean, "Colombian")
        self.assertEqual(data.total_extraction_time, 180)
        self.assertEqual(data.water_temperature, 93.5)
        self.assertEqual(data.grinder_granularity, "Medium")
        self.assertEqual(data.bloom_time, 30)
        self.assertEqual(data.number_of_pours, 3)
        self.assertEqual(data.amount_in_each_pour, 83.33)
        self.assertEqual(data.notes, "First try with new beans")


if __name__ == "__main__":
    unittest.main()
