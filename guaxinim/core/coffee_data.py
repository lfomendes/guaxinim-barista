"""
Data class for coffee preparation parameters.
"""
from dataclasses import dataclass
from typing import Optional


# pylint: disable=too-many-instance-attributes
@dataclass
class CoffeePreparationData:
    """
    A dataclass representing all parameters related to coffee preparation.

    Attributes:
        issue_encountered (str): The main issue or problem with the coffee preparation
        amount_of_coffee (float, optional): Weight of coffee beans in grams
        amount_of_water (float, optional): Volume of water in milliliters
        type_of_bean (str, optional): Type/origin of coffee beans used
        total_extraction_time (int, optional): Total brewing time in seconds
        water_temperature (float, optional): Water temperature in Celsius
        grinder_granularity (str, optional): Coarseness of the ground coffee (Fine/Medium/Coarse)
        bloom_time (int, optional): Time allowed for coffee blooming in seconds
        number_of_pours (int, optional): Number of water pours during brewing
        amount_in_each_pour (float, optional): Volume of water per pour in milliliters
        notes (str, optional): Additional notes or observations about the preparation
    """

    issue_encountered: str
    brewing_method: str
    amount_of_coffee: Optional[float] = None
    amount_of_water: Optional[float] = None
    type_of_bean: Optional[str] = None
    total_extraction_time: Optional[int] = None
    water_temperature: Optional[float] = None
    grinder_granularity: Optional[str] = None
    bloom_time: Optional[int] = None
    number_of_pours: Optional[int] = None
    amount_in_each_pour: Optional[float] = None
    notes: Optional[str] = None
