"""
Guaxinim Bot Module
This module implements the AI-powered coffee assistant that provides
recommendations and answers coffee-related questions using OpenAI's API.
"""

import os
from openai import OpenAI, APIError, APIConnectionError
from dotenv import load_dotenv
from guaxinim.core.coffee_data import CoffeePreparationData

# Load environment variables
load_dotenv(override=True)


class GuaxinimBot:
    """
    A class that handles all AI-powered interactions for the coffee assistant.
    This bot can answer coffee-related questions and provide recommendations
    for improving coffee preparation.
    """

    # OpenAI model configuration
    GPT_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.2  # Lower temperature for more focused and consistent responses

    COFFEE_GUIDE_PROMPT = """You are a professional coffee barista with years of experience.
    You are teaching someone how to make an excellent cup of coffee using the {method} method.
    Please provide a detailed, step-by-step guide that includes:
    1. Required equipment
    2. Recommended coffee-to-water ratio
    3. Grind size recommendation
    4. Water temperature
    5. Detailed brewing steps
    6. Common mistakes to avoid
    7. Tips for achieving the best results

    Format your response in markdown for better readability."""

    def __init__(self):
        """Initialize the GuaxinimBot with API key validation and OpenAI client setup."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI()

    def get_coffee_guide(self, method: str) -> str:
        """
        Get a detailed guide for making coffee using the specified method.

        Args:
            method (str): The coffee brewing method (e.g., 'V60', 'French Press')

        Returns:
            str: Detailed brewing guide from OpenAI
        """
        try:
            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional coffee barista expert.",
                    },
                    {
                        "role": "user",
                        "content": self.COFFEE_GUIDE_PROMPT.format(method=method),
                    },
                ],
                temperature=self.TEMPERATURE,
                max_tokens=1000,
                store=True,
            )
            return response.choices[0].message.content
        except (APIError, APIConnectionError) as e:
            return f"Error getting coffee guide: {str(e)}"

    def ask_guaxinim(self, query: str) -> str:
        """
        Process a coffee-related question and return an AI-generated answer.

        Args:
            query (str): The user's coffee-related question

        Returns:
            str: AI-generated response to the query
        """

        basic_prompt = """
        Check if the following question is coffee-related:
        If yes: Provide a detailed answer
        If no: Reply with 'I only answer questions about coffee.'

        Question: """
        
        try:
            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional coffee barista expert.",
                    },
                    {"role": "user", "content": basic_prompt + query},
                ],
                temperature=self.TEMPERATURE,
                max_tokens=500,
                store=True,
            )
            return response.choices[0].message.content
        except (APIError, APIConnectionError) as e:
            return f"Error processing question: {str(e)}"

    def improve_coffee(self, coffee_data: CoffeePreparationData) -> str:
        """
        Analyze current coffee preparation parameters and suggest improvements.

        Args:
            coffee_data (CoffeePreparationData): Current coffee preparation parameters

        Returns:
            str: AI-generated suggestions for improving coffee preparation
        """
        try:
            # Start with required parameters
            params = [
                f"Issue: {coffee_data.issue_encountered}",
                f"Coffee amount: {coffee_data.amount_of_coffee}g",
                f"Water amount: {coffee_data.amount_of_water}ml",
            ]
            
            # Add optional parameters only if they are provided
            if coffee_data.type_of_bean:
                params.append(f"Bean type: {coffee_data.type_of_bean}")
            if coffee_data.total_extraction_time:
                params.append(f"Extraction time: {coffee_data.total_extraction_time}s")
            if coffee_data.water_temperature:
                params.append(f"Water temp: {coffee_data.water_temperature}°C")
            if coffee_data.grinder_granularity:
                params.append(f"Grind size: {coffee_data.grinder_granularity}")
            if coffee_data.bloom_time:
                params.append(f"Bloom time: {coffee_data.bloom_time}s")
            if coffee_data.number_of_pours:
                params.append(f"Number of pours: {coffee_data.number_of_pours}")
            if coffee_data.amount_in_each_pour:
                params.append(f"Amount per pour: {coffee_data.amount_in_each_pour}ml")
            if coffee_data.notes:
                params.append(f"Additional notes: {coffee_data.notes}")

            prompt = "Analyze these coffee parameters and suggest improvements:\n"
            prompt += "\n".join(params)
            
            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional coffee barista expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.TEMPERATURE,
                max_tokens=500,
                store=True,
            )
            return response.choices[0].message.content
        except (APIError, APIConnectionError) as e:
            return f"Error analyzing coffee parameters: {str(e)}"
