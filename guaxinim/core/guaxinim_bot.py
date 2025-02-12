"""
Guaxinim Bot Module
This module implements the AI-powered coffee assistant that provides
recommendations and answers coffee-related questions using OpenAI's API.
"""

import os
from openai import OpenAI, APIError, APIConnectionError
from dotenv import load_dotenv
from typing import List, Dict
from dataclasses import dataclass
from guaxinim.core.coffee_data import CoffeePreparationData
from src.pdf_processor.similarity_search import DocumentSearcher
from guaxinim.core.logger import logger

# Load environment variables
load_dotenv()


@dataclass
class GuaxinimResponse:
    """Response from GuaxinimBot containing the answer and its sources"""
    answer: str
    sources: List[Dict[str, str]]
    
    @classmethod
    def error(cls, message: str) -> 'GuaxinimResponse':
        """Create an error response"""
        return cls(answer=f"Error: {message}", sources=[])


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

    CONTEXT_PROMPT_TEMPLATE = """Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and your expertise as a professional barista,
    provide a 
     answer to the query.
    Query: {query_str}
    Answer: """

    def __init__(self):
        """Initialize the GuaxinimBot with API key validation and OpenAI client setup."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI()
        try:
            self.searcher = DocumentSearcher('data/json/hoffman_pdf.json')
            logger.info("Document searcher initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize document searcher: {e}")
            self.searcher = None

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

    def _get_relevant_context(self, query: str, k_chunks: int = 5) -> str:
        """Get relevant context from the document database."""
        if not self.searcher:
            logger.warning("Document searcher not available, proceeding without context")
            return ""

        logger.info("Starting context search")
        logger.debug(f"Query: {query}")
        
        # Get relevant chunks and titles
        chunks = self.searcher.search_similar_chunks(query, k=k_chunks)
        titles = self.searcher.search_similar_titles(query, k=2)

        # Log the sources being used
        logger.info("Found relevant content:")
        logger.info("Chunks:")
        for chunk in chunks:
            logger.info(f"- {chunk['title']} (Score: {chunk['similarity_score']:.3f})")

        logger.info("Titles:")
        for title in titles:
            logger.info(f"- {title['title']} (Score: {title['similarity_score']:.3f})")

        # Log detailed chunk content at debug level
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i+1} content:")
            logger.debug(f"Title: {chunk['title']}")
            logger.debug(f"Source: {chunk['source']}")
            logger.debug(f"Content: {chunk['chunk_text'][:200]}...")

        # Format context string
        context_parts = []

        # Add relevant chunks
        for chunk in chunks:
            context_parts.append(
                f"From '{chunk['title']}':\n"
                f"{chunk['chunk_text']}\n"
                f"(Source: {chunk['source']})\n"
            )

        # Add relevant titles if they're different from chunk sources
        chunk_sources = {chunk['source'] for chunk in chunks}
        for title in titles:
            if title['source'] not in chunk_sources:
                context_parts.append(
                    f"Additional relevant article: {title['title']}\n"
                    f"(Source: {title['source']})\n"
                )

        return "\n".join(context_parts)

    def ask_guaxinim(self, query: str) -> GuaxinimResponse:
        """
        Process a coffee-related question and return an AI-generated answer along with sources.

        Args:
            query (str): The user's coffee-related question

        Returns:
            GuaxinimResponse: Object containing the answer and its sources
        """
        try:
            # Get relevant context
            logger.debug(f"TESTE Processing query: {query}")
            sources = []
            
            if self.searcher:
                # Get chunks and titles
                chunks = self.searcher.search_similar_chunks(query, k=5)
                titles = self.searcher.search_similar_titles(query, k=2)
                
                # Combine chunks and titles into sources list
                seen_sources = set()
                for chunk in chunks:
                    if chunk['source'] not in seen_sources:
                        sources.append({
                            'title': chunk['title'],
                            'source': chunk['source']
                        })
                        seen_sources.add(chunk['source'])
                
                for title in titles:
                    if title['source'] not in seen_sources:
                        sources.append({
                            'title': title['title'],
                            'source': title['source']
                        })
                        seen_sources.add(title['source'])
                
                context = self._get_relevant_context(query)
            else:
                context = ""
            
            # Prepare the prompt with context
            if context:
                logger.debug("Using context-based prompt")
                prompt = self.CONTEXT_PROMPT_TEMPLATE.format(
                    context_str=context,
                    query_str=query
                )
            else:
                logger.debug("Using fallback prompt without context")
                prompt = f"As a professional coffee barista, please answer this question: {query}"
            
            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional coffee barista expert. Always cite sources when using information from the provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.TEMPERATURE,
                max_tokens=800,
                store=True,
            )
            
            return GuaxinimResponse(
                answer=response.choices[0].message.content,
                sources=sources
            )
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            return GuaxinimResponse.error(str(e))

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
