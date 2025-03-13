"""
Guaxinim Bot Module
This module implements the AI-powered coffee assistant that provides
recommendations and answers coffee-related questions using OpenAI's API.
"""

import os
from openai import OpenAI, APIError, APIConnectionError
from dotenv import load_dotenv
from typing import List, Dict, Union
from dataclasses import dataclass
from guaxinim.core.coffee_data import CoffeePreparationData
from src.pdf_processor.similarity_search import DocumentSearcher
from guaxinim.core.logger import logger

# Load environment variables
load_dotenv(override=True)

def get_env_var(key: str) -> str:
    """Get environment variable from either .env or Streamlit secrets"""
    # Try to get from streamlit secrets first
    try:
        import streamlit as st
        return st.secrets[key]
    except:
        # Fall back to os.environ
        return os.getenv(key)


@dataclass
class GuaxinimResponse:
    """Response from GuaxinimBot containing the answer and its sources"""
    answer: str
    sources: List[Dict[str, Union[str, List[str]]]]  # Can contain 'title', 'url', and 'tags' fields
    
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
    TEMPERATURE = 0.2
    DEFAULT_MAX_WHOLE_FILES = 2
    SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score for considering a document relevant
    PREVIEW_LENGTH = 500  # Number of characters to show in document previews  # Default number of whole files to return  # Lower temperature for more focused and consistent responses

    COFFEE_GUIDE_PROMPT = """Goal: Create a comprehensive brewing guide for making coffee using the {method} method, ensuring it is detailed enough for a beginner to follow successfully.

Return Format:
Your response MUST be in markdown format with the following sections:

# {method} Brewing Guide

## Required Equipment
* [List of required equipment]

## Specifications
* Coffee-to-Water Ratio: [e.g., 1:16]
* Grind Size: [with specific reference points]
* Water Temperature: [in both Celsius and Fahrenheit]

## Step-by-Step Instructions
1. [First step]
2. [Second step]
...

## Common Mistakes to Avoid
* [First mistake]
* [Second mistake]
...

## Pro Tips
* [First tip]
* [Second tip]
...

Warnings:
- Be extremely precise with measurements and timings
- Ensure water temperature is accurate for the specific method
- Grind size descriptions must be clear and relatable
- All equipment must be standard and commonly available

Context:
You are a professional barista with years of experience teaching beginners. Your guide should be thorough yet approachable."""

    CONTEXT_PROMPT_TEMPLATE = """Goal: Provide a clear, accurate, and helpful answer to a coffee-related question using both provided context and barista expertise.

    Return Format:
    Your response should be structured as follows:
    1. Direct answer to the question (2-3 sentences)
    2. Supporting explanation with technical details (if relevant)
    3. Practical tips or recommendations (if applicable)
    4. References to specific sources from context (if available)

    Warnings:
    - Stick to factual information from the context when available
    - Clearly distinguish between context-based information and general barista knowledge
    - Avoid speculation or unsupported claims
    - Keep the answer focused and relevant to the specific query

    Context:
    ---------------------
    {context_str}
    ---------------------

    Query: {query_str}
    Answer: """

    def __init__(self, max_whole_files: int = None, similarity_field: str = "chunks"):
        """Initialize the GuaxinimBot with API key validation and OpenAI client setup.
        
        Args:
            max_whole_files (int, optional): Maximum number of whole files to return in 'whole file' mode.
                                           Defaults to DEFAULT_MAX_WHOLE_FILES if not specified.
            similarity_field (str, optional): Field to use for similarity search ('chunks' or 'summary').
                                           Defaults to 'chunks'.
        """
        api_key = get_env_var("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
        os.environ["OPENAI_API_KEY"] = api_key  # Set for OpenAI client
        self.client = OpenAI()
        self.max_whole_files = max_whole_files or self.DEFAULT_MAX_WHOLE_FILES
        self.similarity_field = similarity_field
        try:
            self.searcher = DocumentSearcher()
            logger.info("Document searcher initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize document searcher: {e}")
            self.searcher = None

    def get_coffee_guide(self, method: str, rag_return_type: str = "chunks") -> GuaxinimResponse:
        """
        Get a detailed guide for making coffee using the specified method.

        Args:
            method (str): The coffee brewing method (e.g., 'V60', 'French Press')
            rag_return_type (str): Type of context to return ('chunks' or 'whole file')

        Returns:
            GuaxinimResponse: Object containing the guide and its sources
        """
        try:
            # Get relevant context about the brewing method
            query = f"How to make coffee using {method} method"
            sources = []
            context_str = ""
            
            if self.searcher:
                logger.info(f"Searching for relevant context with query: {query}")
                context_str, sources = self._get_relevant_context(query, rag_return_type=rag_return_type)

            # Create the prompt with main guide content first, then add context if available
            prompt = self.COFFEE_GUIDE_PROMPT.format(method=method)
            if context_str:
                prompt += f"\n\nAdditional Context:\n{context_str}"

            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=[{"role": "system", "content": prompt}],
                temperature=self.TEMPERATURE,
                max_tokens=1000,
            )

            return GuaxinimResponse(
                answer=response.choices[0].message.content,
                sources=sources
            )
        except Exception as e:
            error_msg = f"Error getting coffee guide: {str(e)}"
            logger.error(error_msg)
            return GuaxinimResponse.error(error_msg)

    def _get_relevant_context(self, query: str, k_chunks: int = 5, rag_return_type: str = "chunks") -> tuple[str, list]:
        """Get relevant context from the document database.
        
        Args:
            query (str): The search query
            k_chunks (int): Number of chunks to retrieve
            rag_return_type (str): Type of context to return ('chunks' or 'whole file')
            
        Returns:
            tuple[str, list]: Context string and list of sources
        """
        if not self.searcher:
            logger.warning("Document searcher not available, proceeding without context")
            return "", []

        logger.info("Starting context search")
        logger.info(f"Query: {query}")
        
        # Get relevant documents based on similarity field
        if self.similarity_field == "chunks":
            logger.info("Searching for relevant chunks")
            # Search by chunks first, then get related titles
            results = self.searcher.search_similar_chunks(query, k=k_chunks)
            title_results = self.searcher.search_similar_titles(query, k=2)
        else:  # summary mode
            logger.info(f"Searching for relevant summaries with k={k_chunks}")
            # Search by summary first
            results = self.searcher.search_similar_summaries(query, k=k_chunks)
            title_results = results  # Use the same results for titles in summary mode

        # Early return if no results found
        if not results:
            logger.warning("No relevant documents found")
            return "", []

        # Format context string and collect sources
        context_parts = []
        sources = []
        seen_sources = set()

        # Helper function to add a source
        def add_source(result):
            if result['source'] not in seen_sources and result.get('similarity_score', 0) > self.SIMILARITY_THRESHOLD:
                sources.append({
                    'title': result['title'],
                    'url': result['source'],
                    'tags': result.get('tags', [])
                })
                seen_sources.add(result['source'])

        # Helper function to format document content
        def format_document_content(result):
            parts = [f"From '{result['title']}':\n"]
            
            # Add relevant section if available (chunk text or summary)
            relevant_section = result.get('chunk_text') or result.get('summary', '')
            if relevant_section:
                parts.append(f"Relevant section: {relevant_section}\n")
            
            # Add document context if available
            if result.get('full_text'):
                context = result['full_text'][:self.PREVIEW_LENGTH] + '...' if len(result['full_text']) > self.PREVIEW_LENGTH else result['full_text']
                parts.append(f"Document context: {context}\n")
            
            # Add tags if available
            if result.get('tags'):
                parts.append(f"Tags: {', '.join(result['tags'])}\n")
            
            parts.append(f"(Source: {result['source']})\n")
            return ''.join(parts)

        # Process results based on mode
        if rag_return_type == "whole file":
            seen_titles = set()
            for result in results:
                if len(seen_titles) >= self.max_whole_files:
                    break
                    
                if result['title'] not in seen_titles and result.get('similarity_score', 0) > self.SIMILARITY_THRESHOLD:
                    # Add the full text of the document
                    context_parts.append(
                        f"From '{result['title']}':\n"
                        f"{result.get('full_text', '')}\n"
                        f"(Source: {result['source']})\n"
                    )
                    add_source(result)
                    seen_titles.add(result['title'])
        else:  # chunks or summary mode
            # Add relevant chunks/summaries with their context
            for result in results:
                if result.get('similarity_score', 0) > self.SIMILARITY_THRESHOLD:
                    context_parts.append(format_document_content(result))
                    add_source(result)

            # Add relevant titles if they're different from existing sources
            for title in title_results:
                if title.get('similarity_score', 0) > self.SIMILARITY_THRESHOLD and title['source'] not in seen_sources:
                    parts = [f"Additional relevant article: {title['title']}\n"]
                    
                    # Add tags if available
                    if title.get('tags'):
                        parts.append(f"Tags: {', '.join(title['tags'])}\n")
                        
                    parts.append(f"(Source: {title['source']})\n")
                    context_parts.append(''.join(parts))
                    add_source(title)

        # Always return sources even if no context was added
        return "\n".join(context_parts), sources

    def ask_guaxinim(self, query: str, rag_return_type: str = "chunks") -> GuaxinimResponse:
        """
        Process a coffee-related question and return an AI-generated answer along with sources.

        Args:
            query (str): The user's coffee-related question
            rag_return_type (str): Type of context to return ('chunks' or 'whole file')

        Returns:
            GuaxinimResponse: Object containing the answer and its sources
        """
        try:
            # Get relevant context
            logger.debug(f"Processing query: {query}")
            context_str, sources = self._get_relevant_context(query, rag_return_type=rag_return_type)
            
            # Prepare the prompt
            if context_str:
                logger.debug("Using context-based prompt")
                prompt = self.CONTEXT_PROMPT_TEMPLATE.format(
                    context_str=context_str,
                    query_str=query
                )
            else:
                logger.debug("Using fallback prompt without context")
                # Use a simplified version of CONTEXT_PROMPT_TEMPLATE without context section
                prompt = self.CONTEXT_PROMPT_TEMPLATE.format(
                    context_str="No additional context available.",
                    query_str=query
                )
            

            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.TEMPERATURE,
                max_tokens=800,
            )
            
            return GuaxinimResponse(
                answer=response.choices[0].message.content,
                sources=sources
            )
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            return GuaxinimResponse.error(str(e))

    def improve_coffee(self, coffee_data: CoffeePreparationData, rag_return_type: str = "chunks") -> GuaxinimResponse:
        """
        Analyze current coffee preparation parameters and suggest improvements.

        Args:
            coffee_data (CoffeePreparationData): Current coffee preparation parameters
            rag_return_type (str): Type of context to return ('chunks' or 'whole file')

        Returns:
            GuaxinimResponse: Object containing the suggestions and sources
        """
        try:
            # Start with required parameters
            params = [
                f"Issue: {coffee_data.issue_encountered}",
                f"Brewing method: {coffee_data.brewing_method}",
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

            # Get relevant context about the issue and brewing method
            query = f"How to fix {coffee_data.issue_encountered} in {coffee_data.brewing_method} coffee"
            sources = []
            context_str = ""
            
            if self.searcher:
                context_str, sources = self._get_relevant_context(query, rag_return_type=rag_return_type)

            # Create the improvement prompt
            improvement_prompt = f"""Goal: Analyze the current coffee preparation parameters and provide specific suggestions for improvement, focusing on addressing the reported issue.            

            Return Format:
            Your response should be structured as follows:
            1. Issue Analysis (2-3 sentences identifying likely causes)
            2. Key Recommendations (3-5 bullet points)
            3. Detailed Adjustments (specific parameter changes)
            4. Additional Tips (if relevant)

            Warnings:
            - Focus on the most impactful changes first
            - Be specific with measurements and adjustments
            - Explain the reasoning behind each recommendation
            - Reference sources when using contextual information

            Current Parameters:
            {"\n".join(params)}

            Additional Context:
            {context_str if context_str else "No additional context available."}"""

            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=[{"role": "user", "content": improvement_prompt}],
                temperature=self.TEMPERATURE,
                max_tokens=1000
            )
            return GuaxinimResponse(
                answer=response.choices[0].message.content,
                sources=sources
            )
        except Exception as e:
            error_msg = f"Error generating improvement suggestions: {str(e)}"
            logger.error(error_msg)
            return GuaxinimResponse.error(error_msg)
