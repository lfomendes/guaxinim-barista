"""
Implements the similarity search page for the Guaxinim application.
This page allows users to search through coffee-related documents using semantic similarity.
"""

import streamlit as st
from src.pdf_processor.similarity_search import DocumentSearcher

def display_chunk_results(results):
    """Display chunk search results in a nice format."""
    for i, result in enumerate(results, 1):
        with st.expander(f"{i}. {result['title']} (Score: {result['similarity_score']:.3f})"):
            st.write("**Source:**", result['source'])
            st.write("**Relevant Text:**")
            st.write(result['chunk_text'])

def display_title_results(results):
    """Display title search results in a nice format."""
    for i, result in enumerate(results, 1):
        st.write(f"{i}. **{result['title']}**")
        st.write(f"   Score: {result['similarity_score']:.3f}")
        st.write(f"   Source: {result['source']}")
        st.write("---")

def search_coffee_documents():
    """
    Displays the similarity search interface where users can search through coffee-related documents.
    Uses semantic search to find relevant content based on user queries.
    """
    st.header("Search Coffee Knowledge")
    st.write("""
    Search through our coffee knowledge base using natural language. 
    The search will find relevant content based on meaning, not just keywords.
    """)

    # Initialize the document searcher
    try:
        searcher = DocumentSearcher('data/json/hoffman_pdf.json')
    except Exception as e:
        st.error("Error loading document database. Please ensure the document database exists.")
        return

    # Search interface
    query = st.text_input("Enter your search query:", 
                         placeholder="e.g., How can I make my coffee less acidic?")

    if query:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Relevant Chunks")
            chunk_results = searcher.search_similar_chunks(query, k=3)
            display_chunk_results(chunk_results)
            
        with col2:
            st.subheader("Most Related Titles")
            title_results = searcher.search_similar_titles(query, k=2)
            display_title_results(title_results)
    else:
        st.info("Enter a query above to search through our coffee knowledge base.")
