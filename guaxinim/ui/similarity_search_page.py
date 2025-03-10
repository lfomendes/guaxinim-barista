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

def display_summary_results(results):
    """Display summary search results in a nice format."""
    for i, result in enumerate(results, 1):
        st.write(f"{i}. **{result['title']}**")
        st.write(f"   Score: {result['similarity_score']:.3f}")
        st.write(f"   Source: {result['source']}")
        if result.get('summary'):
            st.write(f"   Summary: {result['summary']}")
        st.write("---")

def display_tag_results(results):
    """Display tag search results in a nice format."""
    for i, result in enumerate(results, 1):
        with st.expander(f"{i}. {result['title']}"):
            st.write("**Source:**", result['source'])
            if result.get('summary'):
                st.write("**Summary:**")
                st.write(result['summary'])
            st.write("**Tags:**", ", ".join(result['tags']))

def search_coffee_documents():
    """
    Displays the similarity search interface where users can search through coffee-related documents.
    Uses semantic search to find relevant content based on user queries.
    """
    # Add side menu for RAG settings
    with st.sidebar:
        st.subheader("Search Settings")
        rag_return = st.selectbox(
            "RAG return",
            options=["chunks", "whole file"],
            help="Choose how to return context from the knowledge base"
        )

    st.header("Search Coffee Knowledge")
    st.write("""
    Search through our coffee knowledge base using natural language or browse by topics. 
    The search will find relevant content based on meaning, not just keywords.
    """)

    # Initialize the document searcher
    try:
        searcher = DocumentSearcher()
    except Exception as e:
        st.error("Error loading document database. Please ensure the document database exists.")
        return

    # Create tabs for different search methods
    text_search_tab, tag_search_tab = st.tabs(["Text Search", "Topic Search"])

    with text_search_tab:
        # Text search interface
        query = st.text_input("Enter your search query:", 
                            placeholder="e.g., How can I make my coffee less acidic?")

        if query:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Most Relevant Chunks")
                chunk_results = searcher.search_similar_chunks(query, k=3)
                display_chunk_results(chunk_results)
                
            with col2:
                st.subheader("Most Related Summaries")
                summary_results = searcher.search_similar_summaries(query, k=2)
                display_summary_results(summary_results)
        else:
            st.info("Enter a query above to search through our coffee knowledge base.")

    with tag_search_tab:
        # Tag search interface
        tags_with_freq = searcher.get_tags_with_frequency()
        if tags_with_freq:
            # Format options to show tag and count
            tag_options = [tag for tag, _ in tags_with_freq]
            tag_display = {tag: f"{tag.replace('_', ' ').title()} ({count} documents)" 
                          for tag, count in tags_with_freq}
            
            selected_tag = st.selectbox(
                "Select a topic to explore:",
                options=tag_options,
                format_func=lambda x: tag_display[x]
            )

            if selected_tag:
                count = dict(tags_with_freq)[selected_tag]
                st.subheader(f"Documents about {selected_tag.replace('_', ' ').title()} ({count} documents)")
                tag_results = searcher.search_by_tag(selected_tag, limit=10)
                display_tag_results(tag_results)
        else:
            st.info("No topics found in the knowledge base.")
