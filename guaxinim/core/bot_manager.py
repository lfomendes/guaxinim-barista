"""
Bot Manager Module
Manages the GuaxinimBot instance to ensure a single instance is used across the app.
"""

import streamlit as st
from guaxinim.core.guaxinim_bot import GuaxinimBot

def get_bot(max_whole_files: int = None, similarity_field: str = None) -> GuaxinimBot:
    """Get or create a GuaxinimBot instance.
    
    Args:
        max_whole_files (int, optional): Maximum number of whole files to return in 'whole file' mode.
            If not specified, uses the default value from GuaxinimBot.DEFAULT_MAX_WHOLE_FILES.
        similarity_field (str, optional): Field to use for similarity search ('chunks' or 'summary').
            If not specified, uses 'chunks'.
    """
    # Initialize bot settings in session state if not present
    if 'bot_settings' not in st.session_state:
        st.session_state.bot_settings = {
            'max_whole_files': max_whole_files,
            'similarity_field': similarity_field or 'chunks'
        }
    
    # Check if settings have changed
    settings_changed = (
        max_whole_files != st.session_state.bot_settings['max_whole_files'] or
        similarity_field != st.session_state.bot_settings['similarity_field']
    )
    
    # Create new bot if it doesn't exist or if settings changed
    if 'guaxinim_bot' not in st.session_state or settings_changed:
        st.session_state.guaxinim_bot = GuaxinimBot(
            max_whole_files=max_whole_files,
            similarity_field=similarity_field or "chunks"
        )
        # Update stored settings
        st.session_state.bot_settings = {
            'max_whole_files': max_whole_files,
            'similarity_field': similarity_field or 'chunks'
        }
    
    return st.session_state.guaxinim_bot
