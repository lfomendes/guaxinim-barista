#!/usr/bin/env python3
"""Script to create embeddings for processed documents."""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from guaxinim.core.embeddings_processor import EmbeddingsProcessor

def main():
    """Main function to run the embeddings processor."""
    try:
        processor = EmbeddingsProcessor()
        processor.process_all_documents()
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
