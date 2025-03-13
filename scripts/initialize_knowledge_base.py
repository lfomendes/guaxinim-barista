#!/usr/bin/env python3
"""Script to initialize the knowledge base by processing YouTube channels and PDF files."""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from guaxinim.core.transcript_processor import TranscriptProcessor
from guaxinim.core.embeddings_processor import EmbeddingsProcessor
from crawl_youtube_videos import crawl_channel
from process_pdfs import process_pdfs

def load_channel_list(file_path: str) -> List[str]:
    """Load list of YouTube channels from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('channels', [])

def process_youtube_channels(channels: List[str]):
    """Process all YouTube channels."""
    print("\n=== Processing YouTube Channels ===")
    for channel in channels:
        print(f"\nProcessing channel: {channel}")
        try:
            # Step 1: Crawl channel videos
            print("1. Crawling videos...")
            crawl_channel(channel)

            # Step 2: Process transcripts
            print("2. Processing transcripts...")
            processor = TranscriptProcessor()
            processor.process_channel(channel)

        except Exception as e:
            print(f"Error processing channel {channel}: {str(e)}")
            continue

def process_pdf_directory(pdf_dir: str):
    """Process all PDF files in the specified directory."""
    print("\n=== Processing PDF Files ===")
    try:
        print(f"Processing PDFs from: {pdf_dir}")
        process_pdfs(pdf_dir)
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")

def create_embeddings():
    """Create embeddings for all processed documents."""
    print("\n=== Creating Embeddings ===")
    try:
        processor = EmbeddingsProcessor()
        processor.process_all_documents()
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")

def main():
    """Main function to initialize the knowledge base."""
    parser = argparse.ArgumentParser(description='Initialize knowledge base from YouTube channels and PDF files.')
    parser.add_argument('--channels-file', type=str, required=True,
                      help='Path to JSON file containing list of YouTube channels')
    parser.add_argument('--pdf-dir', type=str, required=True,
                      help='Directory containing PDF files to process')
    
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Ensure OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    # Load channel list
    try:
        channels = load_channel_list(args.channels_file)
    except Exception as e:
        print(f"Error loading channels file: {str(e)}")
        sys.exit(1)

    # Process YouTube channels
    process_youtube_channels(channels)

    # Process PDF files
    process_pdf_directory(args.pdf_dir)

    # Create embeddings for all processed documents
    create_embeddings()

    print("\n=== Knowledge Base Initialization Complete ===")

if __name__ == "__main__":
    main()
