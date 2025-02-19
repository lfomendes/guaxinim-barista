#!/usr/bin/env python3
"""Script to process YouTube transcripts and generate summaries."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from guaxinim.core.transcript_processor import TranscriptProcessor

def main():
    """Main function to run the processor."""
    # Load environment variables
    load_dotenv()
    
    # Ensure OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    # Get channel name from command line argument
    if len(sys.argv) != 2:
        print("Usage: python process_transcripts.py <channel_name>")
        print("Example: python process_transcripts.py jameshoffmann")
        sys.exit(1)

    channel_name = sys.argv[1]
    
    try:
        processor = TranscriptProcessor()
        processor.process_channel(channel_name)
    except Exception as e:
        print(f"Error processing channel: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
