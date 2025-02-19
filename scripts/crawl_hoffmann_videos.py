#!/usr/bin/env python3
"""Script to crawl James Hoffmann's YouTube channel videos and extract transcripts."""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from guaxinim.core.youtube_crawler import YoutubeCrawler

def main():
    """Main function to run the crawler."""
    channel_url = "https://www.youtube.com/@jameshoffmann"
    
    try:
        crawler = YoutubeCrawler(number_of_videos=30)
        crawler.process_channel(channel_url)
    except Exception as e:
        print(f"Error processing channel: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
