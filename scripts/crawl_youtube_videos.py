#!/usr/bin/env python3
"""Script to crawl a YouTube channel's videos and extract transcripts."""
import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from guaxinim.core.youtube_crawler import YoutubeCrawler

def main():
    """Main function to run the crawler."""
    parser = argparse.ArgumentParser(description="Crawl videos from a YouTube channel")
    parser.add_argument("channel_url", help="URL of the YouTube channel to crawl")
    parser.add_argument("--num-videos", type=int, default=30, help="Number of videos to crawl (default: 30)")
    args = parser.parse_args()
    
    try:
        crawler = YoutubeCrawler(number_of_videos=args.num_videos)
        crawler.process_channel(args.channel_url)
    except Exception as e:
        print(f"Error processing channel: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
