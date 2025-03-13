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

def crawl_channel(channel_name: str, num_videos: int = 30, interactive: bool = False):
    """Crawl videos from a YouTube channel.
    
    Args:
        channel_name (str): Name or URL of the YouTube channel
        num_videos (int, optional): Number of videos to crawl. Defaults to 30.
        interactive (bool, optional): If True, prompt for each video. If False, process all videos. Defaults to False.
    """
    try:
        crawler = YoutubeCrawler(number_of_videos=num_videos)
        # If it's not a full URL, construct it
        if not channel_name.startswith('http'):
            channel_url = f'https://www.youtube.com/@{channel_name}'
        else:
            channel_url = channel_name
        crawler.process_channel(channel_url, interactive=interactive)
    except Exception as e:
        print(f"Error crawling channel {channel_name}: {str(e)}")
        raise

def main():
    """Main function to run the crawler."""
    parser = argparse.ArgumentParser(description="Crawl videos from a YouTube channel")
    parser.add_argument("channel_url", help="URL of the YouTube channel to crawl")
    parser.add_argument("--num-videos", type=int, default=30, help="Number of videos to crawl (default: 30)")
    args = parser.parse_args()
    
    try:
        crawl_channel(args.channel_url, args.num_videos)
    except Exception as e:
        print(f"Error processing channel: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
