"""Module for crawling YouTube channels and extracting video transcripts."""
import os
import json
import re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

class YoutubeCrawler:
    """Class to handle YouTube channel crawling and transcript extraction.
    
    Requires YOUTUBE_API_KEY to be set in the .env file.
    """
    
    def __init__(self, number_of_videos: int = 20):
        """Initialize the crawler using YouTube API key from environment variables."""
        load_dotenv()
        from .guaxinim_bot import get_env_var
        api_key = get_env_var('YOUTUBE_API_KEY')
        if not api_key:
            raise ValueError('YOUTUBE_API_KEY not found in environment variables')
            
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.base_dir = Path('data/raw/transcripts')
        self.number_of_videos = number_of_videos
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_channel_id(self, channel_url: str) -> str:
        """Extract channel ID from various forms of YouTube channel URLs."""
        if 'youtube.com/channel/' in channel_url:
            return channel_url.split('youtube.com/channel/')[-1].split('/')[0]
        
        # Handle custom URLs
        if 'youtube.com/c/' in channel_url or 'youtube.com/@' in channel_url:
            channel_name = channel_url.split('/')[-1].replace('@', '')
            request = self.youtube.search().list(
                part='snippet',
                q=channel_name,
                type='channel',
                maxResults=1
            )
            response = request.execute()
            if response['items']:
                return response['items'][0]['id']['channelId']
        
        raise ValueError("Could not extract channel ID from URL")
    
    def get_top_videos(self, channel_id: str) -> List[Dict]:
        """Get the top viewed videos from a channel."""
        request = self.youtube.search().list(
            part='snippet',
            channelId=channel_id,
            maxResults=self.number_of_videos,
            order='viewCount',
            type='video'
        )
        response = request.execute()
        
        videos = []
        for item in response['items']:
            video_id = item['id']['videoId']
            
            # Get video statistics
            video_request = self.youtube.videos().list(
                part='statistics,snippet',
                id=video_id
            )
            video_response = video_request.execute()
            
            if video_response['items']:
                video_data = video_response['items'][0]
                videos.append({
                    'id': video_id,
                    'title': video_data['snippet']['title'],
                    'views': int(video_data['statistics']['viewCount']),
                    'source': f'https://youtube.com/watch?v={video_id}'
                })
        
        return videos
    
    def get_transcript(self, video_id: str) -> List[Dict]:
        """Get transcript for a specific video."""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript
        except TranscriptsDisabled:
            print(f"Transcripts are disabled for video {video_id}")
            return []
        except Exception as e:
            print(f"Error getting transcript for video {video_id}: {str(e)}")
            return []
    
    def sanitize_filename(self, filename: str) -> str:
        """Convert a string to a valid filename by removing or replacing invalid characters."""
        # Remove invalid characters and replace spaces with underscores
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = filename.replace(' ', '_')
        # Ensure the filename is not too long (max 255 characters including extension)
        if len(filename) > 250:  # Leave room for extension
            filename = filename[:250]
        return filename

    def get_channel_name(self, channel_url: str) -> str:
        """Extract channel name from URL."""
        if '@' in channel_url:
            return channel_url.split('@')[-1]
        return 'unknown_channel'

    def process_channel(self, channel_url: str, interactive: bool = True):
        """Process a YouTube channel's top videos.
        
        Args:
            channel_url (str): URL or handle of the YouTube channel
            interactive (bool, optional): If True, prompt for each video. If False, process all videos. Defaults to True.
        """
        try:
            channel_id = self.get_channel_id(channel_url)
            channel_name = self.get_channel_name(channel_url)
            videos = self.get_top_videos(channel_id)
            
            # Create channel directory
            channel_dir = self.base_dir / channel_name
            channel_dir.mkdir(parents=True, exist_ok=True)
            
            for video in videos:
                print(f"\nVideo: {video['title']}")
                print(f"Views: {video['views']}")
                print(f"Source: {video['source']}")
                
                should_process = True
                if interactive:
                    process = input("Process this video? (y/n): ").lower().strip()
                    should_process = process == 'y'
                
                if should_process:
                    transcript = self.get_transcript(video['id'])
                    if transcript:
                        # Create filename from video title
                        safe_title = self.sanitize_filename(video['title'])
                        output_file = channel_dir / f"{safe_title}.json"
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                'video_info': video,
                                'transcript': transcript
                            }, f, indent=2, ensure_ascii=False)
                        print(f"Transcript saved to {output_file}")
                    else:
                        print("No transcript available for this video")
                
        except Exception as e:
            print(f"Error processing channel: {str(e)}")
            raise
