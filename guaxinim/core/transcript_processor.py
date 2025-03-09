"""Module for processing YouTube video transcripts and generating summaries using OpenAI."""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import openai

class TranscriptProcessor:
    """Process YouTube transcripts and generate summaries using OpenAI."""

    def __init__(self):
        """Initialize the processor."""
        self.base_input_dir = Path('data/raw/transcripts')
        self.base_output_dir = Path('data/raw/youtube_processed')
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def merge_transcript(self, transcript: List[Dict]) -> str:
        """Merge transcript segments into a single text."""
        return ' '.join(segment['text'] for segment in transcript)

    def get_summary_and_tags(self, text: str, title: str) -> Tuple[str, List[str]]:
        """Generate summary and tags using OpenAI."""
        prompt = f"""Given the following video title and transcript, please provide:
                    1. A concise summary of the video content (max 200 words)
                    2. 5 relevant tags that best represent the video's content

                    Title: {title}
                    Transcript: {text}

                    Please format your response as follows:
                    SUMMARY:
                    [Your summary here]

                    TAGS:
                    - [tag1]
                    - [tag2]
                    - [tag3]
                    - [tag4]
                    - [tag5]"""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes video transcripts and provides summaries and tags."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Parse the response
        response_text = response.choices[0].message.content
        summary_section = response_text.split('TAGS:')[0].replace('SUMMARY:', '').strip()
        tags_section = response_text.split('TAGS:')[1].strip()
        tags = [tag.strip('- ').strip() for tag in tags_section.split('\n') if tag.strip()]

        return summary_section, tags

    def process_transcript_file(self, file_path: Path) -> Dict:
        """Process a single transcript file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Merge transcript segments
        merged_text = self.merge_transcript(data['transcript'])
        
        # Get summary and tags from OpenAI
        summary, tags = self.get_summary_and_tags(merged_text, data['video_info']['title'])

        # Prepare processed data
        processed_data = {
            'video_info': data['video_info'],
            'full_text': merged_text,
            'summary': summary,
            'tags': tags
        }

        return processed_data

    def process_channel(self, channel_name: str):
        """Process all transcript files for a specific channel."""
        input_dir = self.base_input_dir / channel_name
        output_dir = self.base_output_dir / channel_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            raise ValueError(f"No transcript directory found for channel: {channel_name}")

        for file_path in input_dir.glob('*.json'):
            try:
                print(f"Processing {file_path.name}...")
                processed_data = self.process_transcript_file(file_path)
                
                # Save processed data
                output_file = output_dir / file_path.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
                print(f"Saved processed data to {output_file}")
            
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
