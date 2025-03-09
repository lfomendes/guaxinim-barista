"""Base processor module for handling document processing and OpenAI integration."""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import openai

class BaseProcessor:
    """Base class for processing documents and generating summaries using OpenAI."""

    def __init__(self, input_dir: str, output_dir: str):
        """Initialize the processor with input and output directories."""
        self.base_input_dir = Path(input_dir)
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def get_summary_and_tags(self, text: str, title: str) -> Tuple[str, List[str]]:
        """Generate summary and tags using OpenAI."""
        prompt = f"""Given the following document title and content, please provide:
                    1. A concise summary of the content (max 200 words)
                    2. 5 relevant tags that best represent the document's content

                    Title: {title}
                    Content: {text}

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
                {"role": "system", "content": "You are a helpful assistant that analyzes documents and provides summaries and tags."},
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

    def save_processed_file(self, data: Dict, output_file: Path):
        """Save processed data to a JSON file."""
        with open('{}/{}'.format(self.base_output_dir, output_file), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved processed data to {output_file}")
