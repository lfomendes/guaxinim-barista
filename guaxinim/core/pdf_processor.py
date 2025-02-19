"""Module for processing PDF documents and generating summaries."""
import os
from pathlib import Path
from typing import Dict, List
import PyPDF2
from .base_processor import BaseProcessor

class PDFProcessor(BaseProcessor):
    """Process PDF documents and generate summaries."""

    def __init__(self):
        """Initialize the PDF processor."""
        super().__init__(
            input_dir='data/raw/pdfs',
            output_dir='data/raw/pdf_processed'
        )

    def extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def process_pdf_file(self, file_path: Path) -> Dict:
        """Process a single PDF file."""
        # Extract content
        content = self.extract_pdf_content(str(file_path))
        
        # Split content into lines
        lines = content.split('\n')
        
        # Get title (first line)
        title = lines[0].strip() if lines else "Untitled"
        
        # Find source and its line index
        source = ""
        source_line_index = -1
        for i, line in enumerate(lines):
            if "--source--" in line:
                source = line.replace("--source--", "").strip()
                source_line_index = i
                break
        
        # Remove title and source line from content
        filtered_lines = []
        for i, line in enumerate(lines):
            if i != 0 and i != source_line_index:  # Skip title and source line
                filtered_lines.append(line)
        
        # Join filtered content
        full_text = '\n'.join(filtered_lines)
        
        # Get summary and tags from OpenAI
        summary, tags = self.get_summary_and_tags(full_text, title)

        # Prepare processed data
        processed_data = {
            "title": title,
            "source": source,
            "full_text": full_text,
            "summary": summary,
            "tags": tags
        }

        return processed_data

    def process_folder(self, folder_name: str):
        """Process all PDF files in a specific folder."""
        input_dir = self.base_input_dir / folder_name
        output_dir = self.base_output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            raise ValueError(f"No PDF directory found for folder: {folder_name}")

        for file_path in input_dir.glob('*.pdf'):
            try:
                print(f"Processing {file_path.name}...")
                processed_data = self.process_pdf_file(file_path)
                
                # Save processed data
                output_file = f"{file_path.stem}.json"
                self.save_processed_file(processed_data, output_file)
            
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
