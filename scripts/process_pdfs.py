#!/usr/bin/env python3
"""Script to process PDF documents and generate summaries."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from guaxinim.core.pdf_processor import PDFProcessor

def process_pdfs(folder_path: str):
    """Process all PDF files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
    """
    try:
        processor = PDFProcessor()
        processor.process_folder(folder_path)
    except Exception as e:
        print(f"Error processing PDFs in {folder_path}: {str(e)}")
        raise

def main():
    """Main function to run the processor."""
    # Load environment variables
    load_dotenv()
    
    # Ensure OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    # Get folder name from command line argument
    if len(sys.argv) != 2:
        print("Usage: python process_pdfs.py <folder_name>")
        print("Example: python process_pdfs.py coffee_articles")
        sys.exit(1)

    folder_name = sys.argv[1]
    
    try:
        process_pdfs(folder_name)
    except Exception as e:
        print(f"Error processing folder: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
