import os
import sys
import json
import argparse
import numpy as np
import PyPDF2
from typing import Dict, List, Union
from sentence_transformers import SentenceTransformer

def extract_pdf_content(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately equal size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_embeddings(chunks: List[str], model_name: str = 'all-MiniLM-L6-v2') -> List[List[float]]:
    """
    Create embeddings for a list of text chunks using sentence-transformers.
    
    Args:
        chunks (List[str]): List of text chunks to create embeddings for
        model_name (str): Name of the sentence-transformers model to use
        
    Returns:
        List[List[float]]: List of embeddings for each chunk
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    # Convert numpy arrays to lists for JSON serialization
    return embeddings.tolist()

def process_pdf_folder(folder_path: str, create_embeddings_flag: bool = False, model_name: str = 'all-MiniLM-L6-v2') -> List[Dict]:
    """
    Process all PDFs in the given folder and return a list of dictionaries with their content.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
        
    Returns:
        List[Dict]: List of dictionaries containing processed PDF information
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
        
    results = []
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}")
        
        try:
            content = extract_pdf_content(pdf_path)
            
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
            
            # Remove title and source line from content before creating chunks
            filtered_lines = []
            for i, line in enumerate(lines):
                if i != 0 and i != source_line_index:  # Skip title (index 0) and source line
                    filtered_lines.append(line)
            
            # Create chunks from filtered content
            filtered_content = '\n'.join(filtered_lines)
            chunks = split_into_chunks(filtered_content)
            
            # Create embeddings if requested
            chunk_embeddings = None
            title_embedding = None
            if create_embeddings_flag:
                print(f"Creating embeddings for {filename}...")
                if chunks:
                    chunk_embeddings = create_embeddings(chunks, model_name)
                if title:
                    title_embedding = create_embeddings([title], model_name)[0]  # Get first (and only) embedding
            
            # Create document dictionary
            doc_dict = {
                "title": title,
                "title_embedding": title_embedding,
                "source": source,
                "chunks": chunks,
                "embeddings": chunk_embeddings
            }
            
            results.append(doc_dict)
            print(f"Successfully processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return results

def save_to_file(data: List[Dict], output_path: str) -> None:
    """Save processed documents to a JSON file.

    Args:
        data (List[Dict]): List of processed document dictionaries
        output_path (str): Path where to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved processed documents to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process PDF files in a folder')
    parser.add_argument('folder_path', 
                      help='Path to the folder containing PDF files')
    parser.add_argument('--output_path',
                      help='Path where to save the processed documents (JSON format)')
    parser.add_argument('--chunk-size', 
                      type=int, 
                      default=1000,
                      help='Size of text chunks (default: 1000 characters)')
    parser.add_argument('--create-embeddings',
                      action='store_true',
                      help='Create embeddings for text chunks')
    parser.add_argument('--model-name',
                      default='all-MiniLM-L6-v2',
                      help='Name of the sentence-transformers model to use for embeddings')
    
    args = parser.parse_args()
    
    processed_documents = process_pdf_folder(
        args.folder_path,
        create_embeddings_flag=args.create_embeddings,
        model_name=args.model_name
    )
    
    # Print processing summary
    print("\nProcessing Summary:")
    for doc in processed_documents:
        print(f"\nDocument Title: {doc['title']}")
        print(f"Source: {doc['source']}")
        print(f"Number of chunks: {len(doc['chunks'])}")
    
    # Save to file
    save_to_file(processed_documents, args.output_path)

if __name__ == "__main__":
    main()
