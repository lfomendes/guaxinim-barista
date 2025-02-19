"""Module for creating embeddings from processed documents."""
import json
from pathlib import Path
from typing import Dict, List, Union
from sentence_transformers import SentenceTransformer

class EmbeddingsProcessor:
    """Create and manage embeddings for processed documents."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embeddings processor."""
        self.model = SentenceTransformer(model_name)
        self.base_input_dir = Path('data/raw')
        self.output_dir = Path('data/processed/documents')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
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

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def process_document(self, file_path: Path) -> Dict:
        """Process a single document and create embeddings."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract fields, handling both flat and nested structures
        try:
            if 'video_info' in data:
                # YouTube structure
                title = data['video_info']['title']
                source = data['video_info']['source']
            else:
                # PDF structure
                title = data['title']
                source = data.get('source', '')

            full_text = data['full_text']
            summary = data['summary']
            tags = data['tags']

        except KeyError as e:
            raise ValueError(f"Missing required field in {file_path.name}: {e}") from e

        # Split full text into chunks
        chunks = self.split_into_chunks(full_text)
        
        # Create embeddings
        chunk_embeddings = self.create_embeddings(chunks)
        title_embedding = self.create_embeddings([title])[0]
        summary_embedding = self.create_embeddings([summary])[0]
        
        # Prepare processed data with embeddings
        processed_data = {
            "title": title,
            "title_embedding": title_embedding,
            "source": source,
            "summary": summary,
            "summary_embedding": summary_embedding,
            "tags": tags,
            "chunks": chunks,
            "chunk_embeddings": chunk_embeddings
        }

        return processed_data

    def process_all_documents(self):
        """Process all documents from both PDF and YouTube sources."""
        source_types = ['pdf_processed', 'youtube_processed']
        
        for source_type in source_types:
            input_base = self.base_input_dir / source_type
            if not input_base.exists():
                print(f"Warning: Directory not found: {input_base}")
                continue

            # Process all JSON files in this source type directory
            for file_path in input_base.glob('**/*.json'):
                try:
                    print(f"Processing {file_path.relative_to(self.base_input_dir)}...")
                    processed_data = self.process_document(file_path)
                    
                    # Save processed data with embeddings
                    output_file = self.output_dir / file_path.name
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_data, f, indent=2, ensure_ascii=False)
                    print(f"Saved embeddings to {output_file}")
                
                except Exception as e:
                    print(f"Error processing {file_path.name}: {str(e)}")
