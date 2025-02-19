import json
import numpy as np
import faiss
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer

class DocumentSearcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the document searcher to load documents from the data/processed/documents directory.
        
        Args:
            model_name (str): Name of the sentence-transformer model to use for new queries
        """
        self.documents_dir = 'data/processed/documents'
        self.model = SentenceTransformer(model_name)
        self.documents = self._load_documents()
        
        if not self.documents:
            raise ValueError(f'No documents found in {self.documents_dir}')
            
        # Initialize FAISS indexes for different types of embeddings
        self.embedding_dim = len(self.documents[0]['title_embedding'])  # All embeddings should have same dimension
        self.chunk_index = self._create_chunk_index()
        self.title_index = self._create_title_index()
        self.summary_index = self._create_summary_index()
        
        # Store mappings for quick lookup
        self.chunk_mapping = self._create_chunk_mapping()
        self.title_mapping = self._create_title_mapping()
        self.summary_mapping = self._create_summary_mapping()
    
    def _load_documents(self) -> List[Dict]:
        """Load all JSON documents from the data/processed/documents directory."""
        documents = []
        import os
        if not os.path.exists(self.documents_dir):
            return documents
            
        for filename in os.listdir(self.documents_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.documents_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        doc = json.load(f)
                        documents.extend(doc if isinstance(doc, list) else [doc])
                    except json.JSONDecodeError:
                        print(f'Error loading {filename}: Invalid JSON format')
                        continue
        return documents
    
    def _create_chunk_index(self) -> faiss.IndexFlatL2:
        """Create FAISS index for chunks."""
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Collect all chunk embeddings
        embeddings = []
        for doc in self.documents:
            if doc.get('chunk_embeddings'):
                embeddings.extend(doc['chunk_embeddings'])
        
        if embeddings:
            index.add(np.array(embeddings, dtype=np.float32))
        return index
    
    def _create_title_index(self) -> faiss.IndexFlatL2:
        """Create FAISS index for titles."""
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Collect all title embeddings
        embeddings = []
        for doc in self.documents:
            if doc.get('title_embedding'):
                embeddings.append(doc['title_embedding'])
        
        if embeddings:
            index.add(np.array(embeddings, dtype=np.float32))
        return index
        
    def _create_summary_index(self) -> faiss.IndexFlatL2:
        """Create FAISS index for summaries."""
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Collect all summary embeddings
        embeddings = []
        for doc in self.documents:
            if doc.get('summary_embedding'):
                embeddings.append(doc['summary_embedding'])
        
        if embeddings:
            index.add(np.array(embeddings, dtype=np.float32))
        return index
    
    def _create_chunk_mapping(self) -> List[Tuple[int, int, str]]:
        """Create mapping of chunk index to document and chunk index."""
        mapping = []
        global_idx = 0
        for doc_idx, doc in enumerate(self.documents):
            if doc.get('chunk_embeddings'):
                for chunk_idx in range(len(doc['chunk_embeddings'])):
                    mapping.append((doc_idx, chunk_idx, doc['chunks'][chunk_idx]))
                    global_idx += 1
        return mapping
    
    def _create_title_mapping(self) -> List[Tuple[int, str]]:
        """Create mapping of title index to document index."""
        mapping = []
        for doc_idx, doc in enumerate(self.documents):
            if doc.get('title_embedding'):
                mapping.append((doc_idx, doc['title']))
        return mapping
                
    def _create_summary_mapping(self) -> List[Tuple[int, str]]:
        """Create mapping of summary index to document index."""
        mapping = []
        for doc_idx, doc in enumerate(self.documents):
            if doc.get('summary_embedding'):
                mapping.append((doc_idx, doc['summary']))
        return mapping
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Search for chunks similar to the query.
        
        Args:
            query (str): The search query
            k (int): Number of similar chunks to return
            
        Returns:
            List[Dict]: List of dictionaries containing similar chunks and their metadata
        """
        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS index
        distances, indices = self.chunk_index.search(query_embedding, k)
        
        return self._collect_search_results(distances[0], indices[0], search_type='chunk')
    
    def search_similar_titles(self, query: str, k: int = 5) -> List[Dict]:
        """Search for documents with titles similar to the query.
        
        Args:
            query (str): The search query
            k (int): Number of similar titles to return
            
        Returns:
            List[Dict]: List of dictionaries containing similar documents and their metadata
        """
        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS index
        distances, indices = self.title_index.search(query_embedding, k)
        
        return self._collect_search_results(distances[0], indices[0], search_type='title')
        
    def search_similar_summaries(self, query: str, k: int = 5) -> List[Dict]:
        """Search for documents with summaries similar to the query.
        
        Args:
            query (str): The search query
            k (int): Number of similar summaries to return
            
        Returns:
            List[Dict]: List of dictionaries containing similar documents and their metadata
        """
        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS index
        distances, indices = self.summary_index.search(query_embedding, k)
        
        return self._collect_search_results(distances[0], indices[0], search_type='summary')

    def _collect_search_results(self, distances: np.ndarray, indices: np.ndarray, search_type: str = 'chunk') -> List[Dict]:
        """Collect search results from FAISS search output.
        
        Args:
            distances (np.ndarray): Array of distances from FAISS search
            indices (np.ndarray): Array of indices from FAISS search
            search_type (str): Type of search being performed ('chunk', 'title', or 'summary')
            
        Returns:
            List[Dict]: List of dictionaries containing search results and metadata
        """
        results = []
        for dist, idx in zip(distances, indices):
            if idx != -1:  # Valid index
                if search_type == 'chunk':
                    doc_idx, chunk_idx, chunk_text = self.chunk_mapping[idx]
                    doc = self.documents[doc_idx]
                    result = {
                        'title': doc['title'],
                        'source': doc['source'],
                        'chunk_text': chunk_text,
                        'tags': doc.get('tags', []),
                        'similarity_score': 1 - dist/2
                    }
                elif search_type == 'title':
                    doc_idx, title = self.title_mapping[idx]
                    doc = self.documents[doc_idx]
                    result = {
                        'title': title,
                        'source': doc['source'],
                        'summary': doc.get('summary', ''),
                        'tags': doc.get('tags', []),
                        'similarity_score': 1 - dist/2
                    }
                else:  # summary
                    doc_idx, summary = self.summary_mapping[idx]
                    doc = self.documents[doc_idx]
                    result = {
                        'title': doc['title'],
                        'source': doc['source'],
                        'summary': summary,
                        'tags': doc.get('tags', []),
                        'similarity_score': 1 - dist/2
                    }
                results.append(result)
        return results

def main():
    # Example usage
    searcher = DocumentSearcher()
    
    # Example: Search for similar chunks
    query = "My coffee is too acidic how can I make it better?"
    print(f"\nSearching chunks similar to: '{query}'")
    results = searcher.search_similar_chunks(query, k=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Title: {result['title']}")
        print(f"   Source: {result['source']}")
        print(f"   Tags: {', '.join(result['tags'])}")
        print(f"   Similarity Score: {result['similarity_score']:.3f}")
        print(f"   Chunk: {result['chunk_text'][:200]}...")
    
    # Example: Search for similar titles
    query = "Beginner's guide to coffee grinders"
    print(f"\nSearching titles similar to: '{query}'")
    results = searcher.search_similar_titles(query, k=2)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Title: {result['title']}")
        print(f"   Source: {result['source']}")
        print(f"   Tags: {', '.join(result['tags'])}")
        print(f"   Summary: {result['summary']}")
        print(f"   Similarity Score: {result['similarity_score']:.3f}")
    
    # Example: Search for similar summaries
    query = "Different types of coffee grinders and their features"
    print(f"\nSearching summaries similar to: '{query}'")
    results = searcher.search_similar_summaries(query, k=2)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Title: {result['title']}")
        print(f"   Source: {result['source']}")
        print(f"   Tags: {', '.join(result['tags'])}")
        print(f"   Summary: {result['summary']}")
        print(f"   Similarity Score: {result['similarity_score']:.3f}")

if __name__ == "__main__":
    main()
