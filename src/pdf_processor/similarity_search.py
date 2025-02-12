import json
import numpy as np
import faiss
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer

class DocumentSearcher:
    def __init__(self, json_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the document searcher with a JSON file containing documents and embeddings.
        
        Args:
            json_path (str): Path to the JSON file containing documents and their embeddings
            model_name (str): Name of the sentence-transformer model to use for new queries
        """
        self.model = SentenceTransformer(model_name)
        self.documents = self._load_documents(json_path)
        
        # Initialize FAISS indexes
        self.embedding_dim = len(self.documents[0]['embeddings'][0])  # Get dimension from first embedding
        self.chunk_index = self._create_chunk_index()
        self.title_index = self._create_title_index()
        
        # Store mappings for quick lookup
        self.chunk_mapping = self._create_chunk_mapping()
        self.title_mapping = self._create_title_mapping()
    
    def _load_documents(self, json_path: str) -> List[Dict]:
        """Load documents from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_chunk_index(self) -> faiss.IndexFlatL2:
        """Create FAISS index for chunks."""
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Collect all chunk embeddings
        embeddings = []
        for doc in self.documents:
            if doc['embeddings']:
                embeddings.extend(doc['embeddings'])
        
        if embeddings:
            index.add(np.array(embeddings, dtype=np.float32))
        return index
    
    def _create_title_index(self) -> faiss.IndexFlatL2:
        """Create FAISS index for titles."""
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Collect all title embeddings
        embeddings = []
        for doc in self.documents:
            if doc['title_embedding']:
                embeddings.append(doc['title_embedding'])
        
        if embeddings:
            index.add(np.array(embeddings, dtype=np.float32))
        return index
    
    def _create_chunk_mapping(self) -> List[Tuple[int, int, str]]:
        """Create mapping of chunk index to document and chunk index."""
        mapping = []
        global_idx = 0
        for doc_idx, doc in enumerate(self.documents):
            if doc['embeddings']:
                for chunk_idx in range(len(doc['embeddings'])):
                    mapping.append((doc_idx, chunk_idx, doc['chunks'][chunk_idx]))
                    global_idx += 1
        return mapping
    
    def _create_title_mapping(self) -> List[Dict]:
        """Create mapping of title index to document."""
        return [doc for doc in self.documents if doc['title_embedding']]
    
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
        
        return self._collect_search_results(distances[0], indices[0], include_chunk_text=True)
    
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
        
        return self._collect_search_results(distances[0], indices[0], include_chunk_text=False)

    def _collect_search_results(self, distances: np.ndarray, indices: np.ndarray, include_chunk_text: bool = False) -> List[Dict]:
        """Collect search results from FAISS search output.
        
        Args:
            distances (np.ndarray): Array of distances from FAISS search
            indices (np.ndarray): Array of indices from FAISS search
            include_chunk_text (bool): Whether to include chunk text in results
            
        Returns:
            List[Dict]: List of dictionaries containing search results and metadata
        """
        results = []
        for dist, idx in zip(distances, indices):
            if idx != -1:  # Valid index
                if include_chunk_text:
                    doc_idx, chunk_idx, chunk_text = self.chunk_mapping[idx]
                    doc = self.documents[doc_idx]
                    result = {
                        'title': doc['title'],
                        'source': "".join(doc['source'].split()),
                        'chunk_text': chunk_text,
                        'similarity_score': 1 - dist/2
                    }
                else:
                    doc = self.title_mapping[idx]
                    result = {
                        'title': doc['title'],
                        'source': "".join(doc['source'].split()),
                        'similarity_score': 1 - dist/2
                    }
                results.append(result)
        return results

def main():
    # Example usage
    searcher = DocumentSearcher('data/json/hoffman_pdf.json')
    
    # Example: Search for similar chunks
    query = "My coffe is too acidic how can I make it better?"
    print(f"\nSearching chunks similar to: '{query}'")
    results = searcher.search_similar_chunks(query, k=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Title: {result['title']}")
        print(f"   Source: {result['source']}")
        print(f"   Similarity Score: {result['similarity_score']:.3f}")
        print(f"   Chunk: {result['chunk_text'][:200]}...")
    
    # Example: Search for similar titles
    query = "Desserts to eat with coffee"
    print(f"\nSearching titles similar to: '{query}'")
    results = searcher.search_similar_titles(query, k=2)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Title: {result['title']}")
        print(f"   Source: {result['source']}")
        print(f"   Similarity Score: {result['similarity_score']:.3f}")

if __name__ == "__main__":
    main()
