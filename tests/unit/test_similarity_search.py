"""
Unit tests for the similarity search functionality
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.pdf_processor.similarity_search import DocumentSearcher


class TestDocumentSearcher(unittest.TestCase):
    """Test cases for the DocumentSearcher class."""

    def setUp(self):
        """Set up test cases"""
        # Mock JSON data for testing
        self.test_documents = [
            {
                'title': 'Test Coffee Guide',
                'source': 'https://test.com/coffee-guide',
                'embeddings': [[0.1, 0.2, 0.3]],  # Single chunk embedding
                'chunks': ['How to make great coffee'],
                'title_embedding': [0.4, 0.5, 0.6]
            }
        ]

        # Create patcher for json load
        self.json_patcher = patch('json.load')
        self.mock_json_load = self.json_patcher.start()
        self.mock_json_load.return_value = self.test_documents

        # Initialize searcher with mock data
        with patch('src.pdf_processor.similarity_search.SentenceTransformer') as mock_transformer_cls:
            # Configure the mock transformer
            mock_transformer = MagicMock()
            test_embedding = np.array([[0.1, 0.2, 0.3]])
            mock_transformer.encode.return_value = test_embedding
            mock_transformer_cls.return_value = mock_transformer

            # Mock the file operations
            with patch('builtins.open'):
                self.searcher = DocumentSearcher()

                # Mock the FAISS indices
                self.searcher.chunk_index = MagicMock()
                self.searcher.title_index = MagicMock()

    def tearDown(self):
        """Clean up after tests"""
        self.json_patcher.stop()

    def test_collect_search_results_with_chunks(self):
        """Test collecting search results for chunk search"""
        # Setup test data
        distances = np.array([0.4])  # L2 distance
        indices = np.array([0])  # Index of the test document

        # Call the method
        results = self.searcher._collect_search_results(
            distances=distances,
            indices=indices,
            search_type='chunk'
        )

        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result['title'], 'Test Coffee Guide')
        self.assertEqual(result['source'], 'https://test.com/coffee-guide')
        self.assertEqual(result['chunk_text'], 'How to make great coffee')
        self.assertAlmostEqual(result['similarity_score'], 0.8)  # 1 - 0.4/2

    def test_collect_search_results_with_titles(self):
        """Test collecting search results for title search"""
        # Setup test data
        distances = np.array([0.6])  # L2 distance
        indices = np.array([0])  # Index of the test document

        # Call the method
        results = self.searcher._collect_search_results(
            distances=distances,
            indices=indices,
            search_type='title'
        )

        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result['title'], 'Test Coffee Guide')
        self.assertEqual(result['source'], 'https://test.com/coffee-guide')
        self.assertNotIn('chunk_text', result)
        self.assertAlmostEqual(result['similarity_score'], 0.7)  # 1 - 0.6/2

    def test_collect_search_results_with_invalid_index(self):
        """Test collecting search results with invalid index"""
        # Setup test data with invalid index
        distances = np.array([0.4])
        indices = np.array([-1])  # Invalid index

        # Call the method
        results = self.searcher._collect_search_results(
            distances=distances,
            indices=indices,
            search_type='chunk'
        )

        # Verify results
        self.assertEqual(len(results), 0)  # Should return empty list

    def test_collect_search_results_url_cleaning(self):
        """Test that URLs are properly cleaned in search results"""
        # Setup test data with URL containing whitespace
        self.test_documents[0]['source'] = 'https://test.com/coffee  guide'
        distances = np.array([0.4])
        indices = np.array([0])

        # Call the method
        results = self.searcher._collect_search_results(
            distances=distances,
            indices=indices,
            search_type='chunk'
        )

        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result['source'], 'https://test.com/coffeeguide')

    def test_collect_search_results_empty_input(self):
        """Test collecting search results with empty input"""
        # Setup empty test data
        distances = np.array([])
        indices = np.array([])

        # Call the method
        results = self.searcher._collect_search_results(
            distances=distances,
            indices=indices,
            search_type='chunk'
        )

        # Verify results
        self.assertEqual(len(results), 0)  # Should return empty list


if __name__ == '__main__':
    unittest.main()
