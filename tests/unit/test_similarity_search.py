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
                'title_embedding': [0.4, 0.5, 0.6],
                'summary': 'A guide about coffee brewing',
                'tags': ['coffee_brewing', 'beginner_guide']
            },
            {
                'title': 'Coffee Bean Types',
                'source': 'https://test.com/coffee-beans',
                'embeddings': [[0.7, 0.8, 0.9]],
                'chunks': ['Different types of coffee beans'],
                'title_embedding': [0.1, 0.2, 0.3],
                'summary': 'All about coffee beans',
                'tags': ['coffee_beans', 'beginner_guide']
            },
            {
                'title': 'Advanced Brewing',
                'source': 'https://test.com/advanced-brewing',
                'embeddings': [[0.4, 0.5, 0.6]],
                'chunks': ['Advanced coffee brewing techniques'],
                'title_embedding': [0.7, 0.8, 0.9],
                'summary': 'Advanced brewing methods',
                'tags': ['coffee_brewing', 'advanced_guide']
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

                # Set the test documents
                self.searcher.documents = self.test_documents

                # Mock the FAISS indices
                self.searcher.chunk_index = MagicMock()
                self.searcher.title_index = MagicMock()
                self.searcher.summary_index = MagicMock()
                
                # Initialize mappings
                self.searcher.chunk_mapping = []
                self.searcher.title_mapping = []
                self.searcher.summary_mapping = []
                for doc_idx, doc in enumerate(self.test_documents):
                    if doc.get('chunks'):
                        for chunk_idx, chunk_text in enumerate(doc['chunks']):
                            self.searcher.chunk_mapping.append((doc_idx, chunk_idx, chunk_text))
                    if doc.get('title'):
                        self.searcher.title_mapping.append((doc_idx, doc['title']))
                    if doc.get('summary'):
                        self.searcher.summary_mapping.append((doc_idx, doc['summary']))

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

    def test_collect_search_results_with_summaries(self):
        """Test collecting search results for summary search"""
        # Setup test data
        distances = np.array([0.4])  # L2 distance
        indices = np.array([0])  # Index of the test document

        # Call the method
        results = self.searcher._collect_search_results(
            distances=distances,
            indices=indices,
            search_type='summary'
        )

        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result['title'], 'Test Coffee Guide')
        self.assertEqual(result['source'], 'https://test.com/coffee-guide')
        self.assertEqual(result['summary'], 'A guide about coffee brewing')
        self.assertNotIn('chunk_text', result)
        self.assertAlmostEqual(result['similarity_score'], 0.8)  # 1 - 0.4/2

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

    def test_get_tags_with_frequency(self):
        """Test getting tags with their frequencies"""
        # Get tags with frequencies
        tags_with_freq = self.searcher.get_tags_with_frequency()

        # Verify results
        expected_tags = [
            ('beginner_guide', 2),  # In 2 documents
            ('coffee_brewing', 2),  # In 2 documents
            ('advanced_guide', 1),  # In 1 document
            ('coffee_beans', 1)     # In 1 document
        ]
        self.assertEqual(tags_with_freq, expected_tags)

    def test_get_all_tags(self):
        """Test getting all unique tags from documents"""
        # Get all tags
        tags = self.searcher.get_all_tags()

        # Verify results - should be in same order as get_tags_with_frequency
        expected_tags = ['beginner_guide', 'coffee_brewing', 'advanced_guide', 'coffee_beans']
        self.assertEqual(tags, expected_tags)

    def test_search_by_tag(self):
        """Test searching documents by tag"""
        # Test searching for 'coffee_brewing' tag
        results = self.searcher.search_by_tag('coffee_brewing', limit=10)
        self.assertEqual(len(results), 2)
        titles = sorted(r['title'] for r in results)
        self.assertEqual(titles, ['Advanced Brewing', 'Test Coffee Guide'])

        # Test searching for 'beginner_guide' tag
        results = self.searcher.search_by_tag('beginner_guide')
        self.assertEqual(len(results), 2)
        titles = [r['title'] for r in results]
        self.assertIn('Test Coffee Guide', titles)
        self.assertIn('Coffee Bean Types', titles)

        # Test searching for non-existent tag
        results = self.searcher.search_by_tag('non_existent_tag')
        self.assertEqual(len(results), 0)

        # Test limit parameter
        results = self.searcher.search_by_tag('beginner_guide', limit=1)
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()
