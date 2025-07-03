import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        """Set up test data and preprocessor instance"""
        self.preprocessor = DataPreprocessor()

        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'Database Fundamentals': [3, 5, 7, 2, 6],
            'Computer Architecture': [4, 6, 3, 7, 5],
            'Distributed Computing Systems': [2, 4, 6, 3, 5],
            'Cyber Security': [5, 3, 7, 4, 6],
            'Networking': [3, 7, 4, 5, 2],
            'Software Development': [6, 4, 5, 7, 3],
            'Programming Skills': [7, 6, 4, 3, 5],
            'Project Management': [2, 5, 3, 6, 4],
            'Computer Forensics Fundamentals': [4, 3, 7, 2, 5],
            'Technical Communication': [5, 6, 2, 4, 7],
            'AI ML': [3, 4, 6, 5, 2],
            'Software Engineering': [6, 5, 3, 7, 4],
            'Business Analysis': [2, 7, 4, 3, 6],
            'Data Science': [5, 3, 6, 4, 7],
            'Web Development': [4, 6, 5, 2, 3],
            'Mobile App Development': [7, 2, 3, 5, 6],
            'Cloud Computing': [3, 5, 7, 4, 2],
            'Logical quotient rating': [0.8, 0.6, 0.9, 0.7, 0.5],
            'hackathons': [0.3, 0.8, 0.6, 0.4, 0.7],
            'coding skills rating': [0.9, 0.7, 0.8, 0.6, 0.5],
            'public speaking points': [0.4, 0.6, 0.3, 0.8, 0.7],
            'self-learning capability?': [0.8, 0.9, 0.6, 0.7, 0.5],
            'Extra-courses did': [0.5, 0.7, 0.8, 0.4, 0.6],
            'Introvert': [0.6, 0.3, 0.8, 0.5, 0.7],
            'reading and writing skills': [0.7, 0.8, 0.5, 0.6, 0.4],
            'memory capability score': [0.8, 0.6, 0.9, 0.7, 0.5],
            'smart or hard work': [0.6, 0.9, 0.7, 0.8, 0.5],
            'Management or Technical': [0.4, 0.6, 0.8, 0.5, 0.7],
            'Suggested Job Role': ['Data Scientist', 'Software Developer', 'Web Developer', 
                                 'Network Security Analyst', 'Mobile Application Developer']
        })

    def test_initialization(self):
        """Test DataPreprocessor initialization"""
        self.assertIsInstance(self.preprocessor, DataPreprocessor)
        self.assertIsNone(self.preprocessor.X_train)
        self.assertIsNone(self.preprocessor.y_train)

    def test_load_data(self):
        """Test data loading functionality"""
        # Mock pd.read_csv to return sample data
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.sample_data

            result = self.preprocessor.load_data('dummy_path.csv')

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 5)
            mock_read_csv.assert_called_once_with('dummy_path.csv')

    def test_normalize_technical_skills(self):
        """Test technical skills normalization"""
        technical_skills = ['Database Fundamentals', 'Computer Architecture', 
                           'Distributed Computing Systems', 'Cyber Security']

        normalized_data = self.preprocessor.normalize_technical_skills(
            self.sample_data, technical_skills
        )

        # Check if values are normalized to 0-1 range
        for skill in technical_skills:
            self.assertTrue(all(0 <= val <= 1 for val in normalized_data[skill]))

    def test_preprocess_features(self):
        """Test feature preprocessing"""
        X, y = self.preprocessor.preprocess_features(self.sample_data)

        # Check dimensions
        self.assertEqual(X.shape[0], 5)
        self.assertEqual(X.shape[1], 28)  # 17 technical + 11 personality
        self.assertEqual(len(y), 5)

        # Check if technical skills are normalized
        technical_cols = X.columns[:17]  # First 17 columns are technical skills
        for col in technical_cols:
            self.assertTrue(all(0 <= val <= 1 for val in X[col]))

    def test_split_data(self):
        """Test data splitting"""
        X, y = self.preprocessor.preprocess_features(self.sample_data)

        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)

        # Check if split maintains data integrity
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))

    def test_balance_classes(self):
        """Test class balancing with SMOTE"""
        # Create imbalanced data
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6], 
                         'feature2': [2, 4, 6, 8, 10, 12]})
        y = pd.Series(['A', 'A', 'A', 'A', 'B', 'B'])

        X_balanced, y_balanced = self.preprocessor.balance_classes(X, y)

        # Check if classes are more balanced
        self.assertGreaterEqual(len(X_balanced), len(X))
        self.assertGreaterEqual(len(y_balanced), len(y))

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        # Mock file loading
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.sample_data

            result = self.preprocessor.full_preprocessing_pipeline('dummy_path.csv')

            self.assertIsInstance(result, dict)
            self.assertIn('X_train', result)
            self.assertIn('X_test', result)
            self.assertIn('y_train', result)
            self.assertIn('y_test', result)

    def test_validate_input_data(self):
        """Test input data validation"""
        # Test with valid data
        valid_data = {
            'Database Fundamentals': 5,
            'Computer Architecture': 6,
            'Distributed Computing Systems': 4,
            'Cyber Security': 7,
            'Networking': 3,
            'Software Development': 6,
            'Programming Skills': 7,
            'Project Management': 4,
            'Computer Forensics Fundamentals': 5,
            'Technical Communication': 6,
            'AI ML': 5,
            'Software Engineering': 6,
            'Business Analysis': 4,
            'Data Science': 7,
            'Web Development': 5,
            'Mobile App Development': 6,
            'Cloud Computing': 5,
            'Logical quotient rating': 0.8,
            'hackathons': 0.6,
            'coding skills rating': 0.9,
            'public speaking points': 0.7,
            'self-learning capability?': 0.8,
            'Extra-courses did': 0.6,
            'Introvert': 0.4,
            'reading and writing skills': 0.7,
            'memory capability score': 0.8,
            'smart or hard work': 0.6,
            'Management or Technical': 0.7
        }

        is_valid, message = self.preprocessor.validate_input_data(valid_data)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Data is valid")

        # Test with invalid data (missing feature)
        invalid_data = valid_data.copy()
        del invalid_data['Database Fundamentals']

        is_valid, message = self.preprocessor.validate_input_data(invalid_data)
        self.assertFalse(is_valid)
        self.assertIn("Missing required feature", message)

    def test_get_feature_names(self):
        """Test feature names retrieval"""
        feature_names = self.preprocessor.get_feature_names()

        self.assertEqual(len(feature_names), 28)
        self.assertIn('Database Fundamentals', feature_names)
        self.assertIn('Logical quotient rating', feature_names)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_data('non_existent_file.csv')

        # Test with invalid data types
        with self.assertRaises(ValueError):
            self.preprocessor.normalize_technical_skills("invalid_data", [])

if __name__ == '__main__':
    unittest.main()
