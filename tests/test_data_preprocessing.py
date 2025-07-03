import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import DataPreprocessor
from src.predictor import CareerPredictor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.predictor = CareerPredictor()
        self.predictor.load_model()
        self.data_path = "data/career_map.csv"
        self.expected_features = self.predictor.model.n_features_in_

    def test_preprocess_pipeline(self):
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_pipeline(self.data_path)
        self.assertEqual(X_train.shape[1], self.expected_features)
        self.assertEqual(X_test.shape[1], self.expected_features)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_get_feature_names(self):
        self.assertEqual(len(self.preprocessor.get_feature_names()), self.expected_features)

    def test_get_target_classes(self):
        self.preprocessor.preprocess_pipeline(self.data_path)
        self.assertGreater(len(self.preprocessor.get_target_classes()), 0)

if __name__ == '__main__':
    unittest.main()
