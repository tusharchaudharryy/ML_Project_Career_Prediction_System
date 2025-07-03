import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import DataPreprocessor
from src.model_training import CareerModelTrainer
from src.predictor import CareerPredictor
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class TestCareerModelTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = CareerModelTrainer()
        self.preprocessor = DataPreprocessor()
        self.predictor = CareerPredictor()

        # Use a small test CSV (you must ensure this exists)
        self.test_csv = os.path.join(os.path.dirname(__file__), 'test_data.csv')
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.preprocess_pipeline(self.test_csv)

        # Get expected feature count from predictor
        self.predictor.load_model()
        self.expected_features = self.predictor.model.n_features_in_

    def test_training_and_evaluation(self):
        self.assertEqual(self.X_train.shape[1], self.expected_features)
        self.assertEqual(self.X_test.shape[1], self.expected_features)

        result = self.trainer.train_and_evaluate(self.X_train, self.X_test, self.y_train, self.y_test)

        self.assertIn('best_model', result)
        self.assertIn('accuracy', result)
        self.assertIn('classification_report', result)

        self.assertGreater(result['accuracy'], 0)
        print("\nAccuracy:", result['accuracy'])

    def test_model_saving_and_loading(self):
        model_data = self.trainer.train_and_evaluate(self.X_train, self.X_test, self.y_train, self.y_test)
        self.trainer.save_model(model_data, 'test_model.pkl')

        self.assertTrue(os.path.exists('test_model.pkl'))

        loaded = self.trainer.load_model('test_model.pkl')
        self.assertIn('model', loaded)
        self.assertIn('feature_names', loaded)
        self.assertIn('target_classes', loaded)

    def tearDown(self):
        if os.path.exists('test_model.pkl'):
            os.remove('test_model.pkl')

if __name__ == '__main__':
    unittest.main()
