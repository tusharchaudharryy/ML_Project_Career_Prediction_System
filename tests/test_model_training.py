import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import tempfile

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        """Set up test data and model trainer instance"""
        self.trainer = ModelTrainer()

        # Create sample training data
        np.random.seed(42)  # For reproducible results
        self.X_train = pd.DataFrame({
            f'feature_{i}': np.random.rand(100) for i in range(28)
        })
        self.y_train = pd.Series(np.random.choice(
            ['Data Scientist', 'Software Developer', 'Web Developer', 
             'Network Security Analyst', 'Mobile Application Developer'], 
            size=100
        ))

        self.X_test = pd.DataFrame({
            f'feature_{i}': np.random.rand(30) for i in range(28)
        })
        self.y_test = pd.Series(np.random.choice(
            ['Data Scientist', 'Software Developer', 'Web Developer', 
             'Network Security Analyst', 'Mobile Application Developer'], 
            size=30
        ))

    def test_initialization(self):
        """Test ModelTrainer initialization"""
        self.assertIsInstance(self.trainer, ModelTrainer)
        self.assertIsNone(self.trainer.model)
        self.assertIsInstance(self.trainer.models, dict)
        self.assertIn('random_forest', self.trainer.models)
        self.assertIn('svm', self.trainer.models)

    def test_train_model_random_forest(self):
        """Test training with Random Forest"""
        accuracy = self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        self.assertIsInstance(accuracy, float)
        self.assertGreater(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertIsNotNone(self.trainer.model)

    def test_train_model_svm(self):
        """Test training with SVM"""
        accuracy = self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='svm'
        )

        self.assertIsInstance(accuracy, float)
        self.assertGreater(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertIsNotNone(self.trainer.model)

    def test_train_model_decision_tree(self):
        """Test training with Decision Tree"""
        accuracy = self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='decision_tree'
        )

        self.assertIsInstance(accuracy, float)
        self.assertGreater(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertIsNotNone(self.trainer.model)

    def test_train_model_invalid_algorithm(self):
        """Test training with invalid algorithm"""
        with self.assertRaises(ValueError):
            self.trainer.train_model(
                self.X_train, self.y_train, 
                self.X_test, self.y_test, 
                algorithm='invalid_algorithm'
            )

    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning"""
        best_params = self.trainer.hyperparameter_tuning(
            self.X_train, self.y_train, 
            algorithm='random_forest'
        )

        self.assertIsInstance(best_params, dict)
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)

    def test_cross_validation(self):
        """Test cross-validation"""
        scores = self.trainer.cross_validation(
            self.X_train, self.y_train, 
            algorithm='random_forest'
        )

        self.assertIsInstance(scores, dict)
        self.assertIn('mean_accuracy', scores)
        self.assertIn('std_accuracy', scores)
        self.assertIn('scores', scores)

        # Check score ranges
        self.assertGreater(scores['mean_accuracy'], 0.0)
        self.assertLessEqual(scores['mean_accuracy'], 1.0)
        self.assertGreaterEqual(scores['std_accuracy'], 0.0)

    def test_evaluate_model(self):
        """Test model evaluation"""
        # First train a model
        self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        # Then evaluate it
        metrics = self.trainer.evaluate_model(self.X_test, self.y_test)

        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('classification_report', metrics)
        self.assertIn('confusion_matrix', metrics)

    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        # Train a tree-based model first
        self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        importance = self.trainer.get_feature_importance()

        self.assertIsInstance(importance, pd.DataFrame)
        self.assertEqual(len(importance), 28)  # 28 features
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)

    def test_save_model(self):
        """Test model saving"""
        # Train a model first
        self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            self.trainer.save_model(tmp_file.name)

            # Verify file exists and can be loaded
            self.assertTrue(os.path.exists(tmp_file.name))
            loaded_model = joblib.load(tmp_file.name)
            self.assertIsNotNone(loaded_model)

            # Clean up
            os.unlink(tmp_file.name)

    def test_load_model(self):
        """Test model loading"""
        # Train and save a model first
        self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            self.trainer.save_model(tmp_file.name)

            # Create new trainer and load model
            new_trainer = ModelTrainer()
            new_trainer.load_model(tmp_file.name)

            self.assertIsNotNone(new_trainer.model)

            # Clean up
            os.unlink(tmp_file.name)

    def test_compare_models(self):
        """Test model comparison"""
        results = self.trainer.compare_models(
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )

        self.assertIsInstance(results, dict)
        self.assertIn('random_forest', results)
        self.assertIn('svm', results)
        self.assertIn('decision_tree', results)

        # Check each model result has required metrics
        for model_name, metrics in results.items():
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1_score', metrics)

    def test_predict_single(self):
        """Test single prediction"""
        # Train a model first
        self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        # Make prediction on single sample
        single_sample = self.X_test.iloc[0:1]
        prediction = self.trainer.predict(single_sample)

        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(len(prediction), 1)

    def test_predict_batch(self):
        """Test batch prediction"""
        # Train a model first
        self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        # Make predictions on test set
        predictions = self.trainer.predict(self.X_test)

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_predict_proba(self):
        """Test probability prediction"""
        # Train a model first
        self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        # Get prediction probabilities
        probabilities = self.trainer.predict_proba(self.X_test)

        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape[0], len(self.X_test))

        # Check probabilities sum to 1
        for prob_row in probabilities:
            self.assertAlmostEqual(np.sum(prob_row), 1.0, places=5)

    def test_get_model_info(self):
        """Test model information retrieval"""
        # Train a model first
        self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        model_info = self.trainer.get_model_info()

        self.assertIsInstance(model_info, dict)
        self.assertIn('algorithm', model_info)
        self.assertIn('parameters', model_info)
        self.assertIn('feature_count', model_info)

    def test_error_handling(self):
        """Test error handling for various edge cases"""
        # Test prediction without trained model
        with self.assertRaises(ValueError):
            self.trainer.predict(self.X_test)

        # Test evaluation without trained model
        with self.assertRaises(ValueError):
            self.trainer.evaluate_model(self.X_test, self.y_test)

        # Test feature importance without tree-based model
        self.trainer.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='svm'
        )
        with self.assertRaises(AttributeError):
            self.trainer.get_feature_importance()

    def test_data_validation(self):
        """Test input data validation"""
        # Test with mismatched feature dimensions
        wrong_X = pd.DataFrame({f'feature_{i}': np.random.rand(10) for i in range(20)})

        with self.assertRaises(ValueError):
            self.trainer.train_model(
                wrong_X, self.y_train[:10], 
                self.X_test, self.y_test, 
                algorithm='random_forest'
            )

    def test_random_state_reproducibility(self):
        """Test reproducibility with random state"""
        # Train two models with same random state
        trainer1 = ModelTrainer(random_state=42)
        trainer2 = ModelTrainer(random_state=42)

        acc1 = trainer1.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        acc2 = trainer2.train_model(
            self.X_train, self.y_train, 
            self.X_test, self.y_test, 
            algorithm='random_forest'
        )

        # Results should be identical
        self.assertEqual(acc1, acc2)

if __name__ == '__main__':
    unittest.main()
