# tests/test_predict.py - Tests for prediction module

import pytest
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import CareerPredictor, make_prediction, make_batch_predictions

class TestCareerPredictor:
    """Test suite for CareerPredictor class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.predictor = CareerPredictor()
        
        # Mock model data
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([1])
        self.mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
        
        self.predictor.model = self.mock_model
        self.predictor.feature_names = [f'feature_{i}' for i in range(28)]
        self.predictor.target_classes = ['Software Developer', 'Data Scientist', 'ML Engineer']
        
        # Valid test features (28 features)
        self.valid_features = [0.5] * 17 + [0.6] * 11  # 17 tech + 11 personality
        
    def test_init(self):
        """Test CareerPredictor initialization"""
        predictor = CareerPredictor()
        assert predictor.model is None
        assert predictor.feature_names is None
        assert predictor.target_classes is None
        
        custom_path = 'custom/path/model.pkl'
        predictor_custom = CareerPredictor(custom_path)
        assert predictor_custom.model_path == custom_path
    
    @patch('predict.joblib.load')
    @patch('predict.os.path.exists')
    def test_load_model_success(self, mock_exists, mock_joblib_load):
        """Test successful model loading"""
        mock_exists.return_value = True
        mock_joblib_load.return_value = {
            'model': self.mock_model,
            'feature_names': ['feature_1', 'feature_2'],
            'target_classes': ['Career_A', 'Career_B'],
            'model_type': 'RandomForestClassifier'
        }
        
        result = self.predictor.load_model()
        
        assert result is True
        assert self.predictor.model == self.mock_model
        assert len(self.predictor.feature_names) == 2
        assert len(self.predictor.target_classes) == 2
    
    @patch('predict.os.path.exists')
    def test_load_model_file_not_found(self, mock_exists):
        """Test model loading when file doesn't exist"""
        mock_exists.return_value = False
        
        result = self.predictor.load_model()
        
        assert result is False
        assert self.predictor.model is None
    
    def test_validate_features_valid(self):
        """Test feature validation with valid input"""
        result = self.predictor.validate_features(self.valid_features)
        
        assert result.shape == (1, 28)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_validate_features_invalid_count(self):
        """Test feature validation with wrong number of features"""
        with pytest.raises(ValueError, match="Expected 28 features"):
            self.predictor.validate_features([0.5] * 20)  # Wrong count
    
    def test_validate_features_invalid_range_technical(self):
        """Test feature validation with technical skills out of range"""
        invalid_features = [1.5] + [0.5] * 16 + [0.6] * 11  # First technical skill > 1
        
        with pytest.raises(ValueError, match="Technical skills must be in range"):
            self.predictor.validate_features(invalid_features)
    
    def test_validate_features_invalid_range_personality(self):
        """Test feature validation with personality traits out of range"""
        invalid_features = [0.5] * 17 + [1.5] + [0.6] * 10  # First personality trait > 1
        
        with pytest.raises(ValueError, match="Personality traits must be in range"):
            self.predictor.validate_features(invalid_features)
    
    def test_validate_features_non_list_input(self):
        """Test feature validation with non-list input"""
        with pytest.raises(ValueError, match="Features must be a list or numpy array"):
            self.predictor.validate_features("invalid_input")
    
    def test_predict_success(self):
        """Test successful prediction"""
        result = self.predictor.predict(self.valid_features)
        
        assert 'primary_prediction' in result
        assert 'confidence' in result
        assert 'all_predictions' in result
        assert 'top_3_predictions' in result
        assert 'timestamp' in result
        
        assert result['primary_prediction'] == 'Data Scientist'
        assert result['confidence'] == 0.8
        assert len(result['top_3_predictions']) <= 3
    
    def test_predict_without_proba(self):
        """Test prediction with model that doesn't support predict_proba"""
        # Remove predict_proba method
        delattr(self.mock_model, 'predict_proba')
        
        result = self.predictor.predict(self.valid_features)
        
        assert result['confidence'] == 1.0
        assert len(result['all_predictions']) == 1
    
    def test_predict_model_not_loaded(self):
        """Test prediction when model is not loaded"""
        predictor = CareerPredictor()
        
        with patch.object(predictor, 'load_model', return_value=False):
            with pytest.raises(RuntimeError, match="Model not loaded"):
                predictor.predict(self.valid_features)
    
    def test_predict_batch_success(self):
        """Test batch prediction"""
        features_list = [self.valid_features, self.valid_features]
        
        results = self.predictor.predict_batch(features_list)
        
        assert len(results) == 2
        for i, result in enumerate(results):
            assert result['batch_index'] == i
            assert 'primary_prediction' in result
    
    def test_predict_batch_with_errors(self):
        """Test batch prediction with some invalid inputs"""
        features_list = [
            self.valid_features,  # Valid
            [0.5] * 20,          # Invalid - wrong count
            self.valid_features   # Valid
        ]
        
        results = self.predictor.predict_batch(features_list)
        
        assert len(results) == 3
        assert 'primary_prediction' in results[0]
        assert 'error' in results[1]
        assert 'primary_prediction' in results[2]
    
    def test_get_feature_importance_with_support(self):
        """Test feature importance when model supports it"""
        self.mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3] + [0.05] * 25)
        
        result = self.predictor.get_feature_importance()
        
        assert 'feature_importance' in result
        assert 'sorted_importance' in result
        assert 'top_10_features' in result
        assert len(result['top_10_features']) == 10
    
    def test_get_feature_importance_without_support(self):
        """Test feature importance when model doesn't support it"""
        # Remove feature_importances_ attribute
        if hasattr(self.mock_model, 'feature_importances_'):
            delattr(self.mock_model, 'feature_importances_')
        
        result = self.predictor.get_feature_importance()
        
        assert result is None
    
    def test_explain_prediction(self):
        """Test prediction explanation"""
        self.mock_model.feature_importances_ = np.array([0.1] * 28)
        
        result = self.predictor.explain_prediction(self.valid_features)
        
        assert 'prediction' in result
        assert 'input_features' in result
        assert 'feature_contributions' in result
        assert 'top_contributing_features' in result
        assert len(result['top_contributing_features']) == 5
    
    def test_get_model_info(self):
        """Test getting model information"""
        result = self.predictor.get_model_info()
        
        assert 'model_type' in result
        assert 'feature_count' in result
        assert 'class_count' in result
        assert 'feature_names' in result
        assert 'target_classes' in result
    
    def test_get_model_info_no_model(self):
        """Test getting model info when no model is loaded"""
        predictor = CareerPredictor()
        
        result = predictor.get_model_info()
        
        assert result['status'] == "No model loaded"


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_make_prediction(self):
        """Test convenience function for single prediction"""
        valid_features = [0.5] * 28
        
        with patch('predict.CareerPredictor') as mock_class:
            mock_predictor = Mock()
            mock_predictor.predict.return_value = {'prediction': 'test'}
            mock_class.return_value = mock_predictor
            
            result = make_prediction(valid_features)
            
            mock_predictor.predict.assert_called_once_with(valid_features)
            assert result == {'prediction': 'test'}
    
    def test_make_batch_predictions(self):
        """Test convenience function for batch predictions"""
        features_list = [[0.5] * 28, [0.6] * 28]
        
        with patch('predict.CareerPredictor') as mock_class:
            mock_predictor = Mock()
            mock_predictor.predict_batch.return_value = [{'batch': 'result'}]
            mock_class.return_value = mock_predictor
            
            result = make_batch_predictions(features_list)
            
            mock_predictor.predict_batch.assert_called_once_with(features_list)
            assert result == [{'batch': 'result'}]


@pytest.fixture
def sample_features():
    """Fixture providing sample feature data"""
    return {
        'valid_28_features': [0.5] * 17 + [0.6] * 11,
        'invalid_count': [0.5] * 20,
        'invalid_range_tech': [1.5] + [0.5] * 16 + [0.6] * 11,
        'invalid_range_personality': [0.5] * 17 + [1.5] + [0.6] * 10
    }


def test_edge_cases(sample_features):
    """Test edge cases and boundary conditions"""
    predictor = CareerPredictor()
    
    # Test with minimum values
    min_features = [0.0] * 17 + [0.0] * 11
    result = predictor.validate_features(min_features)
    assert result.shape == (1, 28)
    
    # Test with maximum values
    max_features = [1.0] * 17 + [1.0] * 11
    result = predictor.validate_features(max_features)
    assert result.shape == (1, 28)


def test_integration_example():
    """Integration test example"""
    # This would be a more comprehensive test in a real scenario
    predictor = CareerPredictor()
    
    # Mock successful model loading
    with patch.object(predictor, 'load_model', return_value=True):
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        
        predictor.model = mock_model
        predictor.target_classes = ['Software Developer', 'Data Scientist']
        predictor.feature_names = [f'feature_{i}' for i in range(28)]
        
        # Test prediction pipeline
        features = [0.7] * 17 + [0.5] * 11  # High technical, moderate personality
        result = predictor.predict(features)
        
        assert result['primary_prediction'] == 'Software Developer'
        assert result['confidence'] == 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])