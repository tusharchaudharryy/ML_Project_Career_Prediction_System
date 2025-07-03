import numpy as np
import joblib
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class CareerPredictor:
    """Handles career predictions using trained model"""

    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = None
        self.target_classes = None
        self.model_path = model_path or 'models/career_random_forest_latest.pkl'

    def load_model(self, model_path=None):
        """Load trained model from disk"""
        try:
            path = model_path or self.model_path

            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            model_data = joblib.load(path)
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', [])
            self.target_classes = model_data.get('target_classes', [])

            logger.info(f"Model loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def validate_features(self, features):
        """Validate input features"""
        try:
            if not isinstance(features, (list, np.ndarray)):
                raise ValueError("Features must be a list or numpy array")

            features = np.array(features)
            if len(features) != 28:
                raise ValueError(f"Expected 28 features, got {len(features)}")

            if np.any(features < 0) or np.any(features > 1):
                raise ValueError("All features must be in range [0, 1]")

            return features.reshape(1, -1)

        except Exception as e:
            logger.error(f"Feature validation error: {str(e)}")
            raise

    def predict(self, features):
        """Make career prediction"""
        try:
            if self.model is None:
                if not self.load_model():
                    raise RuntimeError("Model not loaded")

            features_array = self.validate_features(features)
            prediction = self.model.predict(features_array)[0]

            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_array)[0]
                prob_dict = {cls: float(probabilities[i]) for i, cls in enumerate(self.target_classes)}
                sorted_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            else:
                prob_dict = {self.target_classes[prediction]: 1.0}
                sorted_predictions = [(self.target_classes[prediction], 1.0)]

            return {
                'primary_prediction': self.target_classes[prediction],
                'confidence': float(probabilities[prediction]) if hasattr(self.model, 'predict_proba') else 1.0,
                'all_predictions': sorted_predictions,
                'top_3_predictions': sorted_predictions[:3],
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def predict_batch(self, features_list):
        """Predict multiple samples"""
        results = []
        for i, features in enumerate(features_list):
            try:
                result = self.predict(features)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        return results

    def get_feature_importance(self):
        """Return sorted feature importance"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            if not hasattr(self.model, 'feature_importances_'):
                return None

            importance = self.model.feature_importances_
            sorted_importance = sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )
            return {
                'feature_importance': dict(sorted_importance),
                'top_10_features': sorted_importance[:10]
            }

        except Exception as e:
            logger.error(f"Feature importance error: {str(e)}")
            raise

    def explain_prediction(self, features):
        """Explain prediction with feature contributions"""
        try:
            prediction = self.predict(features)
            importance = self.get_feature_importance()
            features_array = self.validate_features(features)

            explanation = {
                'prediction': prediction,
                'input_features': dict(zip(self.feature_names, features_array.flatten())),
                'feature_contributions': {},
                'top_contributing_features': []
            }

            if importance:
                for fname, imp in importance['feature_importance'].items():
                    contribution = imp * explanation['input_features'][fname]
                    explanation['feature_contributions'][fname] = contribution

                sorted_contributions = sorted(
                    explanation['feature_contributions'].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                explanation['top_contributing_features'] = sorted_contributions[:5]

            return explanation

        except Exception as e:
            logger.error(f"Explanation error: {str(e)}")
            raise

    def get_model_info(self):
        """Return summary of loaded model"""
        if self.model is None:
            return {"status": "Model not loaded"}

        return {
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names),
            "class_count": len(self.target_classes),
            "model_path": self.model_path,
            "feature_names": self.feature_names,
            "target_classes": list(self.target_classes)
        }

# ---------- Utility Functions ----------

def make_prediction(features, model_path=None):
    predictor = CareerPredictor(model_path)
    return predictor.predict(features)

def make_batch_predictions(features_list, model_path=None):
    predictor = CareerPredictor(model_path)
    return predictor.predict_batch(features_list)

# ---------- Example Usage ----------

if __name__ == "__main__":
    predictor = CareerPredictor()
    if predictor.load_model():
        # Sample input: 28 normalized features
        sample = [
            0.8, 0.6, 0.4, 0.9, 0.7, 0.8, 0.9, 0.5, 0.3, 0.7,
            0.9, 0.8, 0.6, 0.7, 0.9, 0.8, 0.4,
            0.7, 0.8, 0.6, 0.7, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.7
        ]

        try:
            result = predictor.predict(sample)
            print("\n Prediction:", result['primary_prediction'])
            print(" Top 3 predictions:", result['top_3_predictions'])

            explanation = predictor.explain_prediction(sample)
            print("\n Top contributing features:")
            for feature, contrib in explanation['top_contributing_features']:
                print(f" - {feature}: {contrib:.4f}")

        except Exception as e:
            print(" Error:", e)
    else:
        print(" Failed to load model.")
