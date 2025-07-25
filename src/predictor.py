import numpy as np
import joblib
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Optional: configure logging

class CareerPredictor:
    """Handles career predictions using a trained model"""

    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = None
        self.target_classes = None
        self.model_path = model_path or 'models/career_random_forest_latest.pkl'

    def load_model(self, model_path=None):
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
        try:
            if not isinstance(features, (list, np.ndarray)):
                raise ValueError("Features must be a list or numpy array")

            features = np.array(features, dtype=float)

            # Auto normalize if values are greater than 1 (assuming scale of 1–5)
            if np.any(features > 1):
                features = features / 5.0

            if np.any(features < 0) or np.any(features > 1):
                raise ValueError("All features must be in range [0, 1]")

            if self.model is None:
                self.load_model()

            expected_features = self.model.n_features_in_
            if features.shape[0] > expected_features:
                features = features[:expected_features]
            elif features.shape[0] < expected_features:
                raise ValueError(f"Expected {expected_features} features, got {features.shape[0]}")

            return features.reshape(1, -1)

        except Exception as e:
            logger.error(f"Feature validation error: {str(e)}")
            raise

    def predict(self, features):
        try:
            if self.model is None:
                self.load_model()

            features_array = self.validate_features(features)
            prediction = self.model.predict(features_array)
            # Ensure prediction is a scalar or 1-element array
            pred_idx = int(prediction[0])
            if self.target_classes is not None and len(self.target_classes) > pred_idx:
                predicted_class = self.target_classes[pred_idx]
            else:
                predicted_class = pred_idx

            return {
                'prediction': predicted_class,
                'raw_prediction': pred_idx,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def predict_batch(self, features_list):
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
                    contribution = imp * explanation['input_features'].get(fname, 0)
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
