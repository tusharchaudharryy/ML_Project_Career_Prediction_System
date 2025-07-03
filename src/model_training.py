# src/model_training.py - Model Training Module

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CareerModelTrainer:
    """Handles training of career prediction models"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0

    def initialize_models(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'svm': SVC(kernel='rbf', random_state=42, probability=True),
            'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive_bayes': GaussianNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        logger.info(f"Initialized {len(self.models)} models")

    def train_model(self, model_name, X_train, y_train, X_test=None, y_test=None):
        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")

            model = self.models[model_name]
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            logger.info(f"{model_name} training accuracy: {train_score:.4f}")

            if X_test is not None and y_test is not None:
                test_score = model.score(X_test, y_test)
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)

                return {
                    'model': model,
                    'train_score': train_score,
                    'test_score': test_score,
                    'predictions': y_pred,
                    'classification_report': report
                }

            return {'model': model, 'train_score': train_score}

        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise

    def train_all_models(self, X_train, y_train, X_test=None, y_test=None):
        try:
            self.initialize_models()
            results = {}

            for model_name in self.models:
                result = self.train_model(model_name, X_train, y_train, X_test, y_test)
                results[model_name] = result

                if X_test is not None and y_test is not None:
                    if result['test_score'] > self.best_score:
                        self.best_score = result['test_score']
                        self.best_model = result['model']
                        self.best_model_name = model_name

            logger.info(f"Best model: {self.best_model_name} with accuracy: {self.best_score:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error training all models: {str(e)}")
            raise

    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid=None):
        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")

            model = self.models[model_name]

            if param_grid is None:
                param_grids = {
                    'random_forest': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'svm': {
                        'C': [0.1, 1, 10],
                        'gamma': ['scale', 'auto', 0.1, 1],
                        'kernel': ['rbf', 'linear']
                    },
                    'decision_tree': {
                        'max_depth': [5, 10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'knn': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance']
                    },
                    'logistic_regression': {
                        'C': [0.1, 1, 10],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'lbfgs']
                    }
                }
                param_grid = param_grids.get(model_name, {})

            if not param_grid:
                logger.warning(f"No parameter grid for {model_name}")
                return model

            logger.info(f"Performing hyperparameter tuning for {model_name}...")
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            return grid_search.best_estimator_

        except Exception as e:
            logger.error(f"Error in hyperparameter tuning for {model_name}: {str(e)}")
            raise

    def cross_validate_model(self, model, X, y, cv=5):
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            logger.info(f"CV scores: {scores}")
            logger.info(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            return scores

        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise

    def train_optimized_random_forest(self, X_train, y_train, X_test=None, y_test=None):
        try:
            self.initialize_models()
            logger.info("Training optimized Random Forest model...")
            rf_optimized = self.hyperparameter_tuning('random_forest', X_train, y_train)
            cv_scores = self.cross_validate_model(rf_optimized, X_train, y_train)
            rf_optimized.fit(X_train, y_train)
            train_score = rf_optimized.score(X_train, y_train)

            result = {
                'model': rf_optimized,
                'train_score': train_score,
                'cv_scores': cv_scores,
                'feature_importance': rf_optimized.feature_importances_
            }

            if X_test is not None and y_test is not None:
                test_score = rf_optimized.score(X_test, y_test)
                y_pred = rf_optimized.predict(X_test)
                result.update({
                    'test_score': test_score,
                    'predictions': y_pred,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                })

            logger.info(f"Optimized RF model trained. Train acc: {train_score:.4f}")
            return result

        except Exception as e:
            logger.error(f"Error training optimized Random Forest: {str(e)}")
            raise

    def save_model(self, model, model_name, feature_names=None, target_classes=None):
        try:
            os.makedirs("models", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"models/{model_name}_{timestamp}.pkl"

            model_data = {
                'model': model,
                'feature_names': feature_names,
                'target_classes': target_classes,
                'timestamp': timestamp,
                'model_type': type(model).__name__
            }

            joblib.dump(model_data, filename)
            logger.info(f"Model saved as {filename}")

            latest_filename = f"models/{model_name}_latest.pkl"
            joblib.dump(model_data, latest_filename)
            logger.info(f"Model also saved as {latest_filename}")

            return filename

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filename):
        try:
            model_data = joblib.load(filename)
            logger.info(f"Model loaded from {filename}")
            return model_data

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def evaluate_model(self, model, X_test, y_test, target_classes=None):
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")

            return {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': report,
                'confusion_matrix': cm
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    import pprint

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    trainer = CareerModelTrainer()
    preprocessor = DataPreprocessor()

    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline('data/career_map.csv')

        rf_result = trainer.train_optimized_random_forest(X_train, y_train, X_test, y_test)

        trainer.save_model(
            rf_result['model'],
            'career_random_forest',
            preprocessor.get_feature_names(),
            preprocessor.get_target_classes()
        )

        print("Training completed successfully!")
        print(f" Test accuracy: {rf_result['test_score']:.4f}")
        pprint.pprint(rf_result['classification_report'])

    except Exception as e:
        print(f" Error in training: {str(e)}")
