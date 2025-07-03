import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import logging
import os

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing for career prediction model"""

    def __init__(self, random_state=42):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.target_column = 'role'
        self.random_state = random_state

    def load_data(self, file_path):
        """Load dataset from CSV file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            if df.empty:
                raise ValueError("The input data file is empty.")
            # Normalize column names
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            logger.info(f"Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_features(self, df):
        """Preprocess features for model training"""
        try:
            technical_skills = [
                'database_fundamentals', 'computer_architecture', 'distributed_computing_systems',
                'cyber_security', 'networking', 'software_development', 'programming_skills',
                'project_management', 'computer_forensics_fundamentals', 'technical_communication',
                'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
                'data_science', 'troubleshooting_skills', 'graphics_designing'
            ]

            personality_traits = [
                'openness', 'conscientousness', 'extraversion', 'agreeableness',
                'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
                'self-enhancement', 'self-transcendence'
            ]

            for skill in technical_skills:
                if skill in df.columns:
                    df[skill] = df[skill] / 7.0

            for trait in personality_traits:
                if trait in df.columns:
                    df[trait] = np.clip(df[trait], 0, 1)

            self.feature_columns = technical_skills + personality_traits
            available_features = [col for col in self.feature_columns if col in df.columns]

            missing_features = set(self.feature_columns) - set(available_features)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")

            X = df[available_features].select_dtypes(include=[np.number])
            X = X.fillna(X.mean())

            logger.info(f"Features preprocessed: {X.shape}")
            return X

        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise

    def preprocess_target(self, df):
        """Preprocess target variable"""
        try:
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")

            y = df[self.target_column].fillna('Unknown')
            y_encoded = self.label_encoder.fit_transform(y)
            logger.info(f"Target preprocessed: {len(np.unique(y_encoded))} unique classes")
            return y_encoded

        except Exception as e:
            logger.error(f"Error preprocessing target: {str(e)}")
            raise

    def split_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def balance_data(self, X_train, y_train):
        """Balance training data using oversampling"""
        try:
            logger.info(f"Class distribution before balancing: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            oversampler = RandomOverSampler(random_state=self.random_state)
            X_balanced, y_balanced = oversampler.fit_resample(X_train, y_train)
            logger.info(f"Class distribution after balancing: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
            return X_balanced, y_balanced

        except Exception as e:
            logger.error(f"Error balancing data: {str(e)}")
            raise

    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                return X_train_scaled, X_test_scaled
            return X_train_scaled
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def get_feature_names(self):
        return self.feature_columns

    def get_target_classes(self):
        return self.label_encoder.classes_

    def inverse_transform_target(self, y_encoded):
        return self.label_encoder.inverse_transform(y_encoded)

    def encode_target(self, y):
        return self.label_encoder.transform(y)

    def decode_target(self, y_encoded):
        return self.label_encoder.inverse_transform(y_encoded)

    def preprocess_pipeline(self, file_path, balance=True, scale=True):
        try:
            df = self.load_data(file_path)
            X = self.preprocess_features(df)
            y = self.preprocess_target(df)
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            if balance:
                X_train, y_train = self.balance_data(X_train, y_train)
            if scale:
                X_train, X_test = self.scale_features(X_train, X_test)
            logger.info("Preprocessing pipeline completed successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    preprocessor = DataPreprocessor()
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline('data/career_map.csv')
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Number of classes: {len(preprocessor.get_target_classes())}")
        print(f"Target classes: {preprocessor.get_target_classes()}")
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
