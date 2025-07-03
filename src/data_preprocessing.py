# src/data_preprocessing.py - Data Preprocessing Module

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing for career prediction model"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.target_column = 'role'
        
    def load_data(self, file_path):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(file_path)
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
            # Technical skills features (normalize from 1-7 scale to 0-1)
            technical_skills = [
                'database_fundamentals', 'computer_architecture', 'distributed_computing_systems',
                'cyber_security', 'networking', 'software_development', 'programming_skills',
                'project_management', 'computer_forensics_fundamentals', 'technical_communication',
                'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
                'data_science', 'troubleshooting_skills', 'graphics_designing'
            ]
            
            # Personality traits (already normalized 0-1)
            personality_traits = [
                'openness', 'conscientousness', 'extraversion', 'agreeableness',
                'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
                'self-enhancement', 'self-transcendence'
            ]
            
            # Normalize technical skills
            for skill in technical_skills:
                if skill in df.columns:
                    df[skill] = df[skill] / 7.0
                    
            # Ensure personality traits are in correct range
            for trait in personality_traits:
                if trait in df.columns:
                    df[trait] = np.clip(df[trait], 0, 1)
            
            # Create feature list
            self.feature_columns = technical_skills + personality_traits
            
            # Select only available features
            available_features = [col for col in self.feature_columns if col in df.columns]
            
            if len(available_features) != len(self.feature_columns):
                logger.warning(f"Some features missing. Available: {len(available_features)}, Expected: {len(self.feature_columns)}")
            
            X = df[available_features]
            
            # Handle missing values
            X = X.fillna(X.mean(numeric_only=True))
            
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
            
            y = df[self.target_column]
            
            # Handle missing values in target
            y = y.fillna('Unknown')
            
            # Encode target labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            logger.info(f"Target preprocessed: {len(np.unique(y_encoded))} unique classes")
            return y_encoded
            
        except Exception as e:
            logger.error(f"Error preprocessing target: {str(e)}")
            raise
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def balance_data(self, X_train, y_train, random_state=42):
        """Balance training data using oversampling"""
        try:
            # Check class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            logger.info(f"Class distribution before balancing: {dict(zip(unique, counts))}")
            
            # Apply random oversampling
            oversampler = RandomOverSampler(random_state=random_state)
            X_balanced, y_balanced = oversampler.fit_resample(X_train, y_train)
            
            # Check new distribution
            unique, counts = np.unique(y_balanced, return_counts=True)
            logger.info(f"Class distribution after balancing: {dict(zip(unique, counts))}")
            
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
        """Get list of feature names"""
        return self.feature_columns
    
    def get_target_classes(self):
        """Get list of target classes"""
        return self.label_encoder.classes_
    
    def inverse_transform_target(self, y_encoded):
        """Convert encoded target back to original labels"""
        return self.label_encoder.inverse_transform(y_encoded)
    
    def preprocess_pipeline(self, file_path, balance=True, scale=True):
        """Complete preprocessing pipeline"""
        try:
            # Load data
            df = self.load_data(file_path)
            
            # Preprocess features and target
            X = self.preprocess_features(df)
            y = self.preprocess_target(df)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Balance data if requested
            if balance:
                X_train, y_train = self.balance_data(X_train, y_train)
            
            # Scale features if requested
            if scale:
                X_train, X_test = self.scale_features(X_train, X_test)
            
            logger.info("Preprocessing pipeline completed successfully")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Example preprocessing
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            'data/career_map.csv'
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Number of classes: {len(preprocessor.get_target_classes())}")
        print(f"Target classes: {preprocessor.get_target_classes()}")
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
