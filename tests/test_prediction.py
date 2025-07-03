import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predictor import CareerPredictor
import warnings

warnings.filterwarnings("ignore")

class TestCareerPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = CareerPredictor()
        self.sample_input = {
            "database_fundamentals": 1.0,
            "computer_architecture": 0.8,
            "distributed_computing": 0.8,
            "cyber_security": 0.6,
            "networking": 0.8,
            "development": 0.8,
            "programming_skills": 1.0,
            "project_management": 0.6,
            "computer_forensics": 0.4,
            "technical_communication": 0.8,
            "ai_ml": 1.0,
            "software_engineering": 0.8,
            "business_analysis": 0.6,
            "communication_skills": 1.0,
            "data_science": 0.8,
            "troubleshooting": 0.6,
            "graphics_designing": 0.4,
            "openness": 0.85,
            "conscientiousness": 0.7,
            "extraversion": 0.65,
            "agreeableness": 0.8,
            "emotional_range": 0.45,
            "conversation": 0.8,
            "openness_to_change": 0.7,
            "hedonism": 0.6,
            "self_enhancement": 0.5,
            "self_transcendence": 0.6,
            "conservation": 0.5


        }

    def test_prediction_output(self):
        try:
            input_values = list(self.sample_input.values())  # ✅ convert dict to list
            result = self.predictor.predict(input_values)
            self.assertIsInstance(result, str)
            print("✅ Predicted Career Role:", result)
        except Exception as e:
            self.fail(f"Prediction failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
