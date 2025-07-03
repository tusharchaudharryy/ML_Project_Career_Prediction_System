# tests/test_utils.py - Tests for utility functions

import pytest
import numpy as np
import os
import sys
from unittest.mock import patch
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    validate_input, format_predictions, get_confidence_level,
    format_feature_name, get_skill_description, get_career_description,
    calculate_skill_statistics, get_skill_profile_description,
    sanitize_input, generate_recommendations
)

class TestValidateInput:
    """Test input validation functions"""
    
    def test_validate_input_complete_valid(self):
        """Test validation with complete valid input"""
        form_data = {}
        
        # Technical skills (1-7)
        technical_skills = [
            'database_fundamentals', 'computer_architecture', 'distributed_computing',
            'cyber_security', 'networking', 'development', 'programming_skills',
            'project_management', 'computer_forensics', 'technical_communication',
            'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
            'data_science', 'troubleshooting', 'graphics_designing'
        ]
        
        for skill in technical_skills:
            form_data[skill] = '5'
        
        # Personality traits (0-1)
        personality_traits = [
            'openness', 'conscientiousness', 'extraversion', 'agreeableness',
            'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
            'self_enhancement', 'self_transcendence', 'conservation'
        ]
        
        for trait in personality_traits:
            form_data[trait] = '0.7'
        
        is_valid, message = validate_input(form_data)
        
        assert is_valid is True
        assert message == "Validation successful"
    
    def test_validate_input_missing_fields(self):
        """Test validation with missing fields"""
        form_data = {
            'database_fundamentals': '5',
            'openness': '0.7'
            # Missing most fields
        }
        
        is_valid, message = validate_input(form_data)
        
        assert is_valid is False
        assert "Missing required fields" in message
    
    def test_validate_input_invalid_technical_range(self):
        """Test validation with technical skills out of range"""
        form_data = {}
        
        # Valid data except one invalid technical skill
        technical_skills = [
            'database_fundamentals', 'computer_architecture', 'distributed_computing',
            'cyber_security', 'networking', 'development', 'programming_skills',
            'project_management', 'computer_forensics', 'technical_communication',
            'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
            'data_science', 'troubleshooting', 'graphics_designing'
        ]
        
        for skill in technical_skills:
            form_data[skill] = '5' if skill != 'ai_ml' else '8'  # Invalid value
        
        personality_traits = [
            'openness', 'conscientiousness', 'extraversion', 'agreeableness',
            'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
            'self_enhancement', 'self_transcendence', 'conservation'
        ]
        
        for trait in personality_traits:
            form_data[trait] = '0.7'
        
        is_valid, message = validate_input(form_data)
        
        assert is_valid is False
        assert "must be between 1 and 7" in message
    
    def test_validate_input_invalid_personality_range(self):
        """Test validation with personality traits out of range"""
        form_data = {}
        
        technical_skills = [
            'database_fundamentals', 'computer_architecture', 'distributed_computing',
            'cyber_security', 'networking', 'development', 'programming_skills',
            'project_management', 'computer_forensics', 'technical_communication',
            'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
            'data_science', 'troubleshooting', 'graphics_designing'
        ]
        
        for skill in technical_skills:
            form_data[skill] = '5'
        
        personality_traits = [
            'openness', 'conscientiousness', 'extraversion', 'agreeableness',
            'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
            'self_enhancement', 'self_transcendence', 'conservation'
        ]
        
        for trait in personality_traits:
            form_data[trait] = '0.7' if trait != 'openness' else '1.5'  # Invalid value
        
        is_valid, message = validate_input(form_data)
        
        assert is_valid is False
        assert "must be between 0 and 1" in message
    
    def test_validate_input_non_numeric_values(self):
        """Test validation with non-numeric values"""
        form_data = {
            'database_fundamentals': 'not_a_number',
            'openness': '0.7'
        }
        
        is_valid, message = validate_input(form_data)
        
        assert is_valid is False
        assert "must be a valid number" in message


class TestFormatPredictions:
    """Test prediction formatting functions"""
    
    def test_format_predictions_complete(self):
        """Test formatting with complete prediction result"""
        prediction_result = {
            'primary_prediction': 'Data Scientist',
            'confidence': 0.85,
            'top_3_predictions': [
                ('Data Scientist', 0.85),
                ('Machine Learning Engineer', 0.72),
                ('Software Developer', 0.65)
            ],
            'all_predictions': [
                ('Data Scientist', 0.85),
                ('Machine Learning Engineer', 0.72),
                ('Software Developer', 0.65),
                ('DevOps Engineer', 0.45),
                ('Web Developer', 0.32)
            ]
        }
        
        result = format_predictions(prediction_result)
        
        assert result['primary_career'] == 'Data Scientist'
        assert result['confidence_percentage'] == 85.0
        assert result['confidence_level'] == 'Very High'
        assert len(result['alternative_careers']) == 3
        assert len(result['top_matches']) == 5
        
        # Check alternative careers format
        alt_career = result['alternative_careers'][0]
        assert 'career' in alt_career
        assert 'probability' in alt_career
        assert alt_career['career'] == 'Data Scientist'
        assert alt_career['probability'] == 85.0
    
    def test_format_predictions_error_handling(self):
        """Test formatting with invalid prediction result"""
        invalid_result = {}
        
        result = format_predictions(invalid_result)
        
        assert result['primary_career'] == 'Unknown'
        assert result['confidence_percentage'] == 0
        assert result['confidence_level'] == 'Low'
        assert result['alternative_careers'] == []
        assert result['top_matches'] == []


class TestConfidenceLevel:
    """Test confidence level determination"""
    
    def test_get_confidence_level_very_high(self):
        """Test very high confidence level"""
        assert get_confidence_level(0.85) == 'Very High'
        assert get_confidence_level(1.0) == 'Very High'
    
    def test_get_confidence_level_high(self):
        """Test high confidence level"""
        assert get_confidence_level(0.75) == 'High'
        assert get_confidence_level(0.6) == 'High'
    
    def test_get_confidence_level_medium(self):
        """Test medium confidence level"""
        assert get_confidence_level(0.5) == 'Medium'
        assert get_confidence_level(0.4) == 'Medium'
    
    def test_get_confidence_level_low(self):
        """Test low confidence level"""
        assert get_confidence_level(0.3) == 'Low'
        assert get_confidence_level(0.2) == 'Low'
    
    def test_get_confidence_level_very_low(self):
        """Test very low confidence level"""
        assert get_confidence_level(0.1) == 'Very Low'
        assert get_confidence_level(0.0) == 'Very Low'


class TestFormatFeatureName:
    """Test feature name formatting"""
    
    def test_format_feature_name_snake_case(self):
        """Test snake_case to Title Case conversion"""
        assert format_feature_name('database_fundamentals') == 'Database Skills'
        assert format_feature_name('programming_skills') == 'Programming'
        assert format_feature_name('project_management') == 'Project Management'
    
    def test_format_feature_name_special_cases(self):
        """Test special case replacements"""
        assert format_feature_name('ai_ml') == 'AI/ML'
        assert format_feature_name('openness_to_change') == 'Openness to Change'
        assert format_feature_name('self_enhancement') == 'Self-Enhancement'
        assert format_feature_name('self_transcendence') == 'Self-Transcendence'


class TestDescriptionFunctions:
    """Test description retrieval functions"""
    
    def test_get_skill_description_known_skill(self):
        """Test getting description for known skill"""
        desc = get_skill_description('ai_ml')
        assert 'machine learning' in desc.lower()
        assert 'artificial intelligence' in desc.lower()
    
    def test_get_skill_description_unknown_skill(self):
        """Test getting description for unknown skill"""
        desc = get_skill_description('unknown_skill')
        assert desc == 'Professional skill or personality trait'
    
    def test_get_career_description_known_career(self):
        """Test getting description for known career"""
        desc = get_career_description('Data Scientist')
        assert 'data' in desc.lower()
        assert 'analysis' in desc.lower() or 'analyze' in desc.lower()
    
    def test_get_career_description_unknown_career(self):
        """Test getting description for unknown career"""
        desc = get_career_description('Unknown Career')
        assert desc == 'Technology professional role in the computer science field.'


class TestCalculateSkillStatistics:
    """Test skill statistics calculation"""
    
    def test_calculate_skill_statistics_complete(self):
        """Test statistics calculation with complete data"""
        form_data = {}
        
        # Technical skills
        technical_skills = [
            'database_fundamentals', 'computer_architecture', 'distributed_computing',
            'cyber_security', 'networking', 'development', 'programming_skills',
            'project_management', 'computer_forensics', 'technical_communication',
            'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
            'data_science', 'troubleshooting', 'graphics_designing'
        ]
        
        for i, skill in enumerate(technical_skills):
            form_data[skill] = str(i % 7 + 1)  # Values from 1-7
        
        # Personality traits
        personality_traits = [
            'openness', 'conscientiousness', 'extraversion', 'agreeableness',
            'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
            'self_enhancement', 'self_transcendence', 'conservation'
        ]
        
        for i, trait in enumerate(personality_traits):
            form_data[trait] = str((i % 10 + 1) / 10)  # Values from 0.1-1.0
        
        stats = calculate_skill_statistics(form_data)
        
        assert 'technical_average' in stats
        assert 'technical_max' in stats
        assert 'technical_min' in stats
        assert 'personality_average' in stats
        assert 'top_technical_skills' in stats
        assert 'top_personality_traits' in stats
        assert 'skill_profile' in stats
        
        assert isinstance(stats['technical_average'], float)
        assert isinstance(stats['personality_average'], float)
        assert len(stats['top_technical_skills']) == 5
        assert len(stats['top_personality_traits']) == 5
    
    def test_calculate_skill_statistics_empty_data(self):
        """Test statistics calculation with empty data"""
        form_data = {}
        
        stats = calculate_skill_statistics(form_data)
        
        assert stats['technical_average'] == 0
        assert stats['personality_average'] == 0
        assert stats['top_technical_skills'] == []
        assert stats['top_personality_traits'] == []


class TestSkillProfileDescription:
    """Test skill profile description generation"""
    
    def test_get_skill_profile_description_expert(self):
        """Test profile description for expert-level skills"""
        tech_values = [6.5, 6.8, 6.2, 6.9]  # Expert level
        personality_values = [0.9, 0.8, 0.85]  # Very strong
        
        description = get_skill_profile_description(tech_values, personality_values)
        
        assert 'expert-level' in description
        assert 'very strong' in description
    
    def test_get_skill_profile_description_beginner(self):
        """Test profile description for beginner-level skills"""
        tech_values = [2.5, 3.0, 2.8]  # Beginner level
        personality_values = [0.3, 0.4, 0.35]  # Developing
        
        description = get_skill_profile_description(tech_values, personality_values)
        
        assert 'beginner' in description
        assert 'developing' in description


class TestSanitizeInput:
    """Test input sanitization"""
    
    def test_sanitize_input_clean_string(self):
        """Test sanitizing clean input"""
        clean_input = "This is a clean input"
        result = sanitize_input(clean_input)
        assert result == clean_input
    
    def test_sanitize_input_malicious_characters(self):
        """Test sanitizing input with potentially harmful characters"""
        malicious_input = "<script>alert('xss')</script>"
        result = sanitize_input(malicious_input)
        assert '<' not in result
        assert '>' not in result
        assert 'script' in result  # Text content preserved
    
    def test_sanitize_input_non_string(self):
        """Test sanitizing non-string input"""
        result = sanitize_input(123)
        assert result == "123"


class TestGenerateRecommendations:
    """Test recommendation generation"""
    
    def test_generate_recommendations_low_confidence(self):
        """Test recommendations for low confidence prediction"""
        prediction_result = {
            'primary_prediction': 'Data Scientist',
            'confidence': 0.3,
            'all_predictions': [('Data Scientist', 0.3)]
        }
        form_data = {}
        
        recommendations = generate_recommendations(prediction_result, form_data)
        
        assert len(recommendations) > 0
        skill_dev_rec = next((r for r in recommendations if r['type'] == 'skill_development'), None)
        assert skill_dev_rec is not None
        assert skill_dev_rec['priority'] == 'high'
    
    def test_generate_recommendations_specific_career(self):
        """Test career-specific recommendations"""
        prediction_result = {
            'primary_prediction': 'Data Scientist',
            'confidence': 0.8,
            'all_predictions': [('Data Scientist', 0.8)]
        }
        form_data = {}
        
        recommendations = generate_recommendations(prediction_result, form_data)
        
        career_recs = [r for r in recommendations if r['type'] == 'career_specific']
        assert len(career_recs) > 0
        
        # Check that recommendations are relevant to Data Scientist
        data_sci_rec = career_recs[0]
        assert 'Data Scientist' in data_sci_rec['title']


@pytest.fixture
def sample_form_data():
    """Fixture providing sample form data"""
    data = {}
    
    # Technical skills
    skills = [
        'database_fundamentals', 'computer_architecture', 'distributed_computing',
        'cyber_security', 'networking', 'development', 'programming_skills',
        'project_management', 'computer_forensics', 'technical_communication',
        'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
        'data_science', 'troubleshooting', 'graphics_designing'
    ]
    
    for skill in skills:
        data[skill] = '5'
    
    # Personality traits
    traits = [
        'openness', 'conscientiousness', 'extraversion', 'agreeableness',
        'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
        'self_enhancement', 'self_transcendence', 'conservation'
    ]
    
    for trait in traits:
        data[trait] = '0.7'
    
    return data


def test_integration_validation_and_formatting(sample_form_data):
    """Integration test for validation and formatting"""
    # Validate input
    is_valid, message = validate_input(sample_form_data)
    assert is_valid is True
    
    # Calculate statistics
    stats = calculate_skill_statistics(sample_form_data)
    assert stats['technical_average'] == 5.0
    assert stats['personality_average'] == 0.7
    
    # Test formatting
    prediction_result = {
        'primary_prediction': 'Software Developer',
        'confidence': 0.75,
        'top_3_predictions': [('Software Developer', 0.75), ('Web Developer', 0.65)],
        'all_predictions': [('Software Developer', 0.75), ('Web Developer', 0.65)]
    }
    
    formatted = format_predictions(prediction_result)
    assert formatted['confidence_level'] == 'High'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])