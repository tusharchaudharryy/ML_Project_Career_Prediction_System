# src/utils.py - Utility Functions

import numpy as np
import pandas as pd
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------
# ✅ Input Validation
# ---------------------------------------------

def validate_input(form_data):
    """Validate user input from form data."""
    try:
        technical_skills = [
            'database_fundamentals', 'computer_architecture', 'distributed_computing',
            'cyber_security', 'networking', 'development', 'programming_skills',
            'project_management', 'computer_forensics', 'technical_communication',
            'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
            'data_science', 'troubleshooting', 'graphics_designing'
        ]

        personality_traits = [
            'openness', 'conscientiousness', 'extraversion', 'agreeableness',
            'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
            'self_enhancement', 'self_transcendence', 'conservation'
        ]

        all_fields = technical_skills + personality_traits

        # Check for missing fields
        missing_fields = [f for f in all_fields if f not in form_data or form_data[f] == '']
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"

        # Validate technical skills: 1 to 7
        for skill in technical_skills:
            value = float(form_data[skill])
            if value < 1 or value > 7:
                return False, f"Technical skill '{skill}' must be between 1 and 7"

        # Validate personality traits: 0 to 1
        for trait in personality_traits:
            value = float(form_data[trait])
            if value < 0 or value > 1:
                return False, f"Personality trait '{trait}' must be between 0 and 1"

        return True, "Validation successful"

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False, f"Validation error: {str(e)}"

# ---------------------------------------------
# ✅ Prediction Formatting
# ---------------------------------------------

def format_predictions(prediction_result):
    """Format prediction results for display."""
    try:
        formatted = {
            'primary_career': prediction_result['primary_prediction'],
            'confidence_percentage': round(prediction_result['confidence'] * 100, 1),
            'confidence_level': get_confidence_level(prediction_result['confidence']),
            'alternative_careers': [],
            'top_matches': []
        }

        # Format top 3 predictions
        for career, prob in prediction_result['top_3_predictions']:
            formatted['alternative_careers'].append({
                'career': career,
                'probability': round(prob * 100, 1)
            })

        # Format all predictions (top 5)
        for career, prob in prediction_result['all_predictions'][:5]:
            formatted['top_matches'].append({
                'career': career,
                'probability': round(prob * 100, 1),
                'bar_width': round(prob * 100, 1)
            })

        return formatted

    except Exception as e:
        logger.error(f"Error formatting predictions: {str(e)}")
        return {
            'primary_career': 'Unknown',
            'confidence_percentage': 0,
            'confidence_level': 'Low',
            'alternative_careers': [],
            'top_matches': []
        }

def get_confidence_level(confidence):
    """Convert confidence score into descriptive label."""
    if confidence >= 0.8: return 'Very High'
    if confidence >= 0.6: return 'High'
    if confidence >= 0.4: return 'Medium'
    if confidence >= 0.2: return 'Low'
    return 'Very Low'

# ---------------------------------------------
# ✅ Feature Name Formatting and Descriptions
# ---------------------------------------------

def format_feature_name(feature_name):
    """Convert snake_case to readable feature name."""
    formatted = feature_name.replace('_', ' ').title()
    replacements = {
        'Ai Ml': 'AI/ML',
        'Communication Skills': 'Communication',
        'Programming Skills': 'Programming',
        'Graphics Designing': 'Graphics Design',
        'Self Enhancement': 'Self-Enhancement',
        'Self Transcendence': 'Self-Transcendence',
        'Openness To Change': 'Openness to Change',
        'Database Fundamentals': 'Database Skills',
    }
    return replacements.get(formatted, formatted)

def get_skill_description(skill_name):
    """Return human-readable description for skills and traits."""
    descriptions = {
        'database_fundamentals': 'Understanding of database design, SQL, and data management',
        'computer_architecture': 'Knowledge of computer hardware and system design',
        'distributed_computing': 'Experience with distributed systems and cloud computing',
        'cyber_security': 'Skills in information security and protection',
        'networking': 'Understanding of network protocols and infrastructure',
        'development': 'General software development capabilities',
        'programming_skills': 'Proficiency in programming languages and coding',
        'project_management': 'Ability to manage projects and teams',
        'computer_forensics': 'Digital investigation and security analysis skills',
        'technical_communication': 'Ability to explain technical concepts clearly',
        'ai_ml': 'Machine learning and artificial intelligence expertise',
        'software_engineering': 'Software design and engineering principles',
        'business_analysis': 'Understanding business requirements and processes',
        'communication_skills': 'General communication and interpersonal skills',
        'data_science': 'Data analysis, statistics, and data visualization',
        'troubleshooting': 'Problem-solving and debugging capabilities',
        'graphics_designing': 'Visual design and multimedia skills',
        'openness': 'Openness to experience and new ideas',
        'conscientiousness': 'Organization, discipline, and attention to detail',
        'extraversion': 'Social energy and interaction preferences',
        'agreeableness': 'Cooperation and trust in relationships',
        'emotional_range': 'Emotional stability and stress management',
        'conversation': 'Verbal communication and discussion skills',
        'openness_to_change': 'Adaptability and flexibility',
        'hedonism': 'Pleasure and gratification seeking',
        'self_enhancement': 'Achievement and power motivation',
        'self_transcendence': 'Universal and benevolent values',
        'conservation': 'Traditional and security values'
    }
    return descriptions.get(skill_name, 'Professional skill or personality trait')

def get_career_description(career_name):
    """Get description for career roles."""
    descriptions = {
        'Software Developer': 'Design, develop, and maintain software applications and systems',
        'Data Scientist': 'Analyze complex data to derive business insights and predictions',
        'Machine Learning Engineer': 'Build and deploy machine learning models and AI systems',
        'Cybersecurity Analyst': 'Protect organizations from digital threats and security breaches',
        'Network Engineer': 'Design, implement, and maintain computer networks',
        'Database Administrator': 'Manage and maintain database systems and data integrity',
        'DevOps Engineer': 'Bridge development and operations for continuous delivery',
        'AI Researcher': 'Research and develop new artificial intelligence technologies',
        'Business Analyst': 'Analyze business processes and recommend improvements',
        'Project Manager': 'Plan, execute, and deliver technology projects',
        'System Administrator': 'Manage and maintain computer systems and servers',
        'Cloud Architect': 'Design and implement cloud-based solutions',
        'Mobile Developer': 'Create applications for mobile devices and platforms',
        'Web Developer': 'Build and maintain websites and web applications',
        'Game Developer': 'Design and develop video games and interactive entertainment',
        'UI/UX Designer': 'Design user interfaces and user experiences for digital products'
    }
    return descriptions.get(career_name, 'Technology professional role')

# ---------------------------------------------
# ✅ Statistics & Profile Generator
# ---------------------------------------------

def calculate_skill_statistics(form_data):
    """Calculate user stats and top skills from form data."""
    try:
        tech_fields = [
            'database_fundamentals', 'computer_architecture', 'distributed_computing',
            'cyber_security', 'networking', 'development', 'programming_skills',
            'project_management', 'computer_forensics', 'technical_communication',
            'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
            'data_science', 'troubleshooting', 'graphics_designing'
        ]
        personality_fields = [
            'openness', 'conscientiousness', 'extraversion', 'agreeableness',
            'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
            'self_enhancement', 'self_transcendence', 'conservation'
        ]

        tech_values = [float(form_data[f]) for f in tech_fields]
        personality_values = [float(form_data[f]) for f in personality_fields]

        top_tech = sorted(((f, float(form_data[f])) for f in tech_fields), key=lambda x: x[1], reverse=True)[:5]
        top_personality = sorted(((f, float(form_data[f])) for f in personality_fields), key=lambda x: x[1], reverse=True)[:5]

        return {
            'technical_average': round(np.mean(tech_values), 2),
            'technical_max': round(max(tech_values), 2),
            'technical_min': round(min(tech_values), 2),
            'personality_average': round(np.mean(personality_values), 2),
            'top_technical_skills': [(format_feature_name(f), v) for f, v in top_tech],
            'top_personality_traits': [(format_feature_name(f), v) for f, v in top_personality],
            'skill_profile': get_skill_profile_description(tech_values, personality_values)
        }

    except Exception as e:
        logger.error(f"Skill statistics error: {str(e)}")
        return {}

def get_skill_profile_description(tech_values, personality_values):
    """Generate a description of the user's skill profile."""
    try:
        tech_avg = np.mean(tech_values)
        personality_avg = np.mean(personality_values)

        tech_level = (
            "expert-level" if tech_avg >= 6 else
            "advanced" if tech_avg >= 5 else
            "intermediate" if tech_avg >= 4 else
            "beginner" if tech_avg >= 3 else
            "novice"
        )

        personality_level = (
            "very strong" if personality_avg >= 0.8 else
            "strong" if personality_avg >= 0.6 else
            "moderate" if personality_avg >= 0.4 else
            "developing"
        )

        return f"You demonstrate {tech_level} technical capabilities with {personality_level} personality traits for the tech industry."

    except Exception as e:
        logger.error(f"Profile description error: {str(e)}")
        return "Profile analysis unavailable"

# ---------------------------------------------
# ✅ Misc Utilities
# ---------------------------------------------

def log_prediction(form_data, prediction_result):
    """Log the prediction for analytics or debugging."""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_result['primary_prediction'],
            'confidence': prediction_result['confidence'],
            'user_profile': calculate_skill_statistics(form_data)
        }
        logger.info(f"Prediction logged: {log_entry['prediction']} ({log_entry['confidence']:.3f})")
    except Exception as e:
        logger.error(f"Prediction logging error: {str(e)}")

def sanitize_input(input_string):
    """Sanitize user input to prevent injection or XSS."""
    if not isinstance(input_string, str):
        return str(input_string)
    return re.sub(r'[<>\"\'%;()&+]', '', input_string).strip()

def generate_recommendations(prediction_result, form_data):
    """Generate personalized recommendations based on prediction."""
    try:
        career = prediction_result['primary_prediction']
        confidence = prediction_result['confidence']
        recs = []

        if confidence < 0.5:
            recs.append({
                'type': 'skill_development',
                'title': 'Strengthen Core Skills',
                'description': 'Consider developing stronger technical foundations to improve career clarity.',
                'priority': 'high'
            })

        # Add career-specific recommendations
        suggestions = {
            'Data Scientist': [
                'Strengthen statistical analysis and machine learning skills',
                'Gain experience with data visualization tools',
                'Present findings to non-technical stakeholders'
            ],
            'Software Developer': [
                'Master additional programming languages',
                'Contribute to open-source projects',
                'Learn design patterns and best practices'
            ],
            'Machine Learning Engineer': [
                'Deepen ML algorithms knowledge',
                'Work with cloud platforms for ML deployment',
                'Learn MLOps and model versioning'
            ],
            'Cybersecurity Analyst': [
                'Earn cybersecurity certifications',
                'Practice with penetration testing tools',
                'Keep up with latest digital threats'
            ]
        }

        for desc in suggestions.get(career, []):
            recs.append({
                'type': 'career_specific',
                'title': f'{career} Development',
                'description': desc,
                'priority': 'medium'
            })

        return recs[:5]

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return []
