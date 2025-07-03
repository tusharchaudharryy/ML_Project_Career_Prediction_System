# app.py - Main Flask Application for Tech Career Prediction

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from src.predict import CareerPredictor
from src.utils import validate_input, format_predictions
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize predictor
predictor = CareerPredictor()

# Feature lists for form generation
TECHNICAL_SKILLS = [
    'database_fundamentals', 'computer_architecture', 'distributed_computing',
    'cyber_security', 'networking', 'development', 'programming_skills',
    'project_management', 'computer_forensics', 'technical_communication',
    'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
    'data_science', 'troubleshooting', 'graphics_designing'
]

PERSONALITY_TRAITS = [
    'openness', 'conscientiousness', 'extraversion', 'agreeableness',
    'emotional_range', 'conversation', 'openness_to_change', 'hedonism',
    'self_enhancement', 'self_transcendence', 'conservation'
]

CAREER_ROLES = [
    'Software Developer', 'Data Scientist', 'Machine Learning Engineer',
    'Cybersecurity Analyst', 'Network Engineer', 'Database Administrator',
    'DevOps Engineer', 'AI Researcher', 'Business Analyst', 'Project Manager',
    'System Administrator', 'Cloud Architect', 'Mobile Developer',
    'Web Developer', 'Game Developer', 'UI/UX Designer'
]

@app.route('/')
def home():
    """Main landing page with assessment form"""
    try:
        return render_template('index.html', 
                             technical_skills=TECHNICAL_SKILLS,
                             personality_traits=PERSONALITY_TRAITS,
                             career_roles=CAREER_ROLES)
    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}")
        return "Internal Server Error", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle career prediction request"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate input
        is_valid, error_message = validate_input(form_data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Prepare features array
        features = []
        
        # Technical skills (normalize from 1-7 scale to 0-1)
        for skill in TECHNICAL_SKILLS:
            value = float(form_data.get(skill, 1)) / 7.0
            features.append(value)
        
        # Personality traits (already normalized 0-1)
        for trait in PERSONALITY_TRAITS:
            value = float(form_data.get(trait, 0.5))
            features.append(value)
        
        # Make prediction
        prediction_result = predictor.predict(features)
        
        # Format results
        formatted_result = format_predictions(prediction_result)
        
        return render_template('result.html', 
                             prediction=formatted_result,
                             form_data=form_data)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed. Please try again.'}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        features = data['features']
        
        if len(features) != 28:
            return jsonify({'error': 'Expected 28 features'}), 400
        
        # Make prediction
        prediction_result = predictor.predict(features)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction_result
        })
        
    except Exception as e:
        logger.error(f"Error in API prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load model on startup
    try:
        predictor.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=6000)