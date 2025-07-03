from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import logging
from datetime import datetime
import traceback

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from data_preprocessing import DataPreprocessor
from predictor import CareerPredictor
from utils import (
    validate_input, format_predictions, calculate_skill_statistics,
    generate_recommendations, get_skill_description, get_career_description,
    format_feature_name, log_prediction
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize the predictor
predictor = CareerPredictor()

# Define skill categories for the UI
TECHNICAL_SKILLS = [
    ('database_fundamentals', 'Database Fundamentals'),
    ('computer_architecture', 'Computer Architecture'),
    ('distributed_computing_systems', 'Distributed Computing'),
    ('cyber_security', 'Cybersecurity'),
    ('networking', 'Networking'),
    ('software_development', 'Software Development'),
    ('programming_skills', 'Programming Skills'),
    ('project_management', 'Project Management'),
    ('computer_forensics_fundamentals', 'Computer Forensics'),
    ('technical_communication', 'Technical Communication'),
    ('ai_ml', 'AI/Machine Learning'),
    ('software_engineering', 'Software Engineering'),
    ('business_analysis', 'Business Analysis'),
    ('communication_skills', 'Communication Skills'),
    ('data_science', 'Data Science'),
    ('troubleshooting_skills', 'Troubleshooting'),
    ('graphics_designing', 'Graphics Design')
]

PERSONALITY_TRAITS = [
    ('openness', 'Openness to Experience'),
    ('conscientiousness', 'Conscientiousness'),
    ('extraversion', 'Extraversion'),
    ('agreeableness', 'Agreeableness'),
    ('emotional_range', 'Emotional Stability'),
    ('conversation', 'Conversational Skills'),
    ('openness_to_change', 'Adaptability'),
    ('hedonism', 'Enjoyment Seeking'),
    ('self_enhancement', 'Achievement Orientation'),
    ('self_transcendence', 'Helping Others')
]

@app.route('/')
def index():
    """Main page with the skills assessment form."""
    return render_template('index.html', 
                         technical_skills=TECHNICAL_SKILLS,
                         personality_traits=PERSONALITY_TRAITS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    try:
        # Get form data
        form_data = request.form.to_dict()

        # Validate input
        is_valid, validation_message = validate_input(form_data)
        if not is_valid:
            flash(f'Validation Error: {validation_message}', 'error')
            return redirect(url_for('index'))

        # Prepare features for prediction
        features = []

        # Add technical skills (normalize from 1-7 to 0-1)
        for skill_key, _ in TECHNICAL_SKILLS:
            value = float(form_data.get(skill_key, 1)) / 7.0
            features.append(value)

        # Add personality traits (already 0-1)
        for trait_key, _ in PERSONALITY_TRAITS:
            value = float(form_data.get(trait_key, 0.5))
            features.append(value)

        # Make prediction
        prediction_result = predictor.predict(features)

        # Format results
        formatted_result = format_predictions(prediction_result)

        # Calculate user statistics
        user_stats = calculate_skill_statistics(form_data)

        # Generate recommendations
        recommendations = generate_recommendations(prediction_result, form_data)

        # Log the prediction
        log_prediction(form_data, prediction_result)

        return render_template('results.html',
                             prediction=formatted_result,
                             user_stats=user_stats,
                             recommendations=recommendations,
                             prediction_data=prediction_result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid input data'}), 400

        # Make prediction
        prediction_result = predictor.predict(data['features'])

        # Format for API response
        response = {
            'success': True,
            'prediction': prediction_result,
            'formatted': format_predictions(prediction_result),
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/about')
def about():
    """About page explaining the system."""
    return render_template('about.html')

@app.route('/careers')
def careers():
    """Page showing career information."""
    # Get model info if available
    model_info = predictor.get_model_info()
    careers = model_info.get('target_classes', [])

    career_info = []
    for career in careers:
        career_info.append({
            'name': career,
            'description': get_career_description(career)
        })

    return render_template('careers.html', careers=career_info)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        # Check if model is loaded
        model_status = predictor.get_model_info()

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': predictor.model is not None,
            'model_info': model_status
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Try to load the model on startup
    try:
        if predictor.load_model():
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model not loaded - predictions may not work")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
