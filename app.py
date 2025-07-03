import os
import pickle
from pathlib import Path

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import (StringField, SelectField, SelectMultipleField, IntegerField,
                     SubmitField)
from wtforms.validators import DataRequired, NumberRange
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key')

MODEL_PATH = Path('model.pkl')

CAREERS = [
    'Software Engineer', 'Data Scientist', 'Product Manager',
    'Business Analyst', 'Marketing Manager', 'Financial Analyst',
    'UX Designer', 'DevOps Engineer'
]

EDU_LEVELS = [
    ('High School', 'High School Diploma'),
    ('Bachelor', "Bachelor's"),
    ('Master', "Master's"),
    ('PhD', 'Doctorate (PhD)')
]

# -------------------------------------------------------
# FORMS
# -------------------------------------------------------
class PredictionForm(FlaskForm):
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(14, 65)])
    education = SelectField('Highest Education', choices=EDU_LEVELS,
                            validators=[DataRequired()])
    major = StringField('Major / Field of Study', validators=[DataRequired()])
    interests = SelectMultipleField('Key Interests', coerce=str,
        choices=[('math', 'Math'), ('coding', 'Coding'), ('design', 'Design'),
                 ('business', 'Business'), ('finance', 'Finance'),
                 ('research', 'Research'), ('communication', 'Communication')])
    submit = SubmitField('Predict Career')

# -------------------------------------------------------
# UTILITIES
# -------------------------------------------------------

def train_and_save_dummy_model(path: Path = MODEL_PATH):
    """Create a simple RandomForest model from synthetic data (only for demo)."""
    rng = np.random.default_rng(42)

    n = 200
    X_demo = pd.DataFrame({
        'age': rng.integers(18, 40, size=n),
        'education': rng.choice([e[0] for e in EDU_LEVELS], size=n),
        'major': rng.choice(['CS', 'Business', 'Design', 'Finance', 'Psychology'], size=n),
        'interests': rng.choice(['math', 'coding', 'design', 'business', 'finance',
                                 'research', 'communication'], size=n)
    })
    y_demo = rng.choice(CAREERS, size=n)

    numeric_features = ['age']
    categorical_features = ['education', 'major', 'interests']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_demo, y_demo)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Dummy model trained and saved to {path}")


def load_model():
    if not MODEL_PATH.exists():
        train_and_save_dummy_model()
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


model = load_model()

# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictionForm()
    if form.validate_on_submit():
        data = {
            'age': form.age.data,
            'education': form.education.data,
            'major': form.major.data,
            'interests': ','.join(form.interests.data) if form.interests.data else 'none'
        }
        df = pd.DataFrame([data])
        probs = model.predict_proba(df)[0]
        predictions = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)
        return render_template('results.html', predictions=predictions)
    return render_template('prediction.html', form=form)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(force=True)
    required_keys = {'age', 'education', 'major', 'interests'}
    if not required_keys.issubset(payload):
        return jsonify({'error': 'Missing keys'}), 400
    df = pd.DataFrame([payload])
    probs = model.predict_proba(df)[0]
    return jsonify({cls: float(prob) for cls, prob in zip(model.classes_, probs)})


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)