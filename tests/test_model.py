import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from app import MODEL_PATH, load_model


def test_model_file_exists():
    assert Path(MODEL_PATH).exists(), 'Model file not found. Run app once to generate.'


def test_model_prediction_shape():
    model = load_model()
    sample = pd.DataFrame([{ 'age': 30, 'education': 'Master',
                             'major': 'CS', 'interests': 'coding' }])
    probs = model.predict_proba(sample)
    assert probs.shape == (1, len(model.classes_))
    assert np.isclose(probs.sum(), 1.0)