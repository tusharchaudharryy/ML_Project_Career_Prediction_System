import json

import pytest
from app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({'TESTING': True})
    with flask_app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Find Your Ideal Career Path' in response.data


def test_api_prediction_success(client):
    payload = {
        'age': 25,
        'education': 'Bachelor',
        'major': 'CS',
        'interests': 'coding,math'
    }
    rv = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert rv.status_code == 200
    data = rv.get_json()
    assert isinstance(data, dict)
    # Should contain at least one career key
    assert 'Software Engineer' in data


def test_api_prediction_missing_key(client):
    rv = client.post('/api/predict', json={'age': 30})
    assert rv.status_code == 400