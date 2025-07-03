from app import CAREERS

def test_careers_unique():
    assert len(CAREERS) == len(set(CAREERS))
