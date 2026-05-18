import pytest

def test_forgetting_sign():
    early = 0.5
    final = 0.8
    forgetting = final - early
    assert forgetting > 0 # Performance worsened (leakage went up)
