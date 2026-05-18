import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import evaluator

def test_service_quality():
    assert evaluator.service_quality(100, 0) == 1.0
    assert evaluator.service_quality(100, 50) == 0.5
    assert evaluator.service_quality(100, 100) == 0.0

def test_mitigation_efficiency():
    assert evaluator.mitigation_efficiency(100, 100) == 1.0
    assert evaluator.mitigation_efficiency(100, 50) == 0.5
    assert evaluator.mitigation_efficiency(100, 0) == 0.0
    assert evaluator.mitigation_efficiency(0, 0) == 1.0

def test_safety_violation():
    assert evaluator.safety_violation_count(0.96) == 0
    assert evaluator.safety_violation_count(0.94) == 1
