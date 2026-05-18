import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics.final_metrics import calculate_sla_norm, calculate_robustness_raw, calculate_forgetting_score

def test_sla_norm():
    # Exactly at budget
    assert calculate_sla_norm(25, 500, 0.05) == 1.0
    # Over budget clamps to 1.0
    assert calculate_sla_norm(100, 500, 0.05) == 1.0
    # Under budget
    assert abs(calculate_sla_norm(10, 500, 0.05) - 0.4) < 1e-5

def test_robustness_raw():
    sq = 1.0
    me = 1.0
    leak = 0.0
    cd = 0.0
    sla = 0.0
    # 0.35(1) + 0.25(1) - 0.2(0) - 0.1(0) - 0.1(0) = 0.6
    assert abs(calculate_robustness_raw(sq, me, leak, cd, sla) - 0.6) < 1e-5

def test_forgetting_score():
    assert abs(calculate_forgetting_score(0.8, 0.2) - 0.6) < 1e-5  # Forgot / Worse
    assert abs(calculate_forgetting_score(0.2, 0.8) - -0.6) < 1e-5 # Improved
