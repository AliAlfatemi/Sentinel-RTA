import pytest

def test_robustness_math():
    sq = 1.0
    me = 1.0
    leak = 0.0
    cd = 0.0
    sla_norm = 0.0
    r_score = (0.35 * sq) + (0.25 * me) - (0.20 * leak) - (0.10 * cd) - (0.10 * sla_norm)
    assert r_score == 0.60
