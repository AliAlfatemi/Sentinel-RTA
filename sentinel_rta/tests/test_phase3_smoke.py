import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.mark.skip(reason="Smoke test handled by script")
def test_phase3_smoke():
    pass
