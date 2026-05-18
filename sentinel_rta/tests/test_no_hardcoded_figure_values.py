import os
import pytest

def test_no_hardcoded_figure_values():
    script_path = "scripts/generate_manuscript_result_figures.py"
    with open(script_path, 'r') as f:
        content = f.read()
        
    # Check for suspicious hardcoded arrays that look like results
    # Matplotlib allows hardcoding [1, 2, 3] etc, but we shouldn't see big manual data arrays.
    # A simple proxy is checking if read_csv is used to get the data
    assert "pd.read_csv" in content, "Figure script must read from CSVs, not hardcode data."
    
    # Ensure there are no arrays like [0.999, 0.498, ...] directly in the script
    assert "0.999" not in content, "Found hardcoded result values in figure generation script."
    assert "0.498" not in content, "Found hardcoded result values in figure generation script."
