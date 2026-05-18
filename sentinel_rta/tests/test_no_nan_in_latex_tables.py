import os
import glob
import pytest

def test_no_nan_in_latex_tables():
    out_dir = "results/manuscript_results_package/tables"
    tex_files = glob.glob(os.path.join(out_dir, "*.tex"))
    
    assert len(tex_files) > 0, "No latex tables found."
    
    for f in tex_files:
        with open(f, 'r') as file:
            content = file.read()
            assert "nan" not in content.lower(), f"NaN value found in table: {f}"
