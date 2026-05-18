import os
import pytest
import re

def test_table4_no_duplicate_rows():
    path = "results/manuscript_results_package/tables/table4_hof_ablation.tex"
    assert os.path.exists(path), "Table 4 not found."
    
    with open(path, 'r') as f:
        content = f.readlines()
        
    methods = []
    # Simple regex to extract the first column (method name) before the first &
    for line in content:
        if "&" in line and not line.strip().startswith('%') and not line.strip().startswith('\\'):
            method = line.split("&")[0].strip()
            if method and method != "Method (Extended Preliminary)":
                methods.append(method)
                
    # Check for duplicates
    assert len(methods) == len(set(methods)), f"Duplicate methods found in Table 4: {methods}"
