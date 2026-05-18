import os
import pytest

def test_result_consistency_audit():
    md_path = "results/manuscript_results_package/audits/result_consistency_audit.md"
    csv_path = "results/manuscript_results_package/audits/result_consistency_audit.csv"
    
    assert os.path.exists(md_path), "Consistency audit MD missing."
    assert os.path.exists(csv_path), "Consistency audit CSV missing."
    
    # Check that no smoke test values are explicitly mentioned
    with open(md_path, 'r') as f:
        content = f.read()
    assert "smoke" not in content.lower() or "smoke test values are not used" in content.lower(), "Audit incorrectly references smoke tests."
