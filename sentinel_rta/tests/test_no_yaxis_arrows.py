import os

def test_no_yaxis_arrows():
    with open("sentinel_rta/scripts/generate_manuscript_result_figures.py", "r") as f:
        content = f.read()
    
    assert "ylabel('Service Quality $\\uparrow$')" not in content
    assert "ylabel('Attack Leakage $\\downarrow$')" not in content
    assert "ylabel('SLA Violations $\\downarrow$')" not in content
    
    # Just to make sure no uparrow or downarrow is used in set_ylabel anywhere
    lines = content.split('\n')
    for line in lines:
        if "set_ylabel" in line:
            assert "\\uparrow" not in line, f"Found uparrow in y-axis: {line}"
            assert "\\downarrow" not in line, f"Found downarrow in y-axis: {line}"
