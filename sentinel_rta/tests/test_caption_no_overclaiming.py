import os
import pytest
import re

def test_caption_no_overclaiming():
    target_dir = "results/manuscript_results_package/"
    
    forbidden_words = [
        "guarantee",
        "zero-shot",
        "zero-day",
        "open-world",
        "universal",
        "definitively prove",
        "proves",
        "chaotic",
        "chaotically",
        "superior",
        "artificial",
        "supportss",
        "is reduced in the evaluated setting",
        "supported under configured simulator constraints"
    ]
    
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.endswith('.md') or f.endswith('.tex') or f.endswith('.csv'):
                if f == "final_manual_readiness_audit.md" or f == "final_visual_readiness_audit.md":
                    continue
                path = os.path.join(root, f)
                with open(path, 'r') as file:
                    content = file.read().lower()
                    
                for word in forbidden_words:
                    # Use word boundary to avoid matching "improves" when searching for "proves"
                    if re.search(rf"\b{word}\b", content):
                        assert False, f"Overclaiming word '{word}' found in {path}"
