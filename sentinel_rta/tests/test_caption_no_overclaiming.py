import os


def test_caption_no_overclaiming():
    target_dir = "results/manuscript_results_package"
    forbidden = [
        "100% service quality",
        "zero-day attack",
        "zero-shot guarantee",
        "open-world robustness",
        "universal robustness",
        "definitively prove",
        "production guarantee",
    ]
    for root, _, files in os.walk(target_dir):
        for name in files:
            if name.endswith((".md", ".tex", ".csv")):
                path = os.path.join(root, name)
                text = open(path, encoding="utf-8", errors="ignore").read().lower()
                for phrase in forbidden:
                    assert phrase not in text, f"Forbidden overclaiming phrase {phrase!r} found in {path}"
