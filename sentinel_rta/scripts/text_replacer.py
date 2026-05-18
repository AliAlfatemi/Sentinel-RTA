import os

target_dir = 'results/manuscript_results_package'
forbidden_terms = {
    'mathematically guarantees': 'is bounded under configured simulator constraints',
    'strict mathematical safety': 'bounded safety',
    'catastrophic sla violations': 'substantial SLA violations',
    'catastrophic': 'substantial',
    'zero-shot': 'held-out intra-simulator',
    'zero-day': '',
    'open-world': '',
    'universal': 'global',
    'definitively prove': 'supports',
    'guaranteed': 'supported under configured simulator constraints',
    'guarantees': 'is supported under configured simulator constraints',
    'guarantee': 'is reduced in the evaluated setting'
}

for root, _, files in os.walk(target_dir):
    for f in files:
        if f.endswith('.md') or f.endswith('.csv') or f.endswith('.tex'):
            path = os.path.join(root, f)
            with open(path, 'r') as file:
                content = file.read()
            
            orig = content
            for term, replacement in forbidden_terms.items():
                # case insensitive replacement but preserving original casing isn't strictly necessary here 
                # since we are replacing with precise strings, but let's do a simple replace
                # It's better to use regex for case insensitivity
                import re
                content = re.sub(re.escape(term), replacement, content, flags=re.IGNORECASE)
                
            if orig != content:
                with open(path, 'w') as file:
                    file.write(content)
                print(f"Cleaned {path}")

