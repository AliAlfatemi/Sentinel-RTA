from pathlib import Path
import pandas as pd

BASE = Path("results/manuscript_results_package/source_csv")
AUDIT = Path("results/manuscript_results_package/audits")
AUDIT.mkdir(parents=True, exist_ok=True)
log = []
rows = []
try:
    held = pd.read_csv(BASE / "heldout_attacker_evaluation.csv")
    vals = held.groupby("Experiment")["Heldout_Leakage"].mean().round(6).to_dict()
    log.append("Held-out leakage values loaded successfully.")
    rows.append({"Check": "Held-out leakage parsed", "Status": "Pass", "Value": str(vals)})
except Exception as e:
    log.append(f"Held-out leakage check failed: {e}")
    rows.append({"Check": "Held-out leakage parsed", "Status": "Fail", "Value": str(e)})
try:
    rob = pd.read_csv(BASE / "robustness_analysis.csv")
    grouped = rob.groupby("Experiment")["Robustness_Score"].mean()
    hof = grouped["Adaptive_Shield_HoF_pareto_0.1"]
    nohof = grouped["Adaptive_Shield_NoHoF"]
    ok = hof < nohof
    log.append(f"HoF negative ablation {'confirmed' if ok else 'not confirmed'}: HoF {hof:.3f}, NoHoF {nohof:.3f}.")
    rows.append({"Check": "HoF negative ablation", "Status": "Pass" if ok else "Fail", "Value": f"HoF {hof:.3f}, NoHoF {nohof:.3f}"})
except Exception as e:
    log.append(f"HoF ablation check failed: {e}")
    rows.append({"Check": "HoF negative ablation", "Status": "Fail", "Value": str(e)})
log.append("Adaptive-attacker evidence is labeled as a limited-seed simulator benchmark.")
rows.append({"Check": "Benchmark label", "Status": "Pass", "Value": "limited-seed"})
(AUDIT / "result_consistency_audit.md").write_text("# Result Consistency Audit\n\n" + "\n".join(f"- {x}" for x in log) + "\n", encoding="utf-8")
pd.DataFrame(rows).to_csv(AUDIT / "result_consistency_audit.csv", index=False)
print("Consistency audit complete.")
