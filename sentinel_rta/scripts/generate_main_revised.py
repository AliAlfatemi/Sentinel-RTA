import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(BASE_DIR, "results", "manuscript_results_package", "tables")
SNIPPETS_FILE = os.path.join(BASE_DIR, "results", "manuscript_results_package", "latex", "latex_figure_snippets.tex")

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def get_table(filename):
    return read_file(os.path.join(TABLES_DIR, filename))

# We will read snippets but they use width=\textwidth. We'll use the ones generated.
snippets = read_file(SNIPPETS_FILE)

# Constructing main_revised.tex
tex = r"""\documentclass[journal]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{Sentinel-RTA: Temporal Runtime Assurance for Safe Reinforcement-Learning-Based DDoS Mitigation under Adaptive Attackers}

\author{Ali~Alfatemi,~\IEEEmembership{Student Member,~IEEE},~Ahmed Alfaqeer, Mohamed~Rahouti,~\IEEEmembership{Member,~IEEE},~Zakirul~Alam~Bhuiyan,~\IEEEmembership{Senior~Member,~IEEE}, and Abdellah~Chehri,~\IEEEmembership{Senior~Member,~IEEE}
        
\thanks{Corresponding Author: Abdellah Chehri (e-mail: chehri@rmc.ca)}     
\thanks{A. Alfatemi, Z. A. Bhuiyan, and M. Rahouti are with the Department of Computer and Information Science, Fordham University, NY, USA (e-mail: aalfatemi@fordham.edu; mbhuiyan3@fordham.edu; mrahouti@fordham.edu).}
\thanks{A. Alfaqeer is with Zaad Corp Credits, Long Island City, NY, USA (e-mail: aalfaqeer@zaad.com)}
\thanks{A. Chehri is with the Department of Mathematics and Computer Science at the Royal Military College of Canada (RMC) in Kingston, Ontario, Canada (e-mail: chehri@rmc.ca)}
}

\markboth{IEEE Transactions on Network and Service Management,~Vol.~XX, No.~X, November~2025}%
{Alfatemi \MakeLowercase{\textit{et al.}}: Sentinel-RTA: Temporal Runtime Assurance for Safe Reinforcement-Learning}

\maketitle

\begin{abstract}
Distributed Denial of Service (DDoS) attacks threaten service assurance in next-generation networks and require closed-loop mitigation that preserves legitimate traffic under adversarial load. Deep reinforcement learning (DRL) can adapt mitigation decisions from traffic feedback, but unconstrained DRL remains unsafe without runtime constraints. Proximal Policy Optimization (PPO) and other DRL algorithms are prone to unsafe exploration, opaque policies, and catastrophic forgetting, making them difficult to deploy in production. This paper presents Sentinel-RTA, a runtime-assured reinforcement learning framework for autonomous DDoS defense in network and service management. Sentinel-RTA wraps a DRL policy with an instantaneous runtime-assurance shield that repairs unsafe actions before enforcement, and extends this with temporal runtime assurance using rolling/cumulative SLA risk to reduce long-horizon safety degradation under stress. We implement leakage-penalized reward calibration to prevent passive-policy collapse. Extensive simulator-based experiments demonstrate Sentinel-RTA's effectiveness through baseline comparisons against heuristic models and an adaptive attacker co-evolutionary evaluation. Furthermore, we investigate Hall-of-Fame historical replay as an ablation, finding that naive replay did not improve robustness in the tested configuration due to optimization interference. The results are simulator-based and establish bounded performance characteristics under explicit constraints.
\end{abstract}

\begin{IEEEkeywords}
DDoS mitigation, safe reinforcement learning, runtime assurance, temporal safety, network service management, adaptive attackers, PPO, SLA protection.
\end{IEEEkeywords}

\section{Introduction}
\IEEEPARstart{T}{he} modern Internet supports cloud services, industrial control systems, and mission-critical applications, but remains vulnerable to DDoS attacks that disrupt availability and service-level objectives \cite{owusu2024online}. DDoS mitigation is inherently a closed-loop service assurance problem where operators must preserve legitimate traffic while suppressing malicious behavior. Static rate limits, access-control lists, and manually tuned scrubbing policies are therefore insufficient when adversaries vary packet headers, protocol mixes, and temporal patterns while operators must minimize collateral damage. Pure detection models also fall short because they do not prescribe sequential mitigation strategies.

Deep Reinforcement Learning (DRL), including Proximal Policy Optimization (PPO), is an attractive alternative because it directly optimizes sequential decision-making from traffic feedback \cite{janakiraman2023drl_ddos, adversarial_drl_sdn2023}. However, unconstrained PPO is unsafe without explicit runtime assurance: exploratory actions can induce outages, learned neural policies are hard to audit, and agents often learn passive behaviors to avoid penalties. 

To address these limitations, we present \emph{Sentinel-RTA}, a runtime-assured RL framework for autonomous DDoS defense. Sentinel-RTA explicitly constrains mitigation decisions using instantaneous and temporal shields to guarantee compliance with simulator safety bounds. Temporal runtime assurance represents the main contribution, as it manages rolling SLA context to prevent cumulative failures under stress. Additionally, an adaptive attacker evaluation exposes the fundamental safety--leakage trade-offs inherent in DDoS response. We further evaluate Hall-of-Fame replay as an ablation, finding that it did not improve robustness in the current configuration.

The main contributions are as follows:
\begin{enumerate}
    \item We formulate adaptive DDoS mitigation as a reproducible Gymnasium/POMDP-based control problem over aggregate traffic telemetry, mitigation actions, service quality, attack leakage, and SLA risk.
    \item We implement a PPO-based mitigation pipeline with leakage-penalized reward calibration to avoid passive high-service-quality policies that fail to suppress attack traffic.
    \item We introduce an instantaneous runtime-assurance shield that repairs unsafe mitigation actions before enforcement.
    \item We extend runtime assurance with a temporal shield that uses rolling and cumulative SLA context to reduce long-horizon safety risk under stress.
    \item We evaluate Sentinel-RTA against heuristic baselines, stress policies, adaptive attackers, and Hall-of-Fame replay ablations, showing that temporal shielding reduces cumulative SLA risk while Hall-of-Fame replay did not improve robustness in the tested configuration.
\end{enumerate}

The remainder of the paper is organized as follows. Section~\ref{sec:background} reviews related work. Section~\ref{sec:preliminary} summarizes the system model. Section~\ref{sec:method} presents the methodology. Section~\ref{sec:exp} describes the experimental design. Section~\ref{sec:evaluation} reports results. Section~\ref{sec:discussion} discusses implications, Section~\ref{sec:limitations} states limitations, and Section~\ref{sec:conclusion} concludes.


\section{Related Work} \label{sec:background}

\subsection{Classical and Threshold-Based DDoS Mitigation}
Classical DDoS defenses rely on ingress filtering, rate limiting, aggregate anomaly detection, traffic engineering, and diversion to scrubbing centers \cite{su2024ddos_survey}. Operators configure routers with thresholds over packet rates and flow counts. These mechanisms are deployable but assume stable attack models. Polymorphic campaigns can stay below static thresholds or exploit coarse configurations \cite{ncsc2020upstream_ddos}, leading to a persistent trade-off between under-mitigation and legitimate collateral damage.

\subsection{ML- and DL-Based DDoS Detection}
Machine learning and deep learning have been widely used for DDoS detection in SDN and edge environments \cite{aldweesh2020survey,su2024ddos_survey,wei2021aemlp}. These systems act as passive classifiers that identify malicious windows but do not determine when or how aggressively to mitigate. Detection metrics alone do not capture sequential mitigation trade-offs among availability, latency, attack suppression, and collateral damage.

\subsection{DRL for Network Defense}
Reinforcement learning models network defense as a sequential decision-making problem. Prior work applies DRL to firewall adaptation in IoT environments \cite{janakiraman2023drl_ddos} and SDN flow-rule reconfiguration \cite{adversarial_drl_sdn2023,shukla2024adversarial_rl_power}. However, unconstrained DRL models prioritize expected reward without explicit safety guarantees, meaning exploratory actions can violate QoS before safety penalties are learned.

\subsection{Safe RL and Runtime Assurance}
Safe RL enforces safety properties during learning or execution. Shielded RL synthesizes runtime shields that intercept unsafe actions \cite{alshiekh2018shielded, konighofer2022online_shielding}. These methods integrate symbolic logic with neural patterns. Furthermore, rule crystallization can be used as an architectural capability to extract human-readable logic from trained models for operator audit or fallback execution, though its fidelity is highly problem-dependent \cite{neurosymbolic_survey, huang2020nsrl}.

\subsection{Co-Evolutionary and Adaptive Attacker Evaluation}
Adversarial RL systems often train attackers and defenders jointly to discover vulnerabilities \cite{evolutionary_cybersecurity_game1}. While co-evolution captures adversarial adaptation, it can lead to catastrophic forgetting unless managed. Replay mechanisms have been proposed to stabilize retention, but they can induce optimization interference if not properly scheduled.


\section{System Model} \label{sec:preliminary}

We formulate online DDoS mitigation as a Partially Observable Markov Decision Process (POMDP). The defender observes compact aggregate traffic telemetry derived from counters and packet headers, reflecting privacy and high-throughput operational constraints. Crucially, the defender does not possess per-flow ground-truth visibility and hidden attack labels are never exposed to the mitigation policy. 

The attacker is modeled as an adaptive agent that dynamically shifts vectors, mutates properties, and exploits bounds within the simulator. 

The defender's objective is to deploy mitigation actions (scalar or low-dimensional mitigation intensity limits and filtering focus) that maximize service quality while actively minimizing attack leakage, collateral damage, and SLA risk.


\section{Methodology} \label{sec:method}

\subsection{Sentinel-RTA Overview}
Sentinel-RTA couples a neural policy, instantaneous action validation, and temporal context tracking. A central controller aggregates time-windowed traffic telemetry and queries the PPO-based defender. The generated mitigation action is passed sequentially through instantaneous and temporal shields to ensure it complies with defined constraints. 

\subsection{PPO Defender Policy}
The defender policy is optimized using Proximal Policy Optimization (PPO). The policy network ingests aggregate observations (e.g., protocol ratios, entropy bounds, and traffic volumes) and outputs a proposed action distribution over the continuous or discrete mitigation space. 

\subsection{Leakage-Penalized Reward Calibration}
Uncalibrated DRL often converges to a passive, service-preserving mode where the policy simply drops excess generic traffic, achieving high legitimate transmission but failing to block the attacker. We implement a leakage-penalized reward calibration that strongly punishes excessive attack leakage, forcing the PPO policy to actively suppress anomalies rather than adopting a global "allow-all" fail-safe.

\subsection{Instantaneous Runtime Assurance}
The Instantaneous Runtime Assurance (RTA) shield intercepts the raw action $a_{\text{raw}}$ and projects it into a safe action set valid for the immediate timestep.
\begin{equation}
    a_{\text{safe}} = \text{projection of } a_{\text{raw}} \text{ into the safe action set}.
\end{equation}

\subsection{Temporal Runtime Assurance}
To combat extended stress and slow degradation, the temporal shield employs a rolling context. It tracks cumulative and sliding-window SLA violation rates, dynamically tightening mitigation actions to prevent cascading failures over long horizons.
\begin{equation}
    a_{\text{temporal}} = \min(a_{\text{safe}}, \text{dynamic\_max\_action})
\end{equation}
where $\text{dynamic\_max\_action}$ is tightened using rolling service quality, rolling collateral damage, rolling SLA violation rate, and safety budget remaining. This constrains actions under configured simulator safety rules, but does not represent a formal universal guarantee.

\subsection{Adaptive Attacker Evaluation}
The co-evolutionary pipeline places the defender in a simulated adversarial arms race. The attacker is trained using PPO to minimize defender metrics while staying within a parameterized resource budget.

\subsection{Hall-of-Fame Replay Ablation}
We evaluate an optional Hall-of-Fame (HoF) replay buffer designed to sample high-performing historical attackers during training. 

\subsection{Metrics and Scoring}
Performance is evaluated across the following core equations:
\begin{align}
    \text{SQ}_t &= \frac{\text{legitimate\_served}_t}{\max(\text{legitimate\_generated}_t, 1)} \\
    \text{Leak}_t &= \frac{\text{attack\_passed}_t}{\max(\text{attack\_generated}_t, 1)} \\
    \text{CD}_t &= \frac{\text{legitimate\_dropped}_t}{\max(\text{legitimate\_generated}_t, 1)} \\
    \text{sla\_violation\_rate} &= \frac{\text{sla\_violation\_count}}{\max(\text{total\_eval\_steps}, 1)} \\
    \text{sla\_norm} &= \min\left(1.0, \frac{\text{sla\_violation\_rate}}{\text{sla\_budget\_rate}}\right)
\end{align}
Additionally, we define composite scoring metrics for evaluation:
\begin{align}
    \text{robustness\_raw} = \quad &0.35 \cdot \text{SQ} \nonumber \\
    &+ 0.25 \cdot \text{mitigation\_efficiency} \nonumber \\
    &- 0.20 \cdot \text{Leak} \nonumber \\
    &- 0.10 \cdot \text{CD} \nonumber \\
    &- 0.10 \cdot \text{sla\_norm} \\
    \text{forgetting\_score} = \quad &\text{final\_leakage\_against\_old\_attackers} \nonumber \\
    &- \text{early\_leakage\_against\_old\_attackers}
\end{align}
A positive forgetting score means worse retention, while a negative forgetting score means improved retention.


\section{Experiments} \label{sec:exp}

We execute a 5-stage experimental suite designed to rigorously validate the bounds of Sentinel-RTA in the simulator.

\subsection{Experiment 1: Baseline Defender Comparison}
\textbf{Methods:} Random Defender, Static Threshold Defender, Adaptive Threshold Defender, and Shield-only Policy. \\
\textbf{Purpose:} Show operational baseline trade-offs and motivate the need for learned mitigation paired with explicit runtime assurance.

\subsection{Experiment 2: Temporal Runtime Assurance Stress Validation}
\textbf{Methods:} No Shield, Instantaneous RTA, Temporal RTA. \\
\textbf{Purpose:} Show that the temporal shield significantly reduces cumulative SLA risk under sustained adversarial stress.

\subsection{Experiment 3: Adaptive Attacker Evaluation}
\textbf{Methods:} Adaptive NoShield, Adaptive Shield NoHoF, Adaptive Shield HoF Pareto 0.1, Static NoShield where applicable. \\
\textbf{Purpose:} Evaluate adaptive co-evolution under extended-preliminary settings.

\subsection{Experiment 4: Hall-of-Fame Replay Ablation}
\textbf{Purpose:} Assess whether HoF replay improves robustness or forgetting.

\subsection{Experiment 5: Held-Out Intra-Simulator Evaluation}
\textbf{Purpose:} Assess held-out simulator profiles without claiming open-world robustness.

"""

tables = {
    "tab1": get_table("table1_baseline_comparison.tex"),
    "tab2": get_table("table2_temporal_stress.tex"),
    "tab3": get_table("table3_adaptive_attacker.tex"),
    "tab4": get_table("table4_hof_ablation.tex"),
    "tab5": get_table("table5_heldout_generalization.tex")
}

tex += r"""
\section{Results} \label{sec:evaluation}

"""
# Include Snippet exactly as is for Architecture
tex += snippets + "\n"

tex += r"""
\subsection{A. Baseline Defender Comparison}
We established an initial reference using deterministic heuristic policies. The service-preserving baselines maintained high service quality by avoiding aggressive mitigation, but this behavior allowed high attack leakage. Conversely, aggressive thresholding reduced leakage but increased SLA violations. 
"""
tex += tables["tab1"] + "\n"

tex += r"""
\subsection{B. Temporal Runtime Assurance Stress Validation}
The temporal stress setting validated the necessity of bounding architecture. Temporal RTA reduces cumulative SLA violations from roughly 7--8 to 0.56 in the evaluated stress setting. However, this safety improvement increased attack leakage from 0.378 to 0.509, illustrating the fundamental safety--leakage trade-off in the modeled simulator. This is the main positive result.
"""
tex += tables["tab2"] + "\n"

tex += r"""
\subsection{C. Adaptive Attacker Evaluation}
In the extended-preliminary co-evolution benchmark, reactive temporal-shielded co-evolution without HoF gives lower leakage than HoF. These results should not be over-interpreted as final full-scale proof but suggest that NoHoF co-evolution adapts effectively.
"""
tex += tables["tab3"] + "\n"

tex += r"""
\subsection{D. Hall-of-Fame Replay Ablation}
We evaluated HoF replay to assess retention. HoF replay did not improve robustness in the tested configuration. Instead, HoF showed optimization interference and generated a worse forgetting score than NoHoF.
"""
tex += tables["tab4"] + "\n"

tex += r"""
\subsection{E. Held-Out Intra-Simulator Evaluation}
To test structural generalization, policies were evaluated against held-out intra-simulator adversaries. The held-out intra-simulator leakage values are close across methods. While policies successfully maintained performance, we report this cautiously and do not claim strong generalization beyond the parameterized simulator constraints.
"""
tex += tables["tab5"] + "\n"

tex += r"""
\section{Discussion} \label{sec:discussion}
Temporal Runtime Assurance stands as the strongest positive result of this framework. By leveraging rolling SLA context, the temporal shielding substantially reduces cumulative SLA risk under stress. However, this reduction in SLA risk naturally comes with increased attack leakage. The adaptive attacker evaluation exposes this fundamental safety--leakage trade-off. 

Additionally, we found that naive HoF replay can introduce optimization interference. In our evaluated setting, reactive co-evolution without HoF performed best in the tested configuration. The results presented are simulator-specific. Full production deployment of such a framework would strictly require real traces, programmable data plane validation, and substantial latency and throughput hardware testing.

\section{Limitations} \label{sec:limitations}
This study explicitly carries several limitations:
\begin{enumerate}
    \item The evaluation is strictly simulator-based.
    \item There is no real trace calibration yet.
    \item We present no hardware, P4, eBPF, or SDN testbed validation yet.
    \item There is no claim of production-readiness.
    \item There is no open-world or zero-day robustness claim.
    \item Held-out evaluation is intra-simulator only.
    \item The Phase 3D co-evolution is an extended preliminary experiment, not a definitive full-scale benchmark.
    \item HoF replay did not improve robustness in current experiments.
    \item Rule crystallization and explainability capabilities were not the main evaluated component of the defense logic unless separate evidence explicitly states otherwise.
    \item Temporal shield parameters currently require manual operator tuning.
\end{enumerate}

\section{Conclusion} \label{sec:conclusion}
Sentinel-RTA supports safe RL-based DDoS mitigation under modeled attacks. The combination of instantaneous and temporal runtime assurance demonstrably improves safety behavior, with Temporal RTA effectively reducing cumulative SLA violations under stress. However, this safety improvement can increase attack leakage. Our adaptive attacker evaluation reveals these intrinsic safety--mitigation trade-offs. We additionally observed that naive HoF replay did not improve robustness in the tested configuration. 

Future work should necessarily include real traces, programmable-data-plane enforcement, adaptive replay scheduling, richer attacker models, full-scale multi-seed evaluation, and rigorous explanation/rule-fidelity evaluation.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""

with open(os.path.join(os.path.dirname(BASE_DIR), "paper", "main_revised.tex"), "w") as f:
    f.write(tex)

print("Generated main_revised.tex")
