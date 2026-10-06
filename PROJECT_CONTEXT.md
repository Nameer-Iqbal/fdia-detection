PROJECT: Physics-informed False Data Injection Attack (FDIA) detection in smart grids.

GOAL: Simulate the IEEE 33-bus grid in Julia (JuliaGrid), run AC power flow and WLS state estimation, inject FDIAs, engineer physics-informed features, then train and compare:
 (a) Random Forest (supervised)
 (b) Autoencoder (unsupervised, trained on normal data only)
 (c) Baseline Bad Data Detection (chi-square test, largest normalised residual)

ATTACK TYPES: random (non-stealth), single-bus, multi-bus coordinated, and stealth using z_attack = z + Hc, where z is the measurement vector, H is the Jacobian, and c is a fake change in the state. The actual measurement tampering is a = Hc. Residuals should be unchanged under a perfect stealth attack. We must verify this empirically.

FEATURES: residual-based (residuals, normalised residuals, chi-square), consistency (power balance mismatch, voltage deviation, line flow inconsistency), sensitivity-based (Jacobian/leverage indicators), plus non-residual features (raw measurements, estimated states, deviation from expected values, change from previous sample), because perfect stealth attacks are invisible to residual features alone.

ARCHITECTURE: Julia does all physics and writes CSV files to data/. Python reads those CSVs for ML and evaluation. Julia and Python never communicate live.

REPO LAYOUT:
 julia/      simulation, attacks, features
 python/     models, evaluation
 data/       generated datasets (gitignored if large)
 results/    plots and tables
 report/     writing
 configs/    config files (seeds, parameters)

HOW THE PROJECT IS MARKED (this shapes how we work):
 - 45% technical work: rigorous methodology, thorough analysis, content beyond the basics, innovation.
 - 45% critical discussion: evaluation (not just description), well-supported conclusions, honest reporting of negative results, academic referencing.
 - 5% aims and objectives, which must include social and environmental considerations (e.g. grid reliability, blackout risk, protecting critical infrastructure).
 - 5% written communication: clear, well-labelled figures, tables and equations, APA referencing, academic sources only.

THEREFORE these extras are CORE, not optional:
 1. Exact AC stealth attack a = h(x+c) - h(x) versus the linearised a = H*c, and measure how much residual leakage the linearised version has.
 2. Feature ablation study (feature groups on/off).
 3. Generalisation test (train on some attack types, test on unseen ones).
 4. Limited-meter attacker (attacker can only compromise a subset of meters).
 5. Detection versus attack magnitude.

RULES:
1. Do ONLY the task I give you. Do not build ahead.
2. Never guess JuliaGrid's API. Read its official documentation and the installed package source, and cite what you used.
3. Always fix random seeds and keep parameters in configs/.
4. Never claim something works without running it and showing me the real output.
5. If something is ambiguous or fails, stop and ask me instead of working around it silently.
6. Keep code simple, commented, and readable by a student.
7. Prevent data leakage: fit scalers and thresholds on training data only, and never let near-duplicate samples straddle the train/test split.
8. Keep a running log in report/REPORT_LOG.md. After every task, append: what was done, key numbers, figures produced (with file paths), design decisions and why, problems hit, and any academic sources used (APA format). This will become the report.
9. End every task with a short summary: what you did, what you verified, what failed, what's next.
