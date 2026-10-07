# Project Report Log

**Date:** October 7, 2026

## Task 0: Project Context & Repository Setup
- **What was done:** Initialized repository structure with [PROJECT_CONTEXT.md](file:///c:/Users/LINK%20TRADERS%20LT/OneDrive/Desktop/fdia-detection/PROJECT_CONTEXT.md) and initialized [REPORT_LOG.md](file:///c:/Users/LINK%20TRADERS%20LT/OneDrive/Desktop/fdia-detection/report/REPORT_LOG.md).
- **Key numbers:** N/A (setup phase).
- **Figures produced:** None.
- **Design decisions and why:** Kept repository cleanly decoupled with Julia handling power system physics/data generation and Python handling machine learning evaluation via static CSV exchange.
- **Problems hit:** None.
- **Academic sources used:** None for this setup step.

## Task 1: Project Scaffold and Environment Setup (Halted at Step 3: Julia check)
- **What was done:**
  - Created directory scaffold: `julia/`, `python/`, `data/`, `results/`, `configs/` with `.gitkeep` files in each.
  - Created `.gitignore` ignoring `.venv/`, `__pycache__/`, `data/*.csv`, `*.pt`, `.ipynb_checkpoints/`, while preserving `Manifest.toml`.
  - Created [README.md](file:///c:/Projects/fdia-detection/README.md) with a one-paragraph project description.
  - Verified git (`git version 2.50.0.windows.2`), initialized repository, and performed initial commit without touching global user credentials.
  - Verified Python versions (`python --version`: 3.11.5, `py --version`: 3.14.2).
  - Executed `julia --version` and detected that Julia is not installed or not on the system PATH.
  - Researched JuliaGrid official documentation and compatibility requirements.
  - Execution halted per instructions ("If Julia is not installed or not on PATH, tell me and stop. Do not try to install it.").
- **Software versions:**
  - Git: `git version 2.50.0.windows.2`
  - Python: `Python 3.11.5` (`py`: `Python 3.14.2`)
  - Julia: Not found on PATH (`The term 'julia' is not recognized`)
- **JuliaGrid documentation findings & URL:**
  - URL: [https://mcosovic.github.io/JuliaGrid.jl/stable/](https://mcosovic.github.io/JuliaGrid.jl/stable/)
  - Installation notes: Compatible with **Julia versions 1.10 and later**. Installed via `import Pkg; Pkg.add("JuliaGrid")`.
- **Key numbers:** N/A.
- **Figures produced:** None.
- **Problems hit:** `julia` is not recognized as a command on Windows PowerShell. Per prompt rules, stopped immediately without attempting installation or workaround.
- **Academic sources used:**
  - Cosovic, M., Kundacina, O., Delalic, M., Teskeredzic, A., Raca, D., Mesanovic, A., Miskovic, D., Vukobratovic, D., & Monti, A. (2026). JuliaGrid: An open-source Julia-based framework for power system state estimation. *Sustainable Energy, Grids and Networks*, 45, 102073. https://doi.org/10.1016/j.segan.2025.102073

## Task 1b: Environment Setup Completion & Verification
- **What was done:**
  - Verified Julia installation (`julia version 1.13.1`).
  - Activated project in `julia/` and added `JuliaGrid`, `CSV`, and `DataFrames`.
  - Confirmed standard library `Random` loads properly.
  - Maintained `julia/Project.toml` and `julia/Manifest.toml` under version control (not gitignored).
  - Created Python virtual environment at `.venv` explicitly targeting Python 3.11.5.
  - Created `python/requirements.txt` containing `numpy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`, `seaborn`, `shap`, `jupyter`.
  - Installed dependencies into `.venv` directly via `.venv\Scripts\python.exe -m pip install -r python/requirements.txt`.
  - Implemented verification scripts:
    - `julia/00_check_env.jl`: loads JuliaGrid, CSV, DataFrames, Random and prints versions.
    - `python/00_check_env.py`: imports every package and prints versions.
  - Executed both verification scripts and verified all modules load cleanly without error.
- **Exact Software Versions:**
  - **Julia Environment:**
    - Julia runtime: `1.13.1`
    - `JuliaGrid`: `v0.6.2`
    - `CSV`: `v1.1.0`
    - `DataFrames`: `v1.8.2`
    - `Random`: standard library (stdlib)
  - **Python Environment (`.venv`):**
    - Python runtime: `3.11.5` (MSC v.1936 64-bit AMD64)
    - `numpy`: `2.4.6`
    - `pandas`: `3.0.6`
    - `scikit-learn`: `1.9.1`
    - `torch`: `2.14.1+cpu`
    - `matplotlib`: `3.11.2`
    - `seaborn`: `0.13.2`
    - `shap`: `0.51.0`
    - `jupyter`: `1.1.1` (jupyter_core `5.9.1`, jupyterlab `4.6.4`, ipython `9.17.1`)
- **Key numbers:**
  - Julia package dependencies precompiled: 117 packages.
  - Verification scripts exit status: 0 (clean execution).
- **Figures produced:** None.
- **Problems hit:**
  - PowerShell argument escaping required escaping nested quotes when passing inline expressions to `julia -e`.
  - Julia Pkg experienced a host resolution warning for `pkg.julialang.org` and seamlessly fell back to cloning the `General` registry directly from GitHub.
- **Academic sources used:** None new in this step (retained verified JuliaGrid DOI citation from Task 1).
