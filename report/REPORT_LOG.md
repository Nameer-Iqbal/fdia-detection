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
