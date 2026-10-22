# Physics-Informed False Data Injection Attack (FDIA) Detection in Smart Grids


This project implements a physics-informed detection framework for False Data Injection Attacks (FDIAs) targeting power distribution networks, evaluated on the IEEE 33-bus test feeder. The grid physics, state estimation (AC power flow and Weighted Least Squares), and diverse cyber-attack scenarios (random, targeted, and exact AC/linearised stealth attacks) are simulated in Julia using JuliaGrid, while physics-informed residual, consistency, and sensitivity features are passed to a Python pipeline for benchmarking supervised Random Forest and unsupervised Autoencoder architectures against classical bad data detection baselines.
