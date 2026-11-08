"""
julia/09_mimic_attacks.jl

Distribution-aware ("mimic") exact AC stealth attacks.

Motivation: the attacks in julia/06_generate_dataset.jl draw the state shift c at
random. An exact AC attack is invisible to residual-based BDD, but a random c can
still push the state estimate into a region that is unusual for that hour of day,
which data-driven detectors (Random Forest, Autoencoder, Mahalanobis) learn to flag.
This script tests a stronger attacker who keeps the falsified state TYPICAL:

  For every normal operating point in the target split (test), the attacker picks a
  donor: a genuine normal operating point from a different day at the same hour,
  taken from the donor split (val or train). It then injects the exact AC attack
      c = x_hat_donor - x_hat_self
      a = h(x_hat_self + c) - h(x_hat_self) = h(x_hat_donor) - h(x_hat_self)
      z_attacked = z + a
  and the full AC WLS estimator is re-run from a flat start.

Outputs (paths from configs/mimic_attacks_v2.toml), one pair of files per donor set:
  - attacks file: same column layout as data/attacks_full_v2.csv (label = 1 rows only)
  - sens file:    row_id, leverage_1..163, omega_diag_1..163 (same as 07_sensitivity_features.jl)

Usage:
  julia --project=julia julia/09_mimic_attacks.jl configs/mimic_attacks_v2.toml
"""

using JuliaGrid
using DataFrames
using CSV
using Random
using Statistics
using LinearAlgebra
using Printf
import TOML

include("04_attacks.jl")
using .FDIAttacks

function main()
    println("="^80)
    println("Task 10: Distribution-Aware (Mimic) Exact AC Stealth Attacks")
    println("="^80)

    root = normpath(joinpath(@__DIR__, ".."))
    resolve(p) = isabspath(p) ? p : normpath(joinpath(root, p))
    cfg_path = resolve(length(ARGS) >= 1 ? ARGS[1] : "configs/mimic_attacks_v2.toml")
    cfg = TOML.parsefile(cfg_path)
    gcfg = cfg["general"]

    # ------------------------------------------------------------------
    # 1. Load normal rows and split assignments
    # ------------------------------------------------------------------
    df = CSV.read(resolve(gcfg["dataset_path"]), DataFrame)
    sp = CSV.read(resolve(gcfg["splits_path"]), DataFrame)
    @assert df.row_id == sp.row_id "dataset and splits must be row-aligned"
    df.split = sp.split
    normals = df[df.label .== 0, :]
    targets = normals[normals.split .== gcfg["target_split"], :]
    println("Config:            $cfg_path")
    println("Target normal rows ($(gcfg["target_split"]) split): $(nrow(targets))")

    # ------------------------------------------------------------------
    # 2. Measurement model (identical meter set and noise to the dataset)
    # ------------------------------------------------------------------
    noise = read_noise_config(resolve(gcfg["noise_config"]))
    system = powerSystem(resolve(gcfg["case_path"]))
    acModel!(system)
    pf = newtonRaphson(system)
    powerFlow!(pf; verbose = 0)
    power!(pf)
    monitoring = measurement(system)
    for i = 1:system.bus.number
        addVoltmeter!(monitoring; bus = i, magnitude = pf.voltage.magnitude[i], variance = noise.v^2)
        addWattmeter!(monitoring; bus = i, active = pf.power.injection.active[i], variance = noise.p_inj^2)
        addVarmeter!(monitoring; bus = i, reactive = pf.power.injection.reactive[i], variance = noise.q_inj^2)
    end
    for k = 1:system.branch.number
        if system.branch.layout.status[k] == 1
            addWattmeter!(monitoring; from = k, active = pf.power.from.active[k], variance = noise.p_flow^2)
            addVarmeter!(monitoring; from = k, reactive = pf.power.from.reactive[k], variance = noise.q_flow^2)
        end
    end
    se = gaussNewton(monitoring)
    m = length(se.method.mean)   # 163
    n = 65
    W_diag = diag(Matrix(se.method.precision))
    max_iter = gcfg["se_max_iter"]
    gap = gcfg["min_donor_gap_hours"]

    zc = [Symbol("z_$k") for k in 1:m]
    xc = [Symbol("x_hat_$i") for i in 1:n]

    for set in cfg["sets"]
        println("\n--- Mimic set: $(set["name"]) (donors from '$(set["donor_split"])' normal rows) ---")
        rng = MersenneTwister(gcfg["seed"])
        donors = normals[normals.split .== set["donor_split"], :]
        N = nrow(targets)

        Z = zeros(N, m); X = zeros(N, n); R = zeros(N, m)
        LEV = zeros(N, m); OM = zeros(N, m)
        J = zeros(N); LNR = zeros(N); dJ = zeros(N); est_err = zeros(N)
        v_shift = zeros(N); th_shift = zeros(N); dlf = zeros(N)
        donor_gid = zeros(Int, N); conv = falses(N)

        for (r, row) in enumerate(eachrow(targets))
            # Donor: same hour of day, different day
            cand = donors[(donors.hour_of_day .== row.hour_of_day) .& (abs.(donors.time_idx .- row.time_idx) .>= gap), :]
            d = cand[rand(rng, 1:nrow(cand)), :]
            z = Vector{Float64}(row[zc])
            x_self = Vector{Float64}(row[xc])
            x_donor = Vector{Float64}(d[xc])

            # Exact AC stealth attack moving the estimate onto the donor state
            att = generate_exact_ac_attack(z, x_self, se, x_donor .- x_self)

            # Defender: full AC WLS state estimation from a flat start
            se.method.mean .= att.z_attacked
            setInitialPoint!(se)
            stateEstimation!(se; verbose = 0)
            x_att = full_polar_to_state(se.voltage.magnitude, se.voltage.angle)
            rt = residualTest!(se; threshold = 1000.0)
            conv[r] = (se.method.iteration < max_iter) && !any(isnan.(se.voltage.magnitude))

            Z[r, :] = att.z_attacked
            X[r, :] = x_att
            R[r, :] = se.method.residual
            J[r] = se.method.objective
            LNR[r] = rt.maxNormalizedResidual
            dJ[r] = J[r] - row.objective_J
            est_err[r] = maximum(abs.(x_att .- x_donor))
            v_shift[r] = maximum(abs.(x_donor[33:65] .- x_self[33:65]))
            th_shift[r] = maximum(abs.(x_donor[1:32] .- x_self[1:32]))
            dlf[r] = d.load_factor - row.load_factor
            donor_gid[r] = d.group_id

            # Sensitivity features at the re-estimated state (same formula as script 07)
            lev, om = leverage_and_omega(se, x_att, W_diag)
            LEV[r, :] = lev
            OM[r, :] = om
        end

        # --------------------------------------------------------------
        # 3. Verification printout
        # --------------------------------------------------------------
        @printf("SE convergence:                          %d / %d\n", count(conv), N)
        @printf("Stealth check |J_attacked - J_twin|:     median %.2e, max %.2e (exact AC => ~0)\n", median(abs.(dJ)), maximum(abs.(dJ)))
        @printf("Estimate lands on donor state:           max |x_att - x_donor| = %.2e\n", maximum(est_err))
        @printf("Max voltage shift per attack (pu):       median %.4f, 90th pct %.4f, max %.4f\n",
                median(v_shift), quantile(v_shift, 0.9), maximum(v_shift))
        @printf("Max angle shift per attack (rad):        median %.5f, max %.5f\n", median(th_shift), maximum(th_shift))
        @printf("|donor - target| load factor:            median %.3f, max %.3f\n", median(abs.(dlf)), maximum(abs.(dlf)))

        # --------------------------------------------------------------
        # 4. Save rows in the attacks_full_v2.csv layout + sensitivity file
        # --------------------------------------------------------------
        out = DataFrame(
            row_id = 100000 .+ targets.group_id,      # disjoint from the main dataset's row ids
            group_id = targets.group_id,              # same group as the target normal row
            time_idx = targets.time_idx,
            hour_of_day = targets.hour_of_day,
            load_factor = targets.load_factor,
            label = ones(Int, N),
            attack_type = fill("mimic_exact_ac", N),
            family = fill("mimic", N),
            shift_magnitude_pu = v_shift,
            effective_attack = trues(N),
            converged_se = conv,
            objective_J = J,
            max_norm_residual = LNR,
            delta_J_vs_twin = dJ,
            max_theta_shift_rad = th_shift,
            donor_group_id = donor_gid,
            donor_minus_target_load_factor = dlf,
        )
        for k = 1:m; out[!, Symbol("z_$k")] = Z[:, k]; end
        for i = 1:n; out[!, Symbol("x_hat_$i")] = X[:, i]; end
        for k = 1:m; out[!, Symbol("residual_$k")] = R[:, k]; end
        sens = DataFrame(row_id = out.row_id)
        for k = 1:m; sens[!, Symbol("leverage_$k")] = LEV[:, k]; end
        for k = 1:m; sens[!, Symbol("omega_diag_$k")] = OM[:, k]; end

        CSV.write(resolve(set["attacks_path"]), out)
        CSV.write(resolve(set["sens_path"]), sens)
        println("Saved: $(set["attacks_path"]) ($(nrow(out)) rows), $(set["sens_path"])")
    end
    println("\nMimic attack generation complete.")
end

main()
