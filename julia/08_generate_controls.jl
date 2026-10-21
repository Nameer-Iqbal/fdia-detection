"""
julia/08_generate_controls.jl

Generates NORMAL control rows for Dataset v2 (5,000 samples) by passing
normal measurements through the exact same attack code path with c = 0
(same flat-start re-solve of the Gauss-Newton estimator, same column construction),
labelled 0.
"""

using JuliaGrid
using Printf
using Random
using Statistics
using LinearAlgebra
using SparseArrays
using DataFrames
using CSV
import TOML

include("04_attacks.jl")
using .FDIAttacks

function main()
    println("="^80)
    println("GENERATING NORMAL CONTROL ROWS (c = 0 attack path re-solve)")
    println("="^80)

    config_file = length(ARGS) >= 1 ? ARGS[1] : "configs/dataset_attacks_v2.toml"
    cfg_path = isabspath(config_file) ? config_file : normpath(joinpath(@__DIR__, "..", config_file))
    if !isfile(cfg_path)
        cfg_path = normpath(joinpath(@__DIR__, config_file))
    end
    cfg = TOML.parsefile(cfg_path)

    base_data_path = normpath(joinpath(@__DIR__, "..", cfg["general"]["base_dataset_path"]))
    df_base = CSV.read(base_data_path, DataFrame)
    N_base = nrow(df_base)
    println("Base dataset: $base_data_path ($N_base samples)")

    case_path = normpath(joinpath(@__DIR__, "..", cfg["general"]["case_path"]))
    system = powerSystem(case_path)
    acModel!(system)
    pf = newtonRaphson(system)
    powerFlow!(pf; verbose = 0)
    power!(pf)

    num_buses = system.bus.number
    monitoring = measurement(system)

    v_sigma = 0.005
    p_inj_sigma = 0.010
    q_inj_sigma = 0.010
    p_flow_sigma = 0.008
    q_flow_sigma = 0.008

    for i = 1:num_buses
        addVoltmeter!(monitoring; bus = i, magnitude = pf.voltage.magnitude[i], variance = v_sigma^2)
        addWattmeter!(monitoring; bus = i, active = pf.power.injection.active[i], variance = p_inj_sigma^2)
        addVarmeter!(monitoring; bus = i, reactive = pf.power.injection.reactive[i], variance = q_inj_sigma^2)
    end
    for k = 1:system.branch.number
        if system.branch.layout.status[k] == 1
            addWattmeter!(monitoring; from = k, active = pf.power.from.active[k], variance = p_flow_sigma^2)
            addVarmeter!(monitoring; from = k, reactive = pf.power.from.reactive[k], variance = q_flow_sigma^2)
        end
    end

    se = gaussNewton(monitoring)
    m = length(se.method.mean) # 163
    max_se_iters = cfg["general"]["se_max_iter"]

    # Pre-allocate output columns
    group_ids = zeros(Int, N_base)
    time_indices = zeros(Int, N_base)
    hours_of_day = zeros(Int, N_base)
    load_factors = zeros(Float64, N_base)
    labels = zeros(Int, N_base)
    attack_type_cols = fill("control_normal", N_base)
    family_cols = fill("none", N_base)
    magnitudes = zeros(Float64, N_base)
    shift_magnitudes_pu = zeros(Float64, N_base)
    shift_sigmas = zeros(Float64, N_base)
    shift_ratios = zeros(Float64, N_base)
    signs = zeros(Int, N_base)
    effective_attacks = fill(false, N_base)
    n_att_meters = zeros(Int, N_base)
    att_buses_strs = fill("[]", N_base)
    converged_flags = fill(true, N_base)

    Z_mat = zeros(Float64, N_base, m)
    X_hat_mat = zeros(Float64, N_base, 65)
    Residual_mat = zeros(Float64, N_base, m)
    objectives_J = zeros(Float64, N_base)
    max_norm_residuals = zeros(Float64, N_base)
    Mask_mat = zeros(Int, N_base, m)
    C_mat = zeros(Float64, N_base, 65)
    X_true_mat = zeros(Float64, N_base, 65)

    c_zero = zeros(65)

    println("Processing $N_base samples through exact attack code path with c = 0...")
    t_start = time()

    for g = 1:N_base
        base_row = df_base[g, :]
        t_idx = base_row[:time_idx]
        lf = base_row[:load_factor]
        hr = Float64(mod(t_idx - 1, 24))

        x_true_g = zeros(65)
        for i = 2:33
            x_true_g[i - 1] = base_row[Symbol("theta_true_$i")]
        end
        for i = 1:33
            x_true_g[32 + i] = base_row[Symbol("v_true_$i")]
        end

        z_base_g = zeros(m)
        for k = 1:m
            z_base_g[k] = base_row[Symbol("z_$k")]
        end

        # Initial estimate (as done before generating attack)
        se.method.mean .= z_base_g
        setInitialPoint!(se)
        stateEstimation!(se; verbose = 0)
        x_hat_base = full_polar_to_state(se.voltage.magnitude, se.voltage.angle)

        # EXACT ATTACK CODE PATH with c = 0:
        V_hat, _ = state_to_full_polar(x_hat_base)
        V_tar, th_tar = state_to_full_polar(x_hat_base .+ c_zero)
        a_test = eval_ac_measurement(se, V_tar, th_tar) .- eval_ac_measurement(se, V_hat, state_to_full_polar(x_hat_base)[2])
        z_attacked = z_base_g .+ a_test

        # RE-RUN Full AC Gauss-Newton State Estimation on z_attacked from flat start
        # (identical call to Row 2 in julia/06_generate_dataset.jl)
        se.method.mean .= z_attacked
        setInitialPoint!(se)
        stateEstimation!(se; verbose = 0)

        is_conv = (se.method.iteration < max_se_iters) && !any(isnan.(se.voltage.magnitude))
        x_hat_ctrl = full_polar_to_state(se.voltage.magnitude, se.voltage.angle)
        res_ctrl = residualTest!(se; threshold = 1000.0)

        group_ids[g] = g
        time_indices[g] = t_idx
        hours_of_day[g] = hr
        load_factors[g] = lf
        labels[g] = 0
        converged_flags[g] = is_conv

        Z_mat[g, :] .= z_attacked
        X_hat_mat[g, :] .= x_hat_ctrl
        Residual_mat[g, :] .= se.method.residual
        objectives_J[g] = se.method.objective
        max_norm_residuals[g] = res_ctrl.maxNormalizedResidual
        X_true_mat[g, :] .= x_true_g

        if g % 1000 == 0
            @printf("  Processed %5d / %5d samples (elapsed: %.1fs)...\n", g, N_base, time() - t_start)
        end
    end

    println("Assembly complete. Creating DataFrame...")
    df_out = DataFrame(
        group_id = group_ids,
        time_index = time_indices,
        hour_of_day = hours_of_day,
        load_factor = load_factors,
        label = labels,
        attack_type = attack_type_cols,
        family = family_cols,
        magnitude = magnitudes,
        shift_magnitude_pu = shift_magnitudes_pu,
        shift_sigmas = shift_sigmas,
        shift_ratio = shift_ratios,
        sign = signs,
        effective_attack = effective_attacks,
        n_attacked_meters = n_att_meters,
        attacked_buses = att_buses_strs,
        converged = converged_flags,
        objective_J = objectives_J,
        max_norm_residual = max_norm_residuals
    )

    for k = 1:m
        df_out[!, Symbol("z_$k")] = Z_mat[:, k]
    end
    for k = 1:65
        df_out[!, Symbol("x_hat_$k")] = X_hat_mat[:, k]
    end
    for k = 1:m
        df_out[!, Symbol("residual_$k")] = Residual_mat[:, k]
    end
    for k = 1:m
        df_out[!, Symbol("meter_mask_$k")] = Mask_mat[:, k]
    end
    for k = 1:65
        df_out[!, Symbol("c_$k")] = C_mat[:, k]
    end
    for k = 1:65
        df_out[!, Symbol("x_true_$k")] = X_true_mat[:, k]
    end

    out_path = "data/controls_normal_v2.csv"
    println("Saving control rows to: $out_path ($(nrow(df_out)) rows x $(ncol(df_out)) cols)...")
    CSV.write(out_path, df_out)
    println("Saved $out_path successfully.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
