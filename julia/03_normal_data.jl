"""
julia/03_normal_data.jl

Task 4: Normal Dataset Generation, Operating-Point Time Series,
AC State Estimation, BDD Validation, and Jacobian Extraction.
"""

using JuliaGrid
using Printf
using Random
using Statistics
using LinearAlgebra
using SparseArrays
using DataFrames
using CSV
using SHA
using Dates
import TOML

# Helper function to serialize metadata dictionaries to formatted JSON
function to_json(val, indent=0)
    sp = "  "^indent
    sp_next = "  "^(indent+1)
    if isa(val, Dict)
        keys_sorted = sort(collect(keys(val)))
        items = [string(sp_next, "\"", k, "\": ", to_json(val[k], indent+1)) for k in keys_sorted]
        return string("{\n", join(items, ",\n"), "\n", sp, "}")
    elseif isa(val, AbstractVector)
        if isempty(val)
            return "[]"
        elseif isa(first(val), Union{Number, String, Bool}) && length(val) <= 10
            items = [to_json(v, indent+1) for v in val]
            return string("[", join(items, ", "), "]")
        else
            items = [string(sp_next, to_json(v, indent+1)) for v in val]
            return string("[\n", join(items, ",\n"), "\n", sp, "]")
        end
    elseif isa(val, String)
        return string("\"", escape_string(val), "\"")
    elseif isa(val, Bool)
        return val ? "true" : "false"
    elseif isa(val, Number)
        return isfinite(val) ? string(val) : "null"
    else
        return string("\"", string(val), "\"")
    end
end

function main()
    println("================================================================================")
    println("Task 4: Normal Dataset Generation & BDD Validation on IEEE 33-Bus")
    println("================================================================================")

    # -------------------------------------------------------------------------
    # PART 1: Load Configuration
    # -------------------------------------------------------------------------
    config_path = normpath(joinpath(@__DIR__, "..", "configs", "dataset.toml"))
    if !isfile(config_path)
        error("Config file not found: $config_path")
    end
    cfg = TOML.parsefile(config_path)
    config_raw = read(config_path)
    config_hash = bytes2hex(sha256(config_raw))

    case_rel = cfg["system"]["case_file"]
    case_path = normpath(joinpath(@__DIR__, "..", case_rel))
    if !isfile(case_path)
        error("Case file not found: $case_path")
    end

    master_seed = cfg["dataset"]["master_seed"]
    N_samples = cfg["dataset"]["num_samples"]
    samples_per_day = cfg["dataset"]["samples_per_day"]

    lf_min = cfg["operating_point"]["load_factor_min"]
    lf_max = cfg["operating_point"]["load_factor_max"]
    bus_noise_std = cfg["operating_point"]["per_bus_noise_std"]

    peak_hour = cfg["daily_profile"]["peak_hour"]
    c1 = cfg["daily_profile"]["c1"]
    c2 = cfg["daily_profile"]["c2"]

    max_se_iters = cfg["state_estimation"]["max_iterations"]
    se_tol = cfg["state_estimation"]["tolerance"]
    alpha_levels = cfg["state_estimation"]["alpha_levels"]
    num_load_bins = cfg["state_estimation"]["num_load_bins"]

    v_sigma = cfg["noise"]["voltmeter_std"]
    p_inj_sigma = cfg["noise"]["power_injection_active_std"]
    q_inj_sigma = cfg["noise"]["power_injection_reactive_std"]
    p_flow_sigma = cfg["noise"]["power_flow_active_std"]
    q_flow_sigma = cfg["noise"]["power_flow_reactive_std"]

    out_csv_rel = cfg["output"]["dataset_file"]
    out_meta_rel = cfg["output"]["metadata_file"]
    out_jac_rel = cfg["output"]["jacobian_file"]

    out_csv_path = normpath(joinpath(@__DIR__, "..", out_csv_rel))
    out_meta_path = normpath(joinpath(@__DIR__, "..", out_meta_rel))
    out_jac_path = normpath(joinpath(@__DIR__, "..", out_jac_rel))

    mkpath(dirname(out_csv_path))

    println("Config loaded from:             ", config_path)
    println("Config SHA256:                  ", config_hash)
    println("Dataset samples configured (N): ", N_samples)
    println("Master random seed:             ", master_seed)
    println("Load factor range:              [", lf_min, ", ", lf_max, "]")
    println("Per-bus load noise std:         ", bus_noise_std * 100, "% (constant power factor)")

    # -------------------------------------------------------------------------
    # PART 2: System Setup & Measurement Architecture
    # -------------------------------------------------------------------------
    system = powerSystem(case_path)
    acModel!(system)
    pf = newtonRaphson(system)
    powerFlow!(pf; verbose = 0)
    power!(pf)

    num_buses = system.bus.number
    slack_idx = system.bus.layout.slack
    P0 = copy(system.bus.demand.active)
    Q0 = copy(system.bus.demand.reactive)

    monitoring = measurement(system)
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
    m = length(se.method.mean)
    n = 2 * num_buses - 1 # 65 unknown states (32 angles + 33 magnitudes)
    df = m - n             # 98 degrees of freedom

    std_vec = zeros(m)
    std_vec[1:num_buses] .= v_sigma
    std_vec[(num_buses + 1):(2 * num_buses)] .= p_inj_sigma
    std_vec[(2 * num_buses + 1):(2 * num_buses + 32)] .= p_flow_sigma
    std_vec[(2 * num_buses + 33):(3 * num_buses + 32)] .= q_inj_sigma
    std_vec[(3 * num_buses + 33):m] .= q_flow_sigma

    chi_crit_05 = JuliaGrid.quantile(JuliaGrid.Chisq(df), 1.0 - alpha_levels[1])
    chi_crit_01 = JuliaGrid.quantile(JuliaGrid.Chisq(df), 1.0 - alpha_levels[2])

    println("\nSystem measurements (m):        ", m)
    println("Unknown states (n):             ", n, " (Bus ", slack_idx, " angle reference fixed)")
    println("Degrees of freedom (m - n):     ", df)
    @printf("Chi-Square threshold (alpha=0.05): %.4f\n", chi_crit_05)
    @printf("Chi-Square threshold (alpha=0.01): %.4f\n", chi_crit_01)

    # -------------------------------------------------------------------------
    # PART 3: Operating-Point Time Series & State Estimation Loop
    # -------------------------------------------------------------------------
    rng = MersenneTwister(master_seed)

    # Diurnal profile helper function
    # Evaluates harmonic profile and maps min-to-max linearly to [lf_min, lf_max]
    function compute_diurnal_load_factor(t_idx)
        hour = mod((t_idx - 1) * (24.0 / samples_per_day), 24.0)
        raw_val = c1 * cos(2pi * (hour - peak_hour) / 24.0) + c2 * cos(4pi * (hour - peak_hour) / 24.0)
        # Fundamental + 2nd harmonic: max is at hour = peak_hour (c1 + c2 = 0.70)
        # min is at hour ~ peak_hour - 15 = 5.0 (raw_val ~ -0.35355)
        raw_min = -0.35355339
        raw_max = 0.70
        norm_val = clamp((raw_val - raw_min) / (raw_max - raw_min), 0.0, 1.0)
        return lf_min + norm_val * (lf_max - lf_min)
    end

    # Pre-allocate result containers
    sample_ids = Int[]
    time_indices = Int[]
    load_factors = Float64[]
    pf_converged = Bool[]
    se_converged = Bool[]
    se_iterations = Int[]
    objective_Js = Float64[]
    max_norm_residuals = Float64[]
    min_voltages = Float64[]

    v_true_matrix = zeros(N_samples, num_buses)
    theta_true_matrix = zeros(N_samples, num_buses)
    v_est_matrix = zeros(N_samples, num_buses)
    theta_est_matrix = zeros(N_samples, num_buses)
    z_matrix = zeros(N_samples, m)
    residual_matrix = zeros(N_samples, m)

    pf_failed_count = 0
    se_failed_count = 0

    # Save representative Jacobian for sample 1
    example_H_65 = Matrix{Float64}(undef, 0, 0)
    manual_vs_jg_max_diff = 0.0

    println("\nGenerating ", N_samples, " operating points and executing AC WLS state estimation...")
    t_start = time()

    for t = 1:N_samples
        lf_t = compute_diurnal_load_factor(t)

        # Apply per-bus noise with constant power factor
        for i = 1:num_buses
            if system.bus.layout.type[i] == 1 # PQ load bus
                eta = 1.0 + bus_noise_std * randn(rng)
                p_new = P0[i] * lf_t * eta
                q_new = Q0[i] * lf_t * eta
                updateBus!(system; label = i, active = p_new, reactive = q_new)
            end
        end

        # Run AC Newton-Raphson power flow
        setInitialPoint!(pf)
        powerFlow!(pf; verbose = 0)
        power!(pf)

        is_pf_ok = (pf.method.iteration < 40)
        v_min_t = minimum(pf.voltage.magnitude)
        push!(min_voltages, v_min_t)

        if !is_pf_ok
            pf_failed_count += 1
            @printf("WARNING: Power flow failed to converge at sample %d (LF = %.4f). Recording and skipping.\n", t, lf_t)
            push!(sample_ids, t)
            push!(time_indices, t)
            push!(load_factors, lf_t)
            push!(pf_converged, false)
            push!(se_converged, false)
            push!(se_iterations, pf.method.iteration)
            push!(objective_Js, NaN)
            push!(max_norm_residuals, NaN)
            continue
        end

        # True measurements from converged power flow
        z_true = [
            pf.voltage.magnitude;
            pf.power.injection.active;
            pf.power.from.active[1:32];
            pf.power.injection.reactive;
            pf.power.from.reactive[1:32]
        ]

        # Synthesize noisy measurements z = h(x_true) + e
        noise_t = randn(rng, m) .* std_vec
        z_noisy = z_true .+ noise_t

        # Run AC WLS State Estimation
        se.method.mean .= z_noisy
        setInitialPoint!(se)
        stateEstimation!(se; verbose = 0)

        is_se_ok = (se.method.iteration < max_se_iters)
        if !is_se_ok
            se_failed_count += 1
            @printf("WARNING: State estimation failed to converge at sample %d (LF = %.4f).\n", t, lf_t)
        end

        J_t = se.method.objective

        # Compute largest normalized residual via JuliaGrid's residualTest!
        res_test = residualTest!(se; threshold = 1000.0)
        rN_max_t = res_test.maxNormalizedResidual

        # Cross-check against manual calculation for sample 1
        if t == 1
            slack = system.bus.layout.slack
            keep_cols = [c for c in 1:size(se.method.jacobian, 2) if c != slack]
            H_reduced = se.method.jacobian[:, keep_cols] # 163 x 65
            example_H_65 = Matrix(H_reduced)

            W_diag = se.method.precision
            R_diag = [1.0 / W_diag[i, i] for i in 1:m]
            G_mat = Matrix(transpose(H_reduced) * W_diag * H_reduced)
            G_inv = inv(G_mat)

            c_proj = zeros(m)
            for i in 1:m
                hi = example_H_65[i, :]
                c_proj[i] = dot(hi, G_inv * hi)
            end
            Omega_diag = R_diag .- c_proj
            rN_manual = abs.(se.method.residual) ./ sqrt.(abs.(Omega_diag))
            manual_max_rN = maximum(rN_manual)
            manual_vs_jg_max_diff = abs(manual_max_rN - rN_max_t)
        end

        # Record values
        push!(sample_ids, t)
        push!(time_indices, t)
        push!(load_factors, lf_t)
        push!(pf_converged, is_pf_ok)
        push!(se_converged, is_se_ok)
        push!(se_iterations, se.method.iteration)
        push!(objective_Js, J_t)
        push!(max_norm_residuals, rN_max_t)

        v_true_matrix[t, :] .= pf.voltage.magnitude
        theta_true_matrix[t, :] .= pf.voltage.angle
        v_est_matrix[t, :] .= se.voltage.magnitude
        theta_est_matrix[t, :] .= se.voltage.angle
        z_matrix[t, :] .= z_noisy
        residual_matrix[t, :] .= se.method.residual
    end

    elapsed_time = time() - t_start
    println("Dataset simulation finished in:   ", round(elapsed_time, digits=2), " s (",
            round((elapsed_time / N_samples) * 1000, digits=2), " ms/sample)")
    @printf("Power Flow Convergence:          %d / %d (failures: %d)\n", N_samples - pf_failed_count, N_samples, pf_failed_count)
    @printf("State Estimation Convergence:    %d / %d (failures: %d)\n", N_samples - se_failed_count, N_samples, se_failed_count)
    @printf("Min Voltage Range across dataset:[%.4f, %.4f] pu (sensible: >= 0.90 pu)\n",
            minimum(min_voltages), maximum(min_voltages))
    @printf("Normalized Residual Cross-Check: Diff between manual & JuliaGrid = %.4e\n", manual_vs_jg_max_diff)

    # Sanity guard: Verify no unphysical voltages
    if minimum(min_voltages) < 0.90
        @printf("WARNING: Lowest voltage %.4f pu is below 0.90 pu threshold!\n", minimum(min_voltages))
    else
        println("Physical sanity confirmation:   PASSED (All bus voltages strictly >= 0.90 pu)")
    end

    # -------------------------------------------------------------------------
    # PART 4: BDD Validation Across Whole Dataset
    # -------------------------------------------------------------------------
    println("\n--------------------------------------------------------------------------------")
    println("PART 4: BDD Validation Across Dataset (Degrees of Freedom = 98)")
    println("--------------------------------------------------------------------------------")

    valid_Js = filter(!isnan, objective_Js)
    N_valid = length(valid_Js)

    mean_J = mean(valid_Js)
    std_J = std(valid_Js)
    overall_fa_05 = count(>(chi_crit_05), valid_Js) / N_valid
    overall_fa_01 = count(>(chi_crit_01), valid_Js) / N_valid

    @printf("Overall Valid Samples:           %d\n", N_valid)
    @printf("Theoretical Mean of J:           %d\n", df)
    @printf("Sample Mean of J:                %.4f (Relative diff: %.2f%%)\n", mean_J, abs(mean_J - df)/df * 100)
    @printf("Theoretical Std of J:            %.4f\n", sqrt(2.0 * df))
    @printf("Sample Std of J:                 %.4f\n", std_J)
    @printf("False Alarm Rate (alpha = 0.05): %.4f (Nominal: 0.0500, Count: %d / %d)\n",
            overall_fa_05, count(>(chi_crit_05), valid_Js), N_valid)
    @printf("False Alarm Rate (alpha = 0.01): %.4f (Nominal: 0.0100, Count: %d / %d)\n",
            overall_fa_01, count(>(chi_crit_01), valid_Js), N_valid)
    @printf("Mean Largest Norm Residual:      %.4f\n", mean(filter(!isnan, max_norm_residuals)))

    # 5 Load-Factor Bins
    bin_edges = range(lf_min, lf_max, length = num_load_bins + 1)
    println("\n--- BDD Statistics Across 5 Load-Factor Bins ---")
    bin_results = []
    has_drift = false

    for b = 1:num_load_bins
        b_low = bin_edges[b]
        b_high = bin_edges[b+1]
        idxs = findall(i -> (b == num_load_bins ?
                             (load_factors[i] >= b_low && load_factors[i] <= b_high) :
                             (load_factors[i] >= b_low && load_factors[i] < b_high)), 1:N_samples)
        n_b = length(idxs)
        if n_b > 0
            J_b = objective_Js[idxs]
            mean_b = mean(J_b)
            std_b = std(J_b)
            fa05_b = count(>(chi_crit_05), J_b) / n_b
            fa01_b = count(>(chi_crit_01), J_b) / n_b
            c05 = count(>(chi_crit_05), J_b)
            c01 = count(>(chi_crit_01), J_b)

            @printf("Bin %d [%.3f - %.3f] (N=%3d): Mean J = %6.2f | Std = %5.2f | FA(0.05) = %.4f (%d/%d) | FA(0.01) = %.4f (%d/%d)\n",
                    b, b_low, b_high, n_b, mean_b, std_b, fa05_b, c05, n_b, fa01_b, c01, n_b)

            push!(bin_results, Dict(
                "bin_id" => b,
                "range_min" => b_low,
                "range_max" => b_high,
                "sample_count" => n_b,
                "mean_J" => mean_b,
                "std_J" => std_b,
                "fa_05" => fa05_b,
                "fa_01" => fa01_b
            ))

            # Check if bin mean departs drastically from nominal degrees of freedom
            if abs(mean_b - df) > 10.0
                has_drift = true
            end
        end
    end

    if has_drift
        println("\nALERT: Significant drift in BDD statistics observed across load levels!")
    else
        println("\nBDD Load Invariance:            CONFIRMED (Mean J is stable around 98 across all bins; no load drift)")
    end

    # -------------------------------------------------------------------------
    # PART 5: Output Generation
    # -------------------------------------------------------------------------
    println("\n--------------------------------------------------------------------------------")
    println("PART 5: Exporting Datasets, Metadata, and Representative Jacobian")
    println("--------------------------------------------------------------------------------")

    # 1. Build DataFrame
    df_data = DataFrame(
        sample_id = sample_ids,
        time_idx = time_indices,
        load_factor = load_factors,
        converged_pf = pf_converged,
        converged_se = se_converged,
        se_iterations = se_iterations,
        objective_J = objective_Js,
        max_norm_residual = max_norm_residuals
    )

    # Append true state columns
    for i = 1:num_buses
        df_data[!, Symbol("v_true_$i")] = v_true_matrix[:, i]
    end
    for i = 1:num_buses
        df_data[!, Symbol("theta_true_$i")] = theta_true_matrix[:, i]
    end

    # Append estimated state columns
    for i = 1:num_buses
        df_data[!, Symbol("v_est_$i")] = v_est_matrix[:, i]
    end
    for i = 1:num_buses
        df_data[!, Symbol("theta_est_$i")] = theta_est_matrix[:, i]
    end

    # Append measurement columns z_1 .. z_163
    for k = 1:m
        df_data[!, Symbol("z_$k")] = z_matrix[:, k]
    end

    # Append residual columns residual_1 .. residual_163
    for k = 1:m
        df_data[!, Symbol("residual_$k")] = residual_matrix[:, k]
    end

    # Write CSV
    CSV.write(out_csv_path, df_data)
    println("Normal dataset saved to:        ", out_csv_path, " (", nrow(df_data), " rows x ", ncol(df_data), " cols)")

    # 2. Save Representative Jacobian H_example.csv
    # Column mapping: 32 voltage angles (theta_2..theta_33), followed by 33 voltage magnitudes (V_1..V_33)
    # Bus 1 angle is excluded (slack angle fixed).
    state_names = [["theta_$i" for i = 2:num_buses]; ["v_$i" for i = 1:num_buses]]
    df_H = DataFrame(example_H_65, Symbol.(state_names))
    CSV.write(out_jac_path, df_H)
    println("Sample Jacobian saved to:       ", out_jac_path, " (", nrow(df_H), " rows x ", ncol(df_H), " cols)")

    # 3. Write metadata JSON
    meta_dict = Dict(
        "dataset_name" => "normal_dev",
        "description" => "Baseline normal operating points and AC state estimation dataset on IEEE 33-bus system",
        "date_created" => string(Dates.now()),
        "julia_version" => string(VERSION),
        "juliagrid_version" => string(pkgversion(JuliaGrid)),
        "master_seed" => master_seed,
        "config_sha256" => config_hash,
        "num_samples" => N_samples,
        "num_measurements_m" => m,
        "num_states_n" => n,
        "degrees_of_freedom_df" => df,
        "slack_bus" => slack_idx,
        "voltage_range_min" => minimum(min_voltages),
        "voltage_range_max" => maximum(min_voltages),
        "overall_stats" => Dict(
            "mean_J" => mean_J,
            "std_J" => std_J,
            "false_alarm_05" => overall_fa_05,
            "false_alarm_01" => overall_fa_01,
            "mean_max_norm_residual" => mean(filter(!isnan, max_norm_residuals))
        ),
        "load_factor_bins" => bin_results,
        "jacobian_structure" => Dict(
            "rows" => m,
            "cols" => n,
            "slack_angle_excluded" => true,
            "column_mapping" => Dict(
                "cols_1_to_32" => "Bus voltage angles theta_2 through theta_33 (radians)",
                "cols_33_to_65" => "Bus voltage magnitudes V_1 through V_33 (per-unit)"
            )
        ),
        "measurement_index_mapping" => Dict(
            "indices_1_to_33" => "Bus voltage magnitudes V_1 through V_33 (per-unit)",
            "indices_34_to_66" => "Bus active power injections P_inj_1 through P_inj_33 (per-unit)",
            "indices_67_to_98" => "Branch active power from-flows P_flow_1 through P_flow_32 (per-unit)",
            "indices_99_to_131" => "Bus reactive power injections Q_inj_1 through Q_inj_33 (per-unit)",
            "indices_132_to_163" => "Branch reactive power from-flows Q_flow_1 through Q_flow_32 (per-unit)"
        ),
        "columns_description" => Dict(
            "sample_id" => "1-based integer sample identifier",
            "time_idx" => "Time index corresponding to hourly diurnal profile",
            "load_factor" => "System global base load multiplier lambda_t",
            "converged_pf" => "Boolean flag indicating whether AC Newton-Raphson power flow converged",
            "converged_se" => "Boolean flag indicating whether AC WLS state estimation converged",
            "se_iterations" => "Number of Gauss-Newton iterations required for SE convergence",
            "objective_J" => "WLS objective function value J(x_hat) = sum((z - h)^2 / sigma^2)",
            "max_norm_residual" => "Largest normalized residual across all 163 measurements",
            "v_true_1..33" => "Ground-truth bus voltage magnitudes from AC power flow (per-unit)",
            "theta_true_1..33" => "Ground-truth bus voltage angles from AC power flow (radians)",
            "v_est_1..33" => "Estimated bus voltage magnitudes from AC WLS (per-unit)",
            "theta_est_1..33" => "Estimated bus voltage angles from AC WLS (radians)",
            "z_1..163" => "Noisy measurements presented to the state estimator (per-unit)",
            "residual_1..163" => "Measurement residuals r = z - h(x_hat) (per-unit)"
        )
    )

    open(out_meta_path, "w") do f
        write(f, to_json(meta_dict))
    end
    println("Metadata JSON saved to:         ", out_meta_path)
    println("================================================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
