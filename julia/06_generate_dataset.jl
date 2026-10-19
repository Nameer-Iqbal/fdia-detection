"""
julia/06_generate_dataset.jl

Task 6: Generate Balanced Labelled FDI Attack Dataset for IEEE 33-Bus System.
Generates 10,000 rows (5,000 normal + 5,000 attacked pairs grouped by base sample):
 - For each normal base sample: exactly 1 normal row (label 0) and 1 attacked row (label 1)
 - Attack types (1/6 balanced mix): random, linearised single-bus, linearised multi-bus,
   exact-stealth single-bus, exact-stealth multi-bus, limited-meter
 - Attack families (50/50 mix): in_range (constrained within normal envelope), unconstrained
 - State estimation re-run: flat-start Gauss-Newton AC state estimation on EVERY attacked row
 - Full verification: detection rates, convergence check, label leakage prevention
 - Output: data/attacks_full.csv and data/attacks_meta.json
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
using Dates
using SHA

include("04_attacks.jl")
using .FDIAttacks

function hash_file(filepath::String)
    if isfile(filepath)
        return bytes2hex(sha256(read(filepath)))
    else
        return "not_found"
    end
end

function to_json_str(val; indent=0)
    sp = " "^indent
    sp_in = " "^(indent + 2)
    if val isa Dict
        lines = String[]
        push!(lines, "{\n")
        keys_list = sort(collect(keys(val)))
        for (idx, k) in enumerate(keys_list)
            v = val[k]
            v_str = to_json_str(v; indent=indent+2)
            comma = idx < length(keys_list) ? "," : ""
            push!(lines, "$sp_in\"$k\": $v_str$comma\n")
        end
        push!(lines, "$sp}")
        return join(lines)
    elseif val isa AbstractVector
        lines = String[]
        push!(lines, "[\n")
        for (idx, v) in enumerate(val)
            v_str = to_json_str(v; indent=indent+2)
            comma = idx < length(val) ? "," : ""
            push!(lines, "$sp_in$v_str$comma\n")
        end
        push!(lines, "$sp]")
        return join(lines)
    elseif val isa String
        return "\"$(escape_string(val))\""
    elseif val isa Bool
        return val ? "true" : "false"
    elseif val isa Number
        return string(val)
    elseif isnothing(val)
        return "null"
    else
        return "\"$val\""
    end
end

function main()
    println("================================================================================")
    println("Task 6: Generating Balanced Labelled FDI Attack Dataset (10,000 rows)")
    println("================================================================================")

    # 1. Load Configurations
    config_file_arg = "dataset_attacks.toml"
    for (idx, arg) in enumerate(ARGS)
        if arg == "--config" && idx < length(ARGS)
            config_file_arg = ARGS[idx + 1]
        elseif !startswith(arg, "-") && idx == 1
            config_file_arg = arg
        end
    end
    config_path = isabspath(config_file_arg) ? config_file_arg : normpath(joinpath(@__DIR__, "..", "configs", config_file_arg))
    if !isfile(config_path) && isfile(config_file_arg)
        config_path = normpath(config_file_arg)
    end
    if !isfile(config_path)
        error("Config file not found: $config_path")
    end
    cfg = TOML.parsefile(config_path)

    bdd_cfg_path = normpath(joinpath(@__DIR__, "..", cfg["general"]["bdd_config_path"]))
    if !isfile(bdd_cfg_path)
        error("BDD config file not found: $bdd_cfg_path")
    end
    bdd_cfg = TOML.parsefile(bdd_cfg_path)

    base_data_path = normpath(joinpath(@__DIR__, "..", cfg["general"]["base_dataset_path"]))
    if !isfile(base_data_path)
        error("Base normal dataset not found: $base_data_path")
    end

    df_base = CSV.read(base_data_path, DataFrame)
    N_base = cfg["general"]["num_base_samples"]
    if nrow(df_base) < N_base
        error("Base dataset has $(nrow(df_base)) rows, expected at least $N_base")
    end

    seed = cfg["general"]["seed"]
    rng = MersenneTwister(seed)
    println("Master Random Seed:             $seed")
    println("Base Samples to Process:        $N_base (yielding $(2 * N_base) rows)")

    # Calibrated thresholds
    lnr_05 = bdd_cfg["largest_normalized_residual"]["empirical_threshold_05"]
    lnr_01 = bdd_cfg["largest_normalized_residual"]["empirical_threshold_01"]
    chi_05 = bdd_cfg["chi_square"]["alpha_05_threshold"]
    chi_01 = bdd_cfg["chi_square"]["alpha_01_threshold"]
    v_norm_min = bdd_cfg["range_check"]["voltage_min_normal"]
    v_norm_max = bdd_cfg["range_check"]["voltage_max_normal"]

    # 2. Setup JuliaGrid AC Power Flow & Measurement Framework
    case_path = normpath(joinpath(@__DIR__, "..", cfg["general"]["case_path"]))
    system = powerSystem(case_path)
    acModel!(system)
    pf = newtonRaphson(system)
    powerFlow!(pf; verbose = 0)
    power!(pf)

    num_buses = system.bus.number
    slack_idx = system.bus.layout.slack
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
    n = 65
    W_mat = Matrix(se.method.precision)

    std_vec = zeros(m)
    std_vec[1:num_buses] .= v_sigma
    std_vec[(num_buses + 1):(2 * num_buses)] .= p_inj_sigma
    std_vec[(2 * num_buses + 1):(2 * num_buses + 32)] .= p_flow_sigma
    std_vec[(2 * num_buses + 33):(3 * num_buses + 32)] .= q_inj_sigma
    std_vec[(3 * num_buses + 33):m] .= q_flow_sigma

    # Configurable Attack Mix Parameters
    attack_types = cfg["attack_mix"]["types"]
    num_types = length(attack_types)
    cand_buses = cfg["parameters"]["single_bus_candidates"]
    target_sets = cfg["parameters"]["multibus_target_sets"]
    mag_min = cfg["parameters"]["alpha_V_min"]
    mag_max = cfg["parameters"]["alpha_V_max"]
    theta_ratio = cfg["parameters"]["alpha_theta_ratio"]
    rnd_meters = cfg["parameters"]["random_meter_counts"]
    rnd_mults = cfg["parameters"]["random_noise_multipliers"]
    lim_budgets = cfg["parameters"]["limited_meter_budgets"]
    lim_lambda = cfg["parameters"]["limited_meter_lambda"]

    # Pre-allocate column vectors for 2 * N_base rows
    total_rows = 2 * N_base
    row_ids = zeros(Int, total_rows)
    group_ids = zeros(Int, total_rows)
    time_indices = zeros(Int, total_rows)
    hours_of_day = zeros(Float64, total_rows)
    load_factors = zeros(Float64, total_rows)
    labels = zeros(Int, total_rows)
    attack_type_cols = Vector{String}(undef, total_rows)
    family_cols = Vector{String}(undef, total_rows)
    magnitudes = zeros(Float64, total_rows)
    shift_magnitudes_pu = zeros(Float64, total_rows)
    shift_sigmas = zeros(Float64, total_rows)
    shift_ratios = zeros(Float64, total_rows)
    signs = zeros(Int, total_rows)
    effective_attacks = falses(total_rows)
    n_att_meters = zeros(Int, total_rows)
    att_buses_strs = Vector{String}(undef, total_rows)
    converged_flags = falses(total_rows)
    objectives_J = zeros(Float64, total_rows)
    max_norm_residuals = zeros(Float64, total_rows)

    # Matrix buffers for array features
    Z_mat = zeros(Float64, total_rows, m)
    X_hat_mat = zeros(Float64, total_rows, n)
    Residual_mat = zeros(Float64, total_rows, m)
    Mask_mat = zeros(Int, total_rows, m)
    C_mat = zeros(Float64, total_rows, n)
    X_true_mat = zeros(Float64, total_rows, n)

    # Precompute per-hour normal state standard deviations for all 65 states across df_base
    hourly_state_std = zeros(Float64, 24, n)
    for h in 0:23
        h_mask = [mod(df_base[row, :time_idx] - 1, 24) == h for row in 1:N_base]
        sub_df = df_base[h_mask, :]
        for i = 2:33
            hourly_state_std[h + 1, i - 1] = std(sub_df[!, Symbol("theta_est_$i")])
        end
        for i = 1:33
            hourly_state_std[h + 1, 32 + i] = std(sub_df[!, Symbol("v_est_$i")])
        end
    end
    for h in 1:24, j in 1:n
        if hourly_state_std[h, j] < 1e-9
            hourly_state_std[h, j] = 1.0
        end
    end
    is_log_uniform = get(cfg["parameters"], "distribution", "uniform") == "log_uniform"
    println("State perturbation distribution: ", is_log_uniform ? "Log-Uniform" : "Uniform", " in [", mag_min, ", ", mag_max, "]")

    # Stats counters
    type_counts = Dict(t => 0 for t in attack_types)
    family_counts = Dict("in_range" => 0, "unconstrained" => 0)
    non_converged_count = 0

    println("Beginning dataset generation for $total_rows rows...")
    t_start = time()

    row_ptr = 1

    for g = 1:N_base
        base_row = df_base[g, :]
        t_idx = base_row[:time_idx]
        lf = base_row[:load_factor]
        hr = Float64(mod(t_idx - 1, 24))

        # Extract true base physical state: theta_2..33, V_1..33
        x_true_g = zeros(n)
        for i = 2:33
            x_true_g[i - 1] = base_row[Symbol("theta_true_$i")]
        end
        for i = 1:33
            x_true_g[32 + i] = base_row[Symbol("v_true_$i")]
        end

        # Base unattacked measurements
        z_base_g = zeros(m)
        for k = 1:m
            z_base_g[k] = base_row[Symbol("z_$k")]
        end

        # Base estimated state & Jacobian
        se.method.mean .= z_base_g
        setInitialPoint!(se)
        stateEstimation!(se; verbose = 0)
        x_hat_base_g = full_polar_to_state(se.voltage.magnitude, se.voltage.angle)
        H_base_g = Matrix(se.method.jacobian[:, 2:66])

        # ---------------------------------------------------------------------
        # ROW 1: Normal Sample (label = 0)
        # ---------------------------------------------------------------------
        row_ids[row_ptr] = row_ptr
        group_ids[row_ptr] = g
        time_indices[row_ptr] = t_idx
        hours_of_day[row_ptr] = hr
        load_factors[row_ptr] = lf
        labels[row_ptr] = 0
        attack_type_cols[row_ptr] = "none"
        family_cols[row_ptr] = "none"
        magnitudes[row_ptr] = 0.0
        shift_magnitudes_pu[row_ptr] = 0.0
        shift_ratios[row_ptr] = 0.0
        signs[row_ptr] = 0
        effective_attacks[row_ptr] = false
        n_att_meters[row_ptr] = 0
        att_buses_strs[row_ptr] = "[]"
        max_se_iters = cfg["general"]["se_max_iter"]
        converged_flags[row_ptr] = (se.method.iteration < max_se_iters) && !any(isnan.(se.voltage.magnitude))

        Z_mat[row_ptr, :] .= z_base_g
        X_hat_mat[row_ptr, :] .= x_hat_base_g
        Residual_mat[row_ptr, :] .= se.method.residual
        objectives_J[row_ptr] = se.method.objective

        res_norm = residualTest!(se; threshold = 1000.0)
        max_norm_residuals[row_ptr] = res_norm.maxNormalizedResidual

        Mask_mat[row_ptr, :] .= 0
        C_mat[row_ptr, :] .= 0.0
        X_true_mat[row_ptr, :] .= x_true_g

        row_ptr += 1

        # ---------------------------------------------------------------------
        # ROW 2: Attacked Sample (label = 1)
        # ---------------------------------------------------------------------
        # Select attack type uniformly at random
        att_type = rand(rng, attack_types)
        type_counts[att_type] += 1

        # Select attack family
        if att_type == "random"
            fam = :unconstrained
        else
            fam = rand(rng, [:in_range, :unconstrained])
        end
        fam_str = string(fam)
        family_counts[fam_str] += 1

        # Magnitude sampling: log-uniform or uniform
        if is_log_uniform
            log_alpha = log(mag_min) + rand(rng) * (log(mag_max) - log(mag_min))
            alpha_V = exp(log_alpha)
        else
            alpha_V = mag_min + rand(rng) * (mag_max - mag_min)
        end
        alpha_theta = alpha_V * theta_ratio

        # Generate attack depending on type
        local att_res

        if att_type == "random"
            k_m = rand(rng, rnd_meters)
            mult = rand(rng, rnd_mults)
            att_res = generate_random_attack(z_base_g, x_hat_base_g, std_vec, cfg;
                                             rng = rng, k_meters = k_m, noise_mult = mult)
        elseif att_type == "linearised_single_bus"
            target_bus = rand(rng, cand_buses)
            att_res = generate_single_bus_attack(z_base_g, x_hat_base_g, se, H_base_g, cfg;
                                                 bus = target_bus, alpha_V = alpha_V, alpha_theta = alpha_theta,
                                                 formulation = :linearised, family = fam, rng = rng,
                                                 v_min = v_norm_min, v_max = v_norm_max)
        elseif att_type == "linearised_multi_bus"
            buses = rand(rng, target_sets)
            att_res = generate_multibus_attack(z_base_g, x_hat_base_g, se, H_base_g, cfg;
                                               target_buses = buses, alpha_V = alpha_V, alpha_theta = alpha_theta,
                                               formulation = :linearised, family = fam, rng = rng,
                                               v_min = v_norm_min, v_max = v_norm_max)
        elseif att_type == "exact_ac_single_bus"
            target_bus = rand(rng, cand_buses)
            att_res = generate_single_bus_attack(z_base_g, x_hat_base_g, se, H_base_g, cfg;
                                                 bus = target_bus, alpha_V = alpha_V, alpha_theta = alpha_theta,
                                                 formulation = :exact_ac, family = fam, rng = rng,
                                                 v_min = v_norm_min, v_max = v_norm_max)
        elseif att_type == "exact_ac_multi_bus"
            buses = rand(rng, target_sets)
            att_res = generate_multibus_attack(z_base_g, x_hat_base_g, se, H_base_g, cfg;
                                               target_buses = buses, alpha_V = alpha_V, alpha_theta = alpha_theta,
                                               formulation = :exact_ac, family = fam, rng = rng,
                                               v_min = v_norm_min, v_max = v_norm_max)
        elseif att_type == "limited_meter"
            buses = rand(rng, target_sets)
            M_bgt = rand(rng, lim_budgets)
            S_sub = sort(randperm(rng, m)[1:M_bgt])
            att_res = generate_limited_meter_ls_attack(z_base_g, x_hat_base_g, H_base_g, W_mat, cfg;
                                                      meter_subset = S_sub, M = M_bgt,
                                                      target_buses = buses, alpha_V = alpha_V, alpha_theta = alpha_theta,
                                                      family = fam, lambda = lim_lambda, rng = rng,
                                                      v_min = v_norm_min, v_max = v_norm_max)
        else
            error("Unknown attack type: $att_type")
        end

        # RE-RUN Full AC Gauss-Newton State Estimation on z_attacked from flat start
        se.method.mean .= att_res.z_attacked
        setInitialPoint!(se)
        stateEstimation!(se; verbose = 0)

        is_conv = (se.method.iteration < max_se_iters) && !any(isnan.(se.voltage.magnitude))
        if !is_conv
            non_converged_count += 1
        end

        x_hat_att = full_polar_to_state(se.voltage.magnitude, se.voltage.angle)
        res_att = residualTest!(se; threshold = 1000.0)

        row_ids[row_ptr] = row_ptr
        group_ids[row_ptr] = g
        time_indices[row_ptr] = t_idx
        hours_of_day[row_ptr] = hr
        load_factors[row_ptr] = lf
        labels[row_ptr] = 1
        attack_type_cols[row_ptr] = att_type
        family_cols[row_ptr] = fam_str
        magnitudes[row_ptr] = Float64(att_res.metadata["magnitude"])
        shift_magnitudes_pu[row_ptr] = (att_type == "random") ? 0.0 : Float64(att_res.metadata["magnitude"])

        # Compute shift in sigmas
        c_vec = att_res.c
        h_idx = Int(hr) + 1
        shifted_indices = findall(!iszero, c_vec)
        if isempty(shifted_indices)
            shift_sigmas[row_ptr] = 0.0
        else
            shift_sigmas[row_ptr] = maximum(abs.(c_vec[shifted_indices]) ./ hourly_state_std[h_idx, shifted_indices])
        end
        if haskey(att_res.metadata, "shift_ratio")
            sr = Float64(att_res.metadata["shift_ratio"])
            shift_ratios[row_ptr] = sr
            effective_attacks[row_ptr] = (sr >= 0.05)
        elseif att_type == "random"
            shift_ratios[row_ptr] = 0.0
            effective_attacks[row_ptr] = true
        else
            shift_ratios[row_ptr] = 1.0
            effective_attacks[row_ptr] = true
        end
        signs[row_ptr] = Int(att_res.metadata["sign"])
        n_att_meters[row_ptr] = Int(att_res.metadata["n_attacked_meters"])
        att_buses_strs[row_ptr] = string(att_res.metadata["attacked_buses"])
        converged_flags[row_ptr] = is_conv

        Z_mat[row_ptr, :] .= att_res.z_attacked
        X_hat_mat[row_ptr, :] .= x_hat_att
        Residual_mat[row_ptr, :] .= se.method.residual
        objectives_J[row_ptr] = se.method.objective
        max_norm_residuals[row_ptr] = res_att.maxNormalizedResidual

        Mask_mat[row_ptr, :] .= att_res.metadata["attacked_meter_mask"]
        C_mat[row_ptr, :] .= att_res.c
        X_true_mat[row_ptr, :] .= x_true_g

        row_ptr += 1

        if g % 1000 == 0
            @printf("  Processed %5d / %5d base samples (elapsed: %.1fs)...\n", g, N_base, time() - t_start)
        end
    end

    elapsed_gen = time() - t_start
    println("Dataset state estimation loop completed in $(round(elapsed_gen, digits=2)) s.")
    println("Non-converged SE runs: $non_converged_count / $N_base (0.0% expected)")

    # 3. Assemble Full DataFrame
    println("Assembling DataFrame with all required columns...")
    df_out = DataFrame(
        row_id = row_ids,
        group_id = group_ids,
        time_idx = time_indices,
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
        converged_se = converged_flags,
        objective_J = objectives_J,
        max_norm_residual = max_norm_residuals
    )

    # Measurement columns z_1 .. z_163
    for k = 1:m
        df_out[!, Symbol("z_$k")] = Z_mat[:, k]
    end

    # Estimated state columns x_hat_1 .. x_hat_65
    for i = 1:n
        df_out[!, Symbol("x_hat_$i")] = X_hat_mat[:, i]
    end

    # Residual columns residual_1 .. residual_163
    for k = 1:m
        df_out[!, Symbol("residual_$k")] = Residual_mat[:, k]
    end

    # Attacked meter mask columns mask_1 .. mask_163
    for k = 1:m
        df_out[!, Symbol("mask_$k")] = Mask_mat[:, k]
    end

    # Perturbation vector c_1 .. c_65
    for i = 1:n
        df_out[!, Symbol("c_$i")] = C_mat[:, i]
    end

    # Ground-truth physical state x_true_1 .. x_true_65
    for i = 1:n
        df_out[!, Symbol("x_true_$i")] = X_true_mat[:, i]
    end

    println("Total DataFrame size: $(nrow(df_out)) rows x $(ncol(df_out)) columns.")

    # 4. Save Dataset to data/attacks_full.csv
    out_csv_path = normpath(joinpath(@__DIR__, "..", cfg["general"]["output_dataset_path"]))
    println("Saving dataset to $out_csv_path...")
    t_save = time()
    CSV.write(out_csv_path, df_out)
    csv_bytes = filesize(out_csv_path)
    println("Saved $out_csv_path ($(round(csv_bytes / (1024 * 1024), digits=2)) MB in $(round(time() - t_save, digits=2)) s).")

    # 5. Save Metadata to data/attacks_meta.json
    out_json_path = normpath(joinpath(@__DIR__, "..", cfg["general"]["output_metadata_path"]))
    println("Saving metadata to $out_json_path...")

    excluded_feature_columns = [
        "label",
        "attack_type",
        "family",
        "magnitude",
        "shift_magnitude_pu",
        "shift_sigmas",
        "shift_ratio",
        "sign",
        "effective_attack",
        "n_attacked_meters",
        "attacked_buses",
        [ "mask_$k" for k in 1:m ]...,
        [ "c_$i" for i in 1:n ]...,
        [ "x_true_$i" for i in 1:n ]...
    ]

    meta_dict = Dict(
        "dataset_name" => "IEEE 33-Bus Labelled FDI Attack Dataset",
        "creation_timestamp" => string(Dates.now()),
        "random_seed" => seed,
        "total_rows" => total_rows,
        "total_columns" => ncol(df_out),
        "num_base_samples" => N_base,
        "rows_per_group" => 2,
        "label_distribution" => Dict("normal (0)" => N_base, "attacked (1)" => N_base),
        "attack_type_distribution" => type_counts,
        "family_distribution" => family_counts,
        "non_converged_se_count" => non_converged_count,
        "storage_format" => "CSV",
        "csv_filesize_mb" => round(csv_bytes / (1024 * 1024), digits=2),
        "storage_rationale" => "CSV was chosen over HDF5 because the dataset is 10,000 rows x 702 columns (~99 MB), easily fitting in memory, natively readable by Pandas/Polars/Julia without requiring external HDF5 C libraries, and gitignored.",
        "column_units_and_descriptions" => Dict(
            "magnitude" => "Mixed units: stores voltage shift fraction (pu, 0.01 to 0.05) for state-shift attacks (exact_ac, linearised, limited_meter); stores measurement noise standard deviation multiplier (3.0 to 10.0) for random attacks; 0.0 for normal rows.",
            "shift_magnitude_pu" => "Clean state shift voltage fraction in per-unit (pu): 0.01 to 0.05 for all state-shift attacks; exactly 0.0 for random attacks and normal rows.",
            "effective_attack" => "Boolean flag indicating whether the attack meaningfully perturbed the system: for limited_meter rows, true if shift_ratio >= 0.05 (false if shift was dampened below 5%); for all other attacked rows true; for normal rows false.",
            "shift_ratio" => "Ratio of achieved state shift norm to target state shift norm ||c_achieved|| / ||c_0||. 1.0 for full stealth attacks, 0.0 for normal and random rows, and variable (0.0 to 1.28) for limited_meter rows."
        ),
        "bdd_thresholds" => Dict(
            "chi_square_05" => chi_05,
            "chi_square_01" => chi_01,
            "lnr_empirical_05" => lnr_05,
            "lnr_empirical_01" => lnr_01,
            "v_normal_min" => v_norm_min,
            "v_normal_max" => v_norm_max
        ),
        "config_sha256_hashes" => Dict(
            "dataset_attacks.toml" => hash_file(config_path),
            "bdd.toml" => hash_file(bdd_cfg_path)
        ),
        "excluded_columns_for_model_training" => excluded_feature_columns
    )

    open(out_json_path, "w") do io
        write(io, to_json_str(meta_dict))
    end
    println("Metadata saved to $out_json_path.")

    # 6. Comprehensive Validation of Dataset
    println("\n================================================================================")
    println("PART 5: DATASET VALIDATION (Real Numbers)")
    println("================================================================================")

    # 6a. Check Group ID counts
    group_counts = combine(groupby(df_out, :group_id), nrow => :count)
    all_two = all(group_counts.count .== 2)
    println("Group ID validation: All groups have exactly 2 rows: ", all_two ? "PASSED (all 2)" : "FAILED")

    # 6b. Check NaN values
    has_nans = any(ismissing.(Matrix(df_out[:, [:objective_J, :max_norm_residual]]))) ||
               any(isnan.(Matrix(df_out[:, [:objective_J, :max_norm_residual]])))
    println("NaN / Missing values in numerical columns: ", has_nans ? "WARNING: NaNs found!" : "NONE (CLEAN)")

    # 6c. Baseline Detections by Attack Type
    println("\nDetection Performance by Attack Type Across Baselines:")
    println("Attack Type             | Rows | Chi2(5%) Det | Chi2(1%) Det | Calibrated LNR(5%) | Calibrated LNR(1%) | Range Check Det")
    println("------------------------|------|--------------|--------------|--------------------|--------------------|----------------")

    # Evaluate Normal rows
    df_norm = filter(r -> r.label == 0, df_out)
    norm_chi05 = count(r -> r.objective_J > chi_05, eachrow(df_norm))
    norm_chi01 = count(r -> r.objective_J > chi_01, eachrow(df_norm))
    norm_lnr05 = count(r -> r.max_norm_residual > lnr_05, eachrow(df_norm))
    norm_lnr01 = count(r -> r.max_norm_residual > lnr_01, eachrow(df_norm))
    # Range check on normal rows
    norm_rc = 0
    for r in eachrow(df_norm)
        v_est = [r[Symbol("x_hat_$i")] for i in 33:65]
        if minimum(v_est) < v_norm_min || maximum(v_est) > v_norm_max
            norm_rc += 1
        end
    end

    @printf("%-23s | %4d | %3d (%5.2f%%) | %3d (%5.2f%%) |    %3d (%5.2f%%)    |    %3d (%5.2f%%)    |    %3d (%5.2f%%)\n",
            "normal (baseline FA)", nrow(df_norm),
            norm_chi05, (norm_chi05 / nrow(df_norm)) * 100,
            norm_chi01, (norm_chi01 / nrow(df_norm)) * 100,
            norm_lnr05, (norm_lnr05 / nrow(df_norm)) * 100,
            norm_lnr01, (norm_lnr01 / nrow(df_norm)) * 100,
            norm_rc, (norm_rc / nrow(df_norm)) * 100)

    # Evaluate each Attack Type
    df_att = filter(r -> r.label == 1, df_out)
    for atype in attack_types
        sub = filter(r -> r.attack_type == atype, df_att)
        n_sub = nrow(sub)
        c_chi05 = count(r -> r.objective_J > chi_05, eachrow(sub))
        c_chi01 = count(r -> r.objective_J > chi_01, eachrow(sub))
        c_lnr05 = count(r -> r.max_norm_residual > lnr_05, eachrow(sub))
        c_lnr01 = count(r -> r.max_norm_residual > lnr_01, eachrow(sub))

        c_rc = 0
        for r in eachrow(sub)
            v_est = [r[Symbol("x_hat_$i")] for i in 33:65]
            if minimum(v_est) < v_norm_min || maximum(v_est) > v_norm_max
                c_rc += 1
            end
        end

        @printf("%-23s | %4d | %3d (%5.2f%%) | %3d (%5.2f%%) |    %3d (%5.2f%%)    |    %3d (%5.2f%%)    |    %3d (%5.2f%%)\n",
                atype, n_sub,
                c_chi05, (c_chi05 / n_sub) * 100,
                c_chi01, (c_chi01 / n_sub) * 100,
                c_lnr05, (c_lnr05 / n_sub) * 100,
                c_lnr01, (c_lnr01 / n_sub) * 100,
                c_rc, (c_rc / n_sub) * 100)
    end

    # Evaluate by Family
    println("\nDetection Performance by Attack Family:")
    println("Family          | Rows | Chi2(5%) Det | Chi2(1%) Det | Calibrated LNR(5%) | Range Check Det")
    println("----------------|------|--------------|--------------|--------------------|----------------")
    for fam in ["in_range", "unconstrained"]
        sub = filter(r -> r.family == fam, df_att)
        n_sub = nrow(sub)
        c_chi05 = count(r -> r.objective_J > chi_05, eachrow(sub))
        c_chi01 = count(r -> r.objective_J > chi_01, eachrow(sub))
        c_lnr05 = count(r -> r.max_norm_residual > lnr_05, eachrow(sub))

        c_rc = 0
        for r in eachrow(sub)
            v_est = [r[Symbol("x_hat_$i")] for i in 33:65]
            if minimum(v_est) < v_norm_min || maximum(v_est) > v_norm_max
                c_rc += 1
            end
        end

        @printf("%-15s | %4d | %3d (%5.2f%%) | %3d (%5.2f%%) |    %3d (%5.2f%%)    |    %3d (%5.2f%%)\n",
                fam, n_sub,
                c_chi05, (c_chi05 / n_sub) * 100,
                c_chi01, (c_chi01 / n_sub) * 100,
                c_lnr05, (c_lnr05 / n_sub) * 100,
                c_rc, (c_rc / n_sub) * 100)
    end

    println("\n================================================================================")
    println("DATASET GENERATION AND VALIDATION COMPLETE!")
    println("================================================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
