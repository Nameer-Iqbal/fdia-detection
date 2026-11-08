"""
julia/07_sensitivity_features.jl

Task 7: Compute exact per-row Jacobian H, leverage values (hat matrix diagonal),
and residual covariance diagonal Omega_ii for all 10,000 samples.
Reads data/attacks_full.csv, evaluates H at each row's re-estimated x_hat,
and saves data/sensitivity_features.csv with:
 - row_id
 - leverage_1 .. leverage_163
 - omega_diag_1 .. omega_diag_163

Validates that recomputed max normalized residuals match the stored max_norm_residual column.
"""

using JuliaGrid
using DataFrames
using CSV
using LinearAlgebra
using Printf
import TOML

include("04_attacks.jl")
using .FDIAttacks

function main()
    println("================================================================================")
    println("Task 7: Generating Sensitivity Features (Per-Row Exact H, Leverage & Omega Diag)")
    println("================================================================================")

    # Usage (preferred):  julia --project=julia julia/07_sensitivity_features.jl configs/features_v2.toml
    #   reads [general] dataset_path, sensitivity_output_path and noise_config from the features config.
    # Legacy usage:        julia ... 07_sensitivity_features.jl <attacks.csv> <out.csv> [noise_config.toml]
    root = normpath(joinpath(@__DIR__, ".."))
    resolve(p) = isabspath(p) ? p : normpath(joinpath(root, p))
    if length(ARGS) >= 1 && endswith(ARGS[1], ".toml")
        fcfg = TOML.parsefile(resolve(ARGS[1]))["general"]
        data_rel = fcfg["dataset_path"]
        out_rel = fcfg["sensitivity_output_path"]
        noise_rel = fcfg["noise_config"]
    else
        data_rel = length(ARGS) >= 1 ? ARGS[1] : joinpath("data", "attacks_full.csv")
        out_rel = length(ARGS) >= 2 ? ARGS[2] : joinpath("data", "sensitivity_features.csv")
        noise_rel = length(ARGS) >= 3 ? ARGS[3] : joinpath("configs", "dataset_v2.toml")
    end
    data_path = resolve(data_rel)
    out_path = resolve(out_rel)
    noise = read_noise_config(resolve(noise_rel))
    println("Measurement noise config: ", resolve(noise_rel))
    if !isfile(data_path)
        error("Dataset file not found: $data_path")
    end

    println("Loading dataset from: $data_path")
    df = CSV.read(data_path, DataFrame)
    total_rows = nrow(df)
    println("Total rows to process: $total_rows")

    # Set up power system and measurement model
    case_path = normpath(joinpath(@__DIR__, "..", "data", "cases", "case33bw.m"))
    system = powerSystem(case_path)
    acModel!(system)
    pf = newtonRaphson(system)
    powerFlow!(pf; verbose = 0)
    power!(pf)

    num_buses = system.bus.number
    monitoring = measurement(system)
    v_sigma = noise.v; p_inj_sigma = noise.p_inj; q_inj_sigma = noise.q_inj; p_flow_sigma = noise.p_flow; q_flow_sigma = noise.q_flow
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
    W_diag = diag(Matrix(se.method.precision))
    sigma2 = 1.0 ./ W_diag

    # Pre-allocate output arrays
    row_ids = df[!, :row_id]
    Leverage_mat = zeros(Float64, total_rows, m)
    Omega_mat = zeros(Float64, total_rows, m)

    # Pre-extract x_hat and residuals from DataFrame
    X_hat_all = Matrix{Float64}(df[!, [Symbol("x_hat_$i") for i in 1:n]])
    Residuals_all = Matrix{Float64}(df[!, [Symbol("residual_$i") for i in 1:m]])
    Stored_max_rn = Vector{Float64}(df[!, :max_norm_residual])

    println("Beginning per-row sensitivity computation across $total_rows samples...")
    t0 = time()
    max_diff = 0.0

    for k = 1:total_rows
        x_hat_k = X_hat_all[k, :]
        V_hat, th_hat = state_to_full_polar(x_hat_k)
        
        # Evaluate AC measurement Jacobian at this specific row's estimated state
        eval_ac_measurement(se, V_hat, th_hat)
        H = Matrix(se.method.jacobian[:, 2:66])
        
        # Gain matrix G = H^T W H (65 x 65)
        G = transpose(H) * (W_diag .* H)
        G_inv = inv(G)
        
        # Hat matrix diagonal K_ii = h_i^T G^-1 h_i * W_ii
        # M = H * G^-1 (163 x 65)
        M = H * G_inv
        q = sum(M .* H, dims=2)[:]
        
        lev = q .* W_diag
        omega_d = sigma2 .- q
        
        Leverage_mat[k, :] .= lev
        Omega_mat[k, :] .= omega_d
        
        # Validation check: verify recomputed max normalized residual matches stored column
        r_k = Residuals_all[k, :]
        recomputed_max_rn = maximum(abs.(r_k) ./ sqrt.(omega_d))
        diff_k = abs(recomputed_max_rn - Stored_max_rn[k])
        if diff_k > max_diff
            max_diff = diff_k
        end

        if k % 2000 == 0
            @printf("  Processed %5d / %5d rows (elapsed: %.1fs)...\n", k, total_rows, time() - t0)
        end
    end

    elapsed = time() - t0
    println("Computation completed in $(round(elapsed, digits=2)) s ($(round(elapsed / total_rows * 1000, digits=2)) ms / row).")
    @printf("Validation: Max absolute difference |recomputed_max_rn - stored_max_rn|: %.4e\n", max_diff)
    if max_diff > 1e-6
        error("ALERT: Recomputed normalized residual discrepancy exceeds 1e-6!")
    else
        println("Validation check: PASSED (Exact match with stored max_norm_residual to machine precision).")
    end

    # Build DataFrame
    println("Assembling sensitivity features DataFrame...")
    df_sens = DataFrame(row_id = row_ids)
    for i = 1:m
        df_sens[!, Symbol("leverage_$i")] = Leverage_mat[:, i]
    end
    for i = 1:m
        df_sens[!, Symbol("omega_diag_$i")] = Omega_mat[:, i]
    end

    println("Saving to $out_path...")
    t_save = time()
    CSV.write(out_path, df_sens)
    fsize_mb = filesize(out_path) / (1024 * 1024)
    println("Saved $out_path ($(round(fsize_mb, digits=2)) MB in $(round(time() - t_save, digits=2)) s).")
    println("================================================================================")
    println("Task 7 Sensitivity Features Generation Complete!")
    println("================================================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
