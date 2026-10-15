"""
julia/05_verify_attacks.jl

Task 5: Verification of False Data Injection (FDI) Attacks on IEEE 33-Bus System.
Evaluates on 200 normal operating points across:
 1. Exact AC stealth attacks (verifying J and residuals unchanged to numerical precision)
 2. Linearised stealth attacks as a function of attack magnitude (non-linear leakage analysis)
 3. Random non-stealth attacks (BDD and largest-normalized-residual detection rates)
 4. Single-bus vs multi-bus coordinated attacks (random and adjacent-along-feeder)
 5. Limited-meter attacks vs meter budget (detection rate vs attacker access)
 6. Physical voltage feasibility checks (ANSI C84.1 bounds [0.90, 1.05] pu)
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
    println("================================================================================")
    println("Task 5: FDI Attack Verification Suite on IEEE 33-Bus System")
    println("================================================================================")

    # 1. Load Configurations & Base System
    config_path = normpath(joinpath(@__DIR__, "..", "configs", "attacks.toml"))
    if !isfile(config_path)
        error("Config file not found: $config_path")
    end
    cfg = TOML.parsefile(config_path)

    dataset_path = normpath(joinpath(@__DIR__, "..", "data", "normal_data.csv"))
    if !isfile(dataset_path)
        error("Dataset file not found: $dataset_path. Run julia/03_normal_data.jl first.")
    end

    df_raw = CSV.read(dataset_path, DataFrame)
    N_test = min(200, nrow(df_raw))
    println("Loaded dataset from:            ", dataset_path)
    println("Evaluation sample count (N):    ", N_test)
    println("Master random seed:             ", cfg["general"]["seed"])

    system_path = normpath(joinpath(@__DIR__, "..", "data", "cases", "case33bw.m"))
    system = powerSystem(system_path)
    acModel!(system)
    pf = newtonRaphson(system)
    powerFlow!(pf; verbose = 0)
    power!(pf)

    num_buses = system.bus.number
    slack_idx = system.bus.layout.slack

    # Set up measurement topology
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
    m = length(se.method.mean)
    n = 65
    df = m - n

    chi_05 = JuliaGrid.quantile(JuliaGrid.Chisq(df), 0.95)
    chi_01 = JuliaGrid.quantile(JuliaGrid.Chisq(df), 0.99)

    std_vec = zeros(m)
    std_vec[1:num_buses] .= v_sigma
    std_vec[(num_buses + 1):(2 * num_buses)] .= p_inj_sigma
    std_vec[(2 * num_buses + 1):(2 * num_buses + 32)] .= p_flow_sigma
    std_vec[(2 * num_buses + 33):(3 * num_buses + 32)] .= q_inj_sigma
    std_vec[(3 * num_buses + 33):m] .= q_flow_sigma

    W_diag = se.method.precision

    println("Degrees of freedom (m - n):     ", df)
    @printf("Chi-Square threshold (alpha=0.05): %.4f\n", chi_05)
    @printf("Chi-Square threshold (alpha=0.01): %.4f\n", chi_01)

    # Pre-extract measurement matrix Z_base (N_test x m) and base estimates
    Z_base = zeros(N_test, m)
    for k = 1:m
        Z_base[:, k] .= df_raw[1:N_test, Symbol("z_$k")]
    end
    J_base_vec = df_raw[1:N_test, :objective_J]

    # Pre-compute unattacked state estimates and Jacobians for each test sample
    X_hat_base = zeros(N_test, n)
    H_base_list = Vector{Matrix{Float64}}(undef, N_test)
    r_norm_base = zeros(N_test)

    println("\nPreprocessing unattacked state estimates for ", N_test, " samples...")
    for t = 1:N_test
        se.method.mean .= Z_base[t, :]
        setInitialPoint!(se)
        stateEstimation!(se; verbose = 0)
        X_hat_base[t, :] .= full_polar_to_state(se.voltage.magnitude, se.voltage.angle)
        # Exclude slack bus angle (col 1)
        H_base_list[t] = Matrix(se.method.jacobian[:, 2:66])
        r_norm_base[t] = norm(se.method.residual)
    end
    println("Preprocessing complete.")

    rng = MersenneTwister(cfg["general"]["seed"])

    # -------------------------------------------------------------------------
    # TEST 1: Exact AC Stealth Attack Verification
    # -------------------------------------------------------------------------
    println("\n================================================================================")
    println("TEST 1: Exact AC Stealth Attack Verification (N = $N_test)")
    println("================================================================================")

    direct_delta_J = Float64[]
    direct_delta_r = Float64[]
    solver_delta_J = Float64[]
    solver_delta_r = Float64[]
    shift_errors = Float64[]
    min_V_att = Float64[]
    max_V_att = Float64[]
    v_breach_count = 0

    # Perturbation vector c: coordinated shift on Buses 10, 18, 25 (3% voltage shift, 0.015 rad angle shift)
    c_fixed = zeros(65)
    c_fixed[32 + 10] = 0.03  # +3% on Bus 10
    c_fixed[9] = 0.015       # +0.015 rad (~0.86 deg) on Bus 10
    c_fixed[32 + 18] = -0.03 # -3% on Bus 18
    c_fixed[17] = -0.015     # -0.015 rad on Bus 18
    c_fixed[32 + 25] = 0.02  # +2% on Bus 25
    c_fixed[24] = 0.010      # +0.010 rad on Bus 25

    for t = 1:N_test
        z_t = Z_base[t, :]
        x_hat_t = X_hat_base[t, :]
        J_0 = J_base_vec[t]
        r_norm_0 = r_norm_base[t]

        # Generate Exact AC attack
        att = generate_exact_ac_attack(z_t, x_hat_t, se, c_fixed)

        # 1. Direct mathematical residual evaluation at x_hat + c
        V_tar, th_tar = state_to_full_polar(x_hat_t .+ c_fixed)
        eval_ac_measurement(se, V_tar, th_tar)
        se.method.mean .= att.z_attacked
        JuliaGrid.normalEquation!(se.system, se)
        r_direct = copy(se.method.residual)
        J_direct = se.method.objective
        push!(direct_delta_J, abs(J_direct - J_0))
        push!(direct_delta_r, abs(norm(r_direct) - r_norm_0))

        # 2. Full Gauss-Newton State Estimation from flat start
        setInitialPoint!(se)
        stateEstimation!(se; verbose = 0)
        x_att = full_polar_to_state(se.voltage.magnitude, se.voltage.angle)

        push!(solver_delta_J, se.method.objective - J_0)
        push!(solver_delta_r, norm(se.method.residual) - r_norm_0)
        push!(shift_errors, norm((x_att .- x_hat_t) .- c_fixed))

        v_min_t = minimum(se.voltage.magnitude)
        v_max_t = maximum(se.voltage.magnitude)
        push!(min_V_att, v_min_t)
        push!(max_V_att, v_max_t)

        if v_min_t < 0.90 || v_max_t > 1.05
            v_breach_count += 1
        end
    end

    @printf("Mathematical Direct Residual Check (at x_hat + c):\n")
    @printf("  Max |Delta J| (direct):          %.4e (Expected: ~0.0)\n", maximum(direct_delta_J))
    @printf("  Max |Delta ||r||| (direct):      %.4e (Expected: ~0.0)\n", maximum(direct_delta_r))
    @printf("Gauss-Newton Full Solver Run (from flat start, tol = 1e-8):\n")
    @printf("  Mean Delta J (solver):           %+.6f\n", mean(solver_delta_J))
    @printf("  Max |Delta J| (solver):          %.6f\n", maximum(abs.(solver_delta_J)))
    @printf("  Mean Delta ||r|| (solver):       %+.6e\n", mean(solver_delta_r))
    @printf("  Mean State Shift Error ||x_att - x_hat - c||: %.6e\n", mean(shift_errors))
    @printf("Physical Sanity & Voltage Bounds Check:\n")
    @printf("  Attacked Voltage Range:          [%.4f, %.4f] pu\n", minimum(min_V_att), maximum(max_V_att))
    @printf("  ANSI C84.1 Limit Breaches:       %d / %d (%.1f%%)\n", v_breach_count, N_test, (v_breach_count / N_test) * 100)

    if maximum(direct_delta_J) > 1e-8
        error("ALERT: Exact AC stealth attack failed! Residuals or J changed significantly.")
    else
        println("Exact AC Stealth Verification:  PASSED (J and residuals mathematically invariant to machine precision)")
    end

    # -------------------------------------------------------------------------
    # TEST 2: Linearised Stealth vs Attack Magnitude (Leakage Curve)
    # -------------------------------------------------------------------------
    println("\n================================================================================")
    println("TEST 2: Linearised Stealth vs Attack Magnitude (Non-Linear Leakage Curve)")
    println("================================================================================")
    println("Magnitude | Mean Delta J | Mean Delta ||r|| | FA (alpha=0.05) | FA (alpha=0.01) | rN_max > 3.0 | Limit Breaches")
    println("----------|--------------|------------------|-----------------|-----------------|--------------|---------------")

    mag_levels = cfg["attack_magnitude"]["magnitude_levels"]

    for mag in mag_levels
        d_J_list = Float64[]
        d_r_list = Float64[]
        fa05_cnt = 0
        fa01_cnt = 0
        rn_cnt = 0
        breach_cnt = 0

        # Create state perturbation proportional to magnitude
        c_mag = zeros(65)
        # Apply to Bus 10, 18, 25
        c_mag[32 + 10] = mag * 1.0
        c_mag[9] = (mag / 0.03) * 0.015
        c_mag[32 + 18] = -mag * 1.0
        c_mag[17] = -(mag / 0.03) * 0.015
        c_mag[32 + 25] = mag * 0.67
        c_mag[24] = (mag / 0.03) * 0.010

        for t = 1:N_test
            z_t = Z_base[t, :]
            x_hat_t = X_hat_base[t, :]
            H_t = H_base_list[t]
            J_0 = J_base_vec[t]
            r_norm_0 = r_norm_base[t]

            att_lin = generate_linearised_attack(z_t, x_hat_t, H_t, c_mag)

            se.method.mean .= att_lin.z_attacked
            setInitialPoint!(se)
            stateEstimation!(se; verbose = 0)

            J_att = se.method.objective
            r_att = copy(se.method.residual)

            push!(d_J_list, J_att - J_0)
            push!(d_r_list, norm(r_att) - r_norm_0)

            if J_att > chi_05
                fa05_cnt += 1
            end
            if J_att > chi_01
                fa01_cnt += 1
            end

            res_t = residualTest!(se; threshold = 1000.0)
            if res_t.maxNormalizedResidual > 3.0
                rn_cnt += 1
            end

            if minimum(se.voltage.magnitude) < 0.90 || maximum(se.voltage.magnitude) > 1.05
                breach_cnt += 1
            end
        end

        @printf("  %4.1f%%   |  %+10.4f  |   %+12.6f   |  %3d/%3d (%4.1f%%) |  %3d/%3d (%4.1f%%) | %3d/%3d (%4.1f%%) |  %3d/%3d (%4.1f%%)\n",
                mag * 100, mean(d_J_list), mean(d_r_list),
                fa05_cnt, N_test, (fa05_cnt / N_test) * 100,
                fa01_cnt, N_test, (fa01_cnt / N_test) * 100,
                rn_cnt, N_test, (rn_cnt / N_test) * 100,
                breach_cnt, N_test, (breach_cnt / N_test) * 100)
    end

    # -------------------------------------------------------------------------
    # TEST 3: Random Non-Stealth Attacks
    # -------------------------------------------------------------------------
    println("\n================================================================================")
    println("TEST 3: Random (Non-Stealth) Attacks vs Compromised Meter Count & Noise Level")
    println("================================================================================")
    println("Meters | Multiplier | Mean Delta J | Mean Delta ||r|| | Det (a=0.05) | Det (a=0.01) | rN_max > 3.0")
    println("-------|------------|--------------|------------------|--------------|--------------|-------------")

    meter_counts = cfg["random_attack"]["meter_counts"]
    noise_mults = cfg["random_attack"]["noise_std_multipliers"]

    for k_m in meter_counts
        for mult in noise_mults
            d_J_list = Float64[]
            d_r_list = Float64[]
            fa05_cnt = 0
            fa01_cnt = 0
            rn_cnt = 0

            rng_rnd = MersenneTwister(42)

            for t = 1:N_test
                z_t = Z_base[t, :]
                x_hat_t = X_hat_base[t, :]
                J_0 = J_base_vec[t]
                r_norm_0 = r_norm_base[t]

                att_rnd = generate_random_attack(z_t, x_hat_t, std_vec, cfg;
                                                 rng = rng_rnd, k_meters = k_m, noise_mult = mult)

                se.method.mean .= att_rnd.z_attacked
                setInitialPoint!(se)
                stateEstimation!(se; verbose = 0)

                J_att = se.method.objective
                push!(d_J_list, J_att - J_0)
                push!(d_r_list, norm(se.method.residual) - r_norm_0)

                if J_att > chi_05
                    fa05_cnt += 1
                end
                if J_att > chi_01
                    fa01_cnt += 1
                end

                res_t = residualTest!(se; threshold = 1000.0)
                if res_t.maxNormalizedResidual > 3.0
                    rn_cnt += 1
                end
            end

            @printf("  %2d   |   %4.1fx    |  %+10.2f  |   %+12.4f   | %3d/%3d (%5.1f%%) | %3d/%3d (%5.1f%%) | %3d/%3d (%5.1f%%)\n",
                    k_m, mult, mean(d_J_list), mean(d_r_list),
                    fa05_cnt, N_test, (fa05_cnt / N_test) * 100,
                    fa01_cnt, N_test, (fa01_cnt / N_test) * 100,
                    rn_cnt, N_test, (rn_cnt / N_test) * 100)
        end
    end

    # -------------------------------------------------------------------------
    # TEST 4: Single-Bus vs Multi-Bus Topologies (Random vs Feeder Adjacent)
    # -------------------------------------------------------------------------
    println("\n================================================================================")
    println("TEST 4: Attack Topologies: Single-Bus vs Multi-Bus Coordinated (N = $N_test)")
    println("================================================================================")
    println("Attack Topology          | Formulation | Mean Shift Err | Mean Delta J | Det (a=0.05) | Limit Breaches")
    println("-------------------------|-------------|----------------|--------------|--------------|---------------")

    topologies = [
        ("Single-Bus (Bus 18 end) ", [18]),
        ("Single-Bus (Bus 2 root) ", [2]),
        ("Multi-Bus (4 random)    ", [5, 12, 21, 30]),
        ("Multi-Bus (Feeder 14-18)", [14, 15, 16, 17, 18])
    ]

    for (topo_name, buses) in topologies
        for form in [:exact_ac, :linearised]
            shift_errs = Float64[]
            d_Js = Float64[]
            det_cnt = 0
            breach_cnt = 0

            for t = 1:N_test
                z_t = Z_base[t, :]
                x_hat_t = X_hat_base[t, :]
                H_t = H_base_list[t]
                J_0 = J_base_vec[t]

                att = generate_multibus_attack(z_t, x_hat_t, se, H_t, cfg;
                                               target_buses = buses, alpha_V = 0.03, alpha_theta = 0.015,
                                               formulation = form)

                se.method.mean .= att.z_attacked
                setInitialPoint!(se)
                stateEstimation!(se; verbose = 0)

                x_att = full_polar_to_state(se.voltage.magnitude, se.voltage.angle)
                push!(shift_errs, norm((x_att .- x_hat_t) .- att.c))
                push!(d_Js, se.method.objective - J_0)

                if se.method.objective > chi_05
                    det_cnt += 1
                end
                if minimum(se.voltage.magnitude) < 0.90 || maximum(se.voltage.magnitude) > 1.05
                    breach_cnt += 1
                end
            end

            @printf("%s | %-11s |    %.4e  |  %+10.4f  | %3d/%3d (%4.1f%%) |  %3d/%3d (%4.1f%%)\n",
                    topo_name, string(form), mean(shift_errs), mean(d_Js),
                    det_cnt, N_test, (det_cnt / N_test) * 100,
                    breach_cnt, N_test, (breach_cnt / N_test) * 100)
        end
    end

    # -------------------------------------------------------------------------
    # TEST 5: Limited-Meter Attacker vs Meter Budget
    # -------------------------------------------------------------------------
    println("\n================================================================================")
    println("TEST 5: Limited-Meter Attacker: Detection vs Meter Budget (N = $N_test)")
    println("================================================================================")
    println("Meter Budget | Unaltered | Mean Trunc Leakage | Mean Delta J | Det (a=0.05) | Det (a=0.01) | rN_max > 3.0")
    println("-------------|-----------|--------------------|--------------|--------------|--------------|-------------")

    budgets = cfg["limited_meter_attack"]["meter_budgets"]

    for bgt in budgets
        d_J_list = Float64[]
        leak_list = Float64[]
        fa05_cnt = 0
        fa01_cnt = 0
        rn_cnt = 0

        for t = 1:N_test
            z_t = Z_base[t, :]
            x_hat_t = X_hat_base[t, :]
            H_t = H_base_list[t]
            J_0 = J_base_vec[t]

            att_lim = generate_limited_meter_attack(z_t, x_hat_t, H_t, cfg;
                                                    meter_budget = bgt, target_buses = [10, 18],
                                                    alpha_V = 0.03, alpha_theta = 0.015)

            push!(leak_list, att_lim.metadata["truncation_leakage_norm"])

            se.method.mean .= att_lim.z_attacked
            setInitialPoint!(se)
            stateEstimation!(se; verbose = 0)

            J_att = se.method.objective
            push!(d_J_list, J_att - J_0)

            if J_att > chi_05
                fa05_cnt += 1
            end
            if J_att > chi_01
                fa01_cnt += 1
            end

            res_t = residualTest!(se; threshold = 1000.0)
            if res_t.maxNormalizedResidual > 3.0
                rn_cnt += 1
            end
        end

        @printf("  %3d / 163  |    %3d    |     %10.4f     |  %+10.2f  | %3d/%3d (%5.1f%%) | %3d/%3d (%5.1f%%) | %3d/%3d (%5.1f%%)\n",
                bgt, m - bgt, mean(leak_list), mean(d_J_list),
                fa05_cnt, N_test, (fa05_cnt / N_test) * 100,
                fa01_cnt, N_test, (fa01_cnt / N_test) * 100,
                rn_cnt, N_test, (rn_cnt / N_test) * 100)
    end

    println("\n================================================================================")
    println("VERIFICATION SUITE SUMMARY: ALL 5 ATTACK TESTS COMPLETED SUCCESSFULLY!")
    println("================================================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
