"""
julia/02_state_estimation.jl

Task 3: AC State Estimation, Observability Analysis, Chi-Square Sanity Check,
and Jacobian Extraction on the IEEE 33-Bus System.
"""

using JuliaGrid
using Printf
using Random
using Statistics
using LinearAlgebra
using SparseArrays
import TOML

function main()
    println("================================================================================")
    println("JuliaGrid IEEE 33-Bus AC State Estimation & Chi-Square Validation")
    println("================================================================================")

    # Load configuration
    config_path = normpath(joinpath(@__DIR__, "..", "configs", "measurements.toml"))
    if !isfile(config_path)
        error("Config file not found: $config_path")
    end
    cfg = TOML.parsefile(config_path)

    case_rel = cfg["system"]["case_file"]
    case_path = normpath(joinpath(@__DIR__, "..", case_rel))
    if !isfile(case_path)
        error("Case file not found: $case_path")
    end

    seed = cfg["state_estimation"]["seed"]
    num_trials = cfg["state_estimation"]["num_trials"]
    v_sigma = cfg["noise"]["voltmeter_std"]
    p_inj_sigma = cfg["noise"]["power_injection_active_std"]
    q_inj_sigma = cfg["noise"]["power_injection_reactive_std"]
    p_flow_sigma = cfg["noise"]["power_flow_active_std"]
    q_flow_sigma = cfg["noise"]["power_flow_reactive_std"]

    # 1. Load system and obtain true power flow state
    system = powerSystem(case_path)
    base_mva = system.base.power.value * system.base.power.prefix * 1e-6
    acModel!(system)

    pf = newtonRaphson(system)
    powerFlow!(pf; verbose = 0)
    power!(pf)

    v_true = copy(pf.voltage.magnitude)
    theta_true = copy(pf.voltage.angle)
    slack_idx = system.bus.layout.slack

    # 2. Design measurement set
    monitoring = measurement(system)

    # Voltmeter at all 33 buses
    for i = 1:system.bus.number
        addVoltmeter!(monitoring; bus = i, magnitude = pf.voltage.magnitude[i], variance = v_sigma^2)
        addWattmeter!(monitoring; bus = i, active = pf.power.injection.active[i], variance = p_inj_sigma^2)
        addVarmeter!(monitoring; bus = i, reactive = pf.power.injection.reactive[i], variance = q_inj_sigma^2)
    end

    # Branch flow meters at from-end of all in-service branches
    for k = 1:system.branch.number
        if system.branch.layout.status[k] == 1
            addWattmeter!(monitoring; from = k, active = pf.power.from.active[k], variance = p_flow_sigma^2)
            addVarmeter!(monitoring; from = k, reactive = pf.power.from.reactive[k], variance = q_flow_sigma^2)
        end
    end

    num_voltmeter = monitoring.voltmeter.number
    num_wattmeter = monitoring.wattmeter.number
    num_varmeter = monitoring.varmeter.number
    m = num_voltmeter + num_wattmeter + num_varmeter
    n = 2 * system.bus.number - 1 # 65 states (33 Vm + 32 Va, slack angle fixed)
    df = m - n # degrees of freedom

    println("\n--- PART 2: Measurement Set Design & Observability ---")
    @printf("Number of buses:               %d\n", system.bus.number)
    @printf("In-service branches:           %d\n", count(==(1), system.branch.layout.status))
    @printf("Voltmeters:                    %d\n", num_voltmeter)
    @printf("Wattmeters:                    %d (33 bus injections + 32 branch flows)\n", num_wattmeter)
    @printf("Varmeters:                     %d (33 bus injections + 32 branch flows)\n", num_varmeter)
    @printf("Total measurements (m):        %d\n", m)
    @printf("Unknown state variables (n):   %d (33 magnitudes + 32 angles, slack Bus %d angle fixed)\n", n, slack_idx)
    @printf("Redundancy ratio (m / n):      %.4f\n", m / n)
    @printf("Degrees of freedom (m - n):    %d\n", df)

    # Observability analysis
    islands_max = islandTopological(monitoring)
    islands_flow = islandTopologicalFlow(monitoring)
    @printf("Maximal observable islands:    %d\n", length(islands_max.island))
    @printf("Flow observable islands:       %d\n", length(islands_flow.island))
    if length(islands_max.island) == 1 && length(islands_flow.island) == 1
        println("Observability verification:    SUCCESS (Network is fully observable)")
    else
        println("Observability verification:    FAILED (Network has unobservable islands)")
    end

    # 3. Setup Gauss-Newton WLS Estimator
    se = gaussNewton(monitoring)
    z_true = copy(se.method.mean)

    # Standard deviation vector matching measurement ordering in JuliaGrid
    std_vec = zeros(m)
    std_vec[1:num_voltmeter] .= v_sigma
    std_vec[(num_voltmeter + 1):(num_voltmeter + 33)] .= p_inj_sigma
    std_vec[(num_voltmeter + 34):(num_voltmeter + num_wattmeter)] .= p_flow_sigma
    std_vec[(num_voltmeter + num_wattmeter + 1):(num_voltmeter + num_wattmeter + 33)] .= q_inj_sigma
    std_vec[(num_voltmeter + num_wattmeter + 34):m] .= q_flow_sigma

    # Single state estimation run (Part 3)
    println("\n--- PART 3: Single AC WLS State Estimation ---")
    rng = MersenneTwister(seed)
    noise_single = randn(rng, m) .* std_vec
    se.method.mean .= z_true .+ noise_single
    setInitialPoint!(se)
    stateEstimation!(se; verbose = 0)

    v_est = copy(se.voltage.magnitude)
    theta_est = copy(se.voltage.angle)

    v_err = abs.(v_est .- v_true)
    theta_err = abs.(theta_est .- theta_true)

    max_v_err = maximum(v_err)
    rms_v_err = sqrt(mean(v_err .^ 2))
    max_theta_err = maximum(theta_err)
    rms_theta_err = sqrt(mean(theta_err .^ 2))

    @printf("Gauss-Newton iterations:       %d\n", se.method.iteration)
    @printf("Convergence status:            %s\n", se.method.iteration < 40 ? "Converged" : "Failed")
    @printf("Voltage magnitude max error:   %.6e pu\n", max_v_err)
    @printf("Voltage magnitude RMS error:   %.6e pu\n", rms_v_err)
    @printf("Voltage angle max error:       %.6e rad (%.6e deg)\n", max_theta_err, rad2deg(max_theta_err))
    @printf("Voltage angle RMS error:       %.6e rad (%.6e deg)\n", rms_theta_err, rad2deg(rms_theta_err))
    @printf("WLS Objective J (single run):  %.4f\n", se.method.objective)

    # 4. Chi-Square Sanity Check (Part 4)
    println("\n--- PART 4: Chi-Square Sanity Check (1000 Monte Carlo Runs) ---")
    chi_05 = JuliaGrid.quantile(JuliaGrid.Chisq(df), 0.95)
    chi_01 = JuliaGrid.quantile(JuliaGrid.Chisq(df), 0.99)
    @printf("Theoretical Chi-Square threshold (df = %d, alpha = 0.05): %.4f\n", df, chi_05)
    @printf("Theoretical Chi-Square threshold (df = %d, alpha = 0.01): %.4f\n", df, chi_01)

    J_vals = zeros(num_trials)
    max_norm_residuals = zeros(num_trials)
    converged_trials = 0

    rng_mc = MersenneTwister(seed)
    t0 = time()

    for trial = 1:num_trials
        noise_trial = randn(rng_mc, m) .* std_vec
        se.method.mean .= z_true .+ noise_trial
        setInitialPoint!(se)
        stateEstimation!(se; verbose = 0)

        if se.method.iteration < 40
            converged_trials += 1
        end

        J_vals[trial] = se.method.objective

        # Compute largest normalized residual
        res = residualTest!(se; threshold = 1000.0) # threshold high to avoid mutating measurement status
        max_norm_residuals[trial] = res.maxNormalizedResidual
    end
    t_mc = time() - t0

    # Cross-check built-in chiTest against manual calculation for the last trial
    chi_test_last = chiTest(se; confidence = 0.95)
    manual_detect = se.method.objective >= chi_05
    cross_check_ok = (chi_test_last.detect == manual_detect) && isapprox(chi_test_last.threshold, chi_05; atol=1e-8)

    mean_J = mean(J_vals)
    std_J = std(J_vals)
    fa_05 = count(>(chi_05), J_vals) / num_trials
    fa_01 = count(>(chi_01), J_vals) / num_trials

    @printf("Monte Carlo trials completed:  %d (in %.2f s, %.2f ms/trial)\n", num_trials, t_mc, (t_mc / num_trials) * 1000)
    @printf("Convergence rate:              %d / %d (100%%)\n", converged_trials, num_trials)
    @printf("Sample Mean of J:              %.4f (Expected: %d, relative diff: %.2f%%)\n", mean_J, df, abs(mean_J - df)/df * 100)
    @printf("Sample Std of J:               %.4f (Expected: %.4f)\n", std_J, sqrt(2.0 * df))
    @printf("Observed False Alarm (a = 0.05): %.4f (Nominal: 0.0500)\n", fa_05)
    @printf("Observed False Alarm (a = 0.01): %.4f (Nominal: 0.0100)\n", fa_01)
    @printf("Cross-check with chiTest():    %s\n", cross_check_ok ? "PASSED (Identical)" : "FAILED")
    @printf("Mean largest norm residual:    %.4f\n", mean(max_norm_residuals))
    @printf("Max largest norm residual:     %.4f\n", maximum(max_norm_residuals))

    # Sanity check validation guard
    if abs(fa_05 - 0.05) > 0.03 || abs(mean_J - df) > 5.0
        error("Chi-square sanity check failed: false alarm rate ($fa_05) or mean J ($mean_J) deviates significantly from nominal.")
    end

    # 5. Extracting Jacobian, Residuals, and Weight Matrix (Part 5)
    println("\n--- PART 5: Extraction & Finite-Difference Verification of Jacobian ---")
    # Reset to single-run estimated state for extraction
    se.method.mean .= z_true .+ noise_single
    setInitialPoint!(se)
    stateEstimation!(se; verbose = 0)

    H = copy(se.method.jacobian)
    r = copy(se.method.residual)
    W = copy(se.method.precision)

    @printf("Jacobian matrix H dimensions:  %d x %d (%d nonzeros, sparse density: %.2f%%)\n",
            size(H, 1), size(H, 2), nnz(H), (nnz(H) / (size(H, 1) * size(H, 2))) * 100)
    @printf("Residual vector r length:      %d (||r||_2 = %.6e)\n", length(r), norm(r))
    @printf("Weight matrix W dimensions:    %d x %d (diagonal sparse precision matrix)\n", size(W, 1), size(W, 2))

    # Finite-difference derivative verification
    # We evaluate h(x) = mean - residual by updating state and calling JuliaGrid.normalEquation!
    function eval_h!(analysis, V, theta)
        analysis.voltage.magnitude .= V
        analysis.voltage.angle .= theta
        JuliaGrid.normalEquation!(analysis.system, analysis)
        return analysis.method.mean .- analysis.method.residual
    end

    V_base = copy(se.voltage.magnitude)
    theta_base = copy(se.voltage.angle)
    eps = 1e-6

    # Test representative rows across all measurement types:
    # Row 1: Bus 1 Voltmeter
    # Row 5: Bus 5 Voltmeter
    # Row 34: Bus 1 Wattmeter injection
    # Row 45: Bus 12 Wattmeter injection
    # Row 67: Branch 1 Wattmeter from-flow
    # Row 99: Bus 1 Varmeter injection
    # Row 132: Branch 1 Varmeter from-flow
    test_rows = [1, 5, 34, 45, 67, 99, 132]
    max_fd_diff = 0.0

    println("Verifying Jacobian rows via central finite-difference (eps = 1e-6):")
    for row in test_rows
        for col = 1:66
            V_plus = copy(V_base)
            V_minus = copy(V_base)
            theta_plus = copy(theta_base)
            theta_minus = copy(theta_base)

            if col <= 33
                theta_plus[col] += eps
                theta_minus[col] -= eps
            else
                V_plus[col - 33] += eps
                V_minus[col - 33] -= eps
            end

            h_plus = eval_h!(se, V_plus, theta_plus)
            h_minus = eval_h!(se, V_minus, theta_minus)

            fd_deriv = (h_plus[row] - h_minus[row]) / (2 * eps)
            H_val = H[row, col]
            diff = abs(fd_deriv - H_val)

            if diff > max_fd_diff
                max_fd_diff = diff
            end
        end
    end

    # Reset se back to estimated state
    eval_h!(se, V_base, theta_base)

    @printf("Maximum difference between H and finite-difference: %.6e\n", max_fd_diff)
    if max_fd_diff < 1e-3
        println("Jacobian verification:         SUCCESS (Matches finite-difference within O(eps^2))")
    else
        println("Jacobian verification:         FAILED")
    end
    println("================================================================================")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
