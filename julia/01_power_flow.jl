"""
julia/01_power_flow.jl

Task 2: Load IEEE 33-bus system (case33bw) in JuliaGrid and run ONE AC Newton-Raphson power flow.
"""

using JuliaGrid
using Printf

function main()
    case_path = normpath(joinpath(@__DIR__, "..", "data", "cases", "case33bw.m"))
    if !isfile(case_path)
        error("Case file not found at: $case_path")
    end

    println("================================================================================")
    println("JuliaGrid IEEE 33-Bus System AC Power Flow")
    println("================================================================================")
    println("Loading case file: $case_path")

    # 1. Load the power system case
    system = powerSystem(case_path)
    base_mva = system.base.power.value * system.base.power.prefix * 1e-6

    # 2. Build the AC model
    acModel!(system)

    # 3. Setup and run Newton-Raphson AC power flow
    analysis = newtonRaphson(system)
    powerFlow!(analysis; verbose = 1)
    power!(analysis)

    # Extract metrics
    num_buses = system.bus.number
    num_branches = system.branch.number
    in_service_branches = count(==(1), system.branch.layout.status)

    total_load_p = sum(system.bus.demand.active) * base_mva
    total_load_q = sum(system.bus.demand.reactive) * base_mva

    iterations = analysis.method.iteration
    converged = iterations < 100

    min_v, min_v_idx = findmin(analysis.voltage.magnitude)
    max_v, max_v_idx = findmax(analysis.voltage.magnitude)
    slack_idx = system.bus.layout.slack
    slack_v = analysis.voltage.magnitude[slack_idx]

    # Branch total losses (sum of from-bus and to-bus powers for all lines)
    p_loss = sum(analysis.power.from.active .+ analysis.power.to.active) * base_mva
    q_loss = sum(analysis.power.from.reactive .+ analysis.power.to.reactive) * base_mva

    gen_p = sum(analysis.power.generator.active) * base_mva
    gen_q = sum(analysis.power.generator.reactive) * base_mva

    println("\n================================================================================")
    println("AC Power Flow Verification Results")
    println("================================================================================")
    @printf("Number of buses:               %d\n", num_buses)
    @printf("Number of branches:            %d (%d in-service, %d tie switches open)\n",
            num_branches, in_service_branches, num_branches - in_service_branches)
    @printf("Base MVA:                      %.2f MVA\n", base_mva)
    @printf("Total load active power:       %.4f MW\n", total_load_p)
    @printf("Total load reactive power:     %.4f MVAr\n", total_load_q)
    @printf("Convergence status:            %s\n", converged ? "Converged" : "Failed")
    @printf("Number of iterations:          %d\n", iterations)
    @printf("Slack bus (Bus %d) voltage:    %.4f pu\n", slack_idx, slack_v)
    @printf("Minimum voltage magnitude:     %.6f pu (Bus %d)\n", min_v, min_v_idx)
    @printf("Maximum voltage magnitude:     %.6f pu (Bus %d)\n", max_v, max_v_idx)
    @printf("Total active losses:           %.6f MW (%.3f kW)\n", p_loss, p_loss * 1000.0)
    @printf("Total reactive losses:         %.6f MVAr (%.3f kVAr)\n", q_loss, q_loss * 1000.0)
    @printf("Total generator active output: %.6f MW\n", gen_p)
    @printf("Total generator reactive:      %.6f MVAr\n", gen_q)
    println("================================================================================")

    return (
        num_buses = num_buses,
        num_branches = num_branches,
        total_load_p = total_load_p,
        total_load_q = total_load_q,
        converged = converged,
        iterations = iterations,
        min_v = min_v,
        min_v_idx = min_v_idx,
        max_v = max_v,
        max_v_idx = max_v_idx,
        p_loss = p_loss,
        q_loss = q_loss
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
