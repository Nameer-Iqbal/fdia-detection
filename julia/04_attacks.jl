"""
julia/04_attacks.jl

Task 5: False Data Injection (FDI) Attack Generation Module for AC Power Systems.
Contains functions for:
 - Exact AC stealth attacks: a = h(x_hat + c) - h(x_hat)
 - Linearised stealth attacks: a = H c
 - Random (non-stealth) attacks: unstructured noise on meter subsets
 - Single-bus attacks: c restricted to a single bus
 - Multi-bus coordinated attacks: random and adjacent-along-feeder variants
 - Limited-meter attacks: attacker restricted to modifying a subset of meters
"""

module FDIAttacks

using JuliaGrid
using Random
using LinearAlgebra
using SparseArrays
import TOML

export eval_ac_measurement,
       generate_exact_ac_attack,
       generate_linearised_attack,
       generate_random_attack,
       generate_single_bus_attack,
       generate_multibus_attack,
       generate_limited_meter_attack,
       state_to_full_polar,
       full_polar_to_state

"""
    state_to_full_polar(x, num_buses=33, slack_idx=1)

Converts the 65-element state vector x = [theta_2..theta_33, V_1..V_33]
into full 33-element voltage magnitude and angle vectors (with theta[slack_idx] = 0.0).
"""
function state_to_full_polar(x::AbstractVector{Float64}, num_buses::Int = 33, slack_idx::Int = 1)
    theta_full = zeros(num_buses)
    theta_idx = 1
    for i in 1:num_buses
        if i != slack_idx
            theta_full[i] = x[theta_idx]
            theta_idx += 1
        end
    end
    V_full = copy(x[(num_buses):end])
    return V_full, theta_full
end

"""
    full_polar_to_state(V_full, theta_full, num_buses=33, slack_idx=1)

Converts full 33-element V and theta vectors to 65-element state vector
x = [theta_2..theta_33, V_1..V_33] (excluding slack bus angle).
"""
function full_polar_to_state(V_full::AbstractVector{Float64}, theta_full::AbstractVector{Float64}, num_buses::Int = 33, slack_idx::Int = 1)
    angles = [theta_full[i] for i in 1:num_buses if i != slack_idx]
    return [angles; V_full]
end

"""
    eval_ac_measurement(se, V, theta)

Evaluates the non-linear AC measurement function h(x) given voltage magnitudes V and angles theta.
Reuses JuliaGrid's normalEquation! to compute predicted measurement values mean - residual.
"""
function eval_ac_measurement(analysis::AcStateEstimation, V::AbstractVector{Float64}, theta::AbstractVector{Float64})
    analysis.voltage.magnitude .= V
    analysis.voltage.angle .= theta
    JuliaGrid.normalEquation!(analysis.system, analysis)
    return copy(analysis.method.mean .- analysis.method.residual)
end

"""
    generate_exact_ac_attack(z, x_hat, se, c; metadata=Dict())

Generates an exact AC stealth attack:
  a = h(x_hat + c) - h(x_hat)
  z_attacked = z + a
"""
function generate_exact_ac_attack(
    z::AbstractVector{Float64},
    x_hat::AbstractVector{Float64},
    se::AcStateEstimation,
    c::AbstractVector{Float64};
    metadata::Dict = Dict()
)
    V_hat, theta_hat = state_to_full_polar(x_hat)
    x_target = x_hat .+ c
    V_target, theta_target = state_to_full_polar(x_target)

    h_target = eval_ac_measurement(se, V_target, theta_target)
    h_hat = eval_ac_measurement(se, V_hat, theta_hat)
    a = h_target .- h_hat
    z_attacked = z .+ a

    meta = copy(metadata)
    meta["attack_type"] = "exact_ac"
    meta["c_norm"] = norm(c)
    meta["a_norm"] = norm(a)

    return (z_attacked = z_attacked, a = a, c = c, metadata = meta)
end

"""
    generate_linearised_attack(z, x_hat, H, c; metadata=Dict())

Generates a linearised stealth attack:
  a = H * c
  z_attacked = z + a
where H is the 163 x 65 measurement Jacobian at x_hat.
"""
function generate_linearised_attack(
    z::AbstractVector{Float64},
    x_hat::AbstractVector{Float64},
    H::AbstractMatrix{Float64},
    c::AbstractVector{Float64};
    metadata::Dict = Dict()
)
    a = H * c
    z_attacked = z .+ a

    meta = copy(metadata)
    meta["attack_type"] = "linearised"
    meta["c_norm"] = norm(c)
    meta["a_norm"] = norm(a)

    return (z_attacked = z_attacked, a = a, c = c, metadata = meta)
end

"""
    generate_random_attack(z, x_hat, std_vec, config; rng=Random.GLOBAL_RNG, k_meters=10, noise_mult=10.0)

Generates an unstructured random (non-stealth) attack on a random subset of meters:
  a_i = delta_i for i in S, 0 otherwise
"""
function generate_random_attack(
    z::AbstractVector{Float64},
    x_hat::AbstractVector{Float64},
    std_vec::AbstractVector{Float64},
    config::Dict;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    k_meters::Int = 10,
    noise_mult::Float64 = 10.0
)
    m = length(z)
    meter_indices = randperm(rng, m)[1:k_meters]
    a = zeros(m)
    for idx in meter_indices
        sign_val = rand(rng, (-1.0, 1.0))
        # Perturbation scaled to measurement standard deviation
        a[idx] = sign_val * noise_mult * std_vec[idx]
    end
    z_attacked = z .+ a
    c = zeros(length(x_hat))

    meta = Dict(
        "attack_type" => "random",
        "compromised_meters" => meter_indices,
        "k_meters" => k_meters,
        "noise_multiplier" => noise_mult,
        "a_norm" => norm(a),
        "c_norm" => 0.0
    )

    return (z_attacked = z_attacked, a = a, c = c, metadata = meta)
end

"""
    generate_single_bus_attack(z, x_hat, se, H, config; bus=18, alpha_V=0.03, alpha_theta=0.015, formulation=:exact_ac, rng=Random.GLOBAL_RNG)

Generates a single-bus attack perturbing states associated with a single physical bus.
"""
function generate_single_bus_attack(
    z::AbstractVector{Float64},
    x_hat::AbstractVector{Float64},
    se::AcStateEstimation,
    H::AbstractMatrix{Float64},
    config::Dict;
    bus::Int = 18,
    alpha_V::Float64 = 0.03,
    alpha_theta::Float64 = 0.015,
    formulation::Symbol = :exact_ac,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    c = zeros(65)
    sign_v = rand(rng, (-1.0, 1.0))
    sign_th = rand(rng, (-1.0, 1.0))

    # Bus voltage magnitude state index: 32 + bus
    c[32 + bus] = sign_v * alpha_V * 1.0 # 1.0 nominal pu base

    # Bus voltage angle state index: bus - 1 (if bus != 1, since slack bus 1 angle is fixed)
    if bus > 1
        c[bus - 1] = sign_th * alpha_theta
    end

    meta = Dict(
        "target_bus" => bus,
        "alpha_V" => alpha_V,
        "alpha_theta" => alpha_theta,
        "formulation" => string(formulation)
    )

    if formulation == :exact_ac
        return generate_exact_ac_attack(z, x_hat, se, c; metadata = meta)
    else
        return generate_linearised_attack(z, x_hat, H, c; metadata = meta)
    end
end

"""
    generate_multibus_attack(z, x_hat, se, H, config; target_buses=[14, 15, 16, 17, 18], alpha_V=0.03, alpha_theta=0.015, formulation=:exact_ac, rng=Random.GLOBAL_RNG)

Generates a multi-bus coordinated attack across a specified set of buses.
"""
function generate_multibus_attack(
    z::AbstractVector{Float64},
    x_hat::AbstractVector{Float64},
    se::AcStateEstimation,
    H::AbstractMatrix{Float64},
    config::Dict;
    target_buses::Vector{Int} = [14, 15, 16, 17, 18],
    alpha_V::Float64 = 0.03,
    alpha_theta::Float64 = 0.015,
    formulation::Symbol = :exact_ac,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    c = zeros(65)
    for b in target_buses
        sign_v = rand(rng, (-1.0, 1.0))
        sign_th = rand(rng, (-1.0, 1.0))
        c[32 + b] = sign_v * alpha_V * 1.0
        if b > 1
            c[b - 1] = sign_th * alpha_theta
        end
    end

    meta = Dict(
        "target_buses" => target_buses,
        "num_target_buses" => length(target_buses),
        "alpha_V" => alpha_V,
        "alpha_theta" => alpha_theta,
        "formulation" => string(formulation)
    )

    if formulation == :exact_ac
        return generate_exact_ac_attack(z, x_hat, se, c; metadata = meta)
    else
        return generate_linearised_attack(z, x_hat, H, c; metadata = meta)
    end
end

"""
    generate_limited_meter_attack(z, x_hat, H, config; meter_budget=20, target_buses=[10, 18], alpha_V=0.03, alpha_theta=0.015, rng=Random.GLOBAL_RNG)

Generates an attack where the adversary is restricted to altering at most `meter_budget` meters.
Constructs target state perturbation c and truncates the linear injection a = H * c to the allowed meter set.
"""
function generate_limited_meter_attack(
    z::AbstractVector{Float64},
    x_hat::AbstractVector{Float64},
    H::AbstractMatrix{Float64},
    config::Dict;
    meter_budget::Int = 20,
    target_buses::Vector{Int} = [10, 18],
    alpha_V::Float64 = 0.03,
    alpha_theta::Float64 = 0.015,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    m = length(z)
    c = zeros(65)
    for b in target_buses
        c[32 + b] = alpha_V * 1.0
        if b > 1
            c[b - 1] = alpha_theta
        end
    end

    a_ideal = H * c
    # Identify meters with largest non-zero attack values
    perm_largest = sortperm(abs.(a_ideal), rev=true)
    compromised_set = perm_largest[1:min(meter_budget, m)]

    a_truncated = zeros(m)
    for idx in compromised_set
        a_truncated[idx] = a_ideal[idx]
    end

    z_attacked = z .+ a_truncated

    meta = Dict(
        "attack_type" => "limited_meter",
        "meter_budget" => meter_budget,
        "target_buses" => target_buses,
        "unaltered_meters" => m - length(compromised_set),
        "truncation_leakage_norm" => norm(a_ideal .- a_truncated),
        "a_norm" => norm(a_truncated),
        "c_norm" => norm(c)
    )

    return (z_attacked = z_attacked, a = a_truncated, c = c, metadata = meta)
end

end # module FDIAttacks
