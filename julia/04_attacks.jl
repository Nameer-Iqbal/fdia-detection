"""
julia/04_attacks.jl

Task 5 & 6: False Data Injection (FDI) Attack Generation Module for AC Power Systems.
Contains functions for:
 - Exact AC stealth attacks: a = h(x_hat + c) - h(x_hat)
 - Linearised stealth attacks: a = H c
 - Random (non-stealth) attacks: unstructured noise on meter subsets
 - Single-bus attacks: c restricted to a single bus
 - Multi-bus coordinated attacks: random and adjacent-along-feeder variants
 - Limited-meter attacks: least-squares residual-space projection on meter subset S
 - In-Range vs Unconstrained attack families (preventing voltage shortcut breaches)
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
       generate_limited_meter_ls_attack,
       check_voltage_range,
       state_to_full_polar,
       full_polar_to_state,
       V_NORMAL_MIN,
       V_NORMAL_MAX

# Normal operating voltage envelope from N=5000 normal dataset
const V_NORMAL_MIN = 0.9008926
const V_NORMAL_MAX = 1.0032777

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
    check_voltage_range(V_est; v_min=V_NORMAL_MIN, v_max=V_NORMAL_MAX)

Checks whether any bus voltage lies outside the empirical normal operating range.
Returns (breach::Bool, min_v::Float64, max_v::Float64).
"""
function check_voltage_range(V_est::AbstractVector{Float64}; v_min::Float64 = V_NORMAL_MIN, v_max::Float64 = V_NORMAL_MAX)
    min_v = minimum(V_est)
    max_v = maximum(V_est)
    breach = (min_v < v_min) || (max_v > v_max)
    return (breach = breach, min_v = min_v, max_v = max_v)
end

"""
    sample_c_element(v_hat_i, alpha_V, alpha_theta; family=:unconstrained, rng=Random.GLOBAL_RNG)

Helper to draw state perturbation sign and magnitude for bus i under :in_range or :unconstrained family.
"""
function sample_c_element(
    v_hat_i::Float64,
    alpha_V::Float64,
    alpha_theta::Float64;
    family::Symbol = :unconstrained,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    v_min::Float64 = V_NORMAL_MIN,
    v_max::Float64 = V_NORMAL_MAX
)
    sign_v = rand(rng, (-1.0, 1.0))
    sign_th = rand(rng, (-1.0, 1.0))

    cV = sign_v * alpha_V * 1.0
    cTh = sign_th * alpha_theta
    was_resampled = false

    if family == :in_range
        # Target voltage v_target = v_hat_i + cV
        # Must satisfy v_min <= v_hat_i + cV <= v_max
        if (v_hat_i + cV < v_min) || (v_hat_i + cV > v_max)
            # Try flipping the sign
            if (v_hat_i - cV >= v_min) && (v_hat_i - cV <= v_max)
                cV = -cV
                sign_v = -sign_v
                was_resampled = true
            else
                # If both signs violate, scale down magnitude to fit boundary safely
                margin_up = max(0.0, v_max - v_hat_i)
                margin_down = max(0.0, v_hat_i - v_min)
                if margin_up > margin_down
                    cV = 0.95 * margin_up
                    sign_v = 1.0
                else
                    cV = -0.95 * margin_down
                    sign_v = -1.0
                end
                was_resampled = true
            end
        end
    end

    return (cV = cV, cTh = cTh, sign_v = sign_v, sign_th = sign_th, resampled = was_resampled)
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
    meter_indices = sort(randperm(rng, m)[1:k_meters])
    a = zeros(m)
    signs = zeros(m)
    for idx in meter_indices
        s = rand(rng, (-1.0, 1.0))
        signs[idx] = s
        a[idx] = s * noise_mult * std_vec[idx]
    end
    z_attacked = z .+ a
    c = zeros(length(x_hat))

    mask = zeros(Int, m)
    mask[meter_indices] .= 1

    meta = Dict(
        "attack_type" => "random",
        "family" => "unconstrained",
        "compromised_meters" => meter_indices,
        "n_attacked_meters" => k_meters,
        "attacked_meter_mask" => mask,
        "noise_multiplier" => noise_mult,
        "a_norm" => norm(a),
        "c_norm" => 0.0,
        "magnitude" => noise_mult,
        "sign" => 0,
        "attacked_buses" => Int[]
    )

    return (z_attacked = z_attacked, a = a, c = c, metadata = meta)
end

"""
    generate_single_bus_attack(z, x_hat, se, H, config; bus=18, alpha_V=0.03, alpha_theta=0.015, formulation=:exact_ac, family=:unconstrained, rng=Random.GLOBAL_RNG)

Generates a single-bus attack perturbing states associated with a single physical bus, with family support.
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
    family::Symbol = :unconstrained,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    v_min::Float64 = V_NORMAL_MIN,
    v_max::Float64 = V_NORMAL_MAX
)
    V_hat, _ = state_to_full_polar(x_hat)
    v_base_bus = V_hat[bus]

    draw = sample_c_element(v_base_bus, alpha_V, alpha_theta; family = family, rng = rng, v_min = v_min, v_max = v_max)

    c = zeros(65)
    c[32 + bus] = draw.cV
    if bus > 1
        c[bus - 1] = draw.cTh
    end

    if formulation == :exact_ac
        V_tar, th_tar = state_to_full_polar(x_hat .+ c)
        a_test = eval_ac_measurement(se, V_tar, th_tar) .- eval_ac_measurement(se, V_hat, state_to_full_polar(x_hat)[2])
    else
        a_test = H * c
    end

    # Compromised meter mask (non-zero attack entries)
    m = length(z)
    mask = [abs(a_test[i]) > 1e-6 ? 1 : 0 for i in 1:m]
    n_att = count(==(1), mask)

    meta = Dict(
        "attack_type" => string(formulation, "_single_bus"),
        "family" => string(family),
        "target_buses" => [bus],
        "attacked_buses" => [bus],
        "alpha_V" => alpha_V,
        "alpha_theta" => alpha_theta,
        "magnitude" => alpha_V,
        "sign" => Int(draw.sign_v),
        "formulation" => string(formulation),
        "resampled" => draw.resampled,
        "n_attacked_meters" => n_att,
        "attacked_meter_mask" => mask
    )

    if formulation == :exact_ac
        return generate_exact_ac_attack(z, x_hat, se, c; metadata = meta)
    else
        return generate_linearised_attack(z, x_hat, H, c; metadata = meta)
    end
end

"""
    generate_multibus_attack(z, x_hat, se, H, config; target_buses=[14, 15, 16, 17, 18], alpha_V=0.03, alpha_theta=0.015, formulation=:exact_ac, family=:unconstrained, rng=Random.GLOBAL_RNG)

Generates a multi-bus coordinated attack across target buses, with family support.
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
    family::Symbol = :unconstrained,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    v_min::Float64 = V_NORMAL_MIN,
    v_max::Float64 = V_NORMAL_MAX
)
    V_hat, _ = state_to_full_polar(x_hat)
    c = zeros(65)
    resample_any = false
    overall_sign = 0

    for b in target_buses
        draw = sample_c_element(V_hat[b], alpha_V, alpha_theta; family = family, rng = rng, v_min = v_min, v_max = v_max)
        c[32 + b] = draw.cV
        if b > 1
            c[b - 1] = draw.cTh
        end
        if draw.resampled
            resample_any = true
        end
        if overall_sign == 0
            overall_sign = Int(draw.sign_v)
        end
    end

    if formulation == :exact_ac
        V_tar, th_tar = state_to_full_polar(x_hat .+ c)
        a_test = eval_ac_measurement(se, V_tar, th_tar) .- eval_ac_measurement(se, V_hat, state_to_full_polar(x_hat)[2])
    else
        a_test = H * c
    end

    m = length(z)
    mask = [abs(a_test[i]) > 1e-6 ? 1 : 0 for i in 1:m]
    n_att = count(==(1), mask)

    meta = Dict(
        "attack_type" => string(formulation, "_multi_bus"),
        "family" => string(family),
        "target_buses" => target_buses,
        "attacked_buses" => target_buses,
        "num_target_buses" => length(target_buses),
        "alpha_V" => alpha_V,
        "alpha_theta" => alpha_theta,
        "magnitude" => alpha_V,
        "sign" => overall_sign,
        "formulation" => string(formulation),
        "resampled" => resample_any,
        "n_attacked_meters" => n_att,
        "attacked_meter_mask" => mask
    )

    if formulation == :exact_ac
        return generate_exact_ac_attack(z, x_hat, se, c; metadata = meta)
    else
        return generate_linearised_attack(z, x_hat, H, c; metadata = meta)
    end
end

"""
    generate_limited_meter_attack(z, x_hat, H, config; meter_budget=20, target_buses=[10, 18], alpha_V=0.03, alpha_theta=0.015, rng=Random.GLOBAL_RNG)

Legacy footprint-covering attack: truncates ideal attack vector a = H*c to the top meter_budget largest entries.
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
    perm_largest = sortperm(abs.(a_ideal), rev=true)
    compromised_set = perm_largest[1:min(meter_budget, m)]

    a_truncated = zeros(m)
    for idx in compromised_set
        a_truncated[idx] = a_ideal[idx]
    end

    z_attacked = z .+ a_truncated

    mask = zeros(Int, m)
    mask[compromised_set] .= 1

    meta = Dict(
        "attack_type" => "limited_meter_legacy",
        "meter_budget" => meter_budget,
        "target_buses" => target_buses,
        "attacked_buses" => target_buses,
        "unaltered_meters" => m - length(compromised_set),
        "truncation_leakage_norm" => norm(a_ideal .- a_truncated),
        "a_norm" => norm(a_truncated),
        "c_norm" => norm(c),
        "n_attacked_meters" => length(compromised_set),
        "attacked_meter_mask" => mask,
        "magnitude" => alpha_V,
        "sign" => 1
    )

    return (z_attacked = z_attacked, a = a_truncated, c = c, metadata = meta)
end

"""
    generate_limited_meter_ls_attack(z, x_hat, H, W, config; meter_subset=Int[], M=20, target_buses=[10, 18], alpha_V=0.03, alpha_theta=0.015, family=:unconstrained, lambda=1e5, rng=Random.GLOBAL_RNG)

Redesigned limited-meter attack solving the least-squares residual-space projection problem:
  min_c 1/2 c^T Q c + lambda/2 ||c - c0||^2
  where Q = H_bar^T (I - K)_{:, bar}^T W (I - K)_{:, bar} H_bar
"""
function generate_limited_meter_ls_attack(
    z::AbstractVector{Float64},
    x_hat::AbstractVector{Float64},
    H::AbstractMatrix{Float64},
    W::AbstractMatrix{Float64},
    config::Dict;
    meter_subset::Vector{Int} = Int[],
    M::Int = 20,
    target_buses::Vector{Int} = [10, 18],
    alpha_V::Float64 = 0.03,
    alpha_theta::Float64 = 0.015,
    family::Symbol = :unconstrained,
    lambda::Float64 = 1e5,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    v_min::Float64 = V_NORMAL_MIN,
    v_max::Float64 = V_NORMAL_MAX
)
    m = length(z)
    n = size(H, 2)

    # Determine compromised meter subset S
    if isempty(meter_subset)
        S = sort(randperm(rng, m)[1:min(M, m)])
    else
        S = sort(copy(meter_subset))
    end
    M_actual = length(S)
    S_bar = setdiff(1:m, S)

    # Construct target shift c0
    V_hat, _ = state_to_full_polar(x_hat)
    c0 = zeros(n)
    overall_sign = 0
    resample_any = false

    for b in target_buses
        draw = sample_c_element(V_hat[b], alpha_V, alpha_theta; family = family, rng = rng, v_min = v_min, v_max = v_max)
        c0[32 + b] = draw.cV
        if b > 1
            c0[b - 1] = draw.cTh
        end
        if draw.resampled
            resample_any = true
        end
        if overall_sign == 0
            overall_sign = Int(draw.sign_v)
        end
    end

    G = transpose(H) * W * H
    G_inv = inv(Matrix(G))

    # Optimization
    if M_actual == m
        c_opt = copy(c0)
        a = H * c_opt
        leakage_J = 0.0
    else
        # Projection matrix K = H * G^-1 * H^T * W
        # Residual of injection [H_S c; 0]:
        # Dr = (I - K)[:, S] * H_S * c
        A_res = (I - H * G_inv * transpose(H) * W)[:, S] * H[S, :]
        Q = Matrix(transpose(A_res) * W * A_res)

        # Normal equations: (Q + lambda * I) c_opt = lambda * c0
        c_opt = (Q + lambda * I) \ (lambda * c0)

        a = zeros(m)
        a[S] = H[S, :] * c_opt
    end

    z_attacked = z .+ a

    # Linear model achieved shift
    c_achieved = G \ (transpose(H) * W * a)
    Dr = a - H * c_achieved
    leakage_J = transpose(Dr) * W * Dr

    mask = zeros(Int, m)
    mask[S] .= 1

    meta = Dict(
        "attack_type" => "limited_meter",
        "family" => string(family),
        "target_buses" => target_buses,
        "attacked_buses" => target_buses,
        "meter_budget" => M_actual,
        "compromised_meters" => S,
        "n_attacked_meters" => M_actual,
        "attacked_meter_mask" => mask,
        "target_c" => c0,
        "achieved_c" => c_achieved,
        "c_opt" => c_opt,
        "target_c_norm" => norm(c0),
        "achieved_c_norm" => norm(c_achieved),
        "shift_ratio" => norm(c_achieved) / (norm(c0) + 1e-12),
        "cos_sim" => dot(c_achieved, c0) / (norm(c_achieved) * norm(c0) + 1e-12),
        "linear_leakage_J" => leakage_J,
        "magnitude" => alpha_V,
        "sign" => overall_sign,
        "resampled" => resample_any
    )

    return (z_attacked = z_attacked, a = a, c = c_opt, metadata = meta)
end

end # module FDIAttacks
