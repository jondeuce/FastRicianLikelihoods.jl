####
#### Rician negative log-likelihood
####

@inline function neglogpdf_rician(x::Real, ν::Real, logσ::Real)
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end
@inline neglogpdf_rician(x::Real, ν::Real) = _neglogpdf_rician(promote(x, ν)...)
@inline ∇neglogpdf_rician(x::Real, ν::Real) = _∇neglogpdf_rician(promote(x, ν)...)
@inline ∇²neglogpdf_rician(x::Real, ν::Real) = _∇²neglogpdf_rician(promote(x, ν)...)
@inline ∇²neglogpdf_rician_with_gradient(x::Real, ν::Real) = _∇²neglogpdf_rician_with_gradient(promote(x, ν)...)
@inline ∇³neglogpdf_rician_with_gradient_and_hessian(x::Real, ν::Real) = _∇³neglogpdf_rician_with_gradient_and_hessian(promote(x, ν)...)

@inline pdf_rician(args...) = exp(-neglogpdf_rician(args...))
@inline ∇pdf_rician(args...) = -exp(-neglogpdf_rician(args...)) .* ∇neglogpdf_rician(args...)

#### Internal methods with strict type signatures (enables dual number overloads with single method)

@inline function _neglogpdf_rician(x::D, ν::D) where {D}
    # Negative Rician log-likelihood `-logp(x | ν, σ = 1)`
    z = x * ν
    T = checkedfloattype(z)
    if z < first(logbesseli0x_branches(T))
        return ((x^2 + ν^2) / 2 - logbesseli0_taylor(z)) - log(x)
    elseif z < last(logbesseli0x_branches(T))
        return ((x - ν)^2 / 2 - logbesseli0x_middle(z)) - log(x)
    else
        return ((x - ν)^2 / 2 - logratio(x, ν) / 2 - logbesseli0x_scaled_tail(z)) + T(log2π) / 2
    end
end

@inline function _∇neglogpdf_rician(x::D, ν::D) where {D}
    # Define the univariate normalized Bessel function `Î₀(z)` for `z = x * ν ≥ 0` as
    #
    #   Î₀(z) = I₀(z) / (exp(z) / √2πz)
    #
    # The negative likelihood is then be written as
    #
    #        -logp(x | ν, σ = 1) = (x - ν)^2 / 2 - log(x / ν) / 2 - logÎ₀(x * ν) + log√2π
    #   ∂/∂x -logp(x | ν, σ = 1) = x - ν - 1 / 2x - ∂/∂x logÎ₀(x * ν)
    #   ∂/∂ν -logp(x | ν, σ = 1) = ν - x + 1 / 2ν - ∂/∂ν logÎ₀(x * ν)
    #
    # All that must be approximated then is `d/dz logÎ₀(z)` where `z = x * ν`:
    #
    #   d/dz logÎ₀(z) =  1/2z + (I₁(z) / I₀(z) - 1)
    #                 ≈ -1/8z^2 - 1/8z^3 - 25/128z^4 - 13/32z^5 - 1073/1024z^6 - 103/32z^7 + 𝒪(1/z^8)   (z >> 1)
    #                 ≈  1/2z - 1 + z/2 - z^3/16 + z^5/96 - 11*z^7/6144 + 𝒪(z^9)                        (z << 1)
    #   ∂/∂x logÎ₀(z) = ν * d/dz logÎ₀(z)
    #   ∂/∂ν logÎ₀(z) = x * d/dz logÎ₀(z)
    #
    # Note: there are really three relevant limits: z << 1, z >> 1, and the high-SNR case x ≈ ν ≈ √z >> 1.
    z = x * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z, Val(0))
    if z < neglogpdf_rician_parts_taylor_branch(T)
        ∂x = (x - inv(x)) - r * ν
        ∂ν = ν - r * x
    else
        ∂x = (x - ν) - inv(x) * (one(T) - r_tail)
        ∂ν = (ν - x) + inv(ν) * r_tail
    end

    return (∂x, ∂ν)
end

@scalar_rule _neglogpdf_rician(x, ν) (_∇neglogpdf_rician(x, ν)...,)
@dual_rule_from_frule _neglogpdf_rician(x, ν)

@inline function _∇²neglogpdf_rician(x::D, ν::D) where {D}
    z = x * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z, Val(1))
    if z < neglogpdf_rician_parts_taylor_branch(T)
        ∂xx = inv(x)^2 + (one(T) - ν^2 * r′)
        ∂xν = -(r + z * r′)
        ∂νν = one(T) - x^2 * r′
    else
        ∂xx = one(T) + inv(x)^2 * (one(T) - z^2 * r′)
        ∂xν = -r_tail * (one(T) + r)
        ∂νν = one(T) - x^2 * r′
    end

    return (∂xx, ∂xν, ∂νν)
end

@inline function _∇²neglogpdf_rician_with_gradient(x::D, ν::D) where {D}
    z = x * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z, Val(1))
    if z < neglogpdf_rician_parts_taylor_branch(T)
        x⁻¹ = inv(x)
        ∂x = (x - x⁻¹) - r * ν
        ∂ν = ν - r * x
        ∂xx = x⁻¹ * x⁻¹ + (one(T) - ν^2 * r′)
        ∂xν = -(r + z * r′)
        ∂νν = one(T) - x^2 * r′
    else
        x⁻¹, ν⁻¹ = inv(x), inv(ν)
        ∂x = (x - ν) - x⁻¹ * (one(T) - r_tail)
        ∂ν = (ν - x) + ν⁻¹ * r_tail
        ∂xx = one(T) + x⁻¹ * x⁻¹ * (one(T) - z^2 * r′)
        ∂xν = -r_tail * (one(T) + r)
        ∂νν = one(T) - x^2 * r′
    end

    return (∂x, ∂ν), (∂xx, ∂xν, ∂νν)
end

@inline function _∇³neglogpdf_rician_with_gradient_and_hessian(x::D, ν::D) where {D}
    z = x * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z, Val(2))
    x⁻¹ = inv(x)
    if z < neglogpdf_rician_parts_taylor_branch(T)
        x⁻² = x⁻¹ * x⁻¹
        ∂x = (x - x⁻¹) - r * ν
        ∂ν = ν - r * x
        ∂xx = x⁻² + (one(T) - ν^2 * r′)
        ∂xν = -(r + z * r′)
        ∂νν = one(T) - x^2 * r′
    else
        ν⁻¹ = inv(ν)
        x⁻² = x⁻¹ * x⁻¹
        ∂x = (x - ν) - x⁻¹ * (one(T) - r_tail)
        ∂ν = (ν - x) + ν⁻¹ * r_tail
        ∂xx = one(T) + x⁻² * (one(T) - z^2 * r′)
        ∂xν = -r_tail * (one(T) + r)
        ∂νν = one(T) - x^2 * r′
    end
    ∂xxx = T(-2) * x⁻¹ * x⁻² - ν^3 * r′′
    ∂xxν = -ν * two_r′_plus_z_r′′
    ∂xνν = -x * two_r′_plus_z_r′′
    ∂ννν = -x^3 * r′′

    return (∂x, ∂ν), (∂xx, ∂xν, ∂νν), (∂xxx, ∂xxν, ∂xνν, ∂ννν)
end

@inline function _∇³neglogpdf_rician_with_gradient_and_hessian_ad(x::D, ν::D) where {D}
    (∂x, ∂ν, ∂xx, ∂xν, ∂νν), J = withjacobian(SVector(x, ν)) do p
        local (∂x, ∂ν), (∂xx, ∂xν, ∂νν) = _∂²neglogpdf_rician_with_gradient(p...)
        return SVector(∂x, ∂ν, ∂xx, ∂xν, ∂νν)
    end
    ∂xxx, ∂xxν, ∂xνν, ∂ννν = J[3], J[4], J[5], J[10]

    return (∂x, ∂ν), (∂xx, ∂xν, ∂νν), (∂xxx, ∂xxν, ∂xνν, ∂ννν)
end

# Residual derivative methods

@inline function _neglogpdf_rician_residual(x::D, ν::D, Δx::D) where {D}
    # Negative Rician log-likelihood residual `-logp(x + Δx | ν, σ = 1) - (x - ν)^2 / 2 - log(√2π)`
    Δxν = x - ν
    y = x + Δx
    z = y * ν
    T = checkedfloattype(z)
    if z < first(logbesseli0x_branches(T))
        return Δx * (Δxν + Δx / 2) + z - logbesseli0_taylor(z) - log(y) - T(log2π) / 2
    elseif z < last(logbesseli0x_branches(T))
        return Δx * (Δxν + Δx / 2) - logbesseli0x_middle(z) - log(y) - T(log2π) / 2
    else
        return Δx * (Δxν + Δx / 2) - logratio(y, ν) / 2 - logbesseli0x_scaled_tail(z)
    end
end

@inline function _∇neglogpdf_rician_residual(x::D, ν::D, Δx::D) where {D}
    y = x + Δx
    z = y * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z, Val(0))
    if z < neglogpdf_rician_parts_taylor_branch(T)
        ∂x = ((one(T) - r) * ν + Δx) - inv(y)
        ∂ν = (one(T) - r) * x - r * Δx
    else
        ∂x = Δx - inv(y) * (one(T) - r_tail)
        ∂ν = -(Δx - inv(ν) * r_tail)
    end

    return (∂x, ∂ν)
end

@inline function _∇²neglogpdf_rician_residual_with_gradient(x::D, ν::D, Δx::D) where {D}
    y = x + Δx
    z = y * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z, Val(1))
    if z < neglogpdf_rician_parts_taylor_branch(T)
        y⁻¹ = inv(y)
        ∂x = ((one(T) - r) * ν + Δx) - y⁻¹
        ∂ν = (one(T) - r) * x - r * Δx
        ∂xx = y⁻¹ * y⁻¹ - ν^2 * r′
        ∂xν = one(T) - (r + z * r′)
        ∂νν = -y^2 * r′
    else
        y⁻¹, ν⁻¹ = inv(y), inv(ν)
        ∂x = Δx - y⁻¹ * (one(T) - r_tail)
        ∂ν = -(Δx - ν⁻¹ * r_tail)
        ∂xx = y⁻¹ * y⁻¹ * (one(T) - z^2 * r′)
        ∂xν = one_minus_r_minus_z_r′
        ∂νν = -y^2 * r′
    end

    return (∂x, ∂ν), (∂xx, ∂xν, ∂νν)
end

@inline function _∇³neglogpdf_rician_residual_with_gradient_and_hessian(x::D, ν::D, Δx::D) where {D}
    y = x + Δx
    z = y * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z, Val(2))
    y² = y * y
    y⁻¹ = inv(y)
    y⁻² = y⁻¹ * y⁻¹
    if z < neglogpdf_rician_parts_taylor_branch(T)
        ∂x = ((one(T) - r) * ν + Δx) - y⁻¹
        ∂ν = (one(T) - r) * x - r * Δx
        ∂xx = y⁻² - ν^2 * r′
        ∂xν = one(T) - (r + z * r′)
        ∂νν = -y² * r′
    else
        ν⁻¹ = inv(ν)
        ∂x = Δx - y⁻¹ * (one(T) - r_tail)
        ∂ν = -(Δx - ν⁻¹ * r_tail)
        ∂xx = y⁻² * (one(T) - z^2 * r′)
        ∂xν = one_minus_r_minus_z_r′
        ∂νν = -y² * r′
    end
    ∂xxx = T(-2) * y⁻¹ * y⁻² - ν^3 * r′′
    ∂xxν = -ν * two_r′_plus_z_r′′
    ∂xνν = -y * two_r′_plus_z_r′′
    ∂ννν = -y * y² * r′′

    return (∂x, ∂ν), (∂xx, ∂xν, ∂νν), (∂xxx, ∂xxν, ∂xνν, ∂ννν)
end

# Methods for the "regular part" of the residual's derivatives, where singular terms 1/x, 1/x², 1/x³ in ∂x, ∂xx, ∂xxx have been analytically removed

@inline function _∇²neglogpdf_rician_residual_with_gradient_regular(x::D, ν::D, Δx::D) where {D}
    y = x + Δx
    z = y * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z, Val(1))
    if z < neglogpdf_rician_parts_taylor_branch(T)
        y⁻¹ = inv(y)
        ∂x = (one(T) - r) * ν + Δx
        ∂ν = (one(T) - r) * x - r * Δx
        ∂xx = -ν^2 * r′
        ∂xν = one(T) - (r + z * r′)
        ∂νν = -y^2 * r′
    else
        y⁻¹, ν⁻¹ = inv(y), inv(ν)
        ∂x = y⁻¹ * r_tail + Δx
        ∂ν = ν⁻¹ * r_tail - Δx
        ∂xx = -y⁻¹ * y⁻¹ * z^2 * r′
        ∂xν = one_minus_r_minus_z_r′
        ∂νν = -y^2 * r′
    end

    return (∂x, ∂ν), (∂xx, ∂xν, ∂νν)
end

@inline function _∇³neglogpdf_rician_residual_with_gradient_and_hessian_regular(x::D, ν::D, Δx::D) where {D}
    y = x + Δx
    z = y * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z, Val(2))
    y² = y * y
    y⁻¹ = inv(y)
    y⁻² = y⁻¹ * y⁻¹
    if z < neglogpdf_rician_parts_taylor_branch(T)
        ∂x = (one(T) - r) * ν + Δx
        ∂ν = (one(T) - r) * x - r * Δx
        ∂xx = -ν^2 * r′
        ∂xν = one(T) - (r + z * r′)
        ∂νν = -y² * r′
    else
        ν⁻¹ = inv(ν)
        ∂x = y⁻¹ * r_tail + Δx
        ∂ν = ν⁻¹ * r_tail - Δx
        ∂xx = -y⁻² * z^2 * r′
        ∂xν = one_minus_r_minus_z_r′
        ∂νν = -y² * r′
    end
    ∂xxx = -ν^3 * r′′
    ∂xxν = -ν * two_r′_plus_z_r′′
    ∂xνν = -y * two_r′_plus_z_r′′
    ∂ννν = -y * y² * r′′

    return (∂x, ∂ν), (∂xx, ∂xν, ∂νν), (∂xxx, ∂xxν, ∂xνν, ∂ννν)
end

####
#### Quantized Rician negative log-pdf
####

# Quantized Rician PDF is the integral of the Rician PDF over `(x, x+δ)`.
# This integral is approximated using Gauss-Legendre quadrature.
# Note: Rician PDF is never evaluated at the interval endpoints.
@inline function neglogpdf_qrician(x::Real, ν::Real, logσ::Real, δ::Real, order::Val)
    σ⁻¹ = exp(-logσ)
    return neglogpdf_qrician(σ⁻¹ * x, σ⁻¹ * ν, σ⁻¹ * δ, order)
end
@inline neglogpdf_qrician(n::Int, ν::Real, logσ::Real, δ::Real, order::Val) = neglogpdf_qrician(n * δ, ν, logσ, δ, order)

# Wrapper functions that dispatch to fast path for single point quadrature
@inline neglogpdf_qrician(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _neglogpdf_qrician_midpoint(promote(x, ν, δ)...) : _neglogpdf_qrician(promote(x, ν, δ)..., order)
@inline ∇neglogpdf_qrician(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇neglogpdf_qrician_midpoint(promote(x, ν, δ)...) : _∇neglogpdf_qrician(promote(x, ν, δ)..., order)
@inline ∇neglogpdf_qrician_with_primal(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇neglogpdf_qrician_midpoint_with_primal(promote(x, ν, δ)...) : _∇neglogpdf_qrician_with_primal(promote(x, ν, δ)..., order)
@inline ∇²neglogpdf_qrician(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇²neglogpdf_qrician_midpoint(promote(x, ν, δ)...) : _∇²neglogpdf_qrician(promote(x, ν, δ)..., order)
@inline ∇²neglogpdf_qrician_with_gradient(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇²neglogpdf_qrician_midpoint_with_gradient(promote(x, ν, δ)...) : _∇²neglogpdf_qrician_with_gradient(promote(x, ν, δ)..., order)
@inline ∇²neglogpdf_qrician_with_primal_and_gradient(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇²neglogpdf_qrician_midpoint_with_primal_and_gradient(promote(x, ν, δ)...) : _∇²neglogpdf_qrician_with_primal_and_gradient(promote(x, ν, δ)..., order)
@inline ∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian(Δ::SVector{6, <:Real}, x::Real, ν::Real, δ::Real, order::Val) = _∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian(promote_eltypes(Δ, x, ν, δ)..., order) #TODO: midpoint optimization
@inline ∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian(x::Real, ν::Real, δ::Real, order::Val) = _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian(promote(x, ν, δ)..., order) #TODO: midpoint optimization

# Fast-path for single point quadrature which avoids computing the primal; equivalent to using the midpoint rule approximations for the integrals
@inline _neglogpdf_qrician_midpoint(x::D, ν::D, δ::D) where {D} = _neglogpdf_rician(x + δ / 2, ν) - log(δ)
@inline function _∇neglogpdf_qrician_midpoint(x::D, ν::D, δ::D) where {D}
    ∂x, ∂ν = _∇neglogpdf_rician(x + δ / 2, ν)
    return ∂x, ∂ν, ∂x / 2 - inv(δ)
end
@inline function _∇neglogpdf_qrician_midpoint_with_primal(x::D, ν::D, δ::D) where {D}
    Ω = _neglogpdf_qrician_midpoint(x, ν, δ)
    ∇ = _∇neglogpdf_qrician_midpoint(x, ν, δ)
    return Ω, ∇
end
@inline function _∇²neglogpdf_qrician_midpoint_with_gradient(x::D, ν::D, δ::D) where {D}
    δ⁻¹ = inv(δ)
    x′ = x + δ / 2
    (∇x, ∇ν), (∇xx, ∇xν, ∇νν) = _∇²neglogpdf_rician_with_gradient(x′, ν)
    return (∇x, ∇ν, ∇x / 2 - δ⁻¹), (∇xx, ∇xν, ∇xx / 2, ∇νν, ∇xν / 2, ∇xx / 4 + δ⁻¹ * δ⁻¹)
end
@inline function _∇²neglogpdf_qrician_midpoint_with_primal_and_gradient(x::D, ν::D, δ::D) where {D}
    Ω = _neglogpdf_qrician_midpoint(x, ν, δ)
    ∇, ∇² = _∇²neglogpdf_qrician_midpoint_with_gradient(x, ν, δ)
    return Ω, ∇, ∇²
end
@inline _∇²neglogpdf_qrician_midpoint(x::D, ν::D, δ::D) where {D} = last(_∇²neglogpdf_qrician_midpoint_with_gradient(x, ν, δ))

#### Internal methods with strict type signatures (enables dual number overloads with single method)

@inline function _neglogpdf_qrician(x::D, ν::D, δ::D, order::Val) where {D}
    Ω₀ = neglogf_quadrature_unit_interval(D, order) do t
        δt = δ * t
        return _neglogpdf_rician_residual(x, ν, δt)
    end
    return Ω₀ + ((x - ν)^2 + log2π) / 2 - log(δ)
end

@inline function _∇neglogpdf_qrician_with_primal(x::D, ν::D, δ::D, order::Val) where {D}
    Δxν = x - ν
    logδ, δ⁻¹ = log(δ), inv(δ)
    Ω₀, (E_rx, E_rν, E_rδ) = f_quadrature_weighted_unit_interval(D, order) do t
        δt = δ * t
        rx, rν = _∇neglogpdf_rician_residual(x, ν, δt)
        rδ = t * (rx + Δxν)
        return _neglogpdf_rician_residual(x, ν, δt), SVector{3, D}(rx, rν, rδ)
    end
    Ω = Ω₀ + (Δxν^2 + log2π) / 2 - logδ
    ∂x = E_rx + Δxν
    ∂ν = E_rν - Δxν
    ∂δ = E_rδ - δ⁻¹

    #=
    # Differentiate the approximation:
    # ω(t) = neglogpdf_rician(t, ν)
    #    I = ∫_{x}^{x+δ} exp(-ω(x′)) dx′ = ∫_{0}^{1} exp(-ω(x + δ * t)) * δ dt
    #    Ω = -logI
    #   ∂Ω = -∂(logI) = -∂I / I
    #      = -exp(Ω) * ∫_{0}^{1} ∂(exp(-ω(x + δ * t)) * δ) dt
    # where Ω = -logI is constant w.r.t. ∂.
    δ⁻¹ = inv(δ)
    Ω₀, (∂x, ∂ν, ∂δ) = f_quadrature_weighted_unit_interval(D, order) do t
        x′ = x + δ * t
        ∇x, ∇ν = _∇neglogpdf_rician(x′, ν)
        ∇δ = t * ∇x - δ⁻¹
        return _neglogpdf_rician(x′, ν), SVector{3, D}(∇x, ∇ν, ∇δ)
    end
    Ω = Ω₀ - log(δ)
    =#

    #=
    # Differentiate the approximation (using precomputed Ω)
    ∂x, ∂ν, ∂δ = f_quadrature(zero(x), one(x), order) do t
        δt = δ * t
        x′ = x + δt
        ∇x, ∇ν = _∇neglogpdf_rician(x′, ν)
        dx, dν, dδ = ∇x * δ, ∇ν * δ, ∇x * δt - one(x)
        ∇ = SVector{3, D}(dx, dν, dδ)
        return exp(Ω - _neglogpdf_rician(x′, ν)) * ∇
    end
    =#

    #=
    # Differentiate the approximation for (∂x, ∂ν) and use FTC for ∂δ:
    ∂x, ∂ν = f_quadrature(x, δ, order) do x′
        ∇ = _∇neglogpdf_rician(x′, ν) # differentiate the integrand
        ∇ = SVector{2, D}(∇)
        return exp(Ω - _neglogpdf_rician(x′, ν)) * ∇
    end
    ∂δ = -exp(Ω - _neglogpdf_rician(x + δ, ν)) # by fundamental theorem of calculus
    =#

    #=
    # Differentiate the approximation for ∂ν and use FTC for (∂x, ∂δ):
    ∂ν = f_quadrature(x, δ, order) do x′
        _, ∇ν = _∇neglogpdf_rician(x′, ν) # differentiate the integrand
        return exp(Ω - _neglogpdf_rician(x′, ν)) * ∇ν
    end
    lo, hi = _neglogpdf_rician(x, ν), _neglogpdf_rician(x + δ, ν)
    ∂δ = -exp(Ω - hi) # by fundamental theorem of calculus
    ∂x = lo < hi ? exp(Ω - lo) * -expm1(lo - hi) : exp(Ω - hi) * expm1(hi - lo) # by fundamental theorem of calculus (note: leads to catestrophic cancellation for small δ, but more accurate for large δ)
    =#

    return Ω, (∂x, ∂ν, ∂δ)
end
@inline _∇neglogpdf_qrician(x::D, ν::D, δ::D, order::Val) where {D} = last(_∇neglogpdf_qrician_with_primal(x, ν, δ, order))

@scalar_rule _neglogpdf_qrician(x, ν, δ, order::Val) (_∇neglogpdf_qrician_with_primal(x, ν, δ, order)[2]..., NoTangent())
@dual_rule_from_frule _neglogpdf_qrician(x, ν, δ, !(order::Val))

@inline _∇²neglogpdf_qrician_with_primal_and_gradient(x::D, ν::D, δ::D, order::Val) where {D} = _∇²neglogpdf_qrician_with_primal_and_gradient_one_pass_regular(x, ν, δ, order)
@inline _∇²neglogpdf_qrician_with_gradient(x::D, ν::D, δ::D, order::Val) where {D} = Base.tail(_∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ, order))
@inline _∇²neglogpdf_qrician(x::D, ν::D, δ::D, order::Val) where {D} = last(_∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ, order))

@inline function _∇²neglogpdf_qrician_with_primal_and_gradient_one_pass(x::D, ν::D, δ::D, order::Val) where {D}
    Δxν = x - ν
    logδ, δ⁻¹ = log(δ), inv(δ)
    Ω₀, (E_rx, E_rν, E_rδ, E_hxx, E_hxν, E_hxδ, E_hνν, E_hνδ, E_hδδ) = f_quadrature_weighted_unit_interval(D, order) do t
        δt = δ * t
        (rx, rν), (rxx, rxν, rνν) = _∇²neglogpdf_rician_residual_with_gradient(x, ν, δt)
        rδ = t * (rx + Δxν)
        hxx = rxx - rx * rx
        hxν = rxν - rx * rν
        hνν = rνν - rν * rν
        hxδ = t * ((hxx - Δxν * rx) + 1)
        hνδ = t * ((hxν - Δxν * rν) - 1)
        hδδ = t^2 * ((hxx - Δxν * (2 * rx + Δxν)) + 1)
        return _neglogpdf_rician_residual(x, ν, δt), SVector{9, D}(rx, rν, rδ, hxx, hxν, hxδ, hνν, hνδ, hδδ)
    end
    Ω = Ω₀ + (Δxν^2 + log2π) / 2 - logδ

    ∇x = E_rx + Δxν
    ∇ν = E_rν - Δxν
    ∇δ = E_rδ - δ⁻¹

    ∇xx = (E_hxx + E_rx * E_rx) + 1
    ∇xν = (E_hxν + E_rx * E_rν) - 1
    ∇xδ = E_hxδ + E_rx * E_rδ
    ∇νν = (E_hνν + E_rν * E_rν) + 1
    ∇νδ = E_hνδ + E_rν * E_rδ
    ∇δδ = (E_hδδ + E_rδ * E_rδ) + δ⁻¹ * δ⁻¹

    return Ω, (∇x, ∇ν, ∇δ), (∇xx, ∇xν, ∇xδ, ∇νν, ∇νδ, ∇δδ)
end

@inline function _∇²neglogpdf_qrician_with_primal_and_gradient_one_pass_regular(x::D, ν::D, δ::D, order::Val) where {D}
    Δxν = x - ν
    logδ, δ⁻¹ = log(δ), inv(δ)
    Ω₀, (E_rx, E_rν, E_rδ, E_hxx, E_hxν, E_hxδ, E_hνν, E_hνδ, E_hδδ) = f_quadrature_weighted_unit_interval(D, order) do t
        δt = δ * t
        y = x + δt
        y⁻¹ = inv(y)
        t² = t * t

        (rx_ns, rν), (rxx_ns, rxν, rνν) = _∇²neglogpdf_rician_residual_with_gradient_regular(x, ν, δt)
        rx = rx_ns - y⁻¹
        rδ = t * (rx + Δxν)

        # h-integrands
        y⁻¹_rx_ns = y⁻¹ * rx_ns
        y⁻¹_rν = y⁻¹ * rν
        rx_ns_rν = rx_ns * rν
        rxx_ns_minus_rx_ns² = rxx_ns - rx_ns * rx_ns

        hxx = rxx_ns_minus_rx_ns² + 2 * y⁻¹_rx_ns
        hxν = rxν - rx_ns_rν + y⁻¹_rν
        hνν = rνν - rν * rν
        hxδ = t * (rxx_ns_minus_rx_ns² - Δxν * rx_ns + 1 + y⁻¹ * (2 * rx_ns + Δxν))
        hνδ = t * (rxν - rx_ns_rν - Δxν * rν - 1 + y⁻¹_rν)
        hδδ = t² * (rxx_ns_minus_rx_ns² - Δxν * (2 * rx_ns + Δxν) + 1 + 2 * y⁻¹ * (rx_ns + Δxν))

        return _neglogpdf_rician_residual(x, ν, δt), SVector{9, D}(rx, rν, rδ, hxx, hxν, hxδ, hνν, hνδ, hδδ)
    end

    Ω = Ω₀ + (Δxν^2 + log2π) / 2 - logδ

    ∇x = E_rx + Δxν
    ∇ν = E_rν - Δxν
    ∇δ = E_rδ - δ⁻¹

    ∇xx = (E_hxx + E_rx * E_rx) + one(D)
    ∇xν = (E_hxν + E_rx * E_rν) - one(D)
    ∇xδ = E_hxδ + E_rx * E_rδ
    ∇νν = (E_hνν + E_rν * E_rν) + one(D)
    ∇νδ = E_hνδ + E_rν * E_rδ
    ∇δδ = (E_hδδ + E_rδ * E_rδ) + δ⁻¹ * δ⁻¹

    return Ω, (∇x, ∇ν, ∇δ), (∇xx, ∇xν, ∇xδ, ∇νν, ∇νδ, ∇δδ)
end

#### Quantized Rician third-order derivatives

@inline _∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian(Δ::SVector{6, D}, x::D, ν::D, δ::D, order::Val) where {D} = _∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian_two_pass_regular(Δ, x, ν, δ, order)
@inline _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian(x::D, ν::D, δ::D, order::Val) where {D} = _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_two_pass_regular(x, ν, δ, order)

@inline function _∇³neglogpdf_qrician_jacobian_with_hessian_ad(x::D, ν::D, δ::D, order::Val) where {D}
    H, J = withjacobian(SVector(x, ν, δ)) do p
        local x, ν, δ = p
        _, ∇² = _∇²neglogpdf_qrician_with_gradient(x, ν, δ, order)
        return SVector(∇²)
    end
    return Tuple(H), J
end

@inline function _∇³neglogpdf_qrician_vjp_with_hessian_ad(Δ::SVector{6, D}, x::D, ν::D, δ::D, order::Val) where {D}
    H, J = _∇³neglogpdf_qrician_jacobian_with_hessian_ad(x, ν, δ, order)
    return H, J' * Δ
end

@inline function _∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian_two_pass(Δ::SVector{6, D}, x::D, ν::D, δ::D, order::Val) where {D}
    # Compute ∇Ω(θ) ∈ ℝ³, vech(∇²Ω(θ)) ∈ ℝ⁶, and J'Δ where J(θ) = ∂/∂θ vech(∇²Ω(θ)) and θ = (x, ν, δ).
    Δxν = x - ν
    δ⁻¹ = inv(δ)
    δ⁻² = δ⁻¹ * δ⁻¹
    δ⁻³ = δ⁻² * δ⁻¹
    Δ_Hxx, Δ_Hxν, Δ_Hxδ, Δ_Hνν, Δ_Hνδ, Δ_Hδδ = Δ

    # First-pass computes expectations μ = E[∇_θ r̃]
    Ω₀, (μ_rx, μ_rν, μ_rδ), t_nodes, w_nodes = f_quadrature_weighted_unit_interval(D, order) do t
        Δx = δ * t
        rx, rν = _∇neglogpdf_rician_residual(x, ν, Δx)
        fy = rx + Δxν
        rδ = t * fy
        return _neglogpdf_rician_residual(x, ν, Δx), SVector(rx, rν, rδ)
    end
    Ω = Ω₀ + (Δxν^2 + log2π) / 2 - log(δ)

    # Second-pass computes E_h and E_T using centered gradients c = ∇_θ r̃ - μ
    integrands = map(t_nodes) do t
        h, T = _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_two_pass_integrand((μ_rx, μ_rν, μ_rδ), x, ν, δ, t)

        (Txxx, Txxν, Txνν, Tννν, Txxδ, Txνδ, Tννδ, Txδδ, Tνδδ, Tδδδ) = T
        vjp_x = Txxx * Δ_Hxx + Txxν * Δ_Hxν + Txxδ * Δ_Hxδ + Txνν * Δ_Hνν + Txνδ * Δ_Hνδ + Txδδ * Δ_Hδδ # ∂(h ⋅ Δ)/∂x
        vjp_ν = Txxν * Δ_Hxx + Txνν * Δ_Hxν + Txνδ * Δ_Hxδ + Tννν * Δ_Hνν + Tννδ * Δ_Hνδ + Tνδδ * Δ_Hδδ # ∂(h ⋅ Δ)/∂ν
        vjp_δ = Txxδ * Δ_Hxx + Txνδ * Δ_Hxν + Txδδ * Δ_Hxδ + Tννδ * Δ_Hνν + Tνδδ * Δ_Hνδ + Tδδδ * Δ_Hδδ # ∂(h ⋅ Δ)/∂δ

        return (SVector(h), SVector(vjp_x, vjp_ν, vjp_δ))
    end
    E_h, E_vjp = vecdot(w_nodes, integrands)

    # Assemble outputs g = ∇Ω, H = vech(∇²Ω), and vjp = J'Δ
    E_hxx, E_hxν, E_hxδ, E_hνν, E_hνδ, E_hδδ = E_h
    E_vjp_x, E_vjp_ν, E_vjp_δ = E_vjp

    g = (μ_rx + Δxν, μ_rν - Δxν, μ_rδ - δ⁻¹)
    H = (E_hxx + 1, E_hxν - 1, E_hxδ, E_hνν + 1, E_hνδ, E_hδδ + δ⁻²)
    JᵀΔ = (E_vjp_x, E_vjp_ν, E_vjp_δ - 2 * δ⁻³ * Δ_Hδδ)

    return Ω, g, H, JᵀΔ
end

@inline function _∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian_two_pass_regular(Δ::SVector{6, D}, x::D, ν::D, δ::D, order::Val) where {D}
    # Compute ∇Ω(θ) ∈ ℝ³, vech(∇²Ω(θ)) ∈ ℝ⁶, and J'Δ where J(θ) = ∂/∂θ vech(∇²Ω(θ)) and θ = (x, ν, δ).
    Δxν = x - ν
    δ⁻¹ = inv(δ)
    δ⁻² = δ⁻¹ * δ⁻¹
    δ⁻³ = δ⁻² * δ⁻¹
    Δ_Hxx, Δ_Hxν, Δ_Hxδ, Δ_Hνν, Δ_Hνδ, Δ_Hδδ = Δ

    # First-pass computes expectations μ = E[∇_θ r̃]
    Ω₀, (μ_rx, μ_rν, μ_rδ), t_nodes, w_nodes = f_quadrature_weighted_unit_interval(D, order) do t
        Δx = δ * t
        rx, rν = _∇neglogpdf_rician_residual(x, ν, Δx)
        fy = rx + Δxν
        rδ = t * fy
        return _neglogpdf_rician_residual(x, ν, Δx), SVector(rx, rν, rδ)
    end
    Ω = Ω₀ + (Δxν^2 + log2π) / 2 - log(δ)

    # Second-pass computes E_h and E_T using centered gradients c = ∇_θ r̃ - μ
    integrands = map(t_nodes) do t
        h, T = _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_two_pass_regular_integrand((μ_rx, μ_rν, μ_rδ), x, ν, δ, t)

        (Txxx, Txxν, Txνν, Tννν, Txxδ, Txνδ, Tννδ, Txδδ, Tνδδ, Tδδδ) = T
        vjp_x = Txxx * Δ_Hxx + Txxν * Δ_Hxν + Txxδ * Δ_Hxδ + Txνν * Δ_Hνν + Txνδ * Δ_Hνδ + Txδδ * Δ_Hδδ # ∂(h ⋅ Δ)/∂x
        vjp_ν = Txxν * Δ_Hxx + Txνν * Δ_Hxν + Txνδ * Δ_Hxδ + Tννν * Δ_Hνν + Tννδ * Δ_Hνδ + Tνδδ * Δ_Hδδ # ∂(h ⋅ Δ)/∂ν
        vjp_δ = Txxδ * Δ_Hxx + Txνδ * Δ_Hxν + Txδδ * Δ_Hxδ + Tννδ * Δ_Hνν + Tνδδ * Δ_Hνδ + Tδδδ * Δ_Hδδ # ∂(h ⋅ Δ)/∂δ

        return (SVector(h), SVector(vjp_x, vjp_ν, vjp_δ))
    end
    E_h, E_vjp = vecdot(w_nodes, integrands)

    # Assemble outputs g = ∇Ω, H = vech(∇²Ω), and vjp = J'Δ
    E_hxx, E_hxν, E_hxδ, E_hνν, E_hνδ, E_hδδ = E_h
    E_vjp_x, E_vjp_ν, E_vjp_δ = E_vjp

    g = (μ_rx + Δxν, μ_rν - Δxν, μ_rδ - δ⁻¹)
    H = (E_hxx + 1, E_hxν - 1, E_hxδ, E_hνν + 1, E_hνδ, E_hδδ + δ⁻²)
    JᵀΔ = (E_vjp_x, E_vjp_ν, E_vjp_δ - 2 * δ⁻³ * Δ_Hδδ)

    return Ω, g, H, JᵀΔ
end

@inline function _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_two_pass(x::D, ν::D, δ::D, order::Val) where {D}
    # Compute ∇Ω(θ) ∈ ℝ³, vech(∇²Ω(θ)) ∈ ℝ⁶, and J(θ) ∈ ℝ⁶ˣ³ where J(θ) = ∂/∂θ vech(∇²Ω(θ)) and θ = (x, ν, δ).
    # Notation per paper:
    #   r̃(t, θ) = f(x + δ t, ν) - f_G(x, ν),  Z(θ) = ∫ exp(-r̃) dt,  Ω(θ) = -log Z(θ) - log δ.
    # Working identities (all expectations wrt P(t|θ) ∝ exp(-r̃)):
    #   ∇Ω = E[∇_θ r̃] + ∇_θ f_G - (0, 0, δ⁻¹).
    #   ∇²Ω = E[∇²_θ r̃] - Cov(∇_θ r̃, ∇_θ r̃) + diag(1, 1, δ⁻²) + offdiag(1, -1).
    #   ∂_αβγ Ω = E[∂_αβγ r̃] - Cov(∂_αβ r̃, ∂_γ r̃) - Cov(∂_αγ r̃, ∂_β r̃) - Cov(∂_βγ r̃, ∂_α r̃) + Cov3(∂_α r̃, ∂_β r̃, ∂_γ r̃) - 2 δ⁻³ 1{α=β=γ=δ}.
    # Implementation strategy:
    #   Pass 1: μ = E[∇_θ r̃] (to center all covariance terms);
    #   Pass 2: form integrands for E[h] and E[T] directly using centered quantities, then add constants.
    Δxν = x - ν
    δ⁻¹ = inv(δ)
    δ⁻² = δ⁻¹ * δ⁻¹
    δ⁻³ = δ⁻² * δ⁻¹

    # First-pass computes expectations μ = E[∇_θ r̃]
    Ω₀, (μ_rx, μ_rν, μ_rδ), t_nodes, w_nodes = f_quadrature_weighted_unit_interval(D, order) do t
        Δx = δ * t
        rx, rν = _∇neglogpdf_rician_residual(x, ν, Δx)
        fy = rx + Δxν
        rδ = t * fy
        return _neglogpdf_rician_residual(x, ν, Δx), SVector(rx, rν, rδ)
    end
    Ω = Ω₀ + (Δxν^2 + log2π) / 2 - log(δ)

    # Second-pass computes E_h and E_T using centered gradients c = ∇_θ r̃ - μ
    integrands = map(t_nodes) do t
        h, T = _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_two_pass_integrand((μ_rx, μ_rν, μ_rδ), x, ν, δ, t)
        return (SVector(h), SVector(T))
    end
    E_h, E_T = vecdot(w_nodes, integrands)

    # Assemble outputs g = ∇Ω, H = vech(∇²Ω), and J = ∂/∂θ vech(∇²Ω)
    E_hxx, E_hxν, E_hxδ, E_hνν, E_hνδ, E_hδδ = E_h
    E_Txxx, E_Txxν, E_Txνν, E_Tννν, E_Txxδ, E_Txνδ, E_Tννδ, E_Txδδ, E_Tνδδ, E_Tδδδ = E_T

    ∇x, ∇ν, ∇δ = μ_rx + Δxν, μ_rν - Δxν, μ_rδ - δ⁻¹
    Hxx, Hxν, Hxδ, Hνν, Hνδ, Hδδ = E_hxx + 1, E_hxν - 1, E_hxδ, E_hνν + 1, E_hνδ, E_hδδ + δ⁻²

    g = (∇x, ∇ν, ∇δ)
    H = (Hxx, Hxν, Hxδ, Hνν, Hνδ, Hδδ)
    J = SMatrix{6, 3, D, 18}(
        E_Txxx, E_Txxν, E_Txxδ, E_Txνν, E_Txνδ, E_Txδδ, # ∂H/∂x
        E_Txxν, E_Txνν, E_Txνδ, E_Tννν, E_Tννδ, E_Tνδδ, # ∂H/∂ν
        E_Txxδ, E_Txνδ, E_Txδδ, E_Tννδ, E_Tνδδ, E_Tδδδ - 2 * δ⁻³, # ∂H/∂δ
    )

    return Ω, g, H, J
end

@inline function _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_two_pass_regular(x::D, ν::D, δ::D, order::Val) where {D}
    # Compute ∇Ω(θ) ∈ ℝ³, vech(∇²Ω(θ)) ∈ ℝ⁶, and J(θ) ∈ ℝ⁶ˣ³ where J(θ) = ∂/∂θ vech(∇²Ω(θ)) and θ = (x, ν, δ).
    # Notation per paper:
    #   r̃(t, θ) = f(x + δ t, ν) - f_G(x, ν),  Z(θ) = ∫ exp(-r̃) dt,  Ω(θ) = -log Z(θ) - log δ.
    # Working identities (all expectations wrt P(t|θ) ∝ exp(-r̃)):
    #   ∇Ω = E[∇_θ r̃] + ∇_θ f_G - (0, 0, δ⁻¹).
    #   ∇²Ω = E[∇²_θ r̃] - Cov(∇_θ r̃, ∇_θ r̃) + diag(1, 1, δ⁻²) + offdiag(1, -1).
    #   ∂_αβγ Ω = E[∂_αβγ r̃] - Cov(∂_αβ r̃, ∂_γ r̃) - Cov(∂_αγ r̃, ∂_β r̃) - Cov(∂_βγ r̃, ∂_α r̃) + Cov3(∂_α r̃, ∂_β r̃, ∂_γ r̃) - 2 δ⁻³ 1{α=β=γ=δ}.
    # Implementation strategy:
    #   Pass 1: μ = E[∇_θ r̃] (to center all covariance terms);
    #   Pass 2: form integrands for E[h] and E[T] directly using centered quantities, then add constants.
    Δxν = x - ν
    δ⁻¹ = inv(δ)
    δ⁻² = δ⁻¹ * δ⁻¹
    δ⁻³ = δ⁻² * δ⁻¹

    # First-pass computes expectations μ = E[∇_θ r̃]
    Ω₀, (μ_rx, μ_rν, μ_rδ), t_nodes, w_nodes = f_quadrature_weighted_unit_interval(D, order) do t
        Δx = δ * t
        rx, rν = _∇neglogpdf_rician_residual(x, ν, Δx)
        fy = rx + Δxν
        rδ = t * fy
        return _neglogpdf_rician_residual(x, ν, Δx), SVector(rx, rν, rδ)
    end
    Ω = Ω₀ + (Δxν^2 + log2π) / 2 - log(δ)

    # Second-pass computes E_h and E_T using centered gradients and stable reformulations
    integrands = map(t_nodes) do t
        h, T = _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_two_pass_regular_integrand((μ_rx, μ_rν, μ_rδ), x, ν, δ, t)
        return (SVector(h), SVector(T))
    end
    E_h, E_T = vecdot(w_nodes, integrands)

    # Assemble outputs g = ∇Ω, H = vech(∇²Ω), and J = ∂/∂θ vech(∇²Ω)
    E_hxx, E_hxν, E_hxδ, E_hνν, E_hνδ, E_hδδ = E_h
    E_Txxx, E_Txxν, E_Txνν, E_Tννν, E_Txxδ, E_Txνδ, E_Tννδ, E_Txδδ, E_Tνδδ, E_Tδδδ = E_T

    ∇x, ∇ν, ∇δ = μ_rx + Δxν, μ_rν - Δxν, μ_rδ - δ⁻¹
    Hxx, Hxν, Hxδ, Hνν, Hνδ, Hδδ = E_hxx + 1, E_hxν - 1, E_hxδ, E_hνν + 1, E_hνδ, E_hδδ + δ⁻²

    g = (∇x, ∇ν, ∇δ)
    H = (Hxx, Hxν, Hxδ, Hνν, Hνδ, Hδδ)
    J = SMatrix{6, 3, D, 18}(
        E_Txxx, E_Txxν, E_Txxδ, E_Txνν, E_Txνδ, E_Txδδ, # ∂H/∂x
        E_Txxν, E_Txνν, E_Txνδ, E_Tννν, E_Tννδ, E_Tνδδ, # ∂H/∂ν
        E_Txxδ, E_Txνδ, E_Txδδ, E_Tννδ, E_Tνδδ, E_Tδδδ - 2 * δ⁻³, # ∂H/∂δ
    )

    return Ω, g, H, J
end

@inline function _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_one_pass(x::D, ν::D, δ::D, order::Val) where {D}
    # Compute ∇Ω(θ) ∈ ℝ³, vech(∇²Ω(θ)) ∈ ℝ⁶, and J(θ) ∈ ℝ⁶ˣ³ where J(θ) = ∂/∂θ vech(∇²Ω(θ)) and θ = (x, ν, δ).
    # Notation per paper (one-pass, raw-moment formulation):
    #   r̃(t, θ) = f(x + δ t, ν) - f_G(x, ν),  P(t|θ) ∝ exp(-r̃).
    #   E[·] denotes expectation wrt P(t|θ).
    # Working identities:
    #   ∇Ω = E[∇_θ r̃] + ∇_θ f_G - (0, 0, δ⁻¹).
    #   ∇²Ω = E[∇²_θ r̃] - (E[∇_θ r̃ ∇_θ r̃ᵀ] - μ μᵀ) + diag(1, 1, δ⁻²) + offdiag(1, -1),  μ = E[∇_θ r̃].
    #   ∂_αβγ Ω = E[J_αβγ] + g_αβγ - 2 δ⁻³ 1{α=β=γ=δ}, where
    #   J_αβγ = ∂_αβγ r̃ - (∂_αβ r̃ ∂_γ r̃ + ∂_αγ r̃ ∂_β r̃ + ∂_βγ r̃ ∂_α r̃) + ∂_α r̃ ∂_β r̃ ∂_γ r̃, and
    #   g_αβγ = (μ_γ E[∂_αβ] + μ_β E[∂_αγ] + μ_α E[∂_βγ]) - (μ_γ E[∂_α ∂_β] + μ_β E[∂_α ∂_γ] + μ_α E[∂_β ∂_γ]) + 2 μ_α μ_β μ_γ.
    # Implementation strategy: integrate the minimal raw basis (E[∂], E[∂∂ᵀ], E[∂²], E[J]), then assemble ∇Ω, H = vech(∇²Ω), and J = ∂/∂θ vech(∇²Ω).
    Δxν = x - ν
    δ⁻¹ = inv(δ)
    δ⁻² = δ⁻¹ * δ⁻¹
    δ⁻³ = δ⁻² * δ⁻¹

    # Single pass computes expectations of the minimal basis (∂, vech(∂∂ᵀ), vech(∂²), vech(J))
    Ω₀, (E_∂, E_∂∂ᵀ, E_∂², E_∂³) = f_quadrature_weighted_unit_interval(D, order) do t
        Δx = δ * t
        (rx, rν), (rxx, rxν, rνν), (rxxx, rxxν, rxνν, rννν) = _∇³neglogpdf_rician_residual_with_gradient_and_hessian(x, ν, Δx)

        t² = t * t
        t³ = t² * t
        fy, fyy = rx + Δxν, rxx + 1
        rδ = t * fy
        rxδ, rνδ, rδδ = t * fyy, t * (rxν - 1), t² * fyy
        rxxδ, rxνδ, rννδ = t * rxxx, t * rxxν, t * rxνν
        rxδδ, rνδδ, rδδδ = t² * rxxx, t² * rxxν, t³ * rxxx
        rxrx, rxrν, rxrδ, rνrν, rνrδ, rδrδ = rx^2, rx * rν, rx * rδ, rν^2, rν * rδ, rδ^2

        ∂ = SVector(rx, rν, rδ) # first derivatives
        ∂² = SVector(rxx, rxν, rxδ, rνν, rνδ, rδδ) # vech(∂²)
        ∂∂ᵀ = SVector(rxrx, rxrν, rxrδ, rνrν, rνrδ, rδrδ) # vech(∂ ∂ᵀ)

        Jxxx = rxxx - rx * (3 * rxx - rxrx)
        Jxxν = rxxν - (rxx * rν + rx * (2 * rxν - rxrν))
        Jxνν = rxνν - (rνν * rx + rν * (2 * rxν - rxrν))
        Jννν = rννν - rν * (3 * rνν - rνrν)
        Jxxδ = rxxδ - (rxx * rδ + rx * (2 * rxδ - rxrδ))
        Jxνδ = rxνδ - (rxν * rδ + rxδ * rν + rx * (rνδ - rνrδ))
        Jννδ = rννδ - (rνν * rδ + rν * (2 * rνδ - rνrδ))
        Jxδδ = rxδδ - (rδδ * rx + rδ * (2 * rxδ - rxrδ))
        Jνδδ = rνδδ - (rδδ * rν + rδ * (2 * rνδ - rνrδ))
        Jδδδ = rδδδ - rδ * (3 * rδδ - rδrδ)
        ∂³ = SVector(Jxxx, Jxxν, Jxνν, Jννν, Jxxδ, Jxνδ, Jννδ, Jxδδ, Jνδδ, Jδδδ) # vech(∂³)

        return _neglogpdf_rician_residual(x, ν, Δx), (∂, ∂∂ᵀ, ∂², ∂³)
    end
    Ω = Ω₀ + (Δxν^2 + log2π) / 2 - log(δ)

    # Unpack expectations and compute central moments
    μ_rx, μ_rν, μ_rδ = E_∂
    E_rxx, E_rxν, E_rxδ, E_rνν, E_rνδ, E_rδδ = E_∂²
    E_rxrx, E_rxrν, E_rxrδ, E_rνrν, E_rνrδ, E_rδrδ = E_∂∂ᵀ
    E_Jxxx, E_Jxxν, E_Jxνν, E_Jννν, E_Jxxδ, E_Jxνδ, E_Jννδ, E_Jxδδ, E_Jνδδ, E_Jδδδ = E_∂³

    Cov_rx_rx = E_rxrx - μ_rx * μ_rx
    Cov_rx_rν = E_rxrν - μ_rx * μ_rν
    Cov_rx_rδ = E_rxrδ - μ_rx * μ_rδ
    Cov_rν_rν = E_rνrν - μ_rν * μ_rν
    Cov_rν_rδ = E_rνrδ - μ_rν * μ_rδ
    Cov_rδ_rδ = E_rδrδ - μ_rδ * μ_rδ

    # Assemble primal outputs ∇Ω and vech(∇²Ω)
    ∇x, ∇ν, ∇δ = μ_rx + Δxν, μ_rν - Δxν, μ_rδ - δ⁻¹
    Hxx = E_rxx - Cov_rx_rx + 1
    Hxν = E_rxν - Cov_rx_rν - 1
    Hxδ = E_rxδ - Cov_rx_rδ
    Hνν = E_rνν - Cov_rν_rν + 1
    Hνδ = E_rνδ - Cov_rν_rδ
    Hδδ = E_rδδ - Cov_rδ_rδ + δ⁻²

    # Assemble Jacobian J from third derivatives T_αβγ = ∂_αβγ Ω
    μ_rx², μ_rν², μ_rδ² = μ_rx^2, μ_rν^2, μ_rδ^2
    μ_rx³, μ_rν³, μ_rδ³ = μ_rx² * μ_rx, μ_rν² * μ_rν, μ_rδ² * μ_rδ
    Txxx = E_Jxxx + 3 * μ_rx * (E_rxx - E_rxrx) + 2 * μ_rx³
    Txxν = E_Jxxν + μ_rν * (E_rxx - E_rxrx) + 2 * (μ_rx * (E_rxν - E_rxrν) + μ_rx² * μ_rν)
    Txνν = E_Jxνν + μ_rx * (E_rνν - E_rνrν) + 2 * (μ_rν * (E_rxν - E_rxrν) + μ_rx * μ_rν²)
    Tννν = E_Jννν + 3 * μ_rν * (E_rνν - E_rνrν) + 2 * μ_rν³
    Txxδ = E_Jxxδ + μ_rδ * (E_rxx - E_rxrx) + 2 * (μ_rx * (E_rxδ - E_rxrδ) + μ_rx² * μ_rδ)
    Txνδ = E_Jxνδ + μ_rδ * (E_rxν - E_rxrν) + μ_rν * (E_rxδ - E_rxrδ) + μ_rx * (E_rνδ - E_rνrδ) + 2 * μ_rx * μ_rν * μ_rδ
    Tννδ = E_Jννδ + μ_rδ * (E_rνν - E_rνrν) + 2 * (μ_rν * (E_rνδ - E_rνrδ) + μ_rν² * μ_rδ)
    Txδδ = E_Jxδδ + μ_rx * (E_rδδ - E_rδrδ) + 2 * (μ_rδ * (E_rxδ - E_rxrδ) + μ_rx * μ_rδ²)
    Tνδδ = E_Jνδδ + μ_rν * (E_rδδ - E_rδrδ) + 2 * (μ_rδ * (E_rνδ - E_rνrδ) + μ_rν * μ_rδ²)
    Tδδδ = E_Jδδδ + 3 * μ_rδ * (E_rδδ - E_rδrδ) + 2 * μ_rδ³ - 2 * δ⁻³

    g = (∇x, ∇ν, ∇δ)
    H = (Hxx, Hxν, Hxδ, Hνν, Hνδ, Hδδ)
    J = SMatrix{6, 3, D, 18}(
        Txxx, Txxν, Txxδ, Txνν, Txνδ, Txδδ, # ∂H/∂x
        Txxν, Txνν, Txνδ, Tννν, Tννδ, Tνδδ, # ∂H/∂ν
        Txxδ, Txνδ, Txδδ, Tννδ, Tνδδ, Tδδδ, # ∂H/∂δ
    )

    return Ω, g, H, J
end

# Third-derivative integrand methods

@inline function _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_two_pass_integrand((μ_rx, μ_rν, μ_rδ)::NTuple{3, D}, x::D, ν::D, δ::D, t::D) where {D}
    Δx = δ * t
    (rx, rν), (rxx, rxν, rνν), (rxxx, rxxν, rxνν, rννν) = _∇³neglogpdf_rician_residual_with_gradient_and_hessian(x, ν, Δx)

    # Reconstruct ∂/∂δ-derivatives of r̃ from f-derivatives at y = x + δ t
    t² = t * t
    t³ = t² * t
    fy, fyy = rx + (x - ν), rxx + 1
    rδ = t * fy
    rxδ, rνδ, rδδ = t * fyy, t * (rxν - 1), t² * fyy
    rxxδ, rxνδ, rννδ = t * rxxx, t * rxxν, t * rxνν
    rxδδ, rνδδ, rδδδ = t² * rxxx, t² * rxxν, t³ * rxxx

    # Centered first derivatives c = ∇_θ r̃ - μ
    rx_c, rν_c, rδ_c = rx - μ_rx, rν - μ_rν, rδ - μ_rδ

    # h-integrands: h_αβ = E[∂_αβ r̃ - (∂_α r̃ - μ_α) (∂_β r̃ - μ_β)]
    rxrx_c, rxrν_c, rxrδ_c = rx_c * rx_c, rx_c * rν_c, rx_c * rδ_c
    rνrν_c, rνrδ_c, rδrδ_c = rν_c * rν_c, rν_c * rδ_c, rδ_c * rδ_c
    h = (rxx - rxrx_c, rxν - rxrν_c, rxδ - rxrδ_c, rνν - rνrν_c, rνδ - rνrδ_c, rδδ - rδrδ_c)

    # T-integrands: T_αβγ = E[∂_αβγ r̃]
    #   ∂_αβγ Ω = E[∂_αβγ r̃]
    #           - Cov(∂_αβ r̃, ∂_γ r̃) - Cov(∂_αγ r̃, ∂_β r̃) - Cov(∂_βγ r̃, ∂_α r̃)
    #           + Cov3(∂_α r̃, ∂_β r̃, ∂_γ r̃) - 2 δ⁻³ 1{α=β=γ=δ}.
    Txxx = rxxx - rx_c * (3 * rxx - rxrx_c)
    Txxν = rxxν - (rxx * rν_c + rx_c * (2 * rxν - rxrν_c))
    Txνν = rxνν - (rνν * rx_c + rν_c * (2 * rxν - rxrν_c))
    Tννν = rννν - rν_c * (3 * rνν - rνrν_c)
    Txxδ = rxxδ - (rxx * rδ_c + rx_c * (2 * rxδ - rxrδ_c))
    Txνδ = rxνδ - (rxν * rδ_c + rxδ * rν_c + rx_c * (rνδ - rν_c * rδ_c))
    Tννδ = rννδ - (rνν * rδ_c + rν_c * (2 * rνδ - rνrδ_c))
    Txδδ = rxδδ - (rδδ * rx_c + rδ_c * (2 * rxδ - rxrδ_c))
    Tνδδ = rνδδ - (rδδ * rν_c + rδ_c * (2 * rνδ - rνrδ_c))
    Tδδδ = rδδδ - rδ_c * (3 * rδδ - rδrδ_c)
    T = (Txxx, Txxν, Txνν, Tννν, Txxδ, Txνδ, Tννδ, Txδδ, Tνδδ, Tδδδ)

    return (h, T)
end

@inline function _∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian_two_pass_regular_integrand((μ_rx, μ_rν, μ_rδ)::NTuple{3, D}, x::D, ν::D, δ::D, t::D) where {D}
    Δx = δ * t
    y = x + Δx
    y⁻¹ = inv(y)
    t² = t * t
    t³ = t² * t
    ty⁻¹ = t * y⁻¹

    (rx_ns, rν), (rxx_ns, rxν, rνν), (rxxx_ns, rxxν, rxνν, rννν) = _∇³neglogpdf_rician_residual_with_gradient_and_hessian_regular(x, ν, Δx)

    # Centered derivatives
    rx_ns_c = rx_ns - μ_rx
    rδ_ns_c = t * (rx_ns + (x - ν)) - μ_rδ
    rν_c = rν - μ_rν

    # h-integrands
    rx_ns_c² = rx_ns_c * rx_ns_c
    rν_c² = rν_c * rν_c
    rδ_ns_c² = rδ_ns_c * rδ_ns_c
    rxx_ns_p1 = rxx_ns + 1
    rxνm1 = rxν - 1

    h_xx = rxx_ns - rx_ns_c² + 2 * y⁻¹ * rx_ns_c
    h_xν = rxν - rx_ns_c * rν_c + y⁻¹ * rν_c
    h_νν = rνν - rν_c²
    h_xδ = t * rxx_ns_p1 - rx_ns_c * rδ_ns_c + y⁻¹ * (rδ_ns_c + t * rx_ns_c)
    h_νδ = t * rxνm1 - rν_c * rδ_ns_c + ty⁻¹ * rν_c
    h_δδ = t² * rxx_ns_p1 - rδ_ns_c² + 2 * ty⁻¹ * rδ_ns_c
    h = (h_xx, h_xν, h_xδ, h_νν, h_νδ, h_δδ)

    # T-integrands
    rνδ = t * (rxν - 1)
    rxx_ns_minus_rx_ns_c² = rxx_ns - rx_ns_c²
    rνν_minus_rν_c² = rνν - rν_c²
    rxν_minus_rx_ns_c_rν_c = rxν - rx_ns_c * rν_c
    t_rxx_ns_p1 = t * rxx_ns_p1
    t²_rxx_ns_p1 = t * t_rxx_ns_p1
    rν_c_rδ_ns_c = rν_c * rδ_ns_c

    Txxx = rxxx_ns - rx_ns_c * (3 * rxx_ns - rx_ns_c²) + 3 * y⁻¹ * rxx_ns_minus_rx_ns_c²
    Txxν = rxxν - rν_c * rxx_ns_minus_rx_ns_c² - 2 * rx_ns_c * rxν + 2 * y⁻¹ * rxν_minus_rx_ns_c_rν_c
    Txνν = rxνν + rx_ns_c * (rν_c² - rνν) - 2 * rν_c * rxν + y⁻¹ * rνν_minus_rν_c²
    Tννν = rννν - rν_c * (3 * rνν - rν_c²)
    Txxδ = t * rxxx_ns - 2 * t * rx_ns_c * rxx_ns_p1 + rδ_ns_c * (rx_ns_c² - rxx_ns) + y⁻¹ * (t * (3 * rxx_ns + 2 - rx_ns_c²) - 2 * rx_ns_c * rδ_ns_c)
    Txνδ = t * rxxν - rxν * rδ_ns_c - t_rxx_ns_p1 * rν_c - rx_ns_c * (rνδ - rν_c_rδ_ns_c) + y⁻¹ * (t * rxν + rνδ - rν_c * (rδ_ns_c + t * rx_ns_c))
    Tννδ = t * rxνν - 2 * t * rν_c * rxνm1 + (rν_c² - rνν) * rδ_ns_c + ty⁻¹ * rνν_minus_rν_c²
    Txδδ = t² * rxxx_ns - 2 * t_rxx_ns_p1 * rδ_ns_c - rx_ns_c * t²_rxx_ns_p1 + rx_ns_c * rδ_ns_c² + y⁻¹ * (3 * t²_rxx_ns_p1 - rδ_ns_c * (2 * t * rx_ns_c + rδ_ns_c))
    Tνδδ = t² * rxxν - t²_rxx_ns_p1 * rν_c - rδ_ns_c * (2 * t * rxνm1 - rν_c_rδ_ns_c) + ty⁻¹ * (2 * t * rxνm1 - 2 * rν_c_rδ_ns_c)
    Tδδδ = t³ * rxxx_ns - rδ_ns_c * (3 * t²_rxx_ns_p1 - rδ_ns_c²) + 3 * ty⁻¹ * (t²_rxx_ns_p1 - rδ_ns_c²)
    T = (Txxx, Txxν, Txνν, Tννν, Txxδ, Txνδ, Tννδ, Txδδ, Tνδδ, Tδδδ)

    return (h, T)
end

#### Quantized Rician third-order derivatives using the "Jet" formulation where we differentiate the vector Φ = (∇Ω, vech(∇²Ω))

@inline function _∇³neglogpdf_qrician_jacobian_with_jet_ad(x::D, ν::D, δ::D, order::Val) where {D}
    Φ, JΦ = withjacobian(SVector(x, ν, δ)) do p
        local x, ν, δ = p
        ∇, ∇² = _∇²neglogpdf_qrician_with_gradient(x, ν, δ, order)
        return SVector(∇..., ∇²...)
    end
    return Φ, JΦ
end

@inline function _∇³neglogpdf_qrician_vjp_with_jet_ad(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D}
    Φ, JΦ = _∇³neglogpdf_qrician_jacobian_with_jet_ad(x, ν, δ, order)
    return Φ, JΦ' * Δ
end

@inline function _∇³neglogpdf_qrician_jacobian_with_jet(x::D, ν::D, δ::D, order::Val) where {D}
    # Compute primal, expectation parts, and d(E_ϕ)/dp via quadrature
    Φ, (E_∇ω, E_∇²ω, E_Jϕ_minus_E_ϕ∇ωᵀ) = _∇³neglogpdf_qrician_jacobian_parts_with_jet(x, ν, δ, order)
    E_ϕ = SVector{9, D}(E_∇ω..., E_∇²ω...)
    J_Eϕ = E_Jϕ_minus_E_ϕ∇ωᵀ + E_ϕ * E_∇ω'

    # Apply chain rule to get the full Jacobian JΦ = dΦ/dp, exploiting sparsity of dΦ/dE_ϕ
    ∂x, ∂ν, ∂δ = E_∇ω
    J_Eϕ1, J_Eϕ2, J_Eϕ3 = J_Eϕ[1, :], J_Eϕ[2, :], J_Eϕ[3, :]
    JΦ = J_Eϕ + hcat(
        zeros(SMatrix{3, 3, D}),
        2 * ∂x * J_Eϕ1, ∂ν * J_Eϕ1 + ∂x * J_Eϕ2, ∂δ * J_Eϕ1 + ∂x * J_Eϕ3,
        2 * ∂ν * J_Eϕ2, ∂δ * J_Eϕ2 + ∂ν * J_Eϕ3, 2 * ∂δ * J_Eϕ3,
    )'

    return Φ, JΦ
end

@inline function _∇³neglogpdf_qrician_vjp_with_jet_from_parts(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D}
    # Compute JVP from the Jacobian parts
    Φ, (E_∇ω, E_∇²ω, E_Jϕ_minus_E_ϕ∇ωᵀ) = _∇³neglogpdf_qrician_jacobian_parts_with_jet(x, ν, δ, order)
    E_ϕ = SVector{9, D}(E_∇ω..., E_∇²ω...)

    Δgx, Δgν, Δgδ, ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ = Δ
    Δg = SVector{3, D}(Δgx, Δgν, Δgδ)
    ΔH = SVector{6, D}(ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ)
    Δḡ = SHermitianCompact{3, D, 6}(SVector{6, D}(2 * ΔHxx, ΔHxν, ΔHxδ, 2 * ΔHνν, ΔHνδ, 2 * ΔHδδ)) * E_∇ω
    Δϕ = SVector{9, D}((Δg + Δḡ)..., ΔH...)

    gΦ = E_Jϕ_minus_E_ϕ∇ωᵀ' * Δϕ + E_∇ω * dot(E_ϕ, Δϕ)

    return Φ, gΦ
end

@inline function _∇³neglogpdf_qrician_jacobian_parts_with_jet(x::D, ν::D, δ::D, order::Val) where {D}
    # Define a single integrand that computes all necessary terms for the primal and JVP calculations
    Ω₀, (E_∇ω, E_∇²ω, E_Jϕ_minus_E_ϕ∇ωᵀ) = f_quadrature_weighted_unit_interval(D, order) do t
        local ϕ, Jϕ = _∇³neglogpdf_qrician_inner_jacobian_with_jet(x, ν, δ, t)
        local x′ = x + δ * t
        local ∇x, ∇ν, ∇δ, ∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ = ϕ
        local ∇ω = SVector(∇x, ∇ν, ∇δ)
        local ∇²ω = SVector(∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ)
        local ϕ∇ωᵀ = ϕ * ∇ω'
        return _neglogpdf_rician(x′, ν), (∇ω, ∇²ω, Jϕ - ϕ∇ωᵀ)
    end
    Ω = Ω₀ - log(δ)

    ∂x, ∂ν, ∂δ = E_∇ω
    dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ = E_∇²ω
    Φ = SVector{9, D}(
        ∂x, ∂ν, ∂δ,
        ∂x * ∂x + dxdx, ∂x * ∂ν + dxdν, ∂x * ∂δ + dxdδ,
        ∂ν * ∂ν + dνdν, ∂ν * ∂δ + dνdδ, ∂δ * ∂δ + dδdδ,
    )

    return Φ, (E_∇ω, E_∇²ω, E_Jϕ_minus_E_ϕ∇ωᵀ)
end

@inline function _∇³neglogpdf_qrician_vjp_with_jet_one_pass(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D}
    Δgx, Δgν, Δgδ, ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ = Δ
    Δg = SVector{3, D}(Δgx, Δgν, Δgδ)
    ΔH = SVector{6, D}(ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ)

    # Define a single integrand that computes all necessary terms for the primal and JVP calculations
    Ω₀, (E_∇ω, E_∇²ω, E_JϕᵀΔ_minus_∇ωϕᵀΔ, E_J∇ω_minus_E_∇ω∇ωᵀ) = f_quadrature_weighted_unit_interval(D, order) do t
        local ϕ, Jϕ = _∇³neglogpdf_qrician_inner_jacobian_with_jet(x, ν, δ, t)
        local x′ = x + δ * t
        local ∇x, ∇ν, ∇δ, ∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ = ϕ
        local ∇ω = SVector(∇x, ∇ν, ∇δ)
        local ∇²ω = SVector(∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ)
        local JϕᵀΔ_minus_∇ωϕᵀΔ = Jϕ' * Δ - ∇ω * dot(ϕ, Δ)
        local J∇ω = Jϕ[SOneTo{3}(), :]
        local ∇ω∇ωᵀ = ∇ω * ∇ω'
        return _neglogpdf_rician(x′, ν), (∇ω, ∇²ω, JϕᵀΔ_minus_∇ωϕᵀΔ, J∇ω - ∇ω∇ωᵀ)
    end
    Ω = Ω₀ - log(δ)

    ∂x, ∂ν, ∂δ = E_∇ω
    dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ = E_∇²ω
    Φ = SVector{9, D}(
        ∂x, ∂ν, ∂δ,
        ∂x * ∂x + dxdx, ∂x * ∂ν + dxdν, ∂x * ∂δ + dxdδ,
        ∂ν * ∂ν + dνdν, ∂ν * ∂δ + dνdδ, ∂δ * ∂δ + dδdδ,
    )

    Δḡ = SHermitianCompact{3, D, 6}(SVector{6, D}(2 * ΔHxx, ΔHxν, ΔHxδ, 2 * ΔHνν, ΔHνδ, 2 * ΔHδδ)) * E_∇ω
    gΦ = E_JϕᵀΔ_minus_∇ωϕᵀΔ + E_J∇ω_minus_E_∇ω∇ωᵀ' * Δḡ + E_∇ω * (dot(E_∇ω, Δg + Δḡ) + dot(E_∇²ω, ΔH))

    return Φ, gΦ
end

@inline function _∇³neglogpdf_qrician_vjp_with_jet_two_pass(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D}
    # First pass to compute E[∇ω] needed for Δϕ and covariance term in second integrand
    δ⁻¹ = inv(δ)
    Ω₀, E_∇ω, t_nodes, w_nodes = f_quadrature_weighted_unit_interval(D, order) do t
        local x′ = x + δ * t
        local ∇x, ∇ν = _∇neglogpdf_rician(x′, ν)
        return _neglogpdf_rician(x′, ν), SVector(∇x, ∇ν, t * ∇x - δ⁻¹)
    end
    Ω = Ω₀ - log(δ)

    # Assemble the transformed sensitivity vector Δϕ, which is now constant for the main pass
    Δgx, Δgν, Δgδ, ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ = Δ
    Δg = SVector{3, D}(Δgx, Δgν, Δgδ)
    ΔH = SVector{6, D}(ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ)
    Δḡ = SHermitianCompact{3, D, 6}(SVector{6, D}(2 * ΔHxx, ΔHxν, ΔHxδ, 2 * ΔHνν, ΔHνδ, 2 * ΔHδδ)) * E_∇ω
    Δϕ = SVector{9, D}((Δg + Δḡ)..., ΔH...)

    # Second pass to compute JVP-related terms
    integrands = map(t_nodes) do t
        local ϕ, JϕᵀΔϕ = _∇³neglogpdf_qrician_inner_vjp_with_jet(Δϕ, x, ν, δ, t)
        local ∇x, ∇ν, ∇δ, dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ = ϕ
        local ∇ω = SVector(∇x, ∇ν, ∇δ)
        local ∇²ω = SVector(dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ)
        local gϕ = JϕᵀΔϕ - (∇ω - E_∇ω) * dot(ϕ, Δϕ)
        return (gϕ, ∇²ω)
    end
    E_gΦ, E_∇²ω = vecdot(w_nodes, integrands)

    # Assemble the primal output Φ
    ∂x, ∂ν, ∂δ = E_∇ω
    dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ = E_∇²ω
    Φ = SVector{9, D}(
        ∂x, ∂ν, ∂δ,
        ∂x * ∂x + dxdx, ∂x * ∂ν + dxdν, ∂x * ∂δ + dxdδ,
        ∂ν * ∂ν + dνdν, ∂ν * ∂δ + dνdδ, ∂δ * ∂δ + dδdδ,
    )

    return Φ, E_gΦ
end

@inline function _∇³neglogpdf_qrician_inner_jacobian_with_jet_ad(x::D, ν::D, δ::D, t::D) where {D}
    ϕ, Jϕ = withjacobian(SVector(x, ν, δ)) do p
        local x, ν, δ = p
        x′ = x + δ * t
        δ⁻¹ = inv(δ)
        (∇x, ∇ν), (∇xx, ∇xν, ∇νν) = _∇²neglogpdf_rician_with_gradient(x′, ν)
        ∇δ = t * ∇x - δ⁻¹
        dxdx, dxdν, dνdν = ∇xx - ∇x * ∇x, ∇xν - ∇x * ∇ν, ∇νν - ∇ν * ∇ν
        dxdδ, dνdδ, dδdδ = t * dxdx + ∇x * δ⁻¹, t * dxdν + ∇ν * δ⁻¹, t * (t * dxdx + 2 * ∇x * δ⁻¹)
        return SVector(∇x, ∇ν, ∇δ, dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ)
    end
    return ϕ, Jϕ
end

@inline function _∇³neglogpdf_qrician_inner_jacobian_with_jet(x::D, ν::D, δ::D, t::D) where {D}
    # Compute the core derivatives
    x′ = x + δ * t
    (∇x, ∇ν), (∇xx, ∇xν, ∇νν), (∇xxx, ∇xxν, ∇xνν, ∇ννν) = _∇³neglogpdf_rician_with_gradient_and_hessian(x′, ν)

    # Compute the full 9-element vector ϕ from the core derivatives
    δ⁻¹ = inv(δ)
    ∇δ = t * ∇x - δ⁻¹
    dxdx, dxdν, dνdν = ∇xx - ∇x * ∇x, ∇xν - ∇x * ∇ν, ∇νν - ∇ν * ∇ν
    dxdδ, dνdδ, dδdδ = t * dxdx + ∇x * δ⁻¹, t * dxdν + ∇ν * δ⁻¹, t * (t * dxdx + 2 * ∇x * δ⁻¹)
    ϕ = SVector{9, D}(∇x, ∇ν, ∇δ, dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ)

    # Analytically compute the Jacobian of ϕ w.r.t. p = (x, ν, δ)
    δt, δ⁻², t² = δ * t, δ⁻¹ * δ⁻¹, t^2
    dxdxdx, dxdxdν, dνdνdx, dνdνdν = ∇xxx - 2 * ∇x * ∇xx, ∇xxν - 2 * ∇x * ∇xν, ∇xνν - 2 * ∇ν * ∇xν, ∇ννν - 2 * ∇ν * ∇νν
    dxdνdx, dxdνdν = ∇xxν - ∇xx * ∇ν - ∇x * ∇xν, ∇xνν - ∇xν * ∇ν - ∇x * ∇νν
    Jϕ = SMatrix{9, 3, D}(
        ∇xx, ∇xν, t * ∇xx, dxdxdx, dxdνdx, t * dxdxdx + ∇xx * δ⁻¹, dνdνdx, t * dxdνdx + ∇xν * δ⁻¹, t * (t * dxdxdx + 2 * ∇xx * δ⁻¹),
        ∇xν, ∇νν, t * ∇xν, dxdxdν, dxdνdν, t * dxdxdν + ∇xν * δ⁻¹, dνdνdν, t * dxdνdν + ∇νν * δ⁻¹, t * (t * dxdxdν + 2 * ∇xν * δ⁻¹),
        t * ∇xx, t * ∇xν, t² * ∇xx + δ⁻², t * dxdxdx, t * dxdνdx, t² * dxdxdx + (δt * ∇xx - ∇x) * δ⁻², t * dνdνdx, t² * dxdνdx + (δt * ∇xν - ∇ν) * δ⁻², t * (t² * dxdxdx + 2 * (δt * ∇xx - ∇x) * δ⁻²),
    )

    return ϕ, Jϕ
end

@inline function _∇³neglogpdf_qrician_inner_vjp_with_jet(Δϕ::SVector{9, D}, x::D, ν::D, δ::D, t::D) where {D}
    # Compute the core derivatives
    x′ = x + δ * t
    (∇x, ∇ν), (∇xx, ∇xν, ∇νν), (∇xxx, ∇xxν, ∇xνν, ∇ννν) = _∇³neglogpdf_rician_with_gradient_and_hessian(x′, ν)

    # Compute the full 9-element vector ϕ from the core derivatives
    δ⁻¹ = inv(δ)
    δ⁻² = δ⁻¹ * δ⁻¹
    ∇δ = t * ∇x - δ⁻¹
    dxdx, dxdν, dνdν = ∇xx - ∇x * ∇x, ∇xν - ∇x * ∇ν, ∇νν - ∇ν * ∇ν
    dxdδ, dνdδ, dδdδ = t * dxdx + ∇x * δ⁻¹, t * dxdν + ∇ν * δ⁻¹, t * (t * dxdx + 2 * ∇x * δ⁻¹)
    ϕ = SVector{9, D}(∇x, ∇ν, ∇δ, dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ)

    # Compute the vector-Jacobian product g = Jϕ' * Δϕ without explicitly forming Jϕ
    dxdxdx, dxdxdν, dνdνdx, dνdνdν = ∇xxx - 2 * ∇x * ∇xx, ∇xxν - 2 * ∇x * ∇xν, ∇xνν - 2 * ∇ν * ∇xν, ∇ννν - 2 * ∇ν * ∇νν
    dxdνdx, dxdνdν = ∇xxν - ∇xx * ∇ν - ∇x * ∇xν, ∇xνν - ∇xν * ∇ν - ∇x * ∇νν

    Δϕ_∇x, Δϕ_∇ν, Δϕ_∇δ, Δϕ_dxdx, Δϕ_dxdν, Δϕ_dxdδ, Δϕ_dνdν, Δϕ_dνdδ, Δϕ_dδdδ = Δϕ
    Δϕ_∇xx = δ⁻¹ * (2 * t * Δϕ_dδdδ + Δϕ_dxdδ) + t * Δϕ_∇δ + Δϕ_∇x
    Δϕ_∇xν = δ⁻¹ * Δϕ_dνdδ + Δϕ_∇ν
    Δϕ_dxdxdx = t * (t * Δϕ_dδdδ + Δϕ_dxdδ) + Δϕ_dxdx
    Δϕ_dxdνdx = t * Δϕ_dνdδ + Δϕ_dxdν

    gx = ∇xx * Δϕ_∇xx + ∇xν * Δϕ_∇xν + dxdxdx * Δϕ_dxdxdx + dxdνdx * Δϕ_dxdνdx + dνdνdx * Δϕ_dνdν
    gν = ∇xν * Δϕ_∇xx + ∇νν * Δϕ_∇xν + dxdxdν * Δϕ_dxdxdx + dxdνdν * Δϕ_dxdνdx + dνdνdν * Δϕ_dνdν
    gδ = t * gx + δ⁻² * Δϕ_∇δ - δ⁻² * (∇x * (Δϕ_dxdδ + 2 * t * Δϕ_dδdδ) + ∇ν * Δϕ_dνdδ)
    gϕ = SVector{3, D}(gx, gν, gδ)

    return ϕ, gϕ
end

#### Specialized quadrature rules

function neglogpdf_qrician_direct(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δxν = x - ν
    I = neglogf_quadrature(zero(T), δ, order) do t̂
        t = t̂ + x
        Δ_tν = t̂ + Δxν # numerically stable when x ≈ ν, equivalent to: t - ν = t̂ + (x - ν)
        return Δ_tν^2 / 2 - log(t) - logbesseli0x(t * ν)
    end
    return I
end

function neglogpdf_qrician_right_laguerre_tail(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δxν = x - ν
    Δxν′ = Δxν + δ
    λ = δ * (Δxν + δ / 2)
    I0 = Δxν^2 / 2

    if λ > -log(eps(T))
        I1 = f_laguerre_tail_quadrature(Δxν, order) do t̂
            t = x + t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν)
        end
        I1 = -log(I1)
    else
        I1⁺ = f_laguerre_tail_quadrature(Δxν, order) do t̂
            t = x + t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν)
        end
        I1⁻ = f_laguerre_tail_quadrature(Δxν′, order) do t̂
            t = x + δ + t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν)
        end
        I1 = -log(I1⁺ - exp(-λ) * I1⁻)

        #TODO merging into one call worth it?
        # I1 = f_laguerre_tail_quadrature(Δ, order) do t̂
        #     t = x + t̂
        #     f1 = t * besseli0x(t * ν)
        #     f2 = (t + δ) * besseli0x((t + δ) * ν)
        #     return exp(-t̂^2 / 2) * (f1 - exp(-δ * (t̂ + Δ + δ / 2)) * f2)
        # end
        # I1 = -log(I1)
    end

    I = I0 + I1

    return I
end

function neglogpdf_qrician_right_halfhermite_tail(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δxν = x - ν
    Δxν′ = Δxν + δ
    λ = δ * (Δxν + δ / 2)
    I0 = Δxν^2 / 2

    if λ > -log(eps(T))
        I1 = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + t̂
            return exp(-Δxν * t̂) * t * besseli0x(t * ν)
        end
        I1 = -log(I1) - T(log2π) / 2
    else
        I1⁺ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + t̂
            return exp(-Δxν * t̂) * t * besseli0x(t * ν)
        end
        I1⁻ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + δ + t̂
            return exp(-Δxν′ * t̂) * t * besseli0x(t * ν)
        end
        I1 = -log(I1⁺ - exp(-λ) * I1⁻) - T(log2π) / 2

        #TODO merging into one call worth it?
        # I1 = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
        #     t = x + t̂
        #     f1 = t * besseli0x(t * ν)
        #     f2 = (t + δ) * besseli0x((t + δ) * ν)
        #     return exp(-Δ * t̂) * (f1 - exp(-δ * (t̂ + Δ + δ / 2)) * f2)
        # end
        # I1 = -log(I1) - T(log2π) / 2
    end

    I = I0 + I1

    return I
end

function neglogpdf_qrician_left_laguerre_tail(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δ′ = ν - x
    Δ = Δ′ - δ # NOTE: equivalent to Δ = ν - (x + δ), but DO NOT USE ν - (x + δ) directly, as it may be inaccurate due to cancellation
    λ = δ * (Δ′ - δ / 2) # NOTE: equivalent to λ = δ * (Δ + δ / 2)
    I0 = Δ^2 / 2

    if λ > -log(eps(T))
        I1 = f_laguerre_tail_quadrature(Δ, order) do t̂
            t = x + δ - t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1 = -log(I1)
    else
        I1⁺ = f_laguerre_tail_quadrature(Δ, order) do t̂
            t = x + δ - t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1⁻ = f_laguerre_tail_quadrature(Δ′, order) do t̂
            t = x - t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1⁻ = exp(-λ) * I1⁻
        I1 = -log(I1⁺ - I1⁻)

        #TODO merging into one call worth it?
        # I1 = f_laguerre_tail_quadrature(Δ, order) do t̂
        #     t = x - t̂
        #     I1⁺ = exp(-t̂^2 / 2) * (t + δ) * besseli0x((t + δ) * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        #     I1⁻ = exp(-t̂^2 / 2) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        #     return I1⁺ - exp(-λ) * I1⁻
        # end
        # I1 = -log(I1)
    end

    I = I0 + I1

    return I
end

function neglogpdf_qrician_left_halfhermite_tail(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δ′ = ν - x
    Δ = Δ′ - δ # NOTE: equivalent to Δ = ν - (x + δ), but DO NOT USE ν - (x + δ) directly, as it may be inaccurate due to cancellation
    λ = δ * (Δ′ - δ / 2) # NOTE: equivalent to λ = δ * (Δ + δ / 2)
    I0 = Δ^2 / 2

    if λ > -log(eps(T))
        I1 = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + δ - t̂
            return exp(-Δ * t̂) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1 = -log(I1) - T(log2π) / 2
    else
        I1⁺ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + δ - t̂
            return exp(-Δ * t̂) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1⁻ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x - t̂
            return exp(-Δ′ * t̂) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1 = -log(I1⁺ - exp(-λ) * I1⁻) - T(log2π) / 2

        #TODO merging into one call worth it?
        # I1 = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
        #     t = x - t̂
        #     I1⁺ = exp(-Δ * t̂) * (t + δ) * besseli0x((t + δ) * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        #     I1⁻ = exp(-Δ′ * t̂) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        #     return I1⁺ - exp(-λ) * I1⁻
        # end
        # I1 = -log(I1) - T(log2π) / 2
    end

    I = I0 + I1

    return I
end

#### Gaussian quadrature

const DEFAULT_GAUSSLEGENDRE_ORDER = 16
const DEFAULT_GAUSSLAGUERRE_ORDER = 16

@generated function gausslegendre_unit_interval(::Val{order}, ::Type{T}) where {order, T}
    x, w = GaussLegendre.gausslegendre(order, BigFloat) # compute nodes and weights in `BigFloat`, then convert to type `T`
    x = SVector{order, T}((1 .+ x) ./ 2) # rescale from [-1, 1] to [0, 1]
    w = SVector{order, T}(w ./ 2) # adjust weights to account for rescaling
    return :($x, $w)
end

@generated function gausslaguerre_positive_real_axis(::Val{order}, ::Type{Float64}) where {order}
    x, w = gausslaguerre(order) # note: nodes and weights are hardcoded to Float64 in FastGaussQuadrature.jl
    x = SVector{order, Float64}(x) # nodes lie in [0, ∞)
    w = SVector{order, Float64}(w) # exponentially decreasing weights
    return :($x, $w)
end
@inline gausslaguerre_positive_real_axis(::Val{order}, ::Type{T}) where {order, T} = map(Fix1(convert, SVector{order, T}), gausslaguerre_positive_real_axis(Val(order), Float64))

@generated function gausshalfhermite_positive_real_axis(::Val{order}, ::Type{T}, ::Val{γ}) where {order, T, γ}
    @assert γ > -1 "γ must be greater than -1"
    x, w = gausshalfhermite_gw(order, BigFloat(γ); normalize = true) # compute nodes and weights in `BigFloat`, then convert to type `T`
    x = SVector{order, T}(T.(x)) # nodes lie in [0, ∞)
    w = SVector{order, T}(T.(w)) # exponentially decreasing weights
    return :($x, $w)
end

@inline function f_quadrature(f::F, ::Type{T}, x₀::Real, δ::Real, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, T, order}
    # I = ∫_{x₀}^{x₀ + δ} [f(t)] dt
    x, w = gausslegendre_unit_interval(Val(order), T)
    y = @. f(x₀ + δ * x)
    return vecdot(w, y) * δ
end
@inline f_quadrature(f::F, x₀::Real, δ::Real, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order} = f_quadrature(f, basefloattype(x₀, δ), x₀, δ, Val(order))

@inline function f_quadrature_weighted_unit_interval(f::F, ::Type{T}, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, T, order}
    # I = ∫_{0}^{1} [exp(Ω - ω(t)) f(t)] dt where Ω = -log(∫_{0}^{1} exp(-ω(t)) dt)
    x, w = gausslegendre_unit_interval(Val(order), T)
    ω_and_y = @. f(x)
    ω, y = first.(ω_and_y), last.(ω_and_y)
    Ω = weighted_neglogsumexp(w, ω)
    w′ = @. exp(Ω - ω) * w
    I = vecdot(w′, y)
    return Ω, I, x, w′
end

@inline function neglogf_quadrature(neglogf::F, ::Type{T}, x₀::Real, δ::Real, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, T, order}
    # I = ∫_{x₀}^{x₀ + δ} [f(t)] dt, where f(t) = exp(-neglogf(t))
    x, w = gausslegendre_unit_interval(Val(order), T)
    neglogy = @. neglogf(x₀ + δ * x)
    return weighted_neglogsumexp(w, neglogy) .- log(δ)
end
@inline neglogf_quadrature(neglogf::F, x₀::Real, δ::Real, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order} = neglogf_quadrature(neglogf, basefloattype(x₀, δ), x₀, δ, Val(order))

@inline function neglogf_quadrature_unit_interval(neglogf::F, ::Type{T}, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, T, order}
    # I = ∫_{0}^{1} [f(t)] dt, where f(t) = exp(-neglogf(t))
    x, w = gausslegendre_unit_interval(Val(order), T)
    neglogy = @. neglogf(x)
    return weighted_neglogsumexp(w, neglogy)
end

@inline function f_laguerre_tail_quadrature(f::F, ::Type{T}, λ::Real, ::Val{order} = Val(DEFAULT_GAUSSLAGUERRE_ORDER)) where {F, T, order}
    # I = ∫_{0}^{∞} [exp(-λt) f(t)] dt
    x, w = gausslaguerre_positive_real_axis(Val(order), T)
    y = @. f(x / λ)
    return vecdot(w, y) / λ
end
@inline f_laguerre_tail_quadrature(f::F, λ::Real, ::Val{order} = Val(DEFAULT_GAUSSLAGUERRE_ORDER)) where {F, order} = f_laguerre_tail_quadrature(f, basefloattype(λ), λ, Val(order))

@inline function f_halfhermite_tail_quadrature(f::F, ::Type{T}, ::Val{γ}, ::Val{order} = Val(DEFAULT_GAUSSLAGUERRE_ORDER)) where {F, T, order, γ}
    # I = ∫_{0}^{∞} [x^γ exp(-t^2/2) f(t)] / √(2π) dt
    x, w = gausshalfhermite_positive_real_axis(Val(order), T, Val(γ))
    y = @. f(x)
    return vecdot(w, y)
end
@inline f_halfhermite_tail_quadrature(f::F, ::Val{γ}, ::Val{order} = Val(DEFAULT_GAUSSLAGUERRE_ORDER)) where {F, order, γ} = f_halfhermite_tail_quadrature(f, basefloattype(γ), Val(γ), Val(order))

@inline function weighted_neglogsumexp(w::SVector{N}, y::SVector{N}) where {N}
    min_y = minimum(y)
    ȳ = exp.(min_y .- y)
    return min_y - log(vecdot(w, ȳ))
end

@inline function weighted_neglogsumexp(w::SVector{N}, y::SVector{N, <:SVector{M}}) where {N, M}
    min_y = reduce(BroadcastFunction(min), y) # elementwise minimum
    y = reducehcat(y) # stack as columns
    ȳ = exp.(min_y .- y)
    return min_y .- log.(vecdot(w, ȳ))
end

# Convert vector of vectors in flat matrix. Note that `init` is necessary to get the correct type when `N = 1`, otherwise you get an SVector{M} instead of an SMatrix{M, 1}
@inline reducehcat(y::SVector{N, <:SVector{M, T}}) where {N, M, T} = reduce(hcat, y; init = SMatrix{M, 0, T}())

@generated function splat_tuple_of_sarrays(y::T) where {M, T <: Tuple{Vararg{StaticArray, M}}}
    L = sum(length, T.parameters)
    D = promote_type(eltype.(T.parameters)...)
    exprs = [:(y[$i]...) for i in 1:M]
    return :(SVector{$L, $D}($(exprs...)))
end
@generated function unsplat_tuple_of_sarrays(::Type{T}, y::NTuple{N, D}) where {N, D, M, T <: Tuple{Vararg{StaticArray, M}}}
    exprs = []
    @assert sum(length, T.parameters) == N "sum(length, T.parameters) = $(sum(length, T.parameters)) != N = $N"
    offset = 0
    for Sᵢ in T.parameters
        Lᵢ = length(Sᵢ)
        args = [:(y[$(offset + j)]) for j in 1:Lᵢ]
        push!(exprs, :($StaticArrays.SArray{$(Sᵢ.parameters...)}($(args...))))
        offset += Lᵢ
    end
    return :(tuple($(exprs...)))
end
@inline unsplat_tuple_of_sarrays(::Type{T}, y::SVector{N, D}) where {N, D, M, T <: Tuple{Vararg{StaticArray, M}}} = unsplat_tuple_of_sarrays(T, Tuple(y))

@inline vecdot(w::SVector{N}, y::SVector{N, T}) where {N, M, T <: Tuple{Vararg{StaticArray, M}}} = unsplat_tuple_of_sarrays(T, vecdot(w, map(splat_tuple_of_sarrays, y)))
@inline vecdot(w::SVector{N}, y::SVector{N}) where {N} = dot(w, y)
@inline vecdot(w::SVector{N}, y::SVector{N, <:SVector{M}}) where {N, M} = vecdot(w, reducehcat(y))
@inline vecdot(w::SVector{N}, y::SMatrix{M, N}) where {N, M} = y * w
