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
    #   Î₀(z) = I₀(z) / (exp(z) / √2πz).
    #
    # The negative likelihood is then be written as
    #
    #        -logp(x | ν, σ = 1) = (x - ν)^2 / 2 - log(x / ν) / 2 - logÎ₀(x * ν) + log√2π.
    #   ∂/∂x -logp(x | ν, σ = 1) = x - ν - 1 / 2x - ∂/∂x logÎ₀(x * ν).
    #   ∂/∂ν -logp(x | ν, σ = 1) = ν - x + 1 / 2ν - ∂/∂ν logÎ₀(x * ν).
    #
    # All that must be approximated then is `d/dz logÎ₀(z)` where `z = x * ν`:
    #
    #   d/dz logÎ₀(z) =  1/2z + (I₁(z) / I₀(z) - 1)
    #                 ≈ -1/8z^2 - 1/8z^3 - 25/128z^4 - 13/32z^5 - 1073/1024z^6 - 103/32z^7 + 𝒪(1/z^8)   (z >> 1)
    #                 ≈  1/2z - 1 + z/2 - z^3/16 + z^5/96 - 11*z^7/6144 + 𝒪(z^9)                        (z << 1)
    #   ∂/∂x logÎ₀(z) = ν * d/dz logÎ₀(z)
    #   ∂/∂ν logÎ₀(z) = x * d/dz logÎ₀(z)

    # Note: there are really three relevant limits: z << 1, z >> 1, and x ≈ ν.
    # Could plausibly better account for the latter case, though it is tested quite robustly
    z = x * ν
    T = checkedfloattype(z)

    r, r_tail, r′, r′′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z)
    if z < first(neglogpdf_rician_parts_branches(T))
        ∂x = x - r * ν - inv(x)
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

    r, r_tail, r′, r′′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z)
    if z < first(neglogpdf_rician_parts_branches(T))
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

    r, r_tail, r′, r′′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z)
    if z < first(neglogpdf_rician_parts_branches(T))
        x⁻¹ = inv(x)
        ∂x = x - r * ν - x⁻¹
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

    r, r_tail, r′, r′′, two_r′_plus_z_r′′ = _neglogpdf_rician_parts(z)
    if z < first(neglogpdf_rician_parts_branches(T))
        x⁻¹ = inv(x)
        x⁻² = x⁻¹ * x⁻¹
        ∂x = x - r * ν - x⁻¹
        ∂ν = ν - r * x
        ∂xx = x⁻² + (one(T) - ν^2 * r′)
        ∂xν = -(r + z * r′)
        ∂νν = one(T) - x^2 * r′
    else
        x⁻¹, ν⁻¹ = inv(x), inv(ν)
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
@inline ∇²neglogpdf_qrician(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇²neglogpdf_qrician_midpoint(promote(x, ν, δ)...) : _∇²neglogpdf_qrician(promote(x, ν, δ)..., order)
@inline ∇²neglogpdf_qrician_with_gradient(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇²neglogpdf_qrician_midpoint_with_gradient(promote(x, ν, δ)...) : _∇²neglogpdf_qrician_with_gradient(promote(x, ν, δ)..., order)
@inline ∇²neglogpdf_qrician_with_primal_and_gradient(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇²neglogpdf_qrician_midpoint_with_primal_and_gradient(promote(x, ν, δ)...) : _∇²neglogpdf_qrician_with_primal_and_gradient(promote(x, ν, δ)..., order)

# Fast-path for single point quadrature which avoids computing the primal; equivalent to using the midpoint rule approximations for the integrals
@inline _neglogpdf_qrician_midpoint(x::D, ν::D, δ::D) where {D} = _neglogpdf_rician(x + δ / 2, ν) - log(δ)
@inline function _∇neglogpdf_qrician_midpoint(x::D, ν::D, δ::D) where {D}
    ∂x, ∂ν = _∇neglogpdf_rician(x + δ / 2, ν)
    return ∂x, ∂ν, ∂x / 2 - inv(δ)
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

@inline _neglogpdf_qrician(x::D, ν::D, δ::D, order::Val) where {D} = neglogf_quadrature(Base.Fix2(_neglogpdf_rician, ν), x, δ, order)
@inline _∇neglogpdf_qrician(x::D, ν::D, δ::D, order::Val) where {D} = last(_∇neglogpdf_qrician_with_primal(x, ν, δ, order))

@inline function _∇neglogpdf_qrician_with_primal(x::D, ν::D, δ::D, order::Val) where {D}
    # Differentiate the approximation:
    # ω(t) = neglogpdf_rician(t, ν)
    #    I = ∫_{x}^{x+δ} exp(-ω(x′)) dx′ = ∫_{0}^{1} exp(-ω(x + δ * t)) * δ dt
    #    Ω = -logI
    #   ∂Ω = -∂(logI) = -∂I / I
    #      = -exp(Ω) * ∫_{0}^{1} ∂(exp(-ω(x + δ * t)) * δ) dt
    # where Ω = -logI is constant w.r.t. ∂.
    Ω₀, (∂x, ∂ν, ∂δ) = f_quadrature_weighted_unit_interval(D, order) do t
        x′ = x + δ * t
        ∇x, ∇ν = _∇neglogpdf_rician(x′, ν)
        ∇δ = t * ∇x - inv(δ)
        return _neglogpdf_rician(x′, ν), SVector{3, D}(∇x, ∇ν, ∇δ)
    end
    Ω = Ω₀ - log(δ)

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

@scalar_rule _neglogpdf_qrician(x, ν, δ, order::Val) (_∇neglogpdf_qrician_with_primal(x, ν, δ, order)[2]..., NoTangent())
@dual_rule_from_frule _neglogpdf_qrician(x, ν, δ, !(order::Val))

@inline function _∇²neglogpdf_qrician_with_primal_and_gradient(x::D, ν::D, δ::D, order::Val) where {D}
    # Differentiate the approximation, i.e. differentiate through the quadrature:
    #  ω(t) = neglogpdf_rician(t, ν)
    #     I = ∫_{x}^{x+δ} exp(-ω(x′)) dx′ = ∫_{0}^{1} exp(-ω(x + δ * t)) * δ dt
    #     Ω = -logI
    #    ∂Ω = -∂(logI) = -∂I / I
    #       = -exp(Ω) * ∫_{0}^{1} ∂(exp(-ω(x + δ * t)) * δ) dt
    # ∂₁∂₂Ω = -∂₁∂₂(logI) = -∂₁(∂₂I / I) = (∂₁I)(∂₂I) / I² - ∂₁∂₂I / I
    #       = (∂₁Ω)(∂₂Ω) - exp(Ω) * ∫_{0}^{1} ∂₁∂₂(exp(-ω(x + δ * t)) * δ) dt
    # where Ω = -logI is constant w.r.t. ∂₁ and ∂₂.
    logδ, δ⁻¹ = log(δ), inv(δ)
    Ω₀, (∂x, ∂ν, ∂δ, ∂x∂x, ∂x∂ν, ∂x∂δ, ∂ν∂ν, ∂ν∂δ, ∂δ∂δ) = f_quadrature_weighted_unit_interval(D, order) do t
        x′ = x + δ * t
        (∇x, ∇ν), (∇xx, ∇xν, ∇νν) = _∇²neglogpdf_rician_with_gradient(x′, ν)
        ∇δ = t * ∇x - δ⁻¹
        dxdx, dxdν, dνdν = ∇xx - ∇x * ∇x, ∇xν - ∇x * ∇ν, ∇νν - ∇ν * ∇ν
        dxdδ, dνdδ, dδdδ = t * dxdx + ∇x * δ⁻¹, t * dxdν + ∇ν * δ⁻¹, t * (t * dxdx + 2 * ∇x * δ⁻¹)
        return _neglogpdf_rician(x′, ν), SVector{9, D}(∇x, ∇ν, ∇δ, dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ)
    end
    Ω = Ω₀ - logδ

    return Ω, (∂x, ∂ν, ∂δ), (∂x * ∂x + ∂x∂x, ∂x * ∂ν + ∂x∂ν, ∂x * ∂δ + ∂x∂δ, ∂ν * ∂ν + ∂ν∂ν, ∂ν * ∂δ + ∂ν∂δ, ∂δ * ∂δ + ∂δ∂δ)

    #=
    # Differentiate the approximation (using precomputed Ω)
    (∂x, ∂ν, ∂δ, ∂x∂x, ∂x∂ν, ∂x∂δ, ∂ν∂ν, ∂ν∂δ, ∂δ∂δ) = f_quadrature(zero(x), one(x), order) do t
        δt = δ * t
        x′ = x + δt
        (∇x, ∇ν), (∇xx, ∇xν, ∇νν) = _∇²neglogpdf_rician_with_gradient(x′, ν)
        dx, dν, dδ = ∇x * δ, ∇ν * δ, ∇x * δt - one(x)
        dxdx, dxdν, dνdν = (∇xx - ∇x * ∇x) * δ, (∇xν - ∇x * ∇ν) * δ, (∇νν - ∇ν * ∇ν) * δ
        dxdδ, dνdδ, dδdδ = ∇x - δt * (∇x * ∇x - ∇xx), ∇ν - δt * (∇x * ∇ν - ∇xν), t * (2 * ∇x - δt * (∇x * ∇x - ∇xx))
        integrands = SVector{9, D}(dx, dν, dδ, dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ)
        return exp(Ω - _neglogpdf_rician(x′, ν)) * integrands
    end

    return Ω, (∂x, ∂ν, ∂δ), (∂x * ∂x + ∂x∂x, ∂x * ∂ν + ∂x∂ν, ∂x * ∂δ + ∂x∂δ, ∂ν * ∂ν + ∂ν∂ν, ∂ν * ∂δ + ∂ν∂δ, ∂δ * ∂δ + ∂δ∂δ)
    =#

    #=
    # Differentiate the approximation for (∂x, ∂ν, ∂²xx, ∂²xν, ∂²νν) and use FTC for (∂δ, ∂²xδ, ∂²νδ, ∂²δδ):
    # Derivatives of Ω w.r.t. (x, ν, δ) where Ω = -logI = -log ∫_{x}^{x+δ} F(x′,ν) dx′ = -log ∫_{0}^{δ} F(x+δt,ν) dϵ = -log ∫_{0}^{1} F(x+δt,ν) δ dt,
    # F(x′,ν) = exp(-f(x′,ν)), f(x′,ν) = neglogpdf_rician(x′,ν), and x′ = x+ϵ = x+δt.
    # First derivatives ∂Ω/∂x, ∂Ω/∂ν via quadrature:
    #      ∂Ω/∂α = -∂/∂α (logI) = -∂I/∂α / I = ∫_{0}^{δ} exp(Ω - f(x+ϵ,ν)) ∂/∂α f(x+ϵ,ν) dϵ
    # First derivative ∂Ω/∂δ via quadrature:
    #      ∂Ω/∂δ = ∂/∂δ (-log ∫_{0}^{1} F(x+δt,ν) δ dt) = -∂/∂δ (∫_{0}^{1} F(x+δt,ν) δ dt) / I = -(∫_{0}^{1} F(x+δt,ν) + F_y(x+δt,ν) * δt dt) / I = -(∫_{0}^{δ} F(x+ϵ,ν) + F_y(x+ϵ,ν) * ϵ dϵ) / I / δ
    #            = -(∫_{0}^{δ} exp(Ω - f(x+ϵ,ν)) * (1 - f_y(x+ϵ,ν) * ϵ) dϵ) / δ
    # Second derivatives ∂²Ω/∂x², ∂²Ω/∂x∂ν, ∂²Ω/∂ν² via quadrature:
    #   ∂²Ω/∂α∂β = -∂²/∂α∂β (logI) = -∂/∂α (∂I/∂β / I) = (∂I/∂α)(∂I/∂β) / I^2 - ∂²I/∂α∂β / I
    #      ∂I/∂α = ∫_{0}^{δ} ∂/∂α exp(-f(x+ϵ,ν)) dϵ = ∫_{0}^{δ} exp(-f(x+ϵ,ν)) -∂/∂α f(x+ϵ,ν) dϵ
    #   ∂²I/∂α∂β = ∫_{0}^{δ} ∂²/∂α∂β exp(-f(x+ϵ,ν)) dϵ = ∫_{0}^{δ} exp(-f(x+ϵ,ν)) (∂/∂α f(x+ϵ,ν))(∂/∂β f(x+ϵ,ν)) - ∂²/∂α∂β f(x+ϵ,ν) dϵ
    # Second derivative ∂Ω/∂δ via quadrature:
    # This allows us to integrate the gradient essentially for free, since we need it for the Hessian anyways.
    (∂x, ∂ν, ∂²xx, ∂²xν, ∂²νν) = f_quadrature(x, δ, order) do x′
        ∇, ∇² = _∇²neglogpdf_rician_with_gradient(x′, ν)
        integrands = SVector{5, D}(∇[1], ∇[2], ∇[1] * ∇[1] - ∇²[1], ∇[1] * ∇[2] - ∇²[2], ∇[2] * ∇[2] - ∇²[3]) # ∇ and ∇∇ᵀ - ∇²
        return exp(Ω - _neglogpdf_rician(x′, ν)) * integrands
    end
    ∂²xx = ∂x * ∂x - ∂²xx # d²Ω/dx² = (∂I/∂x)² - ∂²I/∂x²
    ∂²xν = ∂x * ∂ν - ∂²xν # d²Ω/dxdν = (∂I/∂x)(∂I/∂ν) - ∂²I/∂x∂ν
    ∂²νν = ∂ν * ∂ν - ∂²νν # d²Ω/dν² = (∂I/∂ν)² - ∂²I/∂ν²

    # Cross-derivatives d²Ω/dxdδ, d²Ω/dν∂δ, d²Ω/dδ² via FTC:
    #      ∂Ω/∂δ = -exp(Ω - f(x + δ, ν))
    #   ∂²Ω/∂δ∂α = ∂Ω/∂δ * (∂Ω/∂α - ∂/∂α f(x + δ, ν)) where α = x, ν, δ
    ∂δ = -exp(Ω - _neglogpdf_rician(x + δ, ν))
    ∂x⁺, ∂ν⁺ = _∇neglogpdf_rician(x + δ, ν) # note: ∂δ⁺ = ∂x⁺
    ∂²xδ = ∂δ * (∂x - ∂x⁺) # d²Ω/dxdδ = ∂Ω/∂δ * (∂Ω/∂x - ∂x⁺)
    ∂²νδ = ∂δ * (∂ν - ∂ν⁺) # d²Ω/dν∂δ = ∂Ω/∂δ * (∂Ω/∂ν - ∂ν⁺)
    ∂²δδ = ∂δ * (∂δ - ∂x⁺) # d²Ω/dδ² = ∂Ω/∂δ * (∂Ω/∂δ - ∂δ⁺)

    return Ω, (∂x, ∂ν, ∂δ), (∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ)
    =#
end
@inline _∇²neglogpdf_qrician_with_gradient(x::D, ν::D, δ::D, order::Val) where {D} = Base.tail(_∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ, order))
@inline _∇²neglogpdf_qrician(x::D, ν::D, δ::D, order::Val) where {D} = last(_∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ, order))

@inline function _∇²neglogpdf_qrician_with_jacobian_ad(x::D, ν::D, δ::D, order::Val) where {D}
    Φ, JΦ = withjacobian(SVector(x, ν, δ)) do p
        local x, ν, δ = p
        ∇, ∇² = _∇²neglogpdf_qrician_with_gradient(x, ν, δ, order)
        return SVector(∇..., ∇²...)
    end
    return Φ, JΦ
end

@inline function _∇²neglogpdf_qrician_jvp_ad(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D}
    Φ, JΦ = _∇²neglogpdf_qrician_with_jacobian_ad(x, ν, δ, order)
    return Φ, JΦ' * Δ
end

@inline function _∇²neglogpdf_qrician_with_jacobian(x::D, ν::D, δ::D, order::Val) where {D}
    # Compute primal, expectation parts, and d(E_ϕ)/dp via quadrature
    Φ, (E_∇ω, E_∇²ω, E_Jϕ_minus_E_ϕ∇ωᵀ) = _∇²neglogpdf_qrician_jac_parts(x, ν, δ, order)
    E_ϕ = SVector{9, D}(E_∇ω..., E_∇²ω...)
    J_Eϕ = E_Jϕ_minus_E_ϕ∇ωᵀ + E_ϕ * E_∇ω'

    # Apply chain rule to get the full Jacobian JΦ = dΦ/dp, exploiting sparsity of dΦ/dE_ϕ.
    ∂x, ∂ν, ∂δ = E_∇ω
    J_Eϕ1, J_Eϕ2, J_Eϕ3 = J_Eϕ[1, :], J_Eϕ[2, :], J_Eϕ[3, :]
    JΦ = J_Eϕ + hcat(
        zeros(SMatrix{3, 3, D}),
        2 * ∂x * J_Eϕ1, ∂ν * J_Eϕ1 + ∂x * J_Eϕ2, ∂δ * J_Eϕ1 + ∂x * J_Eϕ3,
        2 * ∂ν * J_Eϕ2, ∂δ * J_Eϕ2 + ∂ν * J_Eϕ3, 2 * ∂δ * J_Eϕ3,
    )'

    return Φ, JΦ
end

@inline function _∇²neglogpdf_qrician_jvp_via_jac_parts(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D}
    # Compute JVP from the Jacobian parts
    Φ, (E_∇ω, E_∇²ω, E_Jϕ_minus_E_ϕ∇ωᵀ) = _∇²neglogpdf_qrician_jac_parts(x, ν, δ, order)
    E_ϕ = SVector{9, D}(E_∇ω..., E_∇²ω...)

    Δgx, Δgν, Δgδ, ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ = Δ
    Δg = SVector{3, D}(Δgx, Δgν, Δgδ)
    ΔH = SVector{6, D}(ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ)
    Δḡ = SHermitianCompact{3, D, 6}(SVector{6, D}(2 * ΔHxx, ΔHxν, ΔHxδ, 2 * ΔHνν, ΔHνδ, 2 * ΔHδδ)) * E_∇ω
    Δϕ = SVector{9, D}((Δg + Δḡ)..., ΔH...)

    gΦ = E_Jϕ_minus_E_ϕ∇ωᵀ' * Δϕ + E_∇ω * dot(E_ϕ, Δϕ)

    return Φ, gΦ
end

@inline function _∇²neglogpdf_qrician_jac_parts(x::D, ν::D, δ::D, order::Val) where {D}
    # Define a single integrand that computes all necessary terms for the primal and JVP calculations.
    _, (E_∇ω, E_∇²ω, E_Jϕ_minus_E_ϕ∇ωᵀ) = f_quadrature_weighted_unit_interval(D, order) do t
        local ϕ, Jϕ = _∇²neglogpdf_qrician_inner_jac(x, ν, δ, t)
        local x′ = x + δ * t
        local ∇x, ∇ν, ∇δ, ∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ = ϕ
        local ∇ω = SVector(∇x, ∇ν, ∇δ)
        local ∇²ω = SVector(∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ)
        local ϕ∇ωᵀ = ϕ * ∇ω'
        return _neglogpdf_rician(x′, ν), (∇ω, ∇²ω, Jϕ - ϕ∇ωᵀ)
    end
    ∂x, ∂ν, ∂δ = E_∇ω
    dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ = E_∇²ω
    Φ = SVector{9, D}(
        ∂x, ∂ν, ∂δ,
        ∂x * ∂x + dxdx, ∂x * ∂ν + dxdν, ∂x * ∂δ + dxdδ,
        ∂ν * ∂ν + dνdν, ∂ν * ∂δ + dνdδ, ∂δ * ∂δ + dδdδ,
    )

    return Φ, (E_∇ω, E_∇²ω, E_Jϕ_minus_E_ϕ∇ωᵀ)
end

@inline function _∇²neglogpdf_qrician_jvp_via_one_pass(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D}
    Δgx, Δgν, Δgδ, ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ = Δ
    Δg = SVector{3, D}(Δgx, Δgν, Δgδ)
    ΔH = SVector{6, D}(ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ)

    # Define a single integrand that computes all necessary terms for the primal and JVP calculations.
    _, (E_∇ω, E_∇²ω, E_JϕᵀΔ_minus_∇ωϕᵀΔ, E_J∇ω_minus_E_∇ω∇ωᵀ) = f_quadrature_weighted_unit_interval(D, order) do t
        local ϕ, Jϕ = _∇²neglogpdf_qrician_inner_jac(x, ν, δ, t)
        local x′ = x + δ * t
        local ∇x, ∇ν, ∇δ, ∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ = ϕ
        local ∇ω = SVector(∇x, ∇ν, ∇δ)
        local ∇²ω = SVector(∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ)
        local JϕᵀΔ_minus_∇ωϕᵀΔ = Jϕ' * Δ - ∇ω * dot(ϕ, Δ)
        local J∇ω = Jϕ[SOneTo{3}(), :]
        local ∇ω∇ωᵀ = ∇ω * ∇ω'
        return _neglogpdf_rician(x′, ν), (∇ω, ∇²ω, JϕᵀΔ_minus_∇ωϕᵀΔ, J∇ω - ∇ω∇ωᵀ)
    end

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

@inline function _∇²neglogpdf_qrician_jvp_via_two_pass(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D}
    # First pass to compute E[∇ω] needed for Δϕ and covariance term in second integrand
    _, E_∇ω, t_nodes, w_nodes = f_quadrature_weighted_unit_interval(D, order) do t
        local x′ = x + δ * t
        local ∇x, ∇ν = _∇neglogpdf_rician(x′, ν)
        return _neglogpdf_rician(x′, ν), SVector(∇x, ∇ν, t * ∇x - inv(δ))
    end

    # Assemble the transformed sensitivity vector Δϕ, which is now constant for the main pass
    Δgx, Δgν, Δgδ, ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ = Δ
    Δg = SVector{3, D}(Δgx, Δgν, Δgδ)
    ΔH = SVector{6, D}(ΔHxx, ΔHxν, ΔHxδ, ΔHνν, ΔHνδ, ΔHδδ)
    Δḡ = SHermitianCompact{3, D, 6}(SVector{6, D}(2 * ΔHxx, ΔHxν, ΔHxδ, 2 * ΔHνν, ΔHνδ, 2 * ΔHδδ)) * E_∇ω
    Δϕ = SVector{9, D}((Δg + Δḡ)..., ΔH...)

    # Second pass to compute JVP-related terms
    integrands = map(t_nodes) do t
        local ϕ, JϕᵀΔϕ = _∇²neglogpdf_qrician_inner_jvp(Δϕ, x, ν, δ, t)
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

@inline function _∇²neglogpdf_qrician_inner_jac_ad(x::D, ν::D, δ::D, t::D) where {D}
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
end

@inline function _∇²neglogpdf_qrician_inner_jac(x::D, ν::D, δ::D, t::D) where {D}
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
    δt, δ⁻², t² = δ * t, δ⁻¹^2, t^2
    dxdxdx, dxdxdν, dνdνdx, dνdνdν = ∇xxx - 2 * ∇x * ∇xx, ∇xxν - 2 * ∇x * ∇xν, ∇xνν - 2 * ∇ν * ∇xν, ∇ννν - 2 * ∇ν * ∇νν
    dxdνdx, dxdνdν = ∇xxν - ∇xx * ∇ν - ∇x * ∇xν, ∇xνν - ∇xν * ∇ν - ∇x * ∇νν
    Jϕ = SMatrix{9, 3, D}(
        ∇xx, ∇xν, t * ∇xx, dxdxdx, dxdνdx, t * dxdxdx + ∇xx * δ⁻¹, dνdνdx, t * dxdνdx + ∇xν * δ⁻¹, t * (t * dxdxdx + 2 * ∇xx * δ⁻¹),
        ∇xν, ∇νν, t * ∇xν, dxdxdν, dxdνdν, t * dxdxdν + ∇xν * δ⁻¹, dνdνdν, t * dxdνdν + ∇νν * δ⁻¹, t * (t * dxdxdν + 2 * ∇xν * δ⁻¹),
        t * ∇xx, t * ∇xν, t² * ∇xx + δ⁻², t * dxdxdx, t * dxdνdx, t² * dxdxdx + (δt * ∇xx - ∇x) * δ⁻², t * dνdνdx, t² * dxdνdx + (δt * ∇xν - ∇ν) * δ⁻², t * (t² * dxdxdx + 2 * (δt * ∇xx - ∇x) * δ⁻²),
    )

    return ϕ, Jϕ
end

@inline function _∇²neglogpdf_qrician_inner_jvp(Δϕ::SVector{9, D}, x::D, ν::D, δ::D, t::D) where {D}
    # Compute the core derivatives
    x′ = x + δ * t
    (∇x, ∇ν), (∇xx, ∇xν, ∇νν), (∇xxx, ∇xxν, ∇xνν, ∇ννν) = _∇³neglogpdf_rician_with_gradient_and_hessian(x′, ν)

    # Compute the full 9-element vector ϕ from the core derivatives
    δ⁻¹ = inv(δ)
    δ⁻² = δ⁻¹^2
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

# @inline _∇²neglogpdf_qrician_jvp(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D} = _∇²neglogpdf_qrician_jvp_via_one_pass(Δ, x, ν, δ, order)
@inline _∇²neglogpdf_qrician_jvp(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D} = _∇²neglogpdf_qrician_jvp_via_two_pass(Δ, x, ν, δ, order)
# @inline _∇²neglogpdf_qrician_jvp(Δ::SVector{9, D}, x::D, ν::D, δ::D, order::Val) where {D} = _∇²neglogpdf_qrician_jvp_via_jac_parts(Δ, x, ν, δ, order)

#### Specialized quadrature rules

function neglogpdf_qrician_direct(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δ = x - ν
    I = neglogf_quadrature(zero(T), δ, order) do t̂
        t = t̂ + x
        Δ_tν = t̂ + Δ # numerically stable when x ≈ ν, equivalent to: t - ν = t̂ + (x - ν)
        return Δ_tν^2 / 2 - log(t) - logbesseli0x(t * ν)
    end
    return I
end

function neglogpdf_qrician_right_laguerre_tail(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δ = x - ν
    Δ′ = Δ + δ
    λ = δ * (Δ + δ / 2)
    I0 = Δ^2 / 2

    if λ > -log(eps(T))
        I1 = f_laguerre_tail_quadrature(Δ, order) do t̂
            t = x + t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν)
        end
        I1 = -log(I1)
    else
        I1⁺ = f_laguerre_tail_quadrature(Δ, order) do t̂
            t = x + t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν)
        end
        I1⁻ = f_laguerre_tail_quadrature(Δ′, order) do t̂
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
    Δ = x - ν
    Δ′ = Δ + δ
    λ = δ * (Δ + δ / 2)
    I0 = Δ^2 / 2

    if λ > -log(eps(T))
        I1 = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + t̂
            return exp(-Δ * t̂) * t * besseli0x(t * ν)
        end
        I1 = -log(I1) - T(log2π) / 2
    else
        I1⁺ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + t̂
            return exp(-Δ * t̂) * t * besseli0x(t * ν)
        end
        I1⁻ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + δ + t̂
            return exp(-Δ′ * t̂) * t * besseli0x(t * ν)
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

@generated function gausslegendre_unit_interval(::Val{order}, ::Type{T}) where {order, T <: AbstractFloat}
    x, w = gausslegendre(order)
    x = SVector{order, T}(@. T((1 + x) / 2)) # rescale from [-1, 1] to [0, 1]
    w = SVector{order, T}(@. T(w / 2)) # adjust weights to account for rescaling
    return :($x, $w)
end

@generated function gausslaguerre_positive_real_axis(::Val{order}, ::Type{T}) where {order, T <: AbstractFloat}
    x, w = gausslaguerre(order)
    x = SVector{order, T}(T.(x)) # nodes lie in [0, ∞)
    w = SVector{order, T}(T.(w)) # exponentially decreasing weights
    return :($x, $w)
end

@generated function gausshalfhermite_positive_real_axis(::Val{order}, ::Type{T}, ::Val{γ}) where {order, T <: AbstractFloat, γ}
    @assert γ > -1 "γ must be greater than -1"
    x, w = gausshalfhermite_gw(order, BigFloat(γ); normalize = true)
    x = SVector{order, T}(T.(x)) # nodes lie in [0, ∞)
    w = SVector{order, T}(T.(w)) # exponentially decreasing weights
    return :($x, $w)
end

@inline function f_quadrature(f::F, x₀::Real, δ::Real, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order}
    # I = ∫_{x₀}^{x₀ + δ} [f(t)] dt
    T = checkedfloattype(x₀, δ)
    x, w = gausslegendre_unit_interval(Val(order), T)
    y = @. f(x₀ + δ * x)
    return vecdot(w, y) * δ
end

@inline function f_quadrature_weighted_unit_interval(f::F, ::Type{T}, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, T, order}
    # I = ∫_{0}^{1} [exp(Ω - ω(t)) f(t)] dt where Ω = -log(∫_{0}^{1} exp(-ω(t)) dt)
    x, w = gausslegendre_unit_interval(Val(order), checkedfloattype(T))
    ω_and_y = @. f(x)
    ω, y = first.(ω_and_y), last.(ω_and_y)
    Ω = weighted_neglogsumexp(w, ω)
    w′ = @. exp(Ω - ω) * w
    I = vecdot(w′, y)
    return Ω, I, x, w′
end

@inline function neglogf_quadrature(neglogf::F, x₀::Real, δ::Real, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order}
    # I = ∫_{x₀}^{x₀ + δ} [f(t)] dt, where f(t) = exp(-neglogf(t))
    T = checkedfloattype(x₀, δ)
    x, w = gausslegendre_unit_interval(Val(order), T)
    neglogy = @. neglogf(x₀ + δ * x)
    return weighted_neglogsumexp(w, neglogy) .- log(δ)
end

@inline function f_laguerre_tail_quadrature(f::F, λ::Real, ::Val{order} = Val(DEFAULT_GAUSSLAGUERRE_ORDER)) where {F, order}
    # I = ∫_{0}^{∞} [exp(-λt) f(t)] dt
    T = checkedfloattype(λ)
    x, w = gausslaguerre_positive_real_axis(Val(order), T)
    y = @. f(x / λ)
    return vecdot(w, y) / λ
end

@inline function f_halfhermite_tail_quadrature(f::F, ::Val{γ}, ::Val{order} = Val(DEFAULT_GAUSSLAGUERRE_ORDER)) where {F, order, γ}
    # I = ∫_{0}^{∞} [x^γ exp(-t^2/2) f(t)] / √(2π) dt
    T = checkedfloattype(γ)
    x, w = gausshalfhermite_positive_real_axis(Val(order), T, Val(γ))
    y = @. f(x)
    return vecdot(w, y)
end

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
