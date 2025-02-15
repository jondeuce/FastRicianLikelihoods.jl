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

@inline pdf_rician(args...) = exp(-neglogpdf_rician(args...))
@inline ∇pdf_rician(args...) = -exp(-neglogpdf_rician(args...)) .* ∇neglogpdf_rician(args...)

#### Internal methods with strict type signatures (enables dual number overloads with single method)

@inline _neglogpdf_rician(x::D, ν::D) where {D} = (x - ν)^2 / 2 - log(x) - logbesseli0x(x * ν) # negative Rician log-likelihood `-logp(x | ν, σ = 1)`

@inline function _∇neglogpdf_rician(x::D, ν::D) where {D}
    # Define the univariate normalized Bessel function `Î₀` as
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
    T = checkedfloattype(x, ν)
    z = x * ν
    if z < besseli1i0_low_cutoff(T)
        z² = z^2
        r = z * evalpoly(z², besseli1i0_low_coefs(T)) # r = logÎ₀′(z) + 1 - 1/2z = I₁(z) / I₀(z) ≈ z/2 + 𝒪(z^3)
        ∂x = x - ν * r - inv(x)
        ∂ν = ν - x * r
    elseif z < besseli1i0_mid_cutoff(T)
        z² = z^2
        r = z * evalpoly(z², besseli1i0_mid_num_coefs(T)) / evalpoly(z², besseli1i0_mid_den_coefs(T)) # r = I₁(z) / I₀(z)
        ∂x = x - ν * r - inv(x)
        ∂ν = ν - x * r
    elseif z < besseli1i0_high_cutoff(T)
        z² = z^2
        r = z * evalpoly(z², besseli1i0_high_num_coefs(T)) / evalpoly(z², besseli1i0_high_den_coefs(T)) # r = I₁(z) / I₀(z)
        ∂x = x - ν * r - inv(x)
        ∂ν = ν - x * r
    else
        z⁻¹ = inv(z)
        tmp = z⁻¹ * evalpoly(z⁻¹, besseli1i0c_tail_coefs(T)) # -z * logÎ₀′(z) = -1/2 - z * (I₁(z) / I₀(z) - 1) ≈ 1/8z + 𝒪(1/z^2)
        ∂x = x - ν + (T(-0.5) + tmp) / x
        ∂ν = ν - x + (T(+0.5) + tmp) / ν
    end

    return (∂x, ∂ν)
end

@scalar_rule _neglogpdf_rician(x, ν) (_∇neglogpdf_rician(x, ν)...,)
@dual_rule_from_frule _neglogpdf_rician(x, ν)

@inline function _∇²neglogpdf_rician(x::D, ν::D) where {D}
    z = x * ν
    r, rx, rm1, r²m1, r²m1prx = _besseli1i0_parts(z) # (r, r / z, r - 1, r^2 - 1, r^2 - 1 + r / z) where r = I₁(z) / I₀(z)
    ∂²x = 1 + 1 / x^2 + ν^2 * r²m1prx # ∂²/∂x²
    ∂²ν = 1 + x^2 * r²m1prx # ∂²/∂ν²
    ∂x∂ν = z * r²m1 # ∂²/∂x∂ν
    return (∂²x, ∂x∂ν, ∂²ν)
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

# Generated is overkill, but Zygote fails to infer the output type otherwise
@inline neglogpdf_qrician(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _neglogpdf_qrician_midpoint(promote(x, ν, δ)..., order) : _neglogpdf_qrician(promote(x, ν, δ)..., order)
@inline ∇neglogpdf_qrician(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇neglogpdf_qrician_midpoint(promote(x, ν, δ)..., order) : _∇neglogpdf_qrician(promote(x, ν, δ)..., order)
@inline ∇²neglogpdf_qrician(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇²neglogpdf_qrician_midpoint(promote(x, ν, δ)..., order) : _∇²neglogpdf_qrician(promote(x, ν, δ)..., order)
@inline ∇²neglogpdf_qrician_with_gradient(x::Real, ν::Real, δ::Real, order::Val) = order == Val(1) ? _∇²neglogpdf_qrician_midpoint_with_gradient(promote(x, ν, δ)..., order) : _∇²neglogpdf_qrician_with_gradient(promote(x, ν, δ)..., order)

# Fast-path for single point quadrature which avoids computing the primal; equivalent to using the midpoint rule approximations for the integrals
@inline _neglogpdf_qrician_midpoint(x::D, ν::D, δ::D, ::Val{1}) where {D} = _neglogpdf_rician(x + δ / 2, ν) - log(δ)
@inline function _∇neglogpdf_qrician_midpoint(x::D, ν::D, δ::D, ::Val{1}) where {D}
    ∂x, ∂ν = _∇neglogpdf_rician(x + δ / 2, ν)
    return ∂x, ∂ν, ∂x / 2 - inv(δ)
end
@inline function _∇²neglogpdf_qrician_midpoint_with_gradient(x::D, ν::D, δ::D, ::Val{1}) where {D}
    y = x + δ / 2
    ∇x, ∇ν = _∇neglogpdf_rician(y, ν)
    ∇xx, ∇xν, ∇νν = _∇²neglogpdf_rician(y, ν)
    return (∇x, ∇ν, ∇x / 2 - inv(δ)), (∇xx, ∇xν, ∇xx / 2, ∇νν, ∇xν / 2, ∇xx / 4 + 1 / δ^2)
end
_∇²neglogpdf_qrician_midpoint(x::D, ν::D, δ::D, ::Val{1}) where {D} = _∇²neglogpdf_qrician_midpoint_with_gradient(x, ν, δ, Val(1))[2]

#### Internal methods with strict type signatures (enables dual number overloads with single method)

@inline _neglogpdf_qrician(x::D, ν::D, δ::D, order::Val) where {D} = neglogf_quadrature(Base.Fix2(_neglogpdf_rician, ν), x, δ, order)
@inline _∇neglogpdf_qrician(x::D, ν::D, δ::D, order::Val) where {D} = _∇neglogpdf_qrician_with_primal(x, ν, δ, order)[2]

@inline function _∇neglogpdf_qrician_with_primal(Ω::D, x::D, ν::D, δ::D, order::Val) where {D}
    # Differentiate the approximation:
    #   Ω = -logI = -log(∫_{x}^{x+δ} exp(-neglogpdf_rician(y, ν)) dy) = -log(∫_{0}^{1} exp(-neglogpdf_rician(x + δ * t, ν)) * δ dt)
    #  ∂Ω = -∂(logI) = -∂I / I = ∫_{0}^{1} ∂(-exp(Ω - neglogpdf_rician(x + δ * t, ν)) * δ) dt
    # where Ω = -logI is constant w.r.t. ∂.
    ∂x, ∂ν, ∂δ = f_quadrature(zero(x), one(x), order) do t
        δt = δ * t
        y = x + δt
        ∇x, ∇ν = _∇neglogpdf_rician(y, ν)
        dx, dν, dδ = ∇x * δ, ∇ν * δ, ∇x * δt - one(x)
        ∇ = SVector{3, D}((dx, dν, dδ))
        return exp(Ω - _neglogpdf_rician(y, ν)) * ∇
    end

    #=
    # Differentiate the approximation for (∂x, ∂ν) and use FTC for ∂δ:
    ∂x, ∂ν = f_quadrature(x, δ, order) do y
        ∇ = _∇neglogpdf_rician(y, ν) # differentiate the integrand
        ∇ = SVector{2, D}(∇)
        return exp(Ω - _neglogpdf_rician(y, ν)) * ∇
    end
    ∂δ = -exp(Ω - _neglogpdf_rician(x + δ, ν)) # by fundamental theorem of calculus
    =#

    #=
    # Differentiate the approximation for ∂ν and use FTC for (∂x, ∂δ):
    ∂ν = f_quadrature(x, δ, order) do y
        _, ∇ν = _∇neglogpdf_rician(y, ν) # differentiate the integrand
        return exp(Ω - _neglogpdf_rician(y, ν)) * ∇ν
    end
    lo, hi = _neglogpdf_rician(x, ν), _neglogpdf_rician(x + δ, ν)
    ∂δ = -exp(Ω - hi) # by fundamental theorem of calculus
    ∂x = lo < hi ? exp(Ω - lo) * -expm1(lo - hi) : exp(Ω - hi) * expm1(hi - lo) # by fundamental theorem of calculus (note: leads to catestrophic cancellation for small δ, but more accurate for large δ)
    =#

    return Ω, (∂x, ∂ν, ∂δ)
end
@inline _∇neglogpdf_qrician_with_primal(x::D, ν::D, δ::D, order::Val) where {D} = _∇neglogpdf_qrician_with_primal(_neglogpdf_qrician(x, ν, δ, order), x, ν, δ, order)

@scalar_rule _neglogpdf_qrician(x, ν, δ, order::Val) (_∇neglogpdf_qrician_with_primal(Ω, x, ν, δ, order)[2]..., NoTangent())
@dual_rule_from_frule _neglogpdf_qrician(x, ν, δ, !(order::Val))

@inline function _∇²neglogpdf_qrician_with_gradient(Ω::D, x::D, ν::D, δ::D, order::Val) where {D}
    # Differentiate the approximation, i.e. differentiate through the quadrature:
    #     Ω = -logI = -log(∫_{x}^{x+δ} exp(-neglogpdf_rician(y, ν)) dy) = -log(∫_{0}^{1} exp(-neglogpdf_rician(x + δ * t, ν)) * δ dt)
    #    ∂Ω = -∂(logI) = -∂I / I = ∫_{0}^{1} ∂(-exp(Ω - neglogpdf_rician(x + δ * t, ν)) * δ) dt
    # ∂₁∂₂Ω = -∂₁∂₂(logI) = -∂₁(∂₂I / I) = (∂₁I)(∂₂I) / I² - ∂₁∂₂I / I
    #       = (∂₁Ω)(∂₂Ω) + ∫_{0}^{1} ∂₁∂₂(-exp(Ω - neglogpdf_rician(x + δ * t, ν)) * δ) dt
    # where Ω = -logI is constant w.r.t. ∂₁ and ∂₂.
    (∂x, ∂ν, ∂δ, ∂x∂x, ∂x∂ν, ∂x∂δ, ∂ν∂ν, ∂ν∂δ, ∂δ∂δ) = f_quadrature(zero(x), one(x), order) do t
        δt = δ * t
        y = x + δt
        ∇x, ∇ν = _∇neglogpdf_rician(y, ν)
        ∇xx, ∇xν, ∇νν = _∇²neglogpdf_rician(y, ν)
        dx, dν, dδ = ∇x * δ, ∇ν * δ, ∇x * δt - one(x)
        dxdx, dxdν, dνdν = (∇xx - ∇x * ∇x) * δ, (∇xν - ∇x * ∇ν) * δ, (∇νν - ∇ν * ∇ν) * δ
        dxdδ, dνdδ, dδdδ = ∇x - δt * (∇x * ∇x - ∇xx), ∇ν - δt * (∇x * ∇ν - ∇xν), t * (2 * ∇x - δt * (∇x * ∇x - ∇xx))
        integrands = SVector{9, D}((dx, dν, dδ, dxdx, dxdν, dxdδ, dνdν, dνdδ, dδdδ))
        return exp(Ω - _neglogpdf_rician(y, ν)) * integrands
    end

    return (∂x, ∂ν, ∂δ), (∂x * ∂x + ∂x∂x, ∂x * ∂ν + ∂x∂ν, ∂x * ∂δ + ∂x∂δ, ∂ν * ∂ν + ∂ν∂ν, ∂ν * ∂δ + ∂ν∂δ, ∂δ * ∂δ + ∂δ∂δ)

    #=
    # Differentiate the approximation for (∂x, ∂ν, ∂²xx, ∂²xν, ∂²νν) and use FTC for (∂δ, ∂²xδ, ∂²νδ, ∂²δδ):
    (∂x, ∂ν, ∂²xx, ∂²xν, ∂²νν) = f_quadrature(x, δ, order) do y
        ∇ = _∇neglogpdf_rician(y, ν)
        ∇² = _∇²neglogpdf_rician(y, ν)
        integrands = SVector{5, D}(∇[1], ∇[2], ∇[1]^2 - ∇²[1], ∇[1] * ∇[2] - ∇²[2], ∇[2]^2 - ∇²[3]) # ∇ and ∇∇ᵀ - ∇²
        return exp(Ω - _neglogpdf_rician(y, ν)) * integrands
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

    return (∂x, ∂ν, ∂δ), (∂²xx, ∂²xν, ∂²xδ, ∂²νν, ∂²νδ, ∂²δδ)
    =#
end
@inline _∇²neglogpdf_qrician_with_gradient(x::D, ν::D, δ::D, order::Val) where {D} = _∇²neglogpdf_qrician_with_gradient(_neglogpdf_qrician(x, ν, δ, order), x, ν, δ, order)
@inline _∇²neglogpdf_qrician(x::D, ν::D, δ::D, order::Val) where {D} = _∇²neglogpdf_qrician_with_gradient(x, ν, δ, order)[2]

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
    # I = ∫_{0}^{δ} [f(t)] dt
    T = checkedfloattype(x₀, δ)
    x, w = gausslegendre_unit_interval(Val(order), T)
    y = @. f(muladd(δ, x, x₀))
    return vecdot(w, y) * δ
end

@inline function neglogf_quadrature(neglogf::F, x₀::Real, δ::Real, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order}
    # I = ∫_{0}^{δ} [f(t)] dt, where f(t) = exp(-neglogf(t))
    T = checkedfloattype(x₀, δ)
    x, w = gausslegendre_unit_interval(Val(order), T)
    logy = @. -neglogf(muladd(δ, x, x₀))
    return -weighted_logsumexp(w, logy) .- log(δ)
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

@inline function weighted_logsumexp(w::SVector{N}, logy::SVector{N}) where {N}
    max_ = maximum(logy)
    ȳ = exp.(logy .- max_)
    return log(vecdot(w, ȳ)) + max_
end

@inline function weighted_logsumexp(w::SVector{N}, logy::SVector{N, <:SVector{M}}) where {N, M}
    max_ = reduce(BroadcastFunction(max), logy) # elementwise maximum
    logy = reducehcat(logy) # stack as columns
    ȳ = exp.(logy .- max_)
    return log.(vecdot(w, ȳ)) .+ max_
end

# Convert vector of vectors in flat matrix. Note that `init` is necessary to get the correct type when `N = 1`, otherwise you get an SVector{M} instead of an SMatrix{M, 1}
@inline reducehcat(y::SVector{N, <:SVector{M, T}}) where {N, M, T} = reduce(hcat, y; init = SMatrix{M, 0, T}())

@inline vecdot(w::SVector{N}, y::SVector{N}) where {N} = dot(w, y)
@inline vecdot(w::SVector{N}, y::SVector{N, <:SVector{M}}) where {N, M} = vecdot(w, reducehcat(y))
@inline vecdot(w::SVector{N}, y::SMatrix{M, N}) where {N, M} = y * w
