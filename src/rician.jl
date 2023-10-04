####
#### Rician negative log-pdf
####

#### Utilities

@inline promote_float(x...) = promote(map(float, x)...)

####
#### Quantized Gaussian log-pdf
####

function logpdf_qnormal(x::T, δ::T) where {T <: Union{Float32, Float64}}
    @assert x >= 0 && δ > 0
    if δ <= T(0.15)
        if δ * x > T(0.3)
            return logpdf_qnormal_small_δ_large_δx(x, δ)
        else
            return logpdf_qnormal_small_δ_small_δx(x, δ)
        end
    elseif x > 1
        Φ⁻ = erfcx((x + δ) * invsqrt2)
        Φ⁺ = erfcx(x * invsqrt2)
        return log(Φ⁺ / 2) - abs2(x) / 2 + log1p(-(Φ⁻ / Φ⁺) * exp(-δ * (x + δ / 2))) # x > 1 and x + δ > 1
    elseif x + δ > 1
        return log((1 - erfc((x + δ) * invsqrt2) - erf(x * invsqrt2)) / 2) # 0 <= x <= 1 and x + δ > 1
    else
        return log((erf((x + δ) * invsqrt2) - erf(x * invsqrt2)) / 2) # 0 <= x <= 1 and 0 <= x + δ <= 1
    end
end

function logpdf_qnormal_small_δ_large_δx(x::T, δ::T) where {T <: Union{Float32, Float64}}
    # Small `δ` but large `δ * x` (and therefore large `x`); Taylor expand in `δ` only and integrate termwise:
    #   -x^2 / 2 - T(log2π) / 2 - log(x) + log1p(exp(-δ * x) * (945 / x^10 + 945 * δ / x^9 + 945 * δ^2 / (2 * x^8) + 315 * δ^3 / (2 * x^7) + 315 * δ^4 / (8 * x^6) + 63 * δ^5 / (8 * x^5) + 21 * δ^6 / (16 * x^4) + 3 * δ^7 / (16 * x^3) + 3 * δ^8 / (128 * x^2) + δ^9 / (384 * x) + δ^10 / 3840 - δ^8 / 384 - δ^7 / (48 * x) - 7 * δ^6 / (48 * x^2) - 7 * δ^5 / (8 * x^3) - 35 * δ^4 / (8 * x^4) - 35 * δ^3 / (2 * x^5) - 105 * δ^2 / (2 * x^6) - 105 * δ / x^7 - 105 / x^8 + δ^6 / 48 + δ^5 / (8 * x) + 5 * δ^4 / (8 * x^2) + 5 * δ^3 / (2 * x^3) + 15 * δ^2 / (2 * x^4) + 15 * δ / x^5 + 15 / x^6 - δ^4 / 8 - δ^3 / (2 * x) - 3 * δ^2 / (2 * x^2) - 3 * δ / x^3 - 3 / x^4 + δ^2 / 2 + δ / x + 1 / x^2 - 1) + (-945) / x^10 + 105 / x^8 - 15 / x^6 + 3 / x^4 - 1 / x^2)
    x⁻² = inv(x)^2
    a10 = T(1)/384
    a8 = evalpoly(x⁻², T.((-1, 9)) ./ 48)
    a6 = evalpoly(x⁻², T.((1, -7, 63)) ./ 8)
    a4 = evalpoly(x⁻², T.((-1, 5, -35, 315)) ./ 2)
    a2 = evalpoly(x⁻², T.((1, -3, 15, -105, 945)))
    a0 = muladd(x⁻², a2, T(-1))
    p = (a0, a2 / x, a2 / 2, a4 / x, a4 / 4, a6 / x, a6 / 6, a8 / x, a8 / 8, a10 / x, a10 / 10)
    return -x^2 / 2 - T(log2π) / 2 - log(x) + log1p(exp(-δ * x) * evalpoly(δ, p) - x⁻² * a2)
end

function logpdf_qnormal_small_δ_small_δx(x::T, δ::T) where {T <: Union{Float32, Float64}}
    # Small `δ` and small `δ * x`; use Taylor expansion:
    #   log(δ) - x^2 / 2 - T(log2π) / 2 - (x * δ) / 2 + (x^2 * δ^2) / 24 - (x^4 * δ^4) / 2880 + (x^6 * δ^6) / 181440 - (x^8 * δ^8) / 9676800 + (x^10 * δ^10) / 479001600 - δ^2 / 6 + (x * δ^3) / 24 - (x^2 * δ^4) / 720 - (x^3 * δ^5) / 1440 + (x^4 * δ^6) / 30240 + (x^5 * δ^7) / 60480 - (x^6 * δ^8) / 1209600 - (x^7 * δ^9) / 2419200 + (x^8 * δ^10) / 47900160 + (x^9 * δ^11) / 95800320 + δ^4 / 90 - (x * δ^5) / 720 - (61 * x^2 * δ^6) / 120960 + (x^3 * δ^7) / 15120 + (47 * x^4 * δ^8) / 2419200 - (x^5 * δ^9) / 403200 - (643 * x^6 * δ^10) / 958003200 + (x^7 * δ^11) / 11975040 - δ^6 / 2835 - (19 * x * δ^7) / 120960 + (181 * x^2 * δ^8) / 3628800 + (41 * x^3 * δ^9) / 3628800 - (739 * x^4 * δ^10) / 239500800 - (181 * x^5 * δ^11) / 319334400 - δ^8 / 56700 + (61 * x * δ^9) / 3628800 + (1579 * x^2 * δ^10) / 479001600 - (61 * x^3 * δ^11) / 29937600 + δ^10 / 467775 + (193 * x * δ^11) / 479001600
    δx = δ * x
    x², δ², δx² = x^2, δ^2, δx^2
    p = (
        δx² * evalpoly(δx², (T(1)/24, T(-1)/2880, T(1)/181440, T(-1)/9676800, T(1)/479001600)),
        evalpoly(δx, (T(-1)/6, T(1)/24, T(-1)/720, T(-1)/1440, T(1)/30240, T(1)/60480, T(-1)/1209600, T(-1)/2419200, T(1)/47900160, T(1)/95800320)),
        evalpoly(δx, (T(1)/90, T(-1)/720, T(-61)/120960, T(1)/15120, T(47)/2419200, T(-1)/403200, T(-643)/958003200, T(1)/11975040)),
        evalpoly(δx, (T(-1)/2835, T(-19)/120960, T(181)/3628800, T(41)/3628800, T(-739)/239500800, T(-181)/319334400)),
        evalpoly(δx, (T(-1)/56700, T(61)/3628800, T(1579)/479001600, T(-61)/29937600)),
        evalpoly(δx, (T(1)/467775, T(193)/479001600)),
    )
    return log(δ) - x² / 2 - T(log2π) / 2 - δx / 2 + evalpoly(δ², p)
end

####
#### Rician negative log-likelihood
####

@inline function neglogpdf_rician(x::T, ν::T, logσ::T) where {T <: Real}
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end
@inline neglogpdf_rician(x::Real, ν::Real, logσ::Real) = neglogpdf_rician(promote_float(x, ν, logσ)...)

@inline neglogpdf_rician(x::T, ν::T) where {T <: Union{Float32, Float64}} = (x - ν)^2 / 2 - log(x) - logbesseli0x(x * ν) # negative Rician log-likelihood `-logp(x | ν, σ = 1)`

@inline function ∇neglogpdf_rician(x::T, ν::T) where {T <: Union{Float32, Float64}}
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

@inline pdf_rician(args...) = exp(-neglogpdf_rician(args...))
@inline ∇pdf_rician(x::T, ν::T) where {T <: Union{Float32, Float64}} = -exp(-neglogpdf_rician(x, ν)) .* ∇neglogpdf_rician(x, ν)

@scalar_rule neglogpdf_rician(x, ν) (∇neglogpdf_rician(x, ν)...,)
@dual_rule_from_frule neglogpdf_rician(x, ν)

####
#### Quantized Rician negative log-pdf
####

# Quantized Rician PDF is the integral of the Rician PDF over `(x, x+δ)`.
# This integral is approximated using Gauss-Legendre quadrature.
# Note: Rician PDF is never evaluated at the interval endpoints.
@inline function neglogpdf_qrician(x::T, ν::T, logσ::T, δ::T, order::Val) where {T <: Real}
    σ⁻¹ = exp(-logσ)
    return neglogpdf_qrician(σ⁻¹ * x, σ⁻¹ * ν, σ⁻¹ * δ, order)
end
@inline neglogpdf_qrician(x::Real, ν::Real, logσ::Real, δ::Real, order::Val) = neglogpdf_qrician(promote_float(x, ν, logσ, δ)..., order)
@inline neglogpdf_qrician(n::Int, ν::Real, logσ::Real, δ::Real, order::Val) = neglogpdf_qrician(n * δ, ν, logσ, δ, order)

@inline neglogpdf_qrician(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}} = neglogf_quadrature(Base.Fix2(neglogpdf_rician, ν), x, δ, order)
@inline ∇neglogpdf_qrician(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}} = ∇neglogpdf_qrician_with_primal(x, ν, δ, order)[2]

@inline function ∇neglogpdf_qrician_with_primal(Ω::T, x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    ∂x, ∂ν = f_quadrature(x, δ, order) do y
        ∇ = ∇neglogpdf_rician(y, ν) # differentiate the integrand
        ∇ = SVector{2, T}(∇)
        return exp(Ω - neglogpdf_rician(y, ν)) * ∇
    end
    ∂δ = -exp(Ω - neglogpdf_rician(x + δ, ν)) # by fundamental theorem of calculus
    return Ω, (∂x, ∂ν, ∂δ)
end
@inline ∇neglogpdf_qrician_with_primal(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}} = ∇neglogpdf_qrician_with_primal(neglogpdf_qrician(x, ν, δ, order), x, ν, δ, order)

@scalar_rule neglogpdf_qrician(x, ν, δ, order::Val) (∇neglogpdf_qrician_with_primal(Ω, x, ν, δ, order)[2]..., NoTangent())
@dual_rule_from_frule neglogpdf_qrician(x, ν, δ, !order)

#### Gauss-Legendre quadrature

const DEFAULT_GAUSSLEGENDRE_ORDER = 16

@generated function gausslegendre_unit_interval(::Val{order}, ::Type{T}) where {order, T <: AbstractFloat}
    x, w = gausslegendre(order)
    x = SVector{order, T}(@. T((1 + x) / 2)) # rescale from [-1, 1] to [0, 1]
    w = SVector{order, T}(@. T(w / 2)) # adjust weights to account for rescaling
    return :($x, $w)
end

@inline function f_quadrature(f::F, x₀::T, δ::T, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order, T <: AbstractFloat}
    x, w = gausslegendre_unit_interval(Val(order), T)
    y = @. f(x₀ + δ * x)
    return vecdot(w, y) * δ
end

@inline function neglogf_quadrature(neglogf::F, x₀::T, δ::T, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order, T <: AbstractFloat}
    x, w = gausslegendre_unit_interval(Val(order), T)
    logy = @. -neglogf(x₀ + δ * x)
    return -weighted_logsumexp(w, logy) .- log(δ)
end

@inline function weighted_logsumexp(w::SVector{N, T}, logy::SVector{N, T}) where {N, T <: AbstractFloat}
    max_ = maximum(logy)
    ȳ = exp.(logy .- max_)
    return log(vecdot(w, ȳ)) + max_
end

@inline function weighted_logsumexp(w::SVector{N, T}, logy::SVector{N, SVector{M, T}}) where {N, M, T <: AbstractFloat}
    max_ = reduce(BroadcastFunction(max), logy) # elementwise maximum
    logy = reduce(hcat, logy) # stack as columns
    ȳ = exp.(logy .- max_)
    return log.(vecdot(w, ȳ)) .+ max_
end

@inline vecdot(w::SVector{N, T}, y::SVector{N, T}) where {N, T <: AbstractFloat} = dot(w, y)
@inline vecdot(w::SVector{N, T}, y::SVector{N, SVector{M, T}}) where {N, M, T <: AbstractFloat} = vecdot(w, reduce(hcat, y))
@inline vecdot(w::SVector{N, T}, y::SMatrix{M, N, T}) where {N, M, T <: AbstractFloat} = y * w
