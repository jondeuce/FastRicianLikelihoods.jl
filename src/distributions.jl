####
#### Rice distribution
####

@doc raw"""
    Rice(ν, σ)

The Rice distribution with shape parameter $\nu$ and scale parameter $\sigma$.

The probability density function is
```math
p(x \mid \nu, \sigma) = \frac{x}{\sigma^2} \exp\!\left(-\frac{x^2 + \nu^2}{2\sigma^2}\right) I_0\!\left(\frac{x\nu}{\sigma^2}\right)
```
where $I_0$ is the modified Bessel function of the first kind of order zero, and $x \ge 0$.

# External links

- [Rice distribution Wiki](https://en.wikipedia.org/wiki/Rice_distribution)
"""
struct Rice{T <: Real} <: Distributions.ContinuousUnivariateDistribution
    ν::T
    σ::T
    Rice{T}(ν::Real, σ::Real) where {T <: Real} = new{T}(ν, σ)
end

#### Outer constructors

@inline function Rice(ν::T, σ::T; check_args::Bool = true) where {T <: Real}
    Distributions.@check_args Rice (ν, ν >= zero(ν)) (σ, σ > zero(σ))
    return Rice{T}(ν, σ)
end
@inline Rice(ν::Real, σ::Real; check_args::Bool = true) = Rice(promote(ν, σ)...; check_args)
@inline Rice(ν::Integer, σ::Integer; check_args::Bool = true) = Rice(float(ν), float(σ); check_args)
@inline Rice(ν::Real; check_args::Bool = true) = Rice(ν, one(ν); check_args)
@inline Rice() = Rice{Float64}(0.0, 1.0)

#### Conversions

@inline Base.convert(::Type{Rice{T}}, ν::Real, σ::Real) where {T <: Real} = Rice(T(ν), T(σ))
@inline Base.convert(::Type{Rice{T}}, d::Rice{<:Real}) where {T <: Real} = Rice{T}(T(d.ν), T(d.σ))

# Distributions.@distr_support Rice 0 Inf

@inline Base.minimum(::Union{Rice, Type{<:Rice}}) = 0
@inline Base.maximum(::Union{Rice, Type{<:Rice}}) = Inf

#### Parameters

@inline Distributions.params(d::Rice) = (d.ν, d.σ)
@inline Distributions.partype(::Rice{T}) where {T} = T

@inline Distributions.location(d::Rice) = d.ν
@inline Distributions.scale(d::Rice) = d.σ

@inline Base.eltype(::Type{Rice{T}}) where {T} = T

#### Statistics

@inline mean_rician(ν, σ) = σ * mean_rician(ν / σ)
@inline var_rician(ν, σ) = σ^2 * var_rician(ν / σ)
@inline std_rician(ν, σ) = σ * std_rician(ν / σ)
@inline mode_rician(ν, σ) = σ * mode_rician(ν / σ)
@inline var_mode_rician(ν, σ) = σ^2 * var_mode_rician(ν / σ)

@inline mean_rician(ν) = _mean_rician(float(ν))
@inline var_rician(ν) = _var_rician(float(ν))
@inline std_rician(ν) = sqrt(_var_rician(float(ν)))
@inline mode_rician(ν) = _mode_rician(float(ν))
@inline var_mode_rician(ν) = _var_mode_rician(float(ν))

@inline Distributions.mean(d::Rice) = mean_rician(d.ν, d.σ)
@inline Distributions.mode(d::Rice) = mode_rician(d.ν, d.σ)
# @inline Distributions.median(d::Rice) = ?

@inline Distributions.var(d::Rice) = var_rician(d.ν, d.σ)
@inline Distributions.std(d::Rice) = sqrt(Distributions.var(d))
# @inline Distributions.skewness(d::Rice{T}) where {T <: Real} = ?
# @inline Distributions.kurtosis(d::Rice{T}) where {T <: Real} = ?
# @inline Distributions.entropy(d::Rice) = ?

#### Evaluation

# p(x | ν, σ) = x * I₀(x * ν / σ^2) * exp(-(x^2 + ν^2) / 2σ^2) / σ^2
@inline Distributions.logpdf(d::Rice, x::Real) = _logpdf_rician(promote(x, d.ν, d.σ)...)
@inline Distributions.pdf(d::Rice, x::Real) = exp(Distributions.logpdf(d, x))

@inline function _logpdf_rician(x::D, ν::D, σ::D) where {D}
    (x < zero(x) || isinf(x)) && return oftype(x, -Inf)
    σ⁻¹ = inv(σ)
    return -(_neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν) + log(σ))
end

#### Sampling

@inline Distributions.rand(rng::Random.AbstractRNG, d::Rice{T}) where {T} = hypot(d.ν + d.σ * randn(rng, T), d.σ * randn(rng, T))

#### Utils

@inline _mean_rician(ν) = ν > 1 / √eps(one(ν)) ? ν : sqrthalfπ * laguerre½(-ν^2 / 2)

@inline _var_rician(ν) = 1 - laguerre½²c(ν) # equivalent to: ν^2 + 2σ^2 - π * σ^2 * laguerre½(-(ν / σ)^2 / 2)^2 / 2 where σ = 1

@inline function _mode_rician_crude(ν::Real)
    ν² = ν^2
    twoν⁴ = 2 * ν²^2
    return √(1 + ν² * ((1 + twoν⁴) / (2 + twoν⁴)))
end

@inline function _var_mode_rician_crude(v::Real)
    twoν² = 2 * v^2
    return (1 + twoν²) / (2 + twoν²)
end

@inline function _mode_rician(ν::Real)
    T = checkedfloattype(ν)
    tay, low, med1, med2, med3, med4, med5, tail = mode_rician_branches(T)
    if ν < tay
        return evalpoly(ν^2, mode_rician_coeff_taylor(T))
    elseif ν < low
        return mode_rician_kernel_small_approx(T, ν)
    elseif ν < med1
        return ν + mode_rician_kernel_med_1_approx(T, ν)
    elseif ν < med2
        return ν + mode_rician_kernel_med_2_approx(T, ν)
    elseif ν < med3
        return ν + mode_rician_kernel_med_3_approx(T, ν)
    elseif ν < med4
        return ν + mode_rician_kernel_med_4_approx(T, ν)
    elseif ν < med5
        return ν + mode_rician_kernel_med_5_approx(T, ν)
    elseif ν < tail
        return inv(ν) * mode_rician_kernel_tail_approx(T, ν) + ν
    else
        return inv(ν) * mode_rician_kernel_tail_long_approx(T, ν) + ν
    end
end

function _var_mode_rician(ν::Real)
    T = checkedfloattype(ν)
    tay, low, med1, med2, med3, med4, med5, med6, tail = var_mode_rician_branches(T)
    if ν < tay
        return evalpoly(ν^2, var_mode_rician_coeff_taylor(T))
    elseif ν < low
        return var_mode_rician_kernel_small_approx(T, ν)
    elseif ν < med1
        return var_mode_rician_kernel_med_1_approx(T, ν)
    elseif ν < med2
        return var_mode_rician_kernel_med_2_approx(T, ν)
    elseif ν < med3
        return var_mode_rician_kernel_med_3_approx(T, ν)
    elseif ν < med4
        return var_mode_rician_kernel_med_4_approx(T, ν)
    elseif ν < med5
        return var_mode_rician_kernel_med_5_approx(T, ν)
    elseif ν < med6
        return var_mode_rician_kernel_med_6_approx(T, ν)
    elseif ν < tail
        return var_mode_rician_kernel_tail_approx(T, ν)
    else
        return var_mode_rician_kernel_tail_long_approx(T, ν)
    end
end

function _mean_qrician(ν::T, δ::T, order = Val(16); H = one(T), K = basefloattype(T) === Float64 ? 9.0 : 6.5) where {T}
    K = oftype(ν, K)
    μ = mean_rician(ν)
    δ <= std_rician(ν) / 2 && return μ - δ / 2
    lo, hi = max(zero(T), μ - K), μ + K
    total, mass = zero(T), zero(T)
    for n in floor(Int, lo/δ):floor(Int, hi/δ)
        a, b = max(lo, n * δ), min(hi, (n + 1) * δ)
        b > a || continue
        panels = ceil(Int, (b - a) / H)
        h = (b - a) / panels
        p = sum(j -> f_quadrature(y -> exp(-neglogpdf_rician(y, ν)), a + j * h, h, order), 0:(panels-1))
        total += n * δ * p
        mass += p
    end
    return total / mass
end

@inline mean_qrician(ν::Real, δ::Real) = _mean_qrician(promote_float(ν, δ)...)
@inline mean_qrician(ν::Real, σ::Real, δ::Real) = σ * _mean_qrician(promote_float(ν / σ, δ / σ)...)
