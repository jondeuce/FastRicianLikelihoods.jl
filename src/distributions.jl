####
#### Rice distribution
####

"""
    Rice(ν, σ)

The *Rice distribution* with parameters `ν` and `σ` has probability density function:

```math
f(x; \\nu, \\sigma) = \\frac{x}{\\sigma^2} \\exp\\left( \\frac{-(x^2 + \\nu^2)}{2\\sigma^2} \\right) I_0\\left( \\frac{x\\nu}{\\sigma^2} \\right).
```

External links:

* [Rice distribution on Wikipedia](https://en.wikipedia.org/wiki/Rice_distribution)

"""
struct Rice{T <: Real} <: Distributions.ContinuousUnivariateDistribution
    ν::T
    σ::T
end

#### Outer constructors

@inline Rice(ν::Real, σ::Real) = Rice(promote(ν, σ)...)
@inline Rice(ν::Integer, σ::Integer) = Rice(float(ν), float(σ))
@inline Rice(ν::Real) = Rice(ν, one(typeof(ν)))
@inline Rice() = Rice(0.0, 1.0)

#### Conversions

@inline Base.convert(::Type{Rice{T}}, ν::Real, σ::Real) where {T <: Real} = Rice(T(ν), T(σ))
@inline Base.convert(::Type{Rice{T}}, d::Rice{<:Real}) where {T <: Real} = Rice(T(d.ν), T(d.σ))

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

@inline mean_rician(ν, σ) = (t = ν / σ; return σ * (t > 1/√eps(one(t)) ? t : sqrthalfπ * laguerre½(-t^2 / 2)))
@inline var_rician(ν, σ) = (t = ν / σ; return σ^2 * (1 - laguerre½²c(t))) # equivalent to: ν^2 + 2σ^2 - π * σ^2 * laguerre½(-(ν / σ)^2 / 2)^2 / 2
@inline std_rician(ν, σ) = sqrt(var_rician(ν, σ))

@inline Distributions.mean(d::Rice) = mean_rician(d.ν, d.σ)
# @inline Distributions.mode(d::Rice) = ?
# @inline Distributions.median(d::Rice) = ?

@inline Distributions.var(d::Rice) = var_rician(d.ν, d.σ)
@inline Distributions.std(d::Rice) = sqrt(var(d))
# @inline Distributions.skewness(d::Rice{T}) where {T <: Real} = ?
# @inline Distributions.kurtosis(d::Rice{T}) where {T <: Real} = ?
# @inline Distributions.entropy(d::Rice) = ?

#### Evaluation

# p(x | ν, σ) = x * I₀(x * ν / σ^2) * exp(-(x^2 + ν^2) / 2σ^2) / σ^2
@inline Distributions.logpdf(d::Rice, x::Real) = -neglogpdf_rician(x, d.ν, log(d.σ))
@inline Distributions.pdf(d::Rice, x::Real) = exp(Distributions.logpdf(d, x))

#### Sampling

@inline Distributions.rand(rng::Random.AbstractRNG, d::Rice{T}) where {T} = hypot(d.ν + d.σ * randn(rng, T), d.σ * randn(rng, T))
