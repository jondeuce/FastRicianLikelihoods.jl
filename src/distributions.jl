####
#### Rician Distribution
####

#### Rician distribution: https://en.wikipedia.org/wiki/Rice_distribution

struct Rician{T <: Real} <: Distributions.ContinuousUnivariateDistribution
    ν::T
    σ::T
end

#### Outer constructors

@inline Rician(ν::Real, σ::Real) = Rician(promote(ν, σ)...)
@inline Rician(ν::Integer, σ::Integer) = Rician(float(ν), float(σ))
@inline Rician(ν::Real) = Rician(ν, one(typeof(ν)))
@inline Rician() = Rician(0.0, 1.0)

#### Conversions

@inline Base.convert(::Type{Rician{T}}, ν::Real, σ::Real) where {T <: Real} = Rician(T(ν), T(σ))
@inline Base.convert(::Type{Rician{T}}, d::Rician{<:Real}) where {T <: Real} = Rician(T(d.ν), T(d.σ))

# Distributions.@distr_support Rician 0 Inf

@inline Base.minimum(::Union{Rician, Type{<:Rician}}) = 0
@inline Base.maximum(::Union{Rician, Type{<:Rician}}) = Inf

#### Parameters

@inline Distributions.params(d::Rician) = (d.ν, d.σ)
@inline Distributions.partype(::Rician{T}) where {T} = T

@inline Distributions.location(d::Rician) = d.ν
@inline Distributions.scale(d::Rician) = d.σ

@inline Base.eltype(::Type{Rician{T}}) where {T} = T

#### Statistics

@inline mean_rician(ν, σ) = sqrthalfπ * σ * laguerre½(-(ν / σ)^2 / 2)
@inline Distributions.mean(d::Rician) = mean_rician(d.ν, d.σ)
# @inline Distributions.mode(d::Rician) = ?
# @inline Distributions.median(d::Rician) = ?

@inline var_rician(ν, σ) = ν^2 + 2σ^2 - π * σ^2 * laguerre½(-(ν / σ)^2 / 2)^2 / 2
@inline Distributions.var(d::Rician) = var_rician(d.ν, d.σ)
@inline Distributions.std(d::Rician) = sqrt(var(d))
# @inline Distributions.skewness(d::Rician{T}) where {T <: Real} = ?
# @inline Distributions.kurtosis(d::Rician{T}) where {T <: Real} = ?
# @inline Distributions.entropy(d::Rician) = ?

#### Evaluation

# p(x | ν, σ) = x * I₀(x * ν / σ^2) * exp(-(x^2 + ν^2) / 2σ^2) / σ^2
@inline Distributions.logpdf(d::Rician, x::Real) = -neglogpdf_rician(x, d.ν, log(d.σ))
@inline Distributions.pdf(d::Rician, x::Real) = exp(logpdf(d, x))

#### Sampling

@inline Distributions.rand(rng::Random.AbstractRNG, d::Rician{T}) where {T} = sqrt((d.ν + d.σ * randn(rng, T))^2 + (d.σ * randn(rng, T))^2)
