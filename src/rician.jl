#### Rician negative log-likelihood

@inline function neglogpdf_rician(x, ν, logσ)
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end

@inline neglogpdf_rician(x, ν) = (x - ν)^2 / 2 - log(x) - logbesselix0(x * ν) # Rician pdf with `σ = 1`
@scalar_rule neglogpdf_rician(x, ν) (∇neglogpdf_rician(x, ν)...,)
@define_binary_dual_scalar_rule neglogpdf_rician (neglogpdf_rician, ∇neglogpdf_rician)

@inline function ∇neglogpdf_rician(x, ν)
    # Define the univariate normalized Bessel function Î₀ as
    #   Î₀(z) = I₀(z) / (exp(z) / √2πz).
    # The negative likelihood may then be equivalently written as
    #   -logL = (x - ν)^2 / 2 - log(x / ν) / 2 - logÎ₀(x * ν) + log√2π.
    # All that must be approximated is the derivative of logÎ₀:
    #   d/dz logÎ₀(z) =  1/2z + (I₁(z) / I₀(z) - 1)
    #                 ≈ -1/8z^2 - 1/8z^3 - 25/128z^4 - 13/32z^5 + 𝒪(1/z^6)       (z >> 1)
    #                 ≈  1/2z - 1 + z/2 - z^3/16 + z^5/96 - 11*z^7/6144 + 𝒪(z^9)  (z << 1)
    #   d/dx logÎ₀(z) = ν * d/dz logÎ₀(z)
    #   d/dν logÎ₀(z) = x * d/dz logÎ₀(z)
    x, ν = float(x), float(ν)
    z = x * ν
    T = typeof(z)

    # Note: there are really three relevant limits: z >> 1, z << 1, and x ≈ ν; could possibly better account for the latter case, though this is quite robustly tested below
    if z > 10f0
        z⁻¹ = inv(z)
        tmp = -z⁻¹ * evalpoly(z⁻¹, (1/T(8), 1/T(8), 25/T(128), 13/T(32))) # z * (logÎ₀)′(z) = 1/2 + z * (I₁(z) / I₀(z) - 1)
        ∂x = x - ν - (1/T(2) + tmp) / x
        ∂ν = ν - x + (1/T(2) - tmp) / ν
    elseif z < 0.25f0
        r = z * evalpoly(z^2, (1/T(2), -1/T(16), 1/T(96), -11/T(6144))) # (logÎ₀)′(z) + 1 - 1/2z = I₁(z) / I₀(z)
        tmp = 1 - z * (1 - r) # z * (logÎ₀)′(z) + 1/2 = 1 + z * (I₁(z) / I₀(z) - 1)
        ∂x = x - ν - tmp / x
        ∂ν = ν - x * r
    else
        r = besselix1(z) / besselix0(z) # I₁(z) / I₀(z), accurate for all z
        ∂x = x - ν * r - inv(x)
        ∂ν = ν - x * r
    end

    return (∂x, ∂ν)
end
@inline ∇pdf_rician(x, ν) = -exp(-neglogpdf_rician(x, ν)) .* ∇neglogpdf_rician(x, ν)

#### Rician negative log-cdf

@inline function neglogcdf_rician(x, ν, logσ, δ)
    σ⁻¹ = exp(-logσ)
    return neglogcdf_rician(σ⁻¹ * x, σ⁻¹ * ν, σ⁻¹ * δ)
end

# Integral of the Rician PDF over `(x, x+δ)` using Gauss-Legendre quadrature.
# Consequently, PDF is never evaluated at the endpoints.
@inline neglogcdf_rician(x, ν, δ) = neglogcdf_quadrature(Base.Fix2(neglogpdf_rician, ν), x, δ)

@inline function ∇neglogcdf_rician(Ω, x, ν, δ)
    ∂x, ∂ν = cdf_quadrature(x, δ) do y
        exp(Ω - neglogpdf_rician(y, ν)) .* ∇neglogpdf_rician(y, ν) # differentiating the integrand
    end
    ∂δ = -exp(Ω - neglogpdf_rician(x + δ, ν)) # by fundamental theorem of calculus
    return (∂x, ∂ν, ∂δ)
end
∇neglogcdf_rician(x, ν, δ) = ∇neglogcdf_rician(neglogcdf_rician(x, ν, δ), x, ν, δ)

@scalar_rule neglogcdf_rician(x, ν, δ) (∇neglogcdf_rician(Ω, x, ν, δ)...,)
@define_ternary_dual_scalar_rule neglogcdf_rician (neglogcdf_rician, ∇neglogcdf_rician)

#### Gauss-Legendre quadrature

const DEFAULT_GAUSSLEGENDRE_ORDER = 16

@generated function gausslegendre_svectors(::Val{order}, ::Type{T}) where {order, T <: AbstractFloat}
    x, w = gausslegendre(order)
    x, w = tuple(T.(x)...), tuple(T.(w)...)
    return :(SVector{$order, $T}($x), SVector{order, T}($w))
end

@inline function neglogcdf_quadrature(neglogpdf::F, x₀::T, δ::T, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order, T <: AbstractFloat}
    x, w = gausslegendre_svectors(Val(order), T)
    logf = @. -neglogpdf(x₀ + δ * (1 + x) / 2)
    return -weighted_logsumexp(w, logf) - log(δ / 2)
end

@inline function cdf_quadrature(pdf::F, x₀::T, δ::T, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order, T <: AbstractFloat}
    x, w = gausslegendre_svectors(Val(order), T)
    f = @. pdf(x₀ + δ * (1 + x) / 2)
    return innerprod(w, f) .* δ ./ 2
end

@inline function weighted_logsumexp(w, logf)
    max_ = maximum(logf)
    return log(dot(w, exp.(logf .- max_))) + max_
end

@inline innerprod(x, y) = dot(x, y)
@inline innerprod(x::SVector{N, T}, y::SVector{N, SVector{M, T}}) where {N, M, T} = smatrix(y) * x
@inline innerprod(x::SVector{N, T}, y::SVector{N, NTuple{M, T}}) where {N, M, T} = innerprod(x, SVector{M, T}.(y))
@inline smatrix(x::SVector{N, SVector{M, T}}) where {N, M, T} = hcat(x...)
@inline smatrix(x::SVector{N, NTuple{M, T}}) where {N, M, T} = smatrix(SVector{M, T}.(x))
