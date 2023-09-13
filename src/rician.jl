#### Rician negative log-likelihood

@inline function neglogpdf_rician(x, ν, logσ)
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end

@inline neglogpdf_rician(x, ν) = (x - ν)^2 / 2 - log(x) - logbesseli0x(x * ν) # negative Rician log-likelihood `-logp(x | ν, σ = 1)`
@scalar_rule neglogpdf_rician(x, ν) (∇neglogpdf_rician(x, ν)...,)
@define_binary_dual_scalar_rule neglogpdf_rician (neglogpdf_rician, ∇neglogpdf_rician)

@inline function ∇neglogpdf_rician(x, ν)
    # Define the univariate normalized Bessel function `Î₀` as
    #
    #   Î₀(z) = I₀(z) / (exp(z) / √2πz).
    #
    # The negative likelihood is then be written as
    #
    #   -logp(x | ν, σ = 1) = (x - ν)^2 / 2 - log(x / ν) / 2 - logÎ₀(x * ν) + log√2π.
    #
    # All that must be approximated then is `d/dz logÎ₀(z)` where `z = x * ν`:
    #
    #   d/dz logÎ₀(z) =  1/2z + (I₁(z) / I₀(z) - 1)
    #                 ≈ -1/8z^2 - 1/8z^3 - 25/128z^4 - 13/32z^5 + 𝒪(1/z^6)        (z >> 1)
    #                 ≈  1/2z - 1 + z/2 - z^3/16 + z^5/96 - 11*z^7/6144 + 𝒪(z^9)  (z << 1)
    #   d/dx logÎ₀(z) = ν * d/dz logÎ₀(z)
    #   d/dν logÎ₀(z) = x * d/dz logÎ₀(z)
    x, ν = float(x), float(ν)
    z = x * ν
    T = typeof(z)

    # Note: there are really three relevant limits: z >> 1, z << 1, and x ≈ ν.
    # Could plausibly better account for the latter case, though it is tested quite robustly
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
        r = besseli1x(z) / besseli0x(z) # I₁(z) / I₀(z), accurate for all z
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
@inline neglogcdf_rician(x, ν, δ) = neglogf_quadrature(Base.Fix2(neglogpdf_rician, ν), x, δ)

@inline function ∇neglogcdf_rician_kernel(Ω, x, ν, δ)
    ∂x, ∂ν = f_quadrature(x, δ) do y
        ∇ = ∇neglogpdf_rician(y, ν) # differentiate the integrand
        return exp(Ω - neglogpdf_rician(y, ν)) * SVector(∇)
    end
    ∂δ = -exp(Ω - neglogpdf_rician(x + δ, ν)) # by fundamental theorem of calculus
    return (∂x, ∂ν, ∂δ)
end
@inline ∇neglogcdf_rician(x, ν, δ) = ∇neglogcdf_rician_kernel(neglogcdf_rician(x, ν, δ), x, ν, δ)

@scalar_rule neglogcdf_rician(x, ν, δ) (∇neglogcdf_rician_kernel(Ω, x, ν, δ)...,)
@define_ternary_dual_scalar_rule neglogcdf_rician (neglogcdf_rician, ∇neglogcdf_rician)

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
    logy = hcat(logy...) # stack as columns
    ȳ = exp.(logy .- max_)
    return log.(vecdot(w, ȳ)) .+ max_
end

@inline vecdot(w::SVector{N, T}, y::SVector{N, T}) where {N, T <: AbstractFloat} = dot(w, y)
@inline vecdot(w::SVector{N, T}, y::SVector{N, SVector{M, T}}) where {N, M, T <: AbstractFloat} = vecdot(w, hcat(y...))
@inline vecdot(w::SVector{N, T}, y::SMatrix{M, N, T}) where {N, M, T <: AbstractFloat} = y * w
