#### Utilities

@inline promote_float(x...) = promote(map(float, x)...)

#### Rician negative log-likelihood

@inline function neglogpdf_rician(x::T, ν::T, logσ::T) where {T <: Real}
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end
@inline neglogpdf_rician(x, ν, logσ) = neglogpdf_rician(promote_float(x, ν, logσ)...)

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
        ∂x = x - ν - (T(0.5) - tmp) / x
        ∂ν = ν - x + (T(0.5) + tmp) / ν
    end

    return (∂x, ∂ν)
end

@inline pdf_rician(args...) = exp(-neglogpdf_rician(args...))
@inline ∇pdf_rician(x::T, ν::T) where {T <: Union{Float32, Float64}} = -exp(-neglogpdf_rician(x, ν)) .* ∇neglogpdf_rician(x, ν)

@scalar_rule neglogpdf_rician(x, ν) (∇neglogpdf_rician(x, ν)...,)
@dual_rule_from_frule neglogpdf_rician(x, ν)

#### Rician negative log-cdf

# CDF is approximated by an integral of the Rician PDF over `(x, x+δ)` using Gauss-Legendre quadrature.
# Consequently, PDF is never evaluated at the endpoints.
@inline function neglogcdf_rician(x::T, ν::T, logσ::T, δ::T, order::Val) where {T <: Real}
    σ⁻¹ = exp(-logσ)
    return neglogcdf_rician(σ⁻¹ * x, σ⁻¹ * ν, σ⁻¹ * δ, order)
end
@inline neglogcdf_rician(x, ν, logσ, δ, order::Val) = neglogcdf_rician(promote_float(x, ν, logσ, δ)..., order)

@inline neglogcdf_rician(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}} = neglogf_quadrature(Base.Fix2(neglogpdf_rician, ν), x, δ, order)
@inline ∇neglogcdf_rician(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}} = ∇neglogcdf_rician_with_primal(x, ν, δ, order)[2]

@inline function ∇neglogcdf_rician_with_primal(Ω::T, x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    ∂x, ∂ν = f_quadrature(x, δ, order) do y
        ∇ = ∇neglogpdf_rician(y, ν) # differentiate the integrand
        ∇ = SVector{2, T}(∇)
        return exp(Ω - neglogpdf_rician(y, ν)) * ∇
    end
    ∂δ = -exp(Ω - neglogpdf_rician(x + δ, ν)) # by fundamental theorem of calculus
    return Ω, (∂x, ∂ν, ∂δ)
end
@inline ∇neglogcdf_rician_with_primal(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}} = ∇neglogcdf_rician_with_primal(neglogcdf_rician(x, ν, δ, order), x, ν, δ, order)

@scalar_rule neglogcdf_rician(x, ν, δ, order::Val) (∇neglogcdf_rician_with_primal(Ω, x, ν, δ, order)[2]..., NoTangent())
@dual_rule_from_frule neglogcdf_rician(x, ν, δ, !order)

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
