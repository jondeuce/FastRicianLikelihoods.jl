####
#### Rician negative log-likelihood
####

@inline function neglogpdf_rician(x::Real, ν::Real, logσ::Real)
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end
@inline neglogpdf_rician(x::Real, ν::Real) = (x - ν)^2 / 2 - log(x) - logbesseli0x(x * ν) # negative Rician log-likelihood `-logp(x | ν, σ = 1)`

@inline function ∇neglogpdf_rician(x::Real, ν::Real)
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

@inline pdf_rician(args...) = exp(-neglogpdf_rician(args...))
@inline ∇pdf_rician(x::Real, ν::Real) = -exp(-neglogpdf_rician(x, ν)) .* ∇neglogpdf_rician(x, ν)

@scalar_rule neglogpdf_rician(x, ν) (∇neglogpdf_rician(x, ν)...,)
@dual_rule_from_frule neglogpdf_rician(x, ν)

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

@inline neglogpdf_qrician(x::Real, ν::Real, δ::Real, order::Val) = neglogf_quadrature(Base.Fix2(neglogpdf_rician, ν), x, δ, order)
@inline ∇neglogpdf_qrician(x::Real, ν::Real, δ::Real, order::Val) = ∇neglogpdf_qrician_with_primal(x, ν, δ, order)[2]

@inline function ∇neglogpdf_qrician_with_primal(Ω::Real, x::Real, ν::Real, δ::Real, order::Val)
    ∂x, ∂ν = f_quadrature(x, δ, order) do y
        ∇ = ∇neglogpdf_rician(y, ν) # differentiate the integrand
        ∇ = SVector(promote(∇...))
        return exp(Ω - neglogpdf_rician(y, ν)) * ∇
    end
    ∂δ = -exp(Ω - neglogpdf_rician(x + δ, ν)) # by fundamental theorem of calculus
    return Ω, (∂x, ∂ν, ∂δ)
end
@inline ∇neglogpdf_qrician_with_primal(x::Real, ν::Real, δ::Real, order::Val) = ∇neglogpdf_qrician_with_primal(neglogpdf_qrician(x, ν, δ, order), x, ν, δ, order)

@scalar_rule neglogpdf_qrician(x, ν, δ, order::Val) (∇neglogpdf_qrician_with_primal(Ω, x, ν, δ, order)[2]..., NoTangent())
@dual_rule_from_frule neglogpdf_qrician(x, ν, δ, !(order::Val))

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
    y = @. f(x₀ + δ * x)
    return vecdot(w, y) * δ
end

@inline function neglogf_quadrature(neglogf::F, x₀::Real, δ::Real, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order}
    # I = ∫_{0}^{δ} [f(t)] dt, where f(t) = exp(-neglogf(t))
    T = checkedfloattype(x₀, δ)
    x, w = gausslegendre_unit_interval(Val(order), T)
    logy = @. -neglogf(x₀ + δ * x)
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
    logy = reduce(hcat, logy) # stack as columns
    ȳ = exp.(logy .- max_)
    return log.(vecdot(w, ȳ)) .+ max_
end

@inline vecdot(w::SVector{N}, y::SVector{N}) where {N} = dot(w, y)
@inline vecdot(w::SVector{N}, y::SVector{N, <:SVector{M}}) where {N, M} = vecdot(w, reduce(hcat, y))
@inline vecdot(w::SVector{N}, y::SMatrix{M, N}) where {N, M} = y * w
