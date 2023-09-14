#### Rician negative log-likelihood

@inline function neglogpdf_rician(x, ν, logσ)
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end

@inline neglogpdf_rician(x, ν) = (x - ν)^2 / 2 - log(x) - logbesseli0x(x * ν) # negative Rician log-likelihood `-logp(x | ν, σ = 1)`
@scalar_rule neglogpdf_rician(x, ν) (∇neglogpdf_rician(x, ν)...,)
@define_binary_dual_scalar_rule neglogpdf_rician (neglogpdf_rician, ∇neglogpdf_rician)

@inline function ∇neglogpdf_rician(x::T, ν::T) where {T <: Union{Float32, Float64}}
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
    low, mid = 0.25, 20
    z = x * ν

    # Note: there are really three relevant limits: z << 1, z >> 1, and x ≈ ν.
    # Could plausibly better account for the latter case, though it is tested quite robustly
    if z < low
        r = z * evalpoly(z^2, ∇neglogpdf_rician_small_coefs(T)) # logÎ₀′(z) + 1 - 1/2z = I₁(z) / I₀(z) ≈ z/2 + 𝒪(z^2)
        tmp = muladd(z, r - T(1), T(1)) # z * logÎ₀′(z) + 1/2 = 1 + z * (I₁(z) / I₀(z) - 1)
        ∂x = x - ν - tmp / x
        ∂ν = ν - x * r
    elseif z > mid
        z⁻¹ = inv(z)
        tmp = -z⁻¹ * evalpoly(z⁻¹, ∇neglogpdf_rician_large_coefs(T)) # z * logÎ₀′(z) = 1/2 + z * (I₁(z) / I₀(z) - 1) ≈ -1/8z + 𝒪(1/z^2)
        ∂x = x - ν - (1/T(2) + tmp) / x
        ∂ν = ν - x + (1/T(2) - tmp) / ν
    else
        if T == Float32
            r = z * evalpoly(z, ∇neglogpdf_rician_med_num_coefs(T)) / evalpoly(z, ∇neglogpdf_rician_med_den_coefs(T)) # I₁(z) / I₀(z)
        else
            r = besseli1x(z) / besseli0x(z) # I₁(z) / I₀(z), accurate for all z
        end
        ∂x = x - ν * r - inv(x)
        ∂ν = ν - x * r
    end

    return (∂x, ∂ν)
end
@inline ∇pdf_rician(x, ν) = -exp(-neglogpdf_rician(x, ν)) .* ∇neglogpdf_rician(x, ν)

# Argument ranges: x < 0.25, 0.25 < x < 20, x > 20
∇neglogpdf_rician_small_coefs(::Type{Float32}) = (0.5f0, -0.0624989f0, 0.010394423f0, -0.0016448505f0)
∇neglogpdf_rician_med_num_coefs(::Type{Float32}) = (0.49999794f0, 0.025516365f0, 0.042070463f0, 0.0014473405f0, 0.00044399212f0, -2.700828f-5, 1.0477163f-6, -2.3381961f-8, 2.2788199f-10)
∇neglogpdf_rician_med_den_coefs(::Type{Float32}) = (1.0f0, 0.05099831f0, 0.20924653f0, 0.009116647f0, 0.0063170674f0)
∇neglogpdf_rician_large_coefs(::Type{Float32}) = (0.125f0, 0.124996506f0, 0.1958465f0, 0.378206f0, 1.6191356f0)

∇neglogpdf_rician_small_coefs(::Type{Float64}) = (0.4999999999999999, -0.06249999999994528, 0.010416666662044488, -0.001790364434468454, 0.0003092424332731733, -5.344192059352683e-5, 9.146096503297768e-6, -1.3486016148802724e-6)
∇neglogpdf_rician_large_coefs(::Type{Float64}) = (0.1249999999999998, 0.1250000000010686, 0.19531249903078632, 0.406250343464912, 1.047788945815347, 3.2254412507133896, 11.017629183379448, 66.00330904779759, -342.14096395913083, 11018.560609519136, -99099.55058980543, 555353.7956242649)

#=
# Argument ranges: x < 0.5, 0.5 < x < 12, x > 12
#   TODO: these coefficients are suboptimal, but it's very tricky to choose good branch points and polynomial degrees to get a good fit in the middle region because Remez.jl keeps failing to converge
∇neglogpdf_rician_small_coefs(::Type{Float64}) = (0.49999999999999956, -0.062499999999846664, 0.010416666658408659, -0.0017903644119087664, 0.00030924300509276015, -5.345177452435453e-5, 9.2075534147453e-6, -1.5232462810975076e-6, 1.8974134079174596e-7)
∇neglogpdf_rician_med_num_coefs(::Type{Float64}) = (0.5000000004966851, 0.01122667982473457, 0.047502676822322044, 0.0009191568102651722, 0.0009720899797046758, 9.417723209861922e-6, 3.7280056331105413e-6, -7.20394466617955e-8, -5.957610768213467e-9, 6.062812828729949e-10, -2.636893894683211e-11, 6.079421129186984e-13, -6.056472431823339e-15)
∇neglogpdf_rician_med_den_coefs(::Type{Float64}) = (1.0, 0.022453368903854472, 0.22000531593231876, 0.004645073466972717, 0.008611377388601339, 0.00013182580795664643, 8.107245613176089e-5)
∇neglogpdf_rician_large_coefs(::Type{Float64}) = (0.1249999999999992, 0.12499999999439716, 0.19531250971443673, 0.4062447911496693, 1.049225472064705, 3.0056460412897588, 32.838254585383595, -1420.189706732526, 71539.8082487195, -2.508083002974601e6, 6.4434335431566015e7, -1.2058209528915246e9, 1.6234569743422686e10, -1.526820104308443e11, 9.487442853742372e11, -3.4840019782926743e12, 5.698923220622622e12)
=#

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
