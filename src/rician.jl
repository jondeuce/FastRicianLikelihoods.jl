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
    low, mid = T(0.5), T(15.0)
    z = x * ν

    # Note: there are really three relevant limits: z << 1, z >> 1, and x ≈ ν.
    # Could plausibly better account for the latter case, though it is tested quite robustly
    if z < low
        r = z * evalpoly(z^2, ∇neglogpdf_rician_small_coefs(T)) # logÎ₀′(z) + 1 - 1/2z = I₁(z) / I₀(z) ≈ z/2 + 𝒪(z^2)
        tmp = muladd(z, r - T(1), T(1)) # z * logÎ₀′(z) + 1/2 = 1 + z * (I₁(z) / I₀(z) - 1)
        ∂x = x - ν - tmp / x
        ∂ν = ν - x * r
    elseif z < mid
        z² = z^2
        r = z * evalpoly(z², ∇neglogpdf_rician_med_num_coefs(T)) / evalpoly(z², ∇neglogpdf_rician_med_den_coefs(T)) # r = besseli1x(z) / besseli0x(z) = I₁(z) / I₀(z)
        ∂x = x - ν * r - inv(x)
        ∂ν = ν - x * r
    else
        z⁻¹ = inv(z)
        tmp = -z⁻¹ * evalpoly(z⁻¹, ∇neglogpdf_rician_large_coefs(T)) # z * logÎ₀′(z) = 1/2 + z * (I₁(z) / I₀(z) - 1) ≈ -1/8z + 𝒪(1/z^2)
        ∂x = x - ν - (1/T(2) + tmp) / x
        ∂ν = ν - x + (1/T(2) - tmp) / ν
    end

    return (∂x, ∂ν)
end
@inline ∇pdf_rician(x, ν) = -exp(-neglogpdf_rician(x, ν)) .* ∇neglogpdf_rician(x, ν)

# Argument ranges: x < 0.5, 0.5 < x < 15.0, x > 15.0
∇neglogpdf_rician_small_coefs(::Type{Float32}) = (0.5f0, -0.0624989f0, 0.010394423f0, -0.0016448505f0)
∇neglogpdf_rician_med_num_coefs(::Type{Float32}) = (0.49999985f0, 0.04570739f0, 0.0008458269f0, 3.3240756f-6, 1.2917363f-9)
∇neglogpdf_rician_med_den_coefs(::Type{Float32}) = (1.0f0, 0.21641381f0, 0.007910766f0, 6.735792f-5, 9.917585f-8)
∇neglogpdf_rician_large_coefs(::Type{Float32}) = (0.12500001f0, 0.12498587f0, 0.19689824f0, 0.34546292f0, 1.9343305f0)

# Argument ranges: x < 0.5, 0.5 < x < 15.0, x > 15.0
#   TODO: these coefficients may be suboptimal, but it's very tricky to choose good branch points and polynomial degrees to get a good fit in the middle region because Remez.jl keeps failing to converge
∇neglogpdf_rician_small_coefs(::Type{Float64}) = (0.4999999999999999, -0.06249999999994528, 0.010416666662044488, -0.001790364434468454, 0.0003092424332731733, -5.344192059352683e-5, 9.146096503297768e-6, -1.3486016148802727e-6)
∇neglogpdf_rician_med_num_coefs(::Type{Float64}) = (0.4999999999966969, 0.053203375726198425, 0.0015365472423239004, 1.6933167398372286e-5, 7.74041701268799e-8, 1.3866724411625944e-10, 7.658745381064089e-14, 5.59737149816315e-18)
∇neglogpdf_rician_med_den_coefs(::Type{Float64}) = (1.0, 0.23140675143407377, 0.011165605095174927, 0.00018932214451711994, 1.3186259077732742e-6, 3.787997063328428e-9, 3.9174766677802805e-12, 9.546113545877485e-16)
∇neglogpdf_rician_large_coefs(::Type{Float64}) = (0.12500000000000017, 0.12499999999879169, 0.19531250150899987, 0.4062492562129355, 1.0480435526081948, 3.188906697154322, 14.49393731493799, -164.07408273124315, 10554.066042613813, -363473.66139754397, 9.257867756487977e6, -1.6750893375625065e8, 2.110022217619635e9, -1.7523461611835144e10, 8.611676733884535e10, -1.8844466382522766e11)

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
