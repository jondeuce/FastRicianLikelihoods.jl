module Utils

using Test

using ArbNumerics: ArbNumerics, ArbReal
using FiniteDifferences: FiniteDifferences
using QuadGK: quadgk
using Zygote: Zygote

using FastRicianLikelihoods: FastRicianLikelihoods, ForwardDiff, SpecialFunctions, StaticArrays
using FastRicianLikelihoods: neglogpdf_rician, ∇neglogpdf_rician, ∇²neglogpdf_rician, ∇²neglogpdf_rician_with_gradient, ∇³neglogpdf_rician_with_gradient_and_hessian
using FastRicianLikelihoods: neglogpdf_qrician, ∇neglogpdf_qrician, ∇²neglogpdf_qrician, ∇²neglogpdf_qrician_with_gradient, ∇²neglogpdf_qrician_with_primal_and_gradient, ∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian
using .StaticArrays: SVector, SMatrix

Base.setprecision(BigFloat, 500; base = 2)
ArbNumerics.setworkingprecision(ArbReal; bits = 500)

@inline common_float_type(args::Tuple) = mapfoldl(typeof, common_float_type, args)

@inline common_float_type(::Type{T1}, ::Type{T2}) where {T1, T2} =
    (T1 <: Real && T2 <: Real) ? (@assert T1 === T2; @assert T1 <: Union{Float32, Float64}; T1) :
    (T1 <: Real) ? (@assert T1 <: Union{Float32, Float64}; T1) :
    (T2 <: Real) ? (@assert T2 <: Union{Float32, Float64}; T2) :
    (T1)

arbify(x) = x
arbify(x::Real) = error("Expected typeof(x) = $(typeof(x)) <: Union{Float32, Float64}")
arbify(x::Union{Float32, Float64}) = ArbReal(x)::ArbReal
arbify(f::Function) = function f_arbified(args...; kwargs...)
    T = common_float_type(args)
    xs = arbify.(args)
    y = f(xs...; kwargs...)
    return dearbify(T, y)
end
dearbify(::Type{T}, x::Number) where {T} = convert(T, x)
dearbify(::Type{T}, x::Union{Tuple, NamedTuple, AbstractArray}) where {T} = map(Base.Fix1(dearbify, T), x)

const DEFAULT_CENTRAL_FDM = FiniteDifferences.central_fdm(4, 1)
const DEFAULT_FORWARD_FDM = FiniteDifferences.forward_fdm(4, 1)
∇FD_central(f, args::Real...) = ∇FD(DEFAULT_CENTRAL_FDM, f, args...)
∇FD_forward(f, args::Real...) = ∇FD(DEFAULT_FORWARD_FDM, f, args...)
∇FD(fdm, f, args::Real...) = (FiniteDifferences.grad(fdm, Base.splat(f), [args...])[1]...,)

∇Fwd(f, args::Real...) = Tuple(ForwardDiff.gradient(Base.splat(f), SVector(args)))
∇Zyg(f, args::Real...) = Zygote.gradient(f, args...)

function ∇FD_nonneg(f, x, ϵ; log_transform::Bool)
    @assert x >= 0 "x must be nonnegative; got x = $x"
    if log_transform && x > 0
        # Second order central difference on log(x): df/dx = df/dlogx * dlogx/dx = df/dlogx / x
        logx = log(x)
        return (f(exp(logx + ϵ)) .- f(exp(logx - ϵ))) ./ (2 * ϵ * x)
    elseif x >= ϵ
        # Second order central difference on x: df/dx = (f(x + ϵ) - f(x - ϵ)) / (2 * ϵ)
        return (f(x + ϵ) .- f(x - ϵ)) ./ (2 * ϵ)
    else
        # Second order forward difference on x: df/dx = (-3 * f(x) + 4 * f(x + ϵ) - f(x + 2 * ϵ)) / (2 * ϵ)
        return (-3 .* f(x) .+ 4 .* f(x + ϵ) .- f(x + 2 * ϵ)) ./ (2 * ϵ)
    end
end

@inline infnorm(x::Number) = abs(x)
@inline infnorm(x::AbstractArray) = maximum(abs, x)
@inline infnorm(x::Tuple) = maximum(abs, x)

@inline function minerr(ŷ, y)
    abserr = isinf(y) && ŷ == y ? zero(y) : abs(ŷ - y)
    relerr = y == 0 ? abserr : abserr / abs(y)
    return min(relerr, abserr)
end

#### ArbReal extensions

SpecialFunctions.erfinv(x::ArbReal) = ArbReal(erfinv(big(x)))
SpecialFunctions.erfcinv(x::ArbReal) = ArbReal(erfcinv(big(x)))

FastRicianLikelihoods.besseli2(x::ArbReal) = ArbNumerics.besseli(2, x)
FastRicianLikelihoods.besseli2x(x::ArbReal) = exp(-abs(x)) * ArbNumerics.besseli(2, x)
FastRicianLikelihoods.logbesseli0(x::ArbReal) = log(ArbNumerics.besseli(0, x))
FastRicianLikelihoods.logbesseli0x(x::ArbReal) = log(ArbNumerics.besseli(0, x)) - abs(x)
FastRicianLikelihoods.logbesseli1(x::ArbReal) = log(ArbNumerics.besseli(1, x))
FastRicianLikelihoods.logbesseli1x(x::ArbReal) = log(ArbNumerics.besseli(1, x)) - abs(x)
FastRicianLikelihoods.logbesseli2(x::ArbReal) = log(ArbNumerics.besseli(2, x))
FastRicianLikelihoods.logbesseli2x(x::ArbReal) = log(ArbNumerics.besseli(2, x)) - abs(x)
FastRicianLikelihoods.laguerre½(x::ArbReal) = exp(x / 2) * ((1 - x) * ArbNumerics.besseli(0, -x / 2) - x * ArbNumerics.besseli(1, -x / 2))
FastRicianLikelihoods.besseli1i0(x::ArbReal) = ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x)
FastRicianLikelihoods.besseli1i0x(x::ArbReal) = ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x) / x
FastRicianLikelihoods.besseli1i0m1(x::ArbReal) = ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x) - 1
FastRicianLikelihoods.mean_rician(ν::ArbReal, σ::ArbReal) = σ * √(ArbReal(π) / 2) * FastRicianLikelihoods.laguerre½(-(ν / σ)^2 / 2)
FastRicianLikelihoods.std_rician(ν::ArbReal, σ::ArbReal) = sqrt(ν^2 + 2σ^2 - ArbReal(π) * σ^2 * FastRicianLikelihoods.laguerre½(-(ν / σ)^2 / 2)^2 / 2)
# FastRicianLikelihoods.∂x_laguerre½(x::ArbReal)
# FastRicianLikelihoods.∂x_besseli0x(x::ArbReal)
# FastRicianLikelihoods.∂x_besseli1x(x::ArbReal)

function FastRicianLikelihoods.neglogpdf_rician(x::ArbReal, ν::ArbReal)
    x <= 0 && return ArbReal(Inf) # limit as x -> 0⁺
    return (x^2 + ν^2) / 2 - log(x) - log(ArbNumerics.besseli(0, x * ν))
end

function FastRicianLikelihoods.∇neglogpdf_rician(x::ArbReal, ν::ArbReal)
    x <= 0 && return (ArbReal(Inf), ν) # limit as x -> 0⁺
    I0, I1 = ArbNumerics.besseli(0, x * ν), ArbNumerics.besseli(1, x * ν)
    r = I1 / I0
    ∂x = x - ν * r - 1 / x
    ∂ν = ν - x * r
    return (∂x, ∂ν)
end

function FastRicianLikelihoods.∇²neglogpdf_rician_with_gradient(x::ArbReal, ν::ArbReal)
    x <= 0 && return (ArbReal(-Inf), ν), (ArbReal(Inf), ArbReal(0), ArbReal(1)) # limit as x -> 0⁺
    z = x * ν
    I0, I1 = ArbNumerics.besseli(0, z), ArbNumerics.besseli(1, z)
    r = I1 / I0
    r′ = 1 - r / z - r^2
    ∂x = x - ν * r - 1 / x
    ∂ν = ν - x * r
    ∂²x = 1 + 1 / x^2 - ν^2 * r′
    ∂x∂ν = -r - z * r′
    ∂²ν = 1 - x^2 * r′
    return (∂x, ∂ν), (∂²x, ∂x∂ν, ∂²ν)
end
FastRicianLikelihoods.∇²neglogpdf_rician(x::ArbReal, ν::ArbReal) = FastRicianLikelihoods.∇²neglogpdf_rician_with_gradient(x, ν)[2]

function FastRicianLikelihoods.∇³neglogpdf_rician_with_gradient_and_hessian(x::ArbReal, ν::ArbReal)
    x <= 0 && return (ArbReal(-Inf), ν), (ArbReal(Inf), ArbReal(0), ArbReal(1)), (ArbReal(-Inf), -ν, ArbReal(0), ArbReal(0)) # limit as x -> 0⁺
    z = x * ν
    I0, I1 = ArbNumerics.besseli(0, z), ArbNumerics.besseli(1, z)
    r = I1 / I0
    r′ = 1 - r / z - r^2
    r′′ = -r′ / z + r / z^2 - 2 * r * r′
    ∂x = x - ν * r - 1 / x
    ∂ν = ν - x * r
    ∂²x = 1 + 1 / x^2 - ν^2 * r′
    ∂x∂ν = -r - z * r′
    ∂²ν = 1 - x^2 * r′
    ∂³x = -2 / x^3 - ν^3 * r′′
    ∂²x∂ν = -2 * ν * r′ - z * ν * r′′
    ∂x∂ν² = -2 * x * r′ - z * x * r′′
    ∂³ν = -x^3 * r′′
    return (∂x, ∂ν), (∂²x, ∂x∂ν, ∂²ν), (∂³x, ∂²x∂ν, ∂x∂ν², ∂³ν)
end

function FastRicianLikelihoods.mode_rician(ν::ArbReal; tol = √eps(one(ArbReal)), kwargs...)
    ν <= 0 && return one(ν)
    ν >= 1e32 && return ν + (1 - 3 / 4ν^2) / 2ν # relative error < 1e-180
    f(x) = FastRicianLikelihoods.∇neglogpdf_rician(x, ν)[1]
    ∇f(x) = FastRicianLikelihoods.∇²neglogpdf_rician(x, ν)[1]
    x₀ = ArbReal(FastRicianLikelihoods.mode_rician(Float64(ν)))
    return newton_root_find(f, ∇f, x₀; ftol = 0, xtol = tol, dtol = 0, kwargs...)[1]
end

function FastRicianLikelihoods.var_mode_rician(ν::ArbReal; kwargs...)
    ν <= 0 && return one(ν) / 2
    ν >= 1e30 && return 1 - 1 / 2ν^2 # relative error < 1e-120
    μ = FastRicianLikelihoods.mode_rician(ν; kwargs...)
    ∂²x = FastRicianLikelihoods.∇²neglogpdf_rician(μ, ν)[1]
    return 1 / ∂²x
end

neglogpdf_qrician_arbreal_quadgk_rtol() = ArbReal(1e-60)
neglogpdf_qrician_arbreal_quadgk_order() = 31
neglogpdf_qrician_arbreal_eps() = cbrt(eps(one(ArbReal)))^2
neglogpdf_qrician_arbreal_first_deriv_step_size(u) = ArbReal(u)^ArbReal(1 // 3)
neglogpdf_qrician_arbreal_second_deriv_step_size(u) = ArbReal(u)^ArbReal(2 // 9)
neglogpdf_qrician_arbreal_third_deriv_step_size(u) = ArbReal(u)^ArbReal(4 // 27)

function FastRicianLikelihoods.neglogpdf_qrician(x::ArbReal, ν::ArbReal, δ::ArbReal, ::Val{order}; method::Symbol, quadgk_order::Int = neglogpdf_qrician_arbreal_quadgk_order()) where {order}
    if method === :analytic
        return neglogpdf_qrician_integrate(x, ν, δ; quadgk_order)
    elseif method === :gausslegendre
        f = Base.Fix2(neglogpdf_rician, ν)
        return FastRicianLikelihoods.neglogf_quadrature(f, ArbReal, x, δ, Val(order))
    else
        error("Unsupported method: $method")
    end
end
FastRicianLikelihoods.gausslegendre_unit_interval(order::Val, ::Type{ArbReal}) = map(x -> ArbReal.(x), FastRicianLikelihoods.gausslegendre_unit_interval(order, BigFloat))

function neglogpdf_qrician_integrate(x::ArbReal, ν::ArbReal, δ::ArbReal; quadgk_order::Int = neglogpdf_qrician_arbreal_quadgk_order())
    rtol, atol = neglogpdf_qrician_arbreal_quadgk_rtol(), ArbReal(0)

    # # Naive implementation that is good enough if you are near the mode
    # Ω = neglogpdf_rician(x + δ / 2, ν)
    # I, _ = qrician_integrate(Ω, x, x + δ, ν; rtol, atol, quadgk_order)
    # return Ω - log(I)

    μ = FastRicianLikelihoods.mode_rician(ν)
    σ = √FastRicianLikelihoods.var_mode_rician(ν)
    Δσ = 3 * σ

    if x < μ < x + δ && μ - x > Δσ && x + δ - μ > Δσ
        Ω2 = neglogpdf_rician(x + δ, ν) # minimum on [x + δ, Inf)
        I2, _ = qrician_integrate(Ω2, x + δ, ArbReal(Inf), ν; rtol, atol, quadgk_order)
        if x > 0
            Ω1 = neglogpdf_rician(x, ν) # minimum on [0, x]
            I1, _ = qrician_integrate(Ω1, ArbReal(0), x, ν; rtol, atol, quadgk_order)
            out = -log1p(-(exp(-Ω1) * I1 + exp(-Ω2) * I2))
        else
            out = -log1p(-exp(-Ω2) * I2)
        end
    else
        if x < μ < x + δ
            Ω = neglogpdf_rician(μ, ν) # minimum on [x, x + δ]
        elseif x >= μ
            Ω = neglogpdf_rician(x, ν) # minimum on [x, x + δ]
        else # x + δ <= μ
            Ω = neglogpdf_rician(x + δ, ν) # minimum on [x, x + δ]
        end
        I, _ = qrician_integrate(Ω, x, x + δ, ν; rtol, atol, quadgk_order)
        out = Ω - log(I)
    end

    return out
end

function qrician_integrate(f::Function, Ω::ArbReal, a::ArbReal, b::ArbReal, ν::ArbReal; rtol::ArbReal, atol::ArbReal, quadgk_order::Int = neglogpdf_qrician_arbreal_quadgk_order())
    if isfinite(b)
        I, E = quadgk(a, b; rtol, atol, order = quadgk_order, norm = infnorm) do x̃
            return exp(Ω - neglogpdf_rician(x̃, ν)) * f(x̃)
        end
    else
        # Change of variables: x = a + (1 - t) / t where t ∈ (0, 1)
        #TODO:
        #   Note that quadgk should do this internally; `ArbReal` used to cause this to fail, but it's now fixed on master.
        #   See: https://github.com/JuliaMath/QuadGK.jl/commit/0c479123a0756b79f1056a41302dbf3a35eda7cd
        @assert isfinite(a)
        I, E = quadgk(ArbReal(0), ArbReal(1); rtol, atol, order = quadgk_order, norm = infnorm) do t̃
            x̃ = a + (1 - t̃) / t̃
            return exp(Ω - neglogpdf_rician(x̃, ν)) * f(x̃) / t̃^2
        end
    end
    return I, E
end
qrician_integrate(Ω::ArbReal, args...; kwargs...) = qrician_integrate(_ -> ArbReal(1), Ω, args...; kwargs...)

function FastRicianLikelihoods.∇neglogpdf_qrician(x::ArbReal, ν::ArbReal, δ::ArbReal, ::Val{order}; method::Symbol, quadgk_order::Int = neglogpdf_qrician_arbreal_quadgk_order()) where {order}
    if method === :analytic
        rtol, atol = neglogpdf_qrician_arbreal_quadgk_rtol(), ArbReal(0)
        Ω = neglogpdf_qrician(x, ν, δ, Val(nothing); method = :analytic, quadgk_order)
        ∂ν, _ = qrician_integrate(x̃ -> ∇neglogpdf_rician(x̃, ν)[2], Ω, x, x + δ, ν; rtol, atol, quadgk_order) # differentiate the integrand
        ∂δ = -exp(Ω - neglogpdf_rician(x + δ, ν)) # FTC: ∂Ω/∂δ = ∂/∂δ [-log(F(x+δ) - F(x))] = -∂/∂δ [F(x+δ) - F(x)] / exp(-Ω) = -exp(Ω) * f(x+δ) = -exp(Ω - ω(x+δ))
        ∂x = ∂δ + exp(Ω - neglogpdf_rician(x, ν)) # FTC: ∂Ω/∂x = ∂/∂x [-log(F(x+δ) - F(x))] = -∂/∂x [F(x+δ) - F(x)] / exp(-Ω) = -exp(Ω) * (f(x+δ) - f(x)) = exp(Ω - ω(x)) - exp(Ω - ω(x+δ))
    elseif method === :finitediff
        u = neglogpdf_qrician_arbreal_quadgk_rtol()
        ϵ = neglogpdf_qrician_arbreal_first_deriv_step_size(u)
        ∂x = ∇FD_nonneg(x′ -> neglogpdf_qrician(x′, ν, δ, Val(nothing); method = :analytic, quadgk_order), x, ϵ; log_transform = true)
        ∂ν = ∇FD_nonneg(ν′ -> neglogpdf_qrician(x, ν′, δ, Val(nothing); method = :analytic, quadgk_order), ν, ϵ; log_transform = true)
        ∂δ = ∇FD_nonneg(δ′ -> neglogpdf_qrician(x, ν, δ′, Val(nothing); method = :analytic, quadgk_order), δ, ϵ; log_transform = true)
    elseif method === :gausslegendre
        u = neglogpdf_qrician_arbreal_eps()
        ϵ = neglogpdf_qrician_arbreal_first_deriv_step_size(u)
        ∂x = ∇FD_nonneg(x′ -> neglogpdf_qrician(x′, ν, δ, Val(order); method = :gausslegendre, quadgk_order), x, ϵ; log_transform = true)
        ∂ν = ∇FD_nonneg(ν′ -> neglogpdf_qrician(x, ν′, δ, Val(order); method = :gausslegendre, quadgk_order), ν, ϵ; log_transform = true)
        ∂δ = ∇FD_nonneg(δ′ -> neglogpdf_qrician(x, ν, δ′, Val(order); method = :gausslegendre, quadgk_order), δ, ϵ; log_transform = true)
    else
        error("Unsupported method: $method")
    end
    return (∂x, ∂ν, ∂δ)
end

function FastRicianLikelihoods.∇²neglogpdf_qrician_with_primal_and_gradient(x::ArbReal, ν::ArbReal, δ::ArbReal, ::Val{order}; method::Symbol, quadgk_order::Int = neglogpdf_qrician_arbreal_quadgk_order()) where {order}
    if method === :analytic
        rtol, atol = neglogpdf_qrician_arbreal_quadgk_rtol(), ArbReal(0)
        Ω = neglogpdf_qrician(x, ν, δ, Val(nothing); method = :analytic, quadgk_order)
        ∂x, ∂ν, ∂δ = ∇neglogpdf_qrician(x, ν, δ, Val(order); method = :analytic, quadgk_order)

        p₀ = exp(Ω - neglogpdf_rician(x, ν))
        p₁ = exp(Ω - neglogpdf_rician(x + δ, ν))
        ∇x₀, ∇ν₀ = ∇neglogpdf_rician(x, ν)
        ∇x₁, ∇ν₁ = ∇neglogpdf_rician(x + δ, ν)
        p₀∇x₀ = x == 0 ? -exp(Ω - ν^2 / 2) : p₀ * ∇x₀

        ∂xx = (p₁ * ∇x₁ - p₀∇x₀) + ∂x * ∂x
        ∂xν = (p₁ * ∇ν₁ - p₀ * ∇ν₀) + ∂x * ∂ν
        ∂xδ = p₁ * ∇x₁ + ∂x * ∂δ
        ∂νδ = p₁ * ∇ν₁ + ∂ν * ∂δ
        ∂δδ = p₁ * ∇x₁ + ∂δ * ∂δ

        ∂νν, _ = qrician_integrate(Ω, x, x + δ, ν; rtol, atol, quadgk_order) do y
            (_, ∂ν_y), (_, _, ∂νν_y) = ∇²neglogpdf_rician_with_gradient(y, ν)
            return ∂νν_y - (∂ν_y - ∂ν)^2
        end

    elseif method === :finitediff || method === :gausslegendre
        primal_method = method === :finitediff ? :analytic : :gausslegendre
        Ω = neglogpdf_qrician(x, ν, δ, Val(order); method = primal_method, quadgk_order)

        u = method === :finitediff ? neglogpdf_qrician_arbreal_quadgk_rtol() : neglogpdf_qrician_arbreal_eps()
        ϵ = neglogpdf_qrician_arbreal_second_deriv_step_size(u)
        ∂Ω(x, ν, δ) = ∇neglogpdf_qrician(x, ν, δ, Val(order); method, quadgk_order)
        ∂x, ∂ν, ∂δ = ∂Ω(x, ν, δ)
        ∂xx, ∂νx, ∂δx = ∇FD_nonneg(x′ -> ∂Ω(x′, ν, δ), x, ϵ; log_transform = true)
        ∂xν, ∂νν, ∂δν = ∇FD_nonneg(ν′ -> ∂Ω(x, ν′, δ), ν, ϵ; log_transform = true)
        ∂xδ, ∂νδ, ∂δδ = ∇FD_nonneg(δ′ -> ∂Ω(x, ν, δ′), δ, ϵ; log_transform = true)
        ∂xν, ∂xδ, ∂νδ = (∂xν + ∂νx) / 2, (∂xδ + ∂δx) / 2, (∂νδ + ∂δν) / 2

    else
        error("Unsupported method: $method")
    end
    return Ω, (∂x, ∂ν, ∂δ), (∂xx, ∂xν, ∂xδ, ∂νν, ∂νδ, ∂δδ)
end
FastRicianLikelihoods.∇²neglogpdf_qrician_with_gradient(x::ArbReal, ν::ArbReal, δ::ArbReal, ::Val{order}; kwargs...) where {order} = Base.tail(∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ, Val(order); kwargs...))
FastRicianLikelihoods.∇²neglogpdf_qrician(x::ArbReal, ν::ArbReal, δ::ArbReal, ::Val{order}; kwargs...) where {order} = last(∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ, Val(order); kwargs...))

function FastRicianLikelihoods.∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian(x::ArbReal, ν::ArbReal, δ::ArbReal, ::Val{order}; method::Symbol, quadgk_order::Int = neglogpdf_qrician_arbreal_quadgk_order()) where {order}
    if method === :analytic
        rtol, atol = neglogpdf_qrician_arbreal_quadgk_rtol(), ArbReal(0)
        Ω₀, Ω₁ = neglogpdf_rician(x, ν), neglogpdf_rician(x + δ, ν)
        (∇x₀, ∇ν₀), (∇xx₀, ∇xν₀, ∇νν₀) = ∇²neglogpdf_rician_with_gradient(x, ν)
        (∇x₁, ∇ν₁), (∇xx₁, ∇xν₁, ∇νν₁) = ∇²neglogpdf_rician_with_gradient(x + δ, ν)

        Ω = neglogpdf_qrician(x, ν, δ, Val(nothing); method = :analytic, quadgk_order)
        p₀ = exp(Ω - Ω₀)
        p₁ = exp(Ω - Ω₁)

        # Single pass integral to get all expectations needed
        (E_∂ν, E_∂νν, E_∂ννν, E_∂ν², E_∂ν³, E_∂ν∂νν), _ = qrician_integrate(Ω, x, x + δ, ν; rtol, atol, quadgk_order) do y
            (_, ∂ν_y), (_, _, ∂νν_y), (_, _, _, ∂ννν_y) = ∇³neglogpdf_rician_with_gradient_and_hessian(y, ν)
            return SVector(∂ν_y, ∂νν_y, ∂ννν_y, ∂ν_y^2, ∂ν_y^3, ∂ν_y * ∂νν_y)
        end

        # Intermediate values that behave poorly as x → 0: Ω₀ → Inf, p₀ → 0, ∇x₀ → -Inf, ∇ν₀ → ν, ∇xx₀ → Inf, ∇xν₀ → 0, ∇νν₀ → 1
        p₀∇x₀ = x == 0 ? -exp(Ω - ν^2 / 2) : p₀ * ∇x₀
        p₀∇xx₀_minus_p₀∇x₀² = x == 0 ? zero(ArbReal) : p₀ * (∇xx₀ - ∇x₀ * ∇x₀)

        # First derivatives (gradient)
        Ωx = p₀ - p₁
        Ων = E_∂ν
        Ωδ = -p₁
        g = (Ωx, Ων, Ωδ)

        # Second derivatives (Hessian)
        Hxx = (p₁ * ∇x₁ - p₀∇x₀) + Ωx * Ωx
        Hxν = (p₁ * ∇ν₁ - p₀ * ∇ν₀) + Ωx * Ων
        Hxδ = p₁ * ∇x₁ + Ωx * Ωδ
        Hνν = E_∂νν - (E_∂ν² - E_∂ν^2)
        Hνδ = p₁ * ∇ν₁ + Ων * Ωδ
        Hδδ = p₁ * ∇x₁ + Ωδ * Ωδ
        H = (Hxx, Hxν, Hxδ, Hνν, Hνδ, Hδδ)

        # Third derivatives (Jacobian of the Hessian)
        ∂Hxx∂x = (p₁ * (∇xx₁ + (Ωx - ∇x₁) * ∇x₁) - p₀∇x₀ * Ωx - p₀∇xx₀_minus_p₀∇x₀²) + 2 * Hxx * Ωx
        ∂Hxν∂x = (p₁ * (∇xν₁ + (Ωx - ∇x₁) * ∇ν₁) - p₀ * (∇ν₀ * Ωx + ∇xν₀) + p₀∇x₀ * ∇ν₀) + Hxx * Ων + Hxν * Ωx
        ∂Hxδ∂x = (p₁ * (∇xx₁ + (Ωx - ∇x₁) * ∇x₁)) + Hxx * Ωδ + Hxδ * Ωx
        ∂Hνν∂x = p₁ * (∇νν₁ - (∇ν₁ - Ων)^2 - Hνν) - p₀ * (∇νν₀ - (∇ν₀ - Ων)^2 - Hνν)
        ∂Hνδ∂x = (p₁ * (∇xν₁ + (Ωx - ∇x₁) * ∇ν₁)) + Hxν * Ωδ + Hxδ * Ων
        ∂Hδδ∂x = (p₁ * (∇xx₁ + (Ωx - ∇x₁) * ∇x₁)) + 2 * Hxδ * Ωδ

        ∂Hxx∂ν = (p₁ * (∇xν₁ + (Ων - ∇ν₁) * ∇x₁) - (p₀ * ∇xν₀ + p₀∇x₀ * (Ων - ∇ν₀))) + 2 * Hxν * Ωx
        ∂Hxν∂ν = (p₁ * (∇νν₁ + (Ων - ∇ν₁) * ∇ν₁) - p₀ * (∇νν₀ + ∇ν₀ * (Ων - ∇ν₀))) + Hxν * Ων + Hνν * Ωx
        ∂Hxδ∂ν = (p₁ * (∇xν₁ + (Ων - ∇ν₁) * ∇x₁)) + Hxν * Ωδ + Hνδ * Ωx
        ∂Hνν∂ν = E_∂ννν - 3 * (E_∂ν∂νν - E_∂ν * (E_∂νν - E_∂ν²)) + 2 * E_∂ν^3 + E_∂ν³
        ∂Hνδ∂ν = (p₁ * (∇νν₁ + (Ων - ∇ν₁) * ∇ν₁)) + Hνν * Ωδ + Hνδ * Ων
        ∂Hδδ∂ν = (p₁ * (∇xν₁ + (Ων - ∇ν₁) * ∇x₁)) + 2 * Hνδ * Ωδ

        ∂Hxx∂δ = (p₁ * (∇xx₁ + (Ωδ - ∇x₁) * ∇x₁) - p₀∇x₀ * Ωδ) + 2 * Hxδ * Ωx
        ∂Hxν∂δ = (p₁ * (∇xν₁ + (Ωδ - ∇x₁) * ∇ν₁) - p₀ * ∇ν₀ * Ωδ) + Hxδ * Ων + Hνδ * Ωx
        ∂Hxδ∂δ = (p₁ * (∇xx₁ + (Ωδ - ∇x₁) * ∇x₁)) + Hxδ * Ωδ + Hδδ * Ωx
        ∂Hνν∂δ = p₁ * (∇νν₁ - (∇ν₁ - Ων)^2 - Hνν)
        ∂Hνδ∂δ = (p₁ * (∇xν₁ + (Ωδ - ∇x₁) * ∇ν₁)) + Hνδ * Ωδ + Hδδ * Ων
        ∂Hδδ∂δ = (p₁ * (∇xx₁ + (Ωδ - ∇x₁) * ∇x₁)) + 2 * Hδδ * Ωδ

        J = SMatrix{6, 3, ArbReal, 18}(
            ∂Hxx∂x, ∂Hxν∂x, ∂Hxδ∂x, ∂Hνν∂x, ∂Hνδ∂x, ∂Hδδ∂x,
            ∂Hxx∂ν, ∂Hxν∂ν, ∂Hxδ∂ν, ∂Hνν∂ν, ∂Hνδ∂ν, ∂Hδδ∂ν,
            ∂Hxx∂δ, ∂Hxν∂δ, ∂Hxδ∂δ, ∂Hνν∂δ, ∂Hνδ∂δ, ∂Hδδ∂δ,
        )
    elseif method === :finitediff || method === :gausslegendre
        # Base Ω, g, H using the requested method
        Ω, g, H = ∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ, Val(order); method, quadgk_order)

        # Derivatives of H wrt each parameter (6-tuples)
        u = method === :finitediff ? neglogpdf_qrician_arbreal_quadgk_rtol() : neglogpdf_qrician_arbreal_eps()
        ϵ = neglogpdf_qrician_arbreal_third_deriv_step_size(u)
        Jx = ∇FD_nonneg(x′ -> last(∇²neglogpdf_qrician_with_primal_and_gradient(x′, ν, δ, Val(order); method, quadgk_order)), x, ϵ; log_transform = true)
        Jν = ∇FD_nonneg(ν′ -> last(∇²neglogpdf_qrician_with_primal_and_gradient(x, ν′, δ, Val(order); method, quadgk_order)), ν, ϵ; log_transform = true)
        Jδ = ∇FD_nonneg(δ′ -> last(∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ′, Val(order); method, quadgk_order)), δ, ϵ; log_transform = true)
        J = SMatrix{6, 3, ArbReal, 18}(Jx..., Jν..., Jδ...)
    else
        error("Unsupported method: $method")
    end
    return Ω, g, H, J
end

function FastRicianLikelihoods.neglogpdf_rician(x::ArbReal, ν::ArbReal, logσ::ArbReal)
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end

function FastRicianLikelihoods.neglogpdf_qrician(x::ArbReal, ν::ArbReal, logσ::ArbReal, δ::ArbReal, order::Val)
    σ⁻¹ = exp(-logσ)
    return neglogpdf_qrician(σ⁻¹ * x, σ⁻¹ * ν, σ⁻¹ * δ, order)
end

#### Helpers

function binary_search_root_find(f, a::T, b::T; tol = eps(T), verbose = false) where {T <: Real}
    fa, fb = f(a), f(b)
    iter = 0

    if fa * fb >= 0
        verbose && @warn "Binary search infeasible:" a b fa fb
        error("Root is not guaranteed within the given initial interval.")
    end

    while true
        iter += 1
        verbose && @info "Iteration $iter:" a b fa fb

        mid = (a + b) / 2
        (a == mid || b == mid) && break

        fmid = f(mid)
        if fmid == 0
            return mid
        elseif fa * fmid < 0
            b, fb = mid, fmid
        else
            a, fa = mid, fmid
        end

        abs(a - b) <= tol && break
    end

    root = (a + b) / 2
    verbose && @info "Converged after $iter iter. Approximate Root: $root"

    return root, iter
end

function newton_root_find(f, ∇f, x₀::T; ftol = √(eps(T)), xtol = √(eps(T)), dtol = eps(T), verbose = false) where {T <: Real}
    x = x₀
    iter = 0

    while true
        iter += 1
        f_x = f(x)
        ∇f_x = ∇f(x)

        if abs(∇f_x) < dtol
            verbose && @warn "Newton's method infeasible:" f_x ∇f_x dtol
            error("Derivative is close to zero.")
        end

        Δx = f_x / ∇f_x
        x, x_last = x - Δx, x
        verbose && @info "Iteration $iter:" x x_last Δx f_x ∇f_x

        if abs(f_x) < ftol || abs(x - x_last) < xtol
            verbose && @info "Converged after $iter iterations. Approximate Root: $x"
            return x, iter
        end
    end
end

end # module Utils

using .Utils
