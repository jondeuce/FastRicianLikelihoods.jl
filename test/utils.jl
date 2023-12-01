module Utils

using Test

using ArbNumerics: ArbNumerics, ArbFloat
using FastRicianLikelihoods: FastRicianLikelihoods, neglogpdf_rician, ∇neglogpdf_rician, neglogpdf_qrician, ∇neglogpdf_qrician
using ForwardDiff: ForwardDiff
using Memoize: @memoize
using QuadGK: quadgk
using SpecialFunctions: SpecialFunctions
using StaticArrays: SVector
using ThreadSafeDicts: ThreadSafeDict
using Zygote: Zygote

Base.setprecision(BigFloat, 500)
ArbNumerics.setworkingprecision(ArbFloat; digits = 500, base = 2)
ArbNumerics.setextrabits(128)

@inline common_float_type(args::Tuple) = mapfoldl(typeof, common_float_type, args)

@inline common_float_type(::Type{T1}, ::Type{T2}) where {T1, T2} =
    (T1 <: Real && T2 <: Real) ? (@assert T1 === T2; @assert T1 <: Union{Float32, Float64}; T1) :
    (T1 <: Real) ? (@assert T1 <: Union{Float32, Float64}; T1) :
    (T2 <: Real) ? (@assert T2 <: Union{Float32, Float64}; T2) :
    (T1)

arbify(x) = x
arbify(x::AbstractFloat) = error("Expected typeof(x) = $(typeof(x)) <: Union{Float32, Float64}")
arbify(x::Union{Float32, Float64}) = ArbFloat(x)::ArbFloat
arbify(f::Function) = function f_arbified(args...)
    T = common_float_type(args)
    xs = arbify.(args)
    y = f(xs...)
    return convert.(T, y)
end

∇Zyg(f, args::Real...) = @inferred Zygote.gradient(f, args...)
∇Fwd(f, args::Real...) = @inferred Tuple(ForwardDiff.gradient(Base.splat(f), SVector(args)))

#### ArbFloat extensions

SpecialFunctions.erfinv(x::ArbFloat) = ArbFloat(erfinv(big(x)))
SpecialFunctions.erfcinv(x::ArbFloat) = ArbFloat(erfcinv(big(x)))

FastRicianLikelihoods.besseli2(x::ArbFloat) = ArbNumerics.besseli(2, x)
FastRicianLikelihoods.besseli2x(x::ArbFloat) = exp(-abs(x)) * ArbNumerics.besseli(2, x)
FastRicianLikelihoods.logbesseli0(x::ArbFloat) = log(ArbNumerics.besseli(0, x))
FastRicianLikelihoods.logbesseli0x(x::ArbFloat) = log(ArbNumerics.besseli(0, x)) - abs(x)
FastRicianLikelihoods.logbesseli1(x::ArbFloat) = log(ArbNumerics.besseli(1, x))
FastRicianLikelihoods.logbesseli1x(x::ArbFloat) = log(ArbNumerics.besseli(1, x)) - abs(x)
FastRicianLikelihoods.logbesseli2(x::ArbFloat) = log(ArbNumerics.besseli(2, x))
FastRicianLikelihoods.logbesseli2x(x::ArbFloat) = log(ArbNumerics.besseli(2, x)) - abs(x)
FastRicianLikelihoods.laguerre½(x::ArbFloat) = exp(x / 2) * ((1 - x) * ArbNumerics.besseli(0, -x/2) - x * ArbNumerics.besseli(1, -x/2))
FastRicianLikelihoods.besseli1i0(x::ArbFloat) = ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x)
FastRicianLikelihoods.mean_rician(ν::ArbFloat, σ::ArbFloat) = σ * √(ArbFloat(π) / 2) * FastRicianLikelihoods.laguerre½(-(ν / σ)^2 / 2)
FastRicianLikelihoods.std_rician(ν::ArbFloat, σ::ArbFloat) = sqrt(ν^2 + 2σ^2 - ArbFloat(π) * σ^2 * FastRicianLikelihoods.laguerre½(-(ν / σ)^2 / 2)^2 / 2)
# FastRicianLikelihoods.∂x_laguerre½(x::ArbFloat)
# FastRicianLikelihoods.∂x_besseli0x(x::ArbFloat)
# FastRicianLikelihoods.∂x_besseli1x(x::ArbFloat)

function FastRicianLikelihoods.neglogpdf_rician(x::ArbFloat, ν::ArbFloat)
    x <= 0 && return ArbFloat(Inf)
    return (x^2 + ν^2) / 2 - log(x) - log(ArbNumerics.besseli(0, x * ν))
end

function FastRicianLikelihoods.∇neglogpdf_rician(x::ArbFloat, ν::ArbFloat)
    x <= 0 && return (ArbFloat(Inf), ν)
    I0, I1 = ArbNumerics.besseli(0, x * ν), ArbNumerics.besseli(1, x * ν)
    ∂x = x - ν * (I1 / I0) - 1/x
    ∂ν = ν - x * (I1 / I0)
    return (∂x, ∂ν)
end

function ∇²neglogpdf_rician(x::ArbFloat, ν::ArbFloat)
    I0, I1, I2 = ArbNumerics.besseli(0, x * ν), ArbNumerics.besseli(1, x * ν), ArbNumerics.besseli(2, x * ν)
    ∂²x = 1 + ν^2 * ((I1 / I0)^2 - I2 / 2I0) - ν^2/2 + 1/x^2
    ∂²ν = 1 + x^2 * ((I1 / I0)^2 - I2 / 2I0) - ν^2/2
    return (∂²x, ∂²ν)
end
∇²neglogpdf_rician(x, ν) = oftype.(promote(float(x), float(ν))[1], ∇²neglogpdf_rician(ArbFloat(x), ArbFloat(ν)))

function FastRicianLikelihoods.mode_rician(ν::ArbFloat; tol = √eps(one(ArbFloat)), method = :newton, kwargs...)
    ν <= 0 && return one(ν)
    ν >= 1e32 && return ν + (1 - 3 / 4ν^2) / 2ν # relative error < 1e-180
    f(x) = FastRicianLikelihoods.∇neglogpdf_rician(x, ν)[1]
    ∇f(x) = ∇²neglogpdf_rician(x, ν)[1]
    if method === :binary
        a, b = (ν < 2 ? one(ν) : ν - 1 / ν), (ν < 1 ? 1 + ν : ν + 1 / ν) # empirical range
        return binary_search_root_find(f, a, b; tol, kwargs...)[1]
    else # :newton
        x₀ = ν < 1.2 ? sqrt(1 + ν^2 / 2) : ν + (1 - 3 / 4ν^2) / 2ν # empirical initial guess
        return newton_root_find(f, ∇f, x₀; ftol = 0, xtol = tol, dtol = 0, kwargs...)[1]
    end
end

function FastRicianLikelihoods.var_mode_rician(ν::ArbFloat; kwargs...)
    ν <= 0 && return one(ν) / 2
    ν >= 1e30 && return 1 - 1 / 2ν^2 # relative error < 1e-120
    μ = FastRicianLikelihoods.mode_rician(ν; kwargs...)
    ∂²x = ∇²neglogpdf_rician(μ, ν)[1]
    return 1/∂²x
end

const ARBFLOAT_QUADGK_RTOL_DIGITS = 30
const ARBFLOAT_QUADGK_EPS = exp10(-ArbFloat(ARBFLOAT_QUADGK_RTOL_DIGITS))

neglogpdf_qrician_arbfloat_digits() = ARBFLOAT_QUADGK_RTOL_DIGITS
neglogpdf_qrician_arbfloat_eps() = ARBFLOAT_QUADGK_EPS

const NEGLOGPDF_QRICIAN_CACHE = ThreadSafeDict{Tuple{ArbFloat, ArbFloat, ArbFloat, Int}, ArbFloat}()

@memoize ()->NEGLOGPDF_QRICIAN_CACHE function FastRicianLikelihoods.neglogpdf_qrician(x::ArbFloat, ν::ArbFloat, δ::ArbFloat, order::Int)
    rtol, atol = neglogpdf_qrician_arbfloat_eps(), 0
    μ = FastRicianLikelihoods.mode_rician(ν)

    function integrate(Ω, knots...)
        # ArbFloat returns NaN for 1/0.0 whereas BigFloat returns +Inf, and quadgk relies on the latter internally for handling (semi-)infinite intervals
        I, _ = quadgk(BigFloat.(knots)...; rtol, atol, order) do x̃
            return BigFloat(exp(Ω - neglogpdf_rician(ArbFloat(x̃), ν)))
        end
        return ArbFloat(I)
    end

    if x < μ < x + δ && δ > 3
        Ω2 = neglogpdf_rician(x + δ, ν) # minimum on [x + δ, Inf)
        I2 = integrate(Ω2, x + δ, Inf)
        if x > 0
            Ω1 = neglogpdf_rician(x, ν) # minimum on [0, x]
            I1 = integrate(Ω1, 0, x)
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
        I = integrate(Ω, x, x + δ)
        out = Ω - log(I)
    end

    return out
end
FastRicianLikelihoods.neglogpdf_qrician(x::ArbFloat, ν::ArbFloat, δ::ArbFloat, ::Val{order} = Val(15)) where {order} = neglogpdf_qrician(x, ν, δ, order)

function FastRicianLikelihoods.∇neglogpdf_qrician(x::ArbFloat, ν::ArbFloat, δ::ArbFloat, order::Int)
    ϵ = sqrt(neglogpdf_qrician_arbfloat_eps())
    ∂x = (neglogpdf_qrician(x + ϵ, ν, δ, order) - neglogpdf_qrician(x - ϵ, ν, δ, order)) / 2ϵ
    ∂ν = (neglogpdf_qrician(x, ν + ϵ, δ, order) - neglogpdf_qrician(x, ν - ϵ, δ, order)) / 2ϵ
    ∂δ = (neglogpdf_qrician(x, ν, δ + ϵ, order) - neglogpdf_qrician(x, ν, δ - ϵ, order)) / 2ϵ
    return (∂x, ∂ν, ∂δ)
end
FastRicianLikelihoods.∇neglogpdf_qrician(x::ArbFloat, ν::ArbFloat, δ::ArbFloat, ::Val{order} = Val(15)) where {order} = ∇neglogpdf_qrician(x, ν, δ, order)

function FastRicianLikelihoods.neglogpdf_rician(x::ArbFloat, ν::ArbFloat, logσ::ArbFloat)
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end

function FastRicianLikelihoods.neglogpdf_qrician(x::ArbFloat, ν::ArbFloat, logσ::ArbFloat, δ::ArbFloat, order::Val)
    σ⁻¹ = exp(-logσ)
    return neglogpdf_qrician(σ⁻¹ * x, σ⁻¹ * ν, σ⁻¹ * δ, order)
end

#### Helpers

function binary_search_root_find(f, a::T, b::T; tol = eps(T), verbose = false) where {T}
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

function newton_root_find(f, ∇f, x₀::T; ftol = √(eps(T)), xtol = √(eps(T)), dtol = eps(T), verbose = false) where {T <: AbstractFloat}
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
