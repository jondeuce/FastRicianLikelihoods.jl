module Utils

using Test

using ArbNumerics: ArbNumerics, ArbReal
using FiniteDifferences: FiniteDifferences
using QuadGK: quadgk
using Zygote: Zygote

using FastRicianLikelihoods: FastRicianLikelihoods, ForwardDiff, SpecialFunctions, StaticArrays
using FastRicianLikelihoods: neglogpdf_rician, ∇neglogpdf_rician, neglogpdf_qrician, ∇neglogpdf_qrician
using .StaticArrays: SVector

Base.setprecision(BigFloat, 500)
ArbNumerics.setworkingprecision(ArbReal; digits = 500, base = 2)
ArbNumerics.setextrabits(128)

@inline common_float_type(args::Tuple) = mapfoldl(typeof, common_float_type, args)

@inline common_float_type(::Type{T1}, ::Type{T2}) where {T1, T2} =
    (T1 <: Real && T2 <: Real) ? (@assert T1 === T2; @assert T1 <: Union{Float32, Float64}; T1) :
    (T1 <: Real) ? (@assert T1 <: Union{Float32, Float64}; T1) :
    (T2 <: Real) ? (@assert T2 <: Union{Float32, Float64}; T2) :
    (T1)

arbify(x) = x
arbify(x::Real) = error("Expected typeof(x) = $(typeof(x)) <: Union{Float32, Float64}")
arbify(x::Union{Float32, Float64}) = ArbReal(x)::ArbReal
arbify(f::Function) = function f_arbified(args...)
    T = common_float_type(args)
    xs = arbify.(args)
    y = f(xs...)
    return convert.(T, y)
end

const DEFAULT_CENTRAL_FDM = FiniteDifferences.central_fdm(4, 1)
const DEFAULT_FORWARD_FDM = FiniteDifferences.forward_fdm(4, 1)
∇FD_central(f, args::Real...) = ∇FD(DEFAULT_CENTRAL_FDM, f, args...)
∇FD_forward(f, args::Real...) = ∇FD(DEFAULT_FORWARD_FDM, f, args...)
∇FD(fdm, f, args::Real...) = (FiniteDifferences.grad(fdm, Base.splat(f), [args...])[1]...,)

∇Fwd(f, args::Real...) = Tuple(ForwardDiff.gradient(Base.splat(f), SVector(args)))
∇Zyg(f, args::Real...) = Zygote.gradient(f, args...)

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
    x <= 0 && return ArbReal(Inf)
    return (x^2 + ν^2) / 2 - log(x) - log(ArbNumerics.besseli(0, x * ν))
end

function FastRicianLikelihoods.∇neglogpdf_rician(x::ArbReal, ν::ArbReal)
    x <= 0 && return (ArbReal(Inf), ν)
    I0, I1 = ArbNumerics.besseli(0, x * ν), ArbNumerics.besseli(1, x * ν)
    ∂x = x - ν * (I1 / I0) - 1 / x
    ∂ν = ν - x * (I1 / I0)
    return (∂x, ∂ν)
end

function FastRicianLikelihoods.∇²neglogpdf_rician(x::ArbReal, ν::ArbReal)
    I0, I1 = ArbNumerics.besseli(0, x * ν), ArbNumerics.besseli(1, x * ν)
    ∂²x = 1 - ν^2 + 1 / x^2 + (ν / x) * (I1 / I0) * (1 + ν * x * (I1 / I0))
    ∂²ν = 1 - x^2 + (x / ν) * (I1 / I0) * (1 + ν * x * (I1 / I0))
    ∂x∂ν = x * ν * ((I1 / I0)^2 - 1)
    return (∂²x, ∂x∂ν, ∂²ν)
end
FastRicianLikelihoods.∇²neglogpdf_rician(x, ν) = oftype.(promote(float(x), float(ν))[1], FastRicianLikelihoods.∇²neglogpdf_rician(ArbReal(x), ArbReal(ν)))

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

neglogpdf_qrician_arbreal_eps() = ArbReal(1e-30)

function FastRicianLikelihoods.neglogpdf_qrician(x::ArbReal, ν::ArbReal, δ::ArbReal, order::Int = 21)
    rtol, atol = neglogpdf_qrician_arbreal_eps(), ArbReal(0)
    μ = FastRicianLikelihoods.mode_rician(ν)
    σ = √FastRicianLikelihoods.var_mode_rician(ν)
    Δσ = 3 * σ

    if x < μ < x + δ && μ - x > Δσ && x + δ - μ > Δσ
        Ω2 = neglogpdf_rician(x + δ, ν) # minimum on [x + δ, Inf)
        I2 = qrician_integrate(Ω2, x + δ, ArbReal(Inf), ν; rtol, atol, order)
        if x > 0
            Ω1 = neglogpdf_rician(x, ν) # minimum on [0, x]
            I1 = qrician_integrate(Ω1, ArbReal(0), x, ν; rtol, atol, order)
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
        I = qrician_integrate(Ω, x, x + δ, ν; rtol, atol, order)
        out = Ω - log(I)
    end

    return out
end
FastRicianLikelihoods.neglogpdf_qrician(x::ArbReal, ν::ArbReal, δ::ArbReal, ::Val{order}) where {order} = neglogpdf_qrician(x, ν, δ, order)

function qrician_integrate(f::Function, Ω::ArbReal, a::ArbReal, b::ArbReal, ν::ArbReal; rtol::ArbReal, atol::ArbReal, order::Int)
    if isfinite(b)
        I, E = quadgk(a, b; rtol, atol, order) do x̃
            return exp(Ω - neglogpdf_rician(x̃, ν)) * f(x̃)
        end
    else
        # Change of variables: x = a + (1 - t) / t where t ∈ (0, 1)
        #   #TODO: quadgk should do this internally, but `ArbReal` arguments cause it to fail?
        @assert isfinite(a)
        I, E = quadgk(ArbReal(0), ArbReal(1); rtol, atol, order) do t̃
            x̃ = a + (1 - t̃) / t̃
            return exp(Ω - neglogpdf_rician(x̃, ν)) * f(x̃) / t̃^2
        end
    end
    return I
end
qrician_integrate(Ω::ArbReal, args...; kwargs...) = qrician_integrate(Returns(ArbReal(1)), Ω, args...; kwargs...)

function FastRicianLikelihoods.∇neglogpdf_qrician(x::ArbReal, ν::ArbReal, δ::ArbReal, order::Int = 21, method = :analytic)
    if method === :analytic
        rtol, atol = neglogpdf_qrician_arbreal_eps(), ArbReal(0)
        Ω = neglogpdf_qrician(x, ν, δ, order)
        ∂ν = qrician_integrate(x̃ -> ∇neglogpdf_rician(x̃, ν)[2], Ω, x, x + δ, ν; rtol, atol, order)
        ∂δ = -exp(Ω - neglogpdf_rician(x + δ, ν))
        ∂x = ∂δ + exp(Ω - neglogpdf_rician(x, ν))
    else # method === :numeric
        ϵ = sqrt(neglogpdf_qrician_arbreal_eps())
        ∂x = (neglogpdf_qrician(x + ϵ, ν, δ, order) - neglogpdf_qrician(x - ϵ, ν, δ, order)) / 2ϵ
        ∂ν = (neglogpdf_qrician(x, ν + ϵ, δ, order) - neglogpdf_qrician(x, ν - ϵ, δ, order)) / 2ϵ
        ∂δ = (neglogpdf_qrician(x, ν, δ + ϵ, order) - neglogpdf_qrician(x, ν, δ - ϵ, order)) / 2ϵ
    end
    return (∂x, ∂ν, ∂δ)
end
FastRicianLikelihoods.∇neglogpdf_qrician(x::ArbReal, ν::ArbReal, δ::ArbReal, ::Val{order}) where {order} = ∇neglogpdf_qrician(x, ν, δ, order)

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
