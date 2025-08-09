using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Revise.includet(joinpath(@__DIR__, "../utils.jl"))

using FastRicianLikelihoods
using .Utils: Utils, arbify

using ArbNumerics
using CaratheodoryFejerApprox
using DoubleFloats
using Richardson

const CF = CaratheodoryFejerApprox
const F = FastRicianLikelihoods
const U = Utils

setworkingprecision(ArbFloat, 500)
setextrabits(128)

"""
Wrap f that is written over ArbReal to be callable with Double64 inputs and return Double64 outputs.
Converts real inputs via BigFloat -> ArbReal for accurate special function evaluation, then converts outputs back.
"""
doublify(f::Function) = function f_doubled(args...)
    xs = map(x -> convert(ArbReal, BigFloat(x)), args)
    y = f(xs...)
    return y isa Number ? Double64(y) : map(Double64, y)
end

"""
Run Caratheodory–Fejér minimax over a Double64 domain while evaluating f in ArbReal.
"""
function cf_minimax(f::Function, dom::Tuple{<:Real, <:Real}, m::Integer, n::Integer)
    return minimax(doublify(f), (Double64(dom[1]), Double64(dom[2])), m, n)
end

"""
    find_puiseux_series(f, h=one(ArbReal); alpha=1, order=4, verbose=false)

Computes a generalized power series expansion for `f(x)` of the form

    f(x) ~ pₙ(x) = qₙ(x^α) where qₙ(t) = ∑ᵢ₌₀ⁿ cᵢ tⁱ.

The expansion is about `x₀ = 0` if `α > 0` (e.g. Taylor series in `x` for `α=1`),
and about `x₀ = ∞` if `α < 0` (e.g. asymptotic series in `1/x` for `α=-1`).

The coefficients are found recursively using Richardson extrapolation:

    c₀ = lim_{x→x₀} f(x)
    cₙ = lim_{x→x₀} x^{-n * α} * (f(x) - qₙ₋₁(x^α))
"""
function find_puiseux_series(f, h::T = one(ArbReal); alpha::Number = 1, order::Int = 4, verbose::Bool = false) where {T}
    coeffs = Rational[]
    for i in 0:order
        c, c_err = Richardson.extrapolate(h; x0 = alpha > 0 ? zero(T) : T(Inf), power = abs(alpha)) do x
            return i == 0 ? f(x) : x^(-i * alpha) * (f(x) - evalpoly(x^alpha, coeffs))
        end
        c_relerr = abs(c_err / c)
        verbose && @info "order = $i, c = $(Float64(c)) ± $(Float64(c_err)), relerr = $(Float64(c_relerr))"
        push!(coeffs, rationalize(Int, c))
    end
    verbose && @info "coeffs = $coeffs"
    return coeffs
end

"""
Pretty print CaratheodoryFejerApprox result and generate coefficient function definitions.
Displays the result and prints function definitions for coefficients.
"""
function cf_print_result(res, ::Type{T}; name::String, verbose::Bool = false) where {T}
    # Display the CaratheodoryFejerApprox result (has built-in pretty printing)
    verbose && display(res)

    # Extract coefficients
    p, q = monocoeffs(res; transplant = true)

    # Generate function definitions based on whether it's polynomial or rational
    if length(q) == 1
        # Polynomial case - single function
        println("@inline $(name)_coefs(::Type{$T}) = $(T.((p...,)))")
    else
        # Rational case - separate numerator and denominator functions
        println("@inline $(name)_num_coefs(::Type{$T}) = $(T.((p...,)))")
        println("@inline $(name)_den_coefs(::Type{$T}) = $(T.((q...,)))")
    end

    return p, q
end

"""
Small x ∈ [0, upper] <=> y ∈ [0, upper^2] where x = √y, y = x^2:
f(x) = log(besseli(0, x)) = x^2 * (1/4 + x^2 * P(x^2))
<=> P(y) = (log(besseli(0, x)) - y / 4) / y^2
"""
function logbesseli0_taylor_constants(::Type{T}, m::Int, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = y == 0 ? convert(typeof(y), 1 // 64) : (x = sqrt(y); (log(ArbNumerics.besseli(0, x)) - y / 4) / y^2)
    res = cf_minimax(P, (0.0, upper^2), m, 0)
    cf_print_result(res, T; name = "logbesseli0_taylor", verbose)
    return res
end

function logbesseli0_taylor_constants(; verbose::Bool = false)
    logbesseli0_taylor_constants(Float32, 3, 1.0f0; verbose)
    logbesseli0_taylor_constants(Float64, 8, 1.0; verbose)
    return nothing
end

"""
Middle x ∈ [lower, upper]:
f(x) = log(I0x(x)) = x * P(x)
<=> P(x) = log(I0x(x)) / x = (log(I0(x)) - x) / x
"""
function logbesseli0x_middle_constants(::Type{T}, m::Int, n::Int, lower::Real, upper::Real; branchnum::Int, verbose::Bool = false) where {T}
    P(x::ArbReal) = (log(ArbNumerics.besseli(0, x)) - x) / x
    res = cf_minimax(P, (lower, upper), m, n)
    cf_print_result(res, T; name = "logbesseli0x_branch$branchnum", verbose)
    return res
end

"""
Asymptotic x ∈ [0, 1/upper] <=> y ∈ [0, 1/upper] where x = 1/y, y = 1/x:
f(x) = log(I0x(x)) = P(1/x) / x - log(x) / 2 - log(2π) / 2
<=> P(y) = x * (log(I0x(x)) + log(x) / 2 + log(2π) / 2)
"""
function logbesseli0x_asymptotic_constants(::Type{T}, m::Int, upper::Real; verbose::Bool = false) where {T}
    function P(y::ArbReal)
        y == 0 && return zero(y)
        x = inv(y)
        return (log(ArbNumerics.besseli(0, x)) - x + log(x) / 2 + log(2 * ArbReal(π)) / 2) / y
    end
    res = cf_minimax(P, (0.0, inv(upper)), m, 0)
    cf_print_result(res, T; name = "logbesseli0x_tail", verbose)
    return res
end

function logbesseli0x_all_constants(; verbose::Bool = false)
    branches32 = (1.0f0, 2.0f0, 3.0f0, 4.5f0, 6.25f0)
    branches64 = (1.0, 2.0, 3.25, 5.0, 9.0)

    # Float32 approximants
    logbesseli0_taylor_constants(Float32, 2, branches32[1]; verbose)
    for i in 1:length(branches32)-1
        logbesseli0x_middle_constants(Float32, 5, 0, branches32[i], branches32[i+1]; branchnum = i, verbose)
    end
    logbesseli0x_asymptotic_constants(Float32, 5, branches32[end]; verbose)

    # Float64 approximants
    logbesseli0_taylor_constants(Float64, 8, branches64[1]; verbose)
    for i in 1:length(branches64)-1
        logbesseli0x_middle_constants(Float64, 6, 6, branches64[i], branches64[i+1]; branchnum = i, verbose)
    end
    logbesseli0x_asymptotic_constants(Float64, 18, branches64[end]; verbose)

    return nothing
end

"""
Small x ∈ [0, upper] <=> y ∈ [0, upper^2 / 4] where x = 2√y, y = x^2/4:
f(x) = I0(x) = 1 + y * P(y)
<=> P(y) = (I0(x) - 1) / y
"""
function besseli0x_small_constants(::Type{T}, m::Int, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = y == 0 ? one(y) : (x = 2 * sqrt(y); (ArbNumerics.besseli(0, x) - 1) / y)
    res = cf_minimax(P, (0.0, upper^2 / 4), m, 0)
    cf_print_result(res, T; name = "besseli0x_small", verbose)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = I0x(x) = P(1/x) / √x
<=> P(y) = √x * I0x(x)
"""
function besseli0x_large_constants(::Type{T}, m::Int, lower::Real, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = (x = inv(y); exp(-x) * sqrt(x) * ArbNumerics.besseli(0, x))
    res = cf_minimax(P, (inv(upper), inv(lower)), m, 0)
    cf_print_result(res, T; name = "besseli0x_large", verbose)
    return res
end

function besseli0x_all_constants(; verbose::Bool = false)
    besseli0x_small_constants(Float32, 8, 7.75; verbose)
    besseli0x_large_constants(Float32, 4, 7.75, 50.0; verbose)
    besseli0x_large_constants(Float32, 2, 50.0, Inf; verbose)

    besseli0x_small_constants(Float64, 13, 7.75; verbose)
    besseli0x_large_constants(Float64, 21, 7.75, 500.0; verbose)
    besseli0x_large_constants(Float64, 4, 500.0, Inf; verbose)
    return nothing
end

"""
Small x ∈ [0, upper] <=> y ∈ [0, upper^2 / 4] where x = 2√y, y = x^2/4:
f(x) = 2 * I1(x) / x = 1 + y / 2 + y^2 * P(y)
<=> P(y) = ((2 * I1(x) / x - 1) / (y / 2) - 1) / 2y
"""
function besseli1x_small_constants(::Type{T}, m::Int, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = (x = 2 * sqrt(y); ((2 * ArbNumerics.besseli(1, x) / x - 1) / (y / 2) - 1) / 2y)
    res = cf_minimax(P, (0.0, upper^2 / 4), m, 0)
    cf_print_result(res, T; name = "besseli1x_small", verbose)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = I1x(x) / x = P(1/x) / √x
<=> P(y) = √x * I1x(x)
"""
function besseli1x_large_constants(::Type{T}, m::Int, lower::Real, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = (x = inv(y); exp(-x) * sqrt(x) * ArbNumerics.besseli(1, x))
    res = cf_minimax(P, (inv(upper), inv(lower)), m, 0)
    cf_print_result(res, T; name = "besseli1x_large", verbose)
    return res
end

function besseli1x_all_constants(; verbose::Bool = false)
    besseli1x_small_constants(Float32, 6, 7.75; verbose)
    besseli1x_large_constants(Float32, 4, 7.75, Inf; verbose)
    besseli1x_large_constants(Float32, 2, 50.0, Inf; verbose)

    besseli1x_small_constants(Float64, 12, 7.75; verbose)
    besseli1x_large_constants(Float64, 21, 7.75, Inf; verbose)
    besseli1x_large_constants(Float64, 4, 500.0, Inf; verbose)
    return nothing
end

"""
Small x ∈ [0, upper] <=> y ∈ [0, upper^2 / 4] where x = 2√y, y = x^2/4:
f(x) = I2(x) = y * P(y)
<=> P(y) = I2(x) / y
"""
function besseli2x_small_constants(::Type{T}, m::Int, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = (x = 2 * sqrt(y); ArbNumerics.besseli(2, x) / y)
    res = cf_minimax(P, (0.0, upper^2 / 4), m, 0)
    cf_print_result(res, T; name = "besseli2x_small", verbose)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = I2x(x) = P(1/x) / √x
<=> P(y) = √x * I2x(x)
"""
function besseli2x_large_constants(::Type{T}, m::Int, lower::Real, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = (x = inv(y); exp(-x) * sqrt(x) * ArbNumerics.besseli(2, x))
    res = cf_minimax(P, (inv(upper), inv(lower)), m, 0)
    cf_print_result(res, T; name = "besseli2x_large", verbose)
    return res
end

function besseli2x_all_constants(; verbose::Bool = false)
    besseli2x_small_constants(Float32, 8, 7.75; verbose)
    besseli2x_large_constants(Float32, 4, 7.75, Inf; verbose)
    besseli2x_large_constants(Float32, 2, 50.0, Inf; verbose)

    besseli2x_small_constants(Float64, 13, 7.75; verbose)
    besseli2x_large_constants(Float64, 20, 7.75, Inf; verbose)
    besseli2x_large_constants(Float64, 4, 500.0, Inf; verbose)
    return nothing
end

"""
Small x ∈ [0, upper] <=> y ∈ [0, upper^2] where x = √y, y = x^2:
f(x) = L^2 - t^2 - 1 = P(y)
<=> P(y) = L^2 - t^2 - 1
"""
function laguerre½_small_constants(::Type{T}, m::Int, low::Real; verbose::Bool = false) where {T}
    L½(x) = exp(x / 2) * ((1 - x) * ArbNumerics.besseli(0, -x / 2) - x * ArbNumerics.besseli(1, -x / 2))
    L½scaled(t) = sqrt(ArbReal(pi) / 2) * L½(-t^2 / 2)
    P(y::ArbReal) = (t = sqrt(y); L = L½scaled(t); ((L - t) * (L + t) - 1))
    res = cf_minimax(P, (0.0, low^2), m, 0)
    cf_print_result(res, T; name = "laguerre½²c_small", verbose)
    return res
end

"""
Large t ∈ [0, 1/upper] <=> y ∈ [0, 1/upper] where t = 1/√y, y = 1/t^2:
f(t) = L(t)^2 - t^2 - 1 = P(1/t)
<=> P(y) = L(t)^2 - t^2 - 1
"""
function laguerre½_large_constants(::Type{T}, m::Int, high::Real; verbose::Bool = false) where {T}
    L½(t) = exp(t / 2) * ((1 - t) * ArbNumerics.besseli(0, -t / 2) - t * ArbNumerics.besseli(1, -t / 2))
    L½scaled(t) = sqrt(ArbReal(pi) / 2) * L½(-t^2 / 2)
    P(y::ArbReal) = (t = 1 / sqrt(y); L = L½scaled(t); t^2 * ((L - t) * (L + t) - 1))
    res = cf_minimax(P, (0.0, 1 / high^2), m, 0)
    cf_print_result(res, T; name = "laguerre½²c_large", verbose)
    return res
end

function laguerre½_all_constants(; verbose::Bool = false)
    laguerre½_small_constants(Float32, 10, 3.4; verbose)
    laguerre½_large_constants(Float32, 10, 3.4; verbose)

    laguerre½_small_constants(Float64, 20, 4.3; verbose)
    laguerre½_large_constants(Float64, 20, 4.3; verbose)
    return nothing
end

"""
Small x ∈ [0, upper] <=> y ∈ [0, upper^2] where x = √y, y = x^2:
f(x) = I1(x) / I0(x) = x * P(y)
<=> P(y) = (I1(x) / I0(x)) / x
"""
function besseli1i0_low_constants(::Type{T}, m::Int, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = (x = sqrt(y); ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x) / x)
    res = cf_minimax(P, (0.0, upper^2), m, 0)
    cf_print_result(res, T; name = "besseli1i0_low", verbose)
    return res
end

"""
Middle x ∈ [lower, upper] <=> y ∈ [lower^2, upper^2] where x = √y, y = x^2:
f(x) = I1(x) / I0(x) = P(y) / x
<=> P(y) = x * (I1(x) / I0(x))
"""
function besseli1i0_middle_constants(::Type{T}, m::Int, n::Int, lower::Real, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = (x = sqrt(y); ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x) / x)
    res = cf_minimax(P, (lower^2, upper^2), m, n)
    cf_print_result(res, T; name = "besseli1i0_mid", verbose)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = I1(x) / I0(x) - 1 = (-1/2 - P(1/x) / x) / x
<=> P(y) = x * (-1/2 + x * (1 - I₁(x) / I₀(x)))
"""
function besseli1i0c_tail_constants(::Type{T}, m::Int, lower::Real, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = (x = inv(y); x * (2x * (1 - ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x)) - 1) / 2)
    res = cf_minimax(P, (inv(upper), inv(lower)), m, 0)
    cf_print_result(res, T; name = "besseli1i0c_tail", verbose)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = r(x)^2 - 1 + r(x) / x = -P(1/x) / x^2 where r(x) = I1(x) / I0(x)
<=> P(y) = -x^2 * (r(x)^2 - 1 + r(x) / x)
"""
function besseli1i0sqm1pi1i0x_tail_constants(::Type{T}, m::Int, lower::Real, upper::Real; verbose::Bool = false) where {T}
    P(y::ArbReal) = (x = inv(y); r = ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x); -x^2 * ((r - 1) * (r + 1) + r / x))
    res = cf_minimax(P, (inv(upper), inv(lower)), m, 0)
    cf_print_result(res, T; name = "besseli1i0sqm1pi1i0x_tail", verbose)
    return res
end

function besseli1i0_all_constants(; verbose::Bool = false)
    besseli1i0_low_constants(Float32, 3, 0.5; verbose)
    besseli1i0_middle_constants(Float32, 3, 4, 0.5, 7.75; verbose)
    besseli1i0_middle_constants(Float32, 3, 3, 7.75, 15.0; verbose)
    besseli1i0c_tail_constants(Float32, 4, 15.0, Inf; verbose)
    besseli1i0sqm1pi1i0x_tail_constants(Float32, 4, 15.0, Inf; verbose)

    besseli1i0_low_constants(Float64, 7, 0.5; verbose)
    besseli1i0_middle_constants(Float64, 7, 6, 0.5, 7.75; verbose)
    besseli1i0_middle_constants(Float64, 7, 6, 7.75, 15.0; verbose)
    besseli1i0c_tail_constants(Float64, 15, 15.0, Inf; verbose)
    besseli1i0sqm1pi1i0x_tail_constants(Float64, 15, 15.0, Inf; verbose)
    return nothing
end
