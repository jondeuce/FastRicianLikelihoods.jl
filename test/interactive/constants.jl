using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Revise.includet(joinpath(@__DIR__, "../utils.jl"))
Revise.includet(joinpath(@__DIR__, "autotune.jl"))

using FastRicianLikelihoods
using .Utils: Utils, arbify

using ArbNumerics
using CaratheodoryFejerApprox
using DoubleFloats
using Richardson

const CF = CaratheodoryFejerApprox
const F = FastRicianLikelihoods
const U = Utils

setworkingprecision(ArbReal; bits = 500)

"""
Wrap f that is written over ArbReal to be callable with Double64 inputs and return Double64 outputs.
Converts real inputs via BigFloat -> ArbReal for accurate special function evaluation, then converts outputs back.
"""
doublify(f::Function) = function f_doubled(args...)
    xs = map(x -> convert(ArbReal, BigFloat(x)), args)
    return doublify(f(xs...))
end
doublify(x::Number) = Double64(x)
doublify(x::Tuple) = map(doublify, x)

"""
Run Caratheodory-Fejér minimax over a Double64 domain while evaluating f in ArbReal.
"""
function cf_minimax(f::Function, dom::Tuple{<:Real, <:Real}, m::Integer, n::Integer)
    return minimax(doublify(f), (Double64(dom[1]), Double64(dom[2])), m, n)
end

"""
    puiseux_series(f, h=one(ArbReal); alpha=1, order=4, verbose=false)

Computes a generalized power series expansion for `f(x)` of the form

    f(x) ~ pₙ(x) = qₙ(x^α) where qₙ(t) = ∑ᵢ₌₀ⁿ cᵢ tⁱ.

The expansion is about `x₀ = 0` if `α > 0` (e.g. Taylor series in `x` for `α=1`),
and about `x₀ = ∞` if `α < 0` (e.g. asymptotic series in `1/x` for `α=-1`).

The coefficients are found recursively using Richardson extrapolation:

    c₀ = lim_{x→x₀} f(x)
    cₙ = lim_{x→x₀} x^{-n * α} * (f(x) - qₙ₋₁(x^α))
"""
function puiseux_series(f, h::T = one(ArbReal); alpha::Number = 1, order::Int = 4, verbose::Bool = false) where {T}
    coeffs = Rational[]
    for i in 0:order
        c, c_err = Richardson.extrapolate(h; x0 = alpha > 0 ? zero(T) : T(Inf), power = abs(alpha)) do x
            return i == 0 ? f(x) : (f(x) - evalpoly(x^alpha, coeffs)) / x^(i * alpha)
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
function cf_print_result(res; name::String, T::Type, transplant::Bool = true)
    # Extract coefficients
    p, q = monocoeffs(res; transplant)

    if !transplant
        # Coefficients are in transplant basis for the transformed variable t = scale * x + offset ∈ [-1, 1]
        dom = res.dom
        μ, σ = (dom[1] + dom[2]) / 2, (dom[2] - dom[1]) / 2
        println("@inline $(name)_transplant(::Type{$T}) = $(T.((inv(σ), -μ / σ)))")
    end

    # Generate function definitions based on whether it's polynomial or rational
    if length(q) == 1
        # Polynomial case - single function
        println("@inline $(name)_coefs(::Type{$T}) = $(T.((p...,)))")
    else
        # Rational case - separate numerator and denominator functions
        println("@inline $(name)_num_coefs(::Type{$T}) = $(T.((p...,)))")
        println("@inline $(name)_den_coefs(::Type{$T}) = $(T.((q...,)))")
    end

    return nothing
end

"""
Small x ∈ [0, upper] <=> y ∈ [0, upper^2] where x = √y, y = x^2:
f(x) = log(besseli(0, x)) = x^2 * (1/4 + x^2 * P(x^2))
<=> P(y) = (log(besseli(0, x)) - y / 4) / y^2
"""
function logbesseli0_taylor_constants(m::Int, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = y == 0 ? oftype(y, 1 // 64) : (x = sqrt(y); (log(ArbNumerics.besseli(0, x)) - y / 4) / y^2)
    res = cf_minimax(P, (0.0, upper^2), m, 0)
    verbose && cf_print_result(res; name = "logbesseli0_taylor", kwargs...)
    return res
end

function logbesseli0_taylor_constants(; verbose::Bool = false, kwargs...)
    logbesseli0_taylor_constants(Float32, 3, 1.0f0; verbose, kwargs...)
    logbesseli0_taylor_constants(Float64, 8, 1.0; verbose, kwargs...)
    return nothing
end

"""
Middle x ∈ [lower, upper]:
f(x) = log(I0x(x)) = x * P(x)
<=> P(x) = log(I0x(x)) / x = (log(I0(x)) - x) / x
"""
function logbesseli0x_middle_constants(m::Int, n::Int, lower::Real, upper::Real; branchnum::Int, verbose::Bool = false, kwargs...)
    P(x::ArbReal) = x == 0 ? oftype(y, -1) : (log(ArbNumerics.besseli(0, x)) - x) / x
    res = cf_minimax(P, (lower, upper), m, n)
    verbose && cf_print_result(res; name = "logbesseli0x_branch$branchnum", kwargs...)
    return res
end

"""
Log-scaled modified Bessel function of the first kind:

    Î₀(z) = I₀(z) / (eᶻ / √(2πz))
          = e⁻ᶻ * √(2πz) * I₀(z)
          ~ log(2πz) / 2 - z + z^2/4 - z^4/64 + 𝒪(z^6)  (z << 1)
          ~ 1/8z + 1/16z^2 + 25/384z^3 + 𝒪(z^5)         (z >> 1)
"""
function logbesseli0x_scaled(z)
    I0 = ArbNumerics.besseli(0, z)
    return log(exp(-z) * I0 * sqrt(2 * oftype(z, π) * z))
end

"""
Asymptotic x ∈ [0, 1/upper] <=> y ∈ [0, 1/upper] where x = 1/y, y = 1/x:
f(x) = log(I0x(x)) = P(1/x) / x - log(x) / 2 - log(2π) / 2
<=> P(y) = x * (log(I0x(x)) + log(x) / 2 + log(2π) / 2)
"""
function logbesseli0x_asymptotic_constants(m::Int, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = y == 0 ? oftype(y, 1 // 8) : (z = inv(y); logbesseli0x_scaled(z) / y)
    res = cf_minimax(P, (0.0, 1 / upper), m, 0)
    verbose && cf_print_result(res; name = "logbesseli0x_tail", kwargs...)
    return res
end

function logbesseli0x_manual_constants(; verbose::Bool = false)
    # branches32 = (1.0f0, 2.6f0, 4.2f0, 6.25f0)
    # branches64 = (1.0, 4.0, 8.5, 17.5)
    branches32 = (1.0f0, 2.0f0, 3.0f0, 4.5f0, 6.25f0)
    branches64 = (1.0, 2.15, 3.9, 7.0, 12.0)

    # Float32 approximants
    println("@inline logbesseli0x_branches(::Type{Float32}) = $(Float32.((branches32...,)))")
    logbesseli0_taylor_constants(3, branches32[1]; verbose, T = Float32)
    for i in 1:length(branches32)-1
        logbesseli0x_middle_constants(5, 0, branches32[i], branches32[i+1]; branchnum = i, verbose, T = Float32)
    end
    logbesseli0x_asymptotic_constants(5, branches32[end]; verbose, T = Float32)

    # Float64 approximants
    println("@inline logbesseli0x_branches(::Type{Float64}) = $(Float64.((branches64...,)))")
    logbesseli0_taylor_constants(9, branches64[1]; verbose, T = Float64)
    for i in 1:length(branches64)-1
        logbesseli0x_middle_constants(6, 6, branches64[i], branches64[i+1]; branchnum = i, verbose, T = Float64)
    end
    logbesseli0x_asymptotic_constants(18, branches64[end]; verbose, T = Float64)

    return nothing
end

function logbesseli0x_autotune_constants(; verbose::Bool = false)

    function build_ratapprox((a, b), order, i, n)
        if i == 1
            logbesseli0_taylor_constants(order, b)
        elseif i < n
            logbesseli0x_middle_constants(order, 0, a, b; branchnum = i - 1)
        else # i == n
            logbesseli0x_asymptotic_constants(order, a)
        end
    end

    function print_autotune_results(res; T::Type)
        verbose && display.(res.approximants)
        println("@inline logbesseli0x_branches(::Type{$T}) = $(T.((res.branches...,)))")
        cf_print_result(res.approximants[1]; name = "logbesseli0_taylor", T, transplant = true)
        for i in 2:length(res.approximants)-1
            cf_print_result(res.approximants[i]; name = "logbesseli0x_mid$(i-1)", T, transplant = true)
        end
        cf_print_result(res.approximants[end]; name = "logbesseli0x_tail", T, transplant = true)
    end

    res32 = autotune_minimax_spline(
        build_ratapprox;
        initial_order = 5,
        initial_branches = Double64.([1.0f0, 2.0f0, 3.0f0, 5.0f0, 7.0f0]),
        strategy = :subdivide,
        max_order = 5,
        fixed_left_branch = true,
        verbose = verbose,
        minimax_abs_tol = Double64(eps(Float32)) / 2,
    )
    print_autotune_results(res32; T = Float32)

    res64 = autotune_minimax_spline(
        build_ratapprox;
        initial_order = res32.order,
        initial_branches = Double64.(res32.branches),
        strategy = :increase_order,
        max_order = 18,
        fixed_left_branch = true,
        max_branches = length(res32.branches),
        verbose = verbose,
        minimax_abs_tol = Double64(eps(Float64)) / 4,
    )
    print_autotune_results(res64; T = Float64)

    return nothing
end

"""
Small x ∈ [0, upper] <=> y ∈ [0, upper^2 / 4] where x = 2√y, y = x^2/4:
f(x) = I0(x) = 1 + y * P(y)
<=> P(y) = (I0(x) - 1) / y
"""
function besseli0x_small_constants(m::Int, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = y == 0 ? one(y) : (x = 2 * sqrt(y); (ArbNumerics.besseli(0, x) - 1) / y)
    res = cf_minimax(P, (0.0, upper^2 / 4), m, 0)
    verbose && cf_print_result(res; name = "besseli0x_small", kwargs...)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = I0x(x) = P(1/x) / √x
<=> P(y) = √x * I0x(x)
"""
function besseli0x_large_constants(m::Int, lower::Real, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = (x = inv(y); exp(-x) * sqrt(x) * ArbNumerics.besseli(0, x))
    res = cf_minimax(P, (1 / upper, 1 / lower), m, 0)
    verbose && cf_print_result(res; name = "besseli0x_large", kwargs...)
    return res
end

function besseli0x_manual_constants(; verbose::Bool = false)
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
function besseli1x_small_constants(m::Int, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = (x = 2 * sqrt(y); ((2 * ArbNumerics.besseli(1, x) / x - 1) / (y / 2) - 1) / 2y)
    res = cf_minimax(P, (0.0, upper^2 / 4), m, 0)
    verbose && cf_print_result(res; name = "besseli1x_small", kwargs...)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = I1x(x) / x = P(1/x) / √x
<=> P(y) = √x * I1x(x)
"""
function besseli1x_large_constants(m::Int, lower::Real, upper::Real; verbose::Bool = false)
    P(y::ArbReal) = (x = inv(y); exp(-x) * sqrt(x) * ArbNumerics.besseli(1, x))
    res = cf_minimax(P, (1 / upper, 1 / lower), m, 0)
    verbose && cf_print_result(res; name = "besseli1x_large", kwargs...)
    return res
end

function besseli1x_manual_constants(; verbose::Bool = false)
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
function besseli2x_small_constants(m::Int, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = (x = 2 * sqrt(y); ArbNumerics.besseli(2, x) / y)
    res = cf_minimax(P, (0.0, upper^2 / 4), m, 0)
    verbose && cf_print_result(res; name = "besseli2x_small", kwargs...)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = I2x(x) = P(1/x) / √x
<=> P(y) = √x * I2x(x)
"""
function besseli2x_large_constants(m::Int, lower::Real, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = (x = inv(y); exp(-x) * sqrt(x) * ArbNumerics.besseli(2, x))
    res = cf_minimax(P, (1 / upper, 1 / lower), m, 0)
    verbose && cf_print_result(res; name = "besseli2x_large", kwargs...)
    return res
end

function besseli2x_manual_constants(; verbose::Bool = false)
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
function laguerre½_small_constants(m::Int, low::Real; verbose::Bool = false, kwargs...)
    L½(x) = exp(x / 2) * ((1 - x) * ArbNumerics.besseli(0, -x / 2) - x * ArbNumerics.besseli(1, -x / 2))
    L½scaled(t) = sqrt(ArbReal(pi) / 2) * L½(-t^2 / 2)
    P(y::ArbReal) = (t = sqrt(y); L = L½scaled(t); ((L - t) * (L + t) - 1))
    res = cf_minimax(P, (0.0, low^2), m, 0)
    verbose && cf_print_result(res; name = "laguerre½²c_small", kwargs...)
    return res
end

"""
Large t ∈ [0, 1/upper] <=> y ∈ [0, 1/upper] where t = 1/√y, y = 1/t^2:
f(t) = L(t)^2 - t^2 - 1 = P(1/t)
<=> P(y) = L(t)^2 - t^2 - 1
"""
function laguerre½_large_constants(m::Int, high::Real; verbose::Bool = false, kwargs...)
    L½(t) = exp(t / 2) * ((1 - t) * ArbNumerics.besseli(0, -t / 2) - t * ArbNumerics.besseli(1, -t / 2))
    L½scaled(t) = sqrt(ArbReal(pi) / 2) * L½(-t^2 / 2)
    P(y::ArbReal) = (t = 1 / sqrt(y); L = L½scaled(t); t^2 * ((L - t) * (L + t) - 1))
    res = cf_minimax(P, (0.0, 1 / high^2), m, 0)
    verbose && cf_print_result(res; name = "laguerre½²c_large", kwargs...)
    return res
end

function laguerre½_manual_constants(; verbose::Bool = false)
    laguerre½_small_constants(10, 3.4; verbose, T = Float32)
    laguerre½_large_constants(10, 3.4; verbose, T = Float32)

    laguerre½_small_constants(20, 4.3; verbose, T = Float64)
    laguerre½_large_constants(20, 4.3; verbose, T = Float64)
    return nothing
end

"""
Small x ∈ [0, upper] <=> y ∈ [0, upper^2] where x = √y, y = x^2:
f(x) = I1(x) / I0(x) = x * P(y)
<=> P(y) = (I1(x) / I0(x)) / x
"""
function besseli1i0_low_constants(m::Int, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = (x = sqrt(y); ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x) / x)
    res = cf_minimax(P, (0.0, upper^2), m, 0)
    verbose && cf_print_result(res; name = "besseli1i0_low", kwargs...)
    return res
end

"""
Middle x ∈ [lower, upper] <=> y ∈ [lower^2, upper^2] where x = √y, y = x^2:
f(x) = I1(x) / I0(x) = P(y) / x
<=> P(y) = x * (I1(x) / I0(x))
"""
function besseli1i0_middle_constants(m::Int, n::Int, lower::Real, upper::Real; branchnum::Int, verbose::Bool = false, kwargs...)
    P(y::ArbReal) = (x = sqrt(y); ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x) / x)
    res = cf_minimax(P, (lower^2, upper^2), m, n)
    verbose && cf_print_result(res; name = "besseli1i0_mid$branchnum", kwargs...)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = I1(x) / I0(x) - 1 = (-1/2 - P(1/x) / x) / x
<=> P(y) = x * (-1/2 + x * (1 - I₁(x) / I₀(x)))
"""
function besseli1i0c_tail_constants(m::Int, lower::Real, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = (x = inv(y); x * (2x * (1 - ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x)) - 1) / 2)
    res = cf_minimax(P, (1 / upper, 1 / lower), m, 0)
    verbose && cf_print_result(res; name = "besseli1i0c_tail", kwargs...)
    return res
end

"""
Large x ∈ [lower, upper] <=> y ∈ [1/lower, 1/upper] where x = 1/y, y = 1/x:
f(x) = r(x)^2 - 1 + r(x) / x = -P(1/x) / x^2 where r(x) = I1(x) / I0(x)
<=> P(y) = -x^2 * (r(x)^2 - 1 + r(x) / x)
"""
function besseli1i0sqm1pi1i0x_tail_constants(m::Int, lower::Real, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = (x = inv(y); r = ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x); -x^2 * ((r - 1) * (r + 1) + r / x))
    res = cf_minimax(P, (1 / upper, 1 / lower), m, 0)
    verbose && cf_print_result(res; name = "besseli1i0sqm1pi1i0x_tail", kwargs...)
    return res
end

function besseli1i0_manual_constants(; verbose::Bool = false)
    besseli1i0_low_constants(3, 0.5; verbose, T = Float32)
    besseli1i0_middle_constants(3, 4, 0.5, 7.75; branchnum = 1, verbose, T = Float32)
    besseli1i0_middle_constants(3, 3, 7.75, 15.0; branchnum = 2, verbose, T = Float32)
    besseli1i0c_tail_constants(4, 15.0, Inf; verbose, T = Float32)
    besseli1i0sqm1pi1i0x_tail_constants(4, 15.0, Inf; verbose, T = Float32)

    besseli1i0_low_constants(7, 0.5; verbose, T = Float64)
    besseli1i0_middle_constants(7, 6, 0.5, 7.75; branchnum = 1, verbose, T = Float64)
    besseli1i0_middle_constants(7, 6, 7.75, 15.0; branchnum = 2, verbose, T = Float64)
    besseli1i0c_tail_constants(15, 15.0, Inf; verbose, T = Float64)
    besseli1i0sqm1pi1i0x_tail_constants(15, 15.0, Inf; verbose, T = Float64)
    return nothing
end

function besseli1i0_autotune_constants(; verbose::Bool = false)

    function build_ratapprox((a, b), order, i, n)
        if i == 1
            @assert a == 0 && isfinite(b)
            besseli1i0_low_constants(order, b)
        elseif i <= n - 1
            @assert isfinite(a) && isfinite(b)
            besseli1i0_middle_constants(order, min(order, 4), a, b; branchnum = i - 1)
        else # i == n
            @assert isfinite(a) && b == Inf
            besseli1i0c_tail_constants(order, a, b)
        end
    end

    function print_autotune_results(res; T::Type)
        verbose && display.(res.approximants)
        println("@inline besseli1i0_branches(::Type{$T}) = $(T.((res.branches...,)))")
        cf_print_result(res.approximants[1]; name = "besseli1i0_low", T, transplant = true)
        for i in 2:length(res.approximants)-1
            cf_print_result(res.approximants[i]; name = "besseli1i0_mid$(i-1)", T, transplant = true)# i == 2)
        end
        cf_print_result(res.approximants[end]; name = "besseli1i0c_tail", T, transplant = true)
    end

    res32 = autotune_minimax_spline(
        build_ratapprox;
        initial_order = 3,
        initial_branches = Double64.([0.5, 7.75, 15.0]),
        strategy = :subdivide,
        max_order = 3,
        fixed_left_branch = true,
        fixed_right_branch = false,
        verbose = verbose,
        minimax_abs_tol = Double64(1e-8),
    )
    print_autotune_results(res32; T = Float32)

    res64 = autotune_minimax_spline(
        build_ratapprox;
        initial_order = 8,
        initial_branches = Double64.(res32.branches),
        strategy = :increase_order,
        max_order = 20,
        fixed_left_branch = true,
        fixed_right_branch = false,
        max_branches = length(res32.branches),
        verbose = verbose,
        minimax_abs_tol = Double64(1e-18),
    )
    print_autotune_results(res64; T = Float64)

    return nothing
end

r(z) = ArbNumerics.besseli(1, z) / ArbNumerics.besseli(0, z)
r_and_r′(z) = (_r = r(z); return (_r, 1 - _r / z - _r^2)) # r′(z) = 1 - r(z) / z - r(z)^2
r_and_r′_and_r′′(z) = ((_r, _r′) = r_and_r′(z); return (_r, _r′, -_r′ / z + _r / z^2 - 2 * _r * _r′)) # r′′(z) = -r′(z) / z + r(z) / z^2 - 2 * r(z) * r′(z)
r_and_r′_and_r′′_and_two_r′_plus_z_r′′(z) = ((_r, _r′, _r′′) = r_and_r′_and_r′′(z); return (_r, _r′, _r′′, 2 * _r′ + z * _r′′)) # also returns 2 * r′ + z * r′′
r′(z) = r_and_r′(z)[2]
r′′(z) = r_and_r′_and_r′′(z)[3]

a0(z) = z == 0 ? oftype(z, 1 // 2) : r(z) / z # z << 1: a0 = 1/2 - z^2/16 + z^4/96 + O(z^6)
a1(z) = z == 0 ? oftype(z, -1 // 16) : (2 * a0(z) - 1) / 2z^2 # z << 1: a1 = -1/16 + z^2/96 + O(z^4)
b0(u) = u == 0 ? oftype(u, 1 // 2) : (1 - r(1 / u)) / u # u << 1 <=> z >> 1: b0 = 1/2 + u/8 + u^2/8 + O(u^3)
b1(u) = u == 0 ? oftype(u, 1 // 4) : (2 * b0(u) - 1) / u # u << 1 <=> z >> 1: b1 = 1/4 + u/4 + 25u^2/64 + O(u^3)
b2(u) = u == 0 ? oftype(u, 1) : (4 * b1(u) - 1) / u # u << 1 <=> z >> 1: b2 = 1 + 25u/16 + 13u^2/4 + O(u^3)
b3(u) = u == 0 ? oftype(u, 25 // 16) : (b2(u) - 1) / u # u << 1 <=> z >> 1: b3 = 25/16 + 13u/4 + O(u^2)

function besseli1i0_a1_taylor_constants(m::Int, upper::Real; verbose::Bool = false, kwargs...)
    P(y::ArbReal) = a1(√y) # z = √y <=> y = z^2 <=> a1(z) = P(y) = P(z^2)
    res = cf_minimax(P, (0.0, upper^2), m, 0)
    verbose && cf_print_result(res; name = "besseli1i0_a1_taylor", kwargs...)
    return res
end

function besseli1i0_a1_middle_constants(m::Int, n::Int, lower::Real, upper::Real; branchnum::Int, verbose::Bool = false, kwargs...)
    P(y::ArbReal) = a1(√y) # z = √y <=> y = z^2 <=> a1(z) = P(y) / Q(y) = P(z^2) / Q(z^2)
    res = cf_minimax(P, (lower^2, upper^2), m, n)
    verbose && cf_print_result(res; name = "besseli1i0_a1_branch$branchnum", transplant = false, kwargs...)
    return res
end

function besseli1i0_b3_middle_constants(m::Int, n::Int, lower::Real, upper::Real; branchnum::Int, verbose::Bool = false, kwargs...)
    P(u::ArbReal) = b3(u) # z = 1/u <=> u = 1/z <=> b3(z) = P(u) = P(1/z)
    res = cf_minimax(P, (1 / upper, 1 / lower), m, n)
    verbose && cf_print_result(res; name = "besseli1i0_b3_branch$branchnum", transplant = false, kwargs...)
    return res
end

function besseli1i0_b3_asymptotic_constants(m::Int, lower::Real, upper::Real; branchnum::Int, verbose::Bool = false, kwargs...)
    P(u::ArbReal) = b3(u) # z = 1/u <=> u = 1/z <=> b3(z) = P(u) = P(1/z)
    res = cf_minimax(P, (1 / upper, 1 / lower), m, 0)
    verbose && cf_print_result(res; name = "besseli1i0_b3_tail$branchnum", transplant = true, kwargs...)
    return res
end

function neglogpdf_rician_manual_constants(; verbose::Bool = false)
    branches32 = (2.0f0, 2.75f0, 4.0f0, 6.5f0, 11.0f0, 27.5f0)
    branches64 = (2.0, 3.25, 5.25, 8.75, 15.0, 30.0)

    # Float32 approximants
    verbose && println("@inline neglogpdf_rician_parts_branches(::Type{Float32}) = $(Float32.((branches32...,)))")
    besseli1i0_a1_taylor_constants(7, branches32[1]; verbose, T = Float32)
    for i in 1:length(branches32)-2
        besseli1i0_b3_middle_constants(8, 0, branches32[i], branches32[i+1]; branchnum = i, verbose, T = Float32)
    end
    besseli1i0_b3_asymptotic_constants(6, branches32[end-1], branches32[end]; branchnum = 1, verbose, T = Float32)
    besseli1i0_b3_asymptotic_constants(4, branches32[end], Inf; branchnum = 2, verbose, T = Float32)

    # Float64 approximants
    verbose && println("@inline neglogpdf_rician_parts_branches(::Type{Float64}) = $(Float64.((branches64...,)))")
    besseli1i0_a1_taylor_constants(20, branches64[1]; verbose, T = Float64)
    for i in 1:length(branches64)-2
        besseli1i0_b3_middle_constants(20, 0, branches64[i], branches64[i+1]; branchnum = i, verbose, T = Float64)
    end
    besseli1i0_b3_asymptotic_constants(16, branches64[end-1], branches64[end]; branchnum = 1, verbose, T = Float64)
    besseli1i0_b3_asymptotic_constants(14, branches64[end], Inf; branchnum = 2, verbose, T = Float64)

    return nothing
end

function neglogpdf_rician_autotune_constants(; verbose::Bool = false)

    function build_ratapprox((a, b), order, i, n)
        if i == 1
            @assert a == 0
            besseli1i0_a1_taylor_constants(order, b)
        elseif i <= n - 2
            besseli1i0_b3_middle_constants(order, 0, a, b; branchnum = i - 1)
        elseif i == n - 1
            besseli1i0_b3_asymptotic_constants(order, a, b; branchnum = 1)
        else # i == n
            @assert b == Inf
            besseli1i0_b3_asymptotic_constants(order, a, b; branchnum = 2)
        end
    end

    function print_autotune_results(res; T::Type)
        verbose && display.(res.approximants)
        println("@inline neglogpdf_rician_parts_branches(::Type{$T}) = $(T.((res.branches...,)))")
        cf_print_result(res.approximants[1]; name = "besseli1i0_a1_taylor", T, transplant = true)
        for i in 2:length(res.approximants)-2
            cf_print_result(res.approximants[i]; name = "besseli1i0_b3_mid$(i-1)", T, transplant = false)
        end
        cf_print_result(res.approximants[end-1]; name = "besseli1i0_b3_high", T, transplant = true)
        cf_print_result(res.approximants[end]; name = "besseli1i0_b3_tail", T, transplant = true)
    end

    res32 = autotune_minimax_spline(
        build_ratapprox;
        initial_order = 7,
        initial_branches = Double64.([2.0f0, 2.75f0, 4.0f0, 6.5f0, 11.0f0, 30.0f0]),
        strategy = :subdivide,
        max_order = 8,
        fixed_left_branch = true,
        fixed_right_branch = true,
        verbose = verbose,
        minimax_abs_tol = Double64(eps(Float32)) / 2,
    )
    print_autotune_results(res32; T = Float32)

    res64 = autotune_minimax_spline(
        build_ratapprox;
        initial_order = max(14, res32.order),
        initial_branches = Double64.([res32.branches[1:end-1]; 100.0f0]),
        strategy = :increase_order,
        max_order = 20,
        fixed_left_branch = true,
        fixed_right_branch = true,
        max_branches = length(res32.branches),
        verbose = verbose,
        minimax_abs_tol = Double64(eps(Float64)) / 4,
    )
    print_autotune_results(res64; T = Float64)

    return nothing
end
