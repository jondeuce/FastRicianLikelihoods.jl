# Usage: julia --project=test test/interactive/coefficients.jl [output.jl]
# Generates src/coefficients.jl by default.

using ArbNumerics: ArbNumerics, ArbReal, setworkingprecision
using CaratheodoryFejerApprox: minimax, monocoeffs
using DoubleFloats: DoubleFloats, Double64
using Random: Random

const REFERENCE_BITS = 500
const FIT_SEED = 119

ratio(z) = ArbNumerics.besseli(1, z) / ArbNumerics.besseli(0, z)

function log_taylor(y)
    iszero(y) && return -one(y) / 64
    return (log(ArbNumerics.besseli(0, sqrt(y))) - y / 4) / y^2
end

log_middle(z) = (log(ArbNumerics.besseli(0, z)) - z) / z

function log_tail(u)
    iszero(u) && return one(u) / 8
    z = inv(u)
    return log(exp(-z) * ArbNumerics.besseli(0, z) * sqrt(2 * oftype(z, π) * z)) / u
end

function bessel2_small(y)
    iszero(y) && return one(y) / 2
    return ArbNumerics.besseli(2, 2 * sqrt(y)) / y
end

function bessel2_tail(u)
    iszero(u) && return inv(sqrt(2 * oftype(u, π)))
    z = inv(u)
    return sqrt(z) * exp(-z) * ArbNumerics.besseli(2, z)
end

function laguerre_correction(y)
    z = y / 4
    L = sqrt(oftype(y, π) / 2) * exp(-z) * ((1 + y / 2) * ArbNumerics.besseli(0, z) + y / 2 * ArbNumerics.besseli(1, z))
    return L^2 - y - 1
end

laguerre_small(y) = laguerre_correction(4 * y)

function laguerre_tail(u)
    iszero(u) && return one(u) / 2
    return laguerre_correction(inv(u)) / u
end

ratio_small(y) = iszero(y) ? one(y) / 2 : ratio(sqrt(y)) / sqrt(y)

function ratio_tail(u)
    iszero(u) && return one(u) / 8
    return ((1 - ratio(inv(u))) - u / 2) / u^2
end

a1(y) = iszero(y) ? -one(y) / 16 : (ratio_small(y) - one(y) / 2) / y

function b3(u)
    iszero(u) && return oftype(u, 25 // 16)
    b0 = (1 - ratio(inv(u))) / u
    b1 = (2 * b0 - 1) / u
    b2 = (4 * b1 - 1) / u
    return (b2 - 1) / u
end

const MODE_TAYLOR = (1 // 1, 1 // 4, 1 // 16, 1 // 192, -67 // 12288, -817 // 245760, -10109 // 17694720, 210407 // 495452160, 45860611 // 126835752960, 185952481 // 1956894474240)

function mode_reference(ν; max_iterations = 50)
    iszero(ν) && return one(ν)
    ν² = ν^2
    x = sqrt(1 + ν² * (1 + 2 * ν²^2) / (2 + 2 * ν²^2))
    for _ in 1:max_iterations
        z = x * ν
        r = ratio(z)
        r′ = 1 - r / z - r^2
        step = (x - ν * r - inv(x)) / (1 + inv(x)^2 - ν² * r′)
        x -= step
        abs(Float64(step)) <= exp2(-(REFERENCE_BITS - 100)) * max(1, abs(Float64(x))) && return x
    end
    error("Mode reference did not converge for ν = $ν")
end

const VAR_MODE_TAYLOR = (1 // 2, 1 // 4, 1 // 16, -1 // 64, -71 // 3072, -217 // 24576, 4357 // 2949120, 25063 // 7077888, 27151871 // 15854469120, -40160509 // 761014517760)

function var_mode_reference(ν)
    iszero(ν) && return one(ν) / 2
    x = mode_reference(ν)
    z = x * ν
    r = ratio(z)
    r′ = 1 - r / z - r^2
    return inv(1 + inv(x)^2 - ν^2 * r′)
end

function fit_approximant(f, domain, m, n)
    target(x) = Double64(f(ArbReal(BigFloat(x))))
    return minimax(target, Double64.(domain), m, n)
end

function write_table(io, name, T, values)
    println(io, "@inline ", name, "(::Type{", T, "}) = ", repr((T.(values)...,)))
end

# Note: reported fit errors precede coefficient rounding.
function write_approximant(io, name, T, res; centered = false, suffix = "coefs", transplant = true)
    p, q = monocoeffs(res; transplant = !centered)
    all(isfinite, p) && all(isfinite, q) || error("Nonfinite coefficients for $name ($T)")
    m, n = length(p) - 1, length(q) - 1
    println(io, "# ", name, ": domain = ", repr(res.dom), ", degree = ", (m, n), ", fit error = ", Float64(res.err))
    if n == 0
        write_table(io, Symbol(name, "_", suffix), T, p)
    else
        write_table(io, Symbol(name, "_num_", suffix), T, p)
        write_table(io, Symbol(name, "_den_", suffix), T, q)
    end
    if centered && transplant
        a, b = res.dom
        write_table(io, Symbol(name, "_transplant"), T, (2 / (b - a), -(b + a) / (b - a)))
    end
    println(T, " ", name, " (", m, ",", n, ") ", Float64(res.err))
    flush(stdout)
    return res
end

function write_fit(io, name, T, f, domain, m, n = 0; kwargs...)
    return write_approximant(io, name, T, fit_approximant(f, domain, m, n); kwargs...)
end

# Fixed-domain fits.
function generate_static(io, T)
    small, med, large = T === Float32 ? (8, 4, 2) : (13, 20, 4)
    write_fit(io, :besseli2_small, T, bessel2_small, (0, Double64(7.75)^2 / 4), small)
    write_fit(io, :besseli2_med, T, bessel2_tail, (0, inv(Double64(7.75))), med)

    branch = T === Float32 ? 50 : 500
    write_fit(io, :besseli2_large, T, bessel2_tail, (0, inv(Double64(branch))), large)

    degrees = T === Float32 ? (8, 10, 12, 9) : (16, 18, 24, 18)
    write_fit(io, :laguerre½²c_small1, T, laguerre_small, (0, 1), degrees[1])
    write_fit(io, :laguerre½²c_small2, T, laguerre_correction, (4, 12), degrees[2]; centered = true)
    write_fit(io, :laguerre½²c_small3, T, laguerre_correction, (12, 36), degrees[3]; centered = true)
    write_fit(io, :laguerre½²c_large, T, laguerre_tail, (0, Double64(1) / 36), degrees[4]; centered = true)
end

# Dynamic-domain fits.
intervals(branches) = zip((zero(eltype(branches)), branches...), (branches..., oftype(first(branches), Inf)))

function build_intervals(build, branches, degrees)
    return [build(domain, degree, i, length(degrees)) for (i, (domain, degree)) in enumerate(zip(intervals(branches), degrees))]
end

function balance_branches(build, branches, degree, fixed_left, fixed_right; max_iterations = 25)
    fits = build_intervals(build, branches, fill(degree, length(branches) + 1))

    # Balance movable regions to within a factor of two; fixed endpoints remain untouched.
    active = (1+fixed_left):(length(fits)-fixed_right)
    errors = [max(fit.err, eps(Double64)) for fit in fits]
    spread(errs) = maximum(errs[active]) / minimum(errs[active])

    # Damped log-coordinate steps: limit relative movement and backtrack on worse balance.
    step = Double64(0.1)
    for _ in 1:max_iterations
        spread(errors) <= 2 && break
        proposal = copy(branches)
        for j in (1+fixed_left):(length(branches)-fixed_right)
            left = j == 1 ? branches[j] / 2 : branches[j-1]
            right = j == length(branches) ? 2 * branches[j] : branches[j+1]
            width = min(log(branches[j] / left), log(right / branches[j]))
            shift = clamp(step * log(errors[j+1] / errors[j]), -width / 4, width / 4)
            proposal[j] *= exp(shift)
        end

        proposed_fits = build_intervals(build, proposal, fill(degree, length(proposal) + 1))
        proposed_errors = [max(fit.err, eps(Double64)) for fit in proposed_fits]
        if spread(proposed_errors) < spread(errors)
            branches, fits, errors = proposal, proposed_fits, proposed_errors
            step = min(Double64(0.5), step * Double64(1.25))
        else
            step /= 2
            step < 1e-4 && break
        end
    end

    return branches, fits
end

# Balance cutoffs, increase degree to meet the absolute tolerance, then reduce each piece on rounded domains.
function search_approximants(build, T, seeds, degree, tol; fixed_left = false, fixed_right = false, max_degree = 25, max_branches = length(seeds), max_iterations = 50)
    branches = collect(Double64, seeds)
    for iteration in 1:max_iterations
        branches, fits = balance_branches(build, branches, degree, fixed_left, fixed_right)
        errors = [fit.err for fit in fits]
        println("search ", T, " iteration ", iteration, ": degree ", degree, ", branches ", length(branches), ", max error ", Float64(maximum(errors)), " / ", Float64(tol))
        flush(stdout)

        if maximum(errors) <= tol
            # Round the domains and minimize each piece's degree at the requested accuracy.
            degrees = fill(degree, length(fits))
            branches = Double64.(T.(branches))
            for (i, domain) in enumerate(intervals(branches))
                lo, hi = 0, degree
                while lo < hi
                    mid = (lo + hi) ÷ 2
                    fit = build(domain, mid, i, length(fits))
                    if fit.err <= tol
                        hi = mid
                    else
                        lo = mid + 1
                    end
                end
                degrees[i] = lo

                fits[i] = build(domain, lo, i, length(fits))
                while fits[i].err > tol && degrees[i] < max_degree
                    degrees[i] += 1
                    fits[i] = build(domain, degrees[i], i, length(fits))
                end
            end

            maximum(fit.err for fit in fits) <= tol || error("Rounded domains do not meet the fit tolerance")
            return (; branches = Tuple(T.(branches)), fits)

        elseif degree < max_degree
            # Increase approximation capacity without adding regions.
            degree += 1

        elseif length(branches) < max_branches
            # Split the worst adjustable region after reaching the degree limit.
            i = fixed_left + argmax(errors[(1+fixed_left):(end-fixed_right)])
            boundary = i == 1 ? first(branches) / 1.5 : i == length(fits) ? last(branches) * 1.5 : sqrt(branches[i-1] * branches[i])
            insert!(branches, i, boundary)

        else
            # Neither degree nor region count can increase further.
            error("Approximant search exhausted its degree and branch limits")
        end
    end
    error("Approximant search did not converge")
end

function generate_logbesseli(io, T)
    function build((a, b), m, i, n)
        i == 1 && return fit_approximant(log_taylor, (0, b^2), m, 0)
        i == n && return fit_approximant(log_tail, (0, inv(a)), m, 0)
        return fit_approximant(log_middle, (a, b), m, 0)
    end

    # Reserve error budget for rounding the log-Bessel coefficients and coordinate maps.
    degree = T === Float32 ? 5 : 16
    res = search_approximants(build, T, (1, 2, 3, 5, 7), degree, Double64(eps(T)) / (T === Float32 ? 2 : 8); fixed_left = true)

    write_table(io, :logbesseli0x_branches, T, res.branches)
    write_approximant(io, :logbesseli0_taylor, T, first(res.fits))
    for i in 1:(length(res.branches)-1)
        write_approximant(io, Symbol("logbesseli0x_mid", i), T, res.fits[i+1]; centered = true)
    end
    write_approximant(io, :logbesseli0x_tail, T, last(res.fits))

    return nothing
end

function generate_ratio(io, T)
    function build((a, b), m, i, n)
        i == 1 && return fit_approximant(ratio_small, (0, b^2), m, 0)
        i == n && return fit_approximant(ratio_tail, (0, inv(a)), m, 0)
        return fit_approximant(ratio_small, (a^2, b^2), m, min(m, T === Float32 ? 3 : 4))
    end

    degree, tolerance = T === Float32 ? (3, Double64(1e-8)) : (11, Double64(eps(T)) / 2048)
    res = search_approximants(build, T, (0.5, 5, 10, 20, 40), degree, tolerance; fixed_left = true)

    write_table(io, :besseli1i0_branches, T, res.branches)
    write_approximant(io, :besseli1i0_low, T, first(res.fits))
    for i in 1:(length(res.branches)-1)
        write_approximant(io, Symbol("besseli1i0_mid", i), T, res.fits[i+1])
    end
    write_approximant(io, :besseli1i0c_tail, T, last(res.fits))

    return nothing
end

function generate_parts(io, T)
    function build((a, b), m, i, n)
        i == 1 && return fit_approximant(a1, (0, b^2), T === Float32 ? 6 : 14, 0)
        return fit_approximant(b3, (isinf(b) ? zero(a) : inv(b), inv(a)), m, 0)
    end

    tail = T === Float32 ? 30 : 100
    seeds = (1.25, 2, 3, 4.5, 7, 12, 20, tail)
    degree = T === Float32 ? 7 : 17
    res = search_approximants(build, T, seeds, degree, Double64(eps(T)) / (T === Float32 ? 32 : 4); fixed_left = true, fixed_right = true)

    write_table(io, :neglogpdf_rician_parts_branches, T, res.branches)
    println(io, "@inline neglogpdf_rician_parts_taylor_branch(::Type{", T, "}) = first(neglogpdf_rician_parts_branches(", T, "))")
    write_approximant(io, :a1_taylor, T, first(res.fits))
    for i in 1:6
        write_approximant(io, Symbol("b3_mid", i), T, res.fits[i+1]; centered = true)
    end
    write_approximant(io, :b3_high, T, res.fits[end-1])
    write_approximant(io, :b3_tail, T, last(res.fits))

    return nothing
end

function write_wrapper(io, name, T, res; transform = "x")
    value = length(res.q) == 1 ? "evalpoly(y, $(name)_coeffs($T))" : "evalpoly(y, $(name)_num_coeffs($T)) / evalpoly(y, $(name)_den_coeffs($T))"
    println(io, "@inline ", name, "_approx(::Type{", T, "}, x::Real) = (y = ", transform, "; ", value, ")")
end

function taylor_cutoff(f, T, coefficients; max_iterations = 50)
    lo, hi = zero(Double64), one(Double64)
    for _ in 1:max_iterations # reduce the initial unit bracket to a width of 2^(-max_iterations)
        x = (lo + hi) / 2
        a = ArbReal(BigFloat(x))
        err = abs(f(a) - evalpoly(a^2, ArbReal.(coefficients)))
        if err <= ArbReal(eps(T)) / 4
            lo = x
        else
            hi = x
        end
    end
    return T(lo)
end

function generate_mode(io, T, name, f, taylor)
    coefficients = taylor[1:(T===Float32 ? 4 : 8)]
    cutoff = taylor_cutoff(f, T, coefficients)
    tail_limit = name === :mode_rician ? (T === Float32 ? 10 : 50) : (T === Float32 ? 12 : 25)
    middle(ν) = name === :mode_rician ? f(ν) - ν : f(ν)
    small(y) = f(sqrt(y))

    function build((a, b), m, i, n)
        i == 1 && return fit_approximant(small, (0, b^2), m + (T === Float32 ? 2 : 3), 0)
        if i >= n - 1
            function tail(y)
                iszero(y) && return name === :mode_rician ? one(y) / 2 : one(y)
                ν = oftype(y, BigFloat(a)) / sqrt(y)
                return name === :mode_rician ? ν * (f(ν) - ν) : f(ν)
            end
            return fit_approximant(tail, (0, 1), i == n ? m : m + (T === Float32 ? 0 : 4), 0)
        end
        return fit_approximant(middle, (a, b), m, min(m, T === Float32 ? 2 : 4))
    end

    seeds = name === :mode_rician ? (0.8, 1.2, 1.6, 2, 3, 4, tail_limit) : (0.7, 1, 1.4, 1.8, 2.3, 3, 4, tail_limit)
    degree = T === Float32 ? 3 : name === :mode_rician ? 9 : 10
    res = search_approximants(build, T, seeds, degree, Double64(eps(T)) / 2; fixed_right = true)

    write_table(io, Symbol(name, "_branches"), T, (cutoff, res.branches...))
    write_table(io, Symbol(name, "_coeff_taylor"), T, coefficients)
    println(io, "@inline ", name, "_coeff_taylor_cutoff(::Type{", T, "}) = first(", name, "_branches(", T, "))")
    println(io, "@inline ", name, "_coeff_long_tail_cutoff(::Type{", T, "}) = last(", name, "_branches(", T, "))")

    smallname = Symbol(name, "_kernel_small")
    write_approximant(io, smallname, T, first(res.fits); suffix = "coeffs")
    write_wrapper(io, smallname, T, first(res.fits); transform = "abs2(x)")

    for i in 1:(length(res.branches)-2)
        midname = Symbol(name, "_kernel_med_", i)
        fit = res.fits[i+1]
        write_approximant(io, midname, T, fit; centered = true, suffix = "coeffs", transplant = false)

        a, b = fit.dom
        s, h = T(2 / (b - a)), T(-(b + a) / (b - a))
        write_wrapper(io, midname, T, fit; transform = "muladd($(repr(s)), x, $(repr(h)))")
    end

    for (suffix, scale, fit) in (("tail", res.branches[end-1], res.fits[end-1]), ("tail_long", res.branches[end], res.fits[end]))
        fitname = Symbol(name, "_kernel_", suffix)
        write_approximant(io, fitname, T, fit; suffix = "coeffs")
        write_wrapper(io, fitname, T, fit; transform = "abs2($(repr(scale)) / x)")
    end

    return nothing
end

function generate_coefficients(io)
    println(io, "# Generated by test/interactive/coefficients.jl; do not edit.")
    for T in (Float32, Float64)
        generate_static(io, T)
        generate_logbesseli(io, T)
        generate_ratio(io, T)
        generate_parts(io, T)
        T === Float32 && println(io, "@inline mode_rician_coeff_taylor(::Type{T}) where {T} = T.(", repr(MODE_TAYLOR), ")")
        generate_mode(io, T, :mode_rician, mode_reference, MODE_TAYLOR)
        T === Float32 && println(io, "@inline var_mode_rician_coeff_taylor(::Type{T}) where {T} = T.(", repr(VAR_MODE_TAYLOR), ")")
        generate_mode(io, T, :var_mode_rician, var_mode_reference, VAR_MODE_TAYLOR)
    end
end

function main(output)
    Random.seed!(FIT_SEED) # ArnoldiMethod uses a random starting vector for Double64 fits.
    setworkingprecision(ArbReal; bits = REFERENCE_BITS)
    setprecision(BigFloat, REFERENCE_BITS) do
        dir = dirname(abspath(output))
        mkpath(dir)
        tmp = tempname(dir)
        open(generate_coefficients, tmp, "w")
        mv(tmp, output; force = true)
    end
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    output = isempty(ARGS) ? joinpath(@__DIR__, "..", "..", "src", "coefficients.jl") : only(ARGS)
    main(output)
end
