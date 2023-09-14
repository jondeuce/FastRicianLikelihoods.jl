Pkg.activate(@__DIR__)

using FastRicianLikelihoods
const F = FastRicianLikelihoods

using Remez
using ArbNumerics

setworkingprecision(ArbFloat, 500)
setextrabits(128)

arb(f) = x -> convert(typeof(x), f(ArbFloat(x)))

function logbesseli0_small_constants(a=1/1e16)
    # Small x < 1.0:
    #   log(besseli(0, x)) = x^2 * P(x^2)
    #   => P(y) = log(besseli(0, x)) / x^2 where x = √y
    P4(y) = (x = sqrt(y); log(ArbNumerics.besseli(0, x)) / y)
    N, D, E, X = ratfn_minimax(arb(P4), (a, 1.0), 4, 0)
    @info "logbesseli0_small_constants: x < 3.75" E=Float32(E) N=(Float32.(N)...,)
end

function besseli0x_small_constants()
    # Small x < 3.75:
    #   besselix(0, x) = exp(-|x|) * besseli(0, x)
    #                  = exp(-|x|) * P((|x| / 3.75)^2)
    #   => P(y) = besseli(0, x) where x = 3.75 * √y
    P6(y) = (x = 3.75 * sqrt(y); ArbNumerics.besseli(0, x))
    N, D, E, X = ratfn_minimax(arb(P6), (0.0, 1.0), 6, 0)
    @info "besseli0x_small_constants: x < 3.75" E=Float32(E) N=(Float32.(N)...,)
end

function besseli0x_large_constants(a=1/1e16)
    # Large x > 3.75:
    #   besselix(0, x) = exp(-|x|) * besseli(0, x)
    #                  = P(3.75 / |x|) / √|x|
    #   => P(y) = exp(-|x|) * √|x| * besseli(0, x) where x = 3.75 / y
    P8(y) = (x = 3.75 / y; exp(-x) * sqrt(x) * ArbNumerics.besseli(0, x))
    N, D, E, X = ratfn_minimax(arb(P8), (a, 1.0), 8, 0)
    @info "besseli0x_large_constants: x > 3.75" E=Float32(E) N=(Float32.(N)...,)
end

function besseli1x_small_constants(a=1/1e16)
    # Small x < 3.75:
    #   besselix(1, x) = exp(-|x|) * besseli(1, x)
    #                  = exp(-|x|) * |x| * P((|x| / 3.75)^2)
    #   => P(y) = besseli(1, x) / |x| where x = 3.75 * √y
    P6(y) = (x = 3.75 * sqrt(y); ArbNumerics.besseli(1, x) / x)
    N, D, E, X = ratfn_minimax(arb(P6), (a, 1.0), 6, 0)
    @info "besseli1x_small_constants: x < 3.75" E=Float32(E) N=(Float32.(N)...,)
end

function besseli1x_large_constants(a=1/1e16)
    # Large x > 3.75:
    #   besselix(1, x) = exp(-|x|) * besseli(1, x)
    #                  = P(3.75 / |x|) / √|x|
    #   => P(y) = exp(-|x|) * √|x| * besseli(1, x) where x = 3.75 / y
    P8(y) = (x = 3.75 / y; exp(-x) * sqrt(x) * ArbNumerics.besseli(1, x))
    N, D, E, X = ratfn_minimax(arb(P8), (a, 1.0), 8, 0)
    @info "besseli1x_large_constants: x > 3.75" E=Float32(E) N=(Float32.(N)...,)
end

function besseli2x_small_constants(a=1/1e16)
    # Small x < 3.75:
    #   besselix(2, x) = exp(-|x|) * besseli(2, x)
    #                  = exp(-|x|) * |x|^2 * P((|x| / 3.75)^2)
    #   => P(y) = besseli(2, x) / |x|^2 where x = 3.75 * √y
    P6(y) = (x = 3.75 * sqrt(y); ArbNumerics.besseli(2, x) / x^2)
    N, D, E, X = ratfn_minimax(arb(P6), (a, 1.0), 6, 0)
    @info "besseli2x_small_constants: x < 3.75" E=Float32(E) N=(Float32.(N)...,)
end

function besseli2x_large_constants(a=1/1e16)
    # Large x > 3.75:
    #   besselix(2, x) = exp(-|x|) * besseli(2, x)
    #                  = P(3.75 / |x|) / √|x|
    #   => P(y) = exp(-|x|) * √|x| * besseli(2, x) where x = 3.75 / y
    P8(y) = (x = 3.75 / y; exp(-x) * sqrt(x) * ArbNumerics.besseli(2, x))
    N, D, E, X = ratfn_minimax(arb(P8), (a, 1.0), 8, 0)
    @info "besseli2x_large_constants: x > 3.75" E=Float32(E) N=(Float32.(N)...,)
end

function ∇neglogpdf_rician_large_constants(a=1/1e16)
    # Large z > 10f0:
    #   z * (logÎ₀)′(z) = 1/2 + z * (I₁(z) / I₀(z) - 1)
    #                   = -z⁻¹ * P(z⁻¹)
    #   => P(y) = -z * (1/2 + z * (I₁(z) / I₀(z) - 1)) where z = 1 / y
    P4(y) = (z = inv(y); -z * (0.5 + z * (ArbNumerics.besseli(1, z) / ArbNumerics.besseli(0, z) - 1)))
    N, D, E, X = ratfn_minimax(arb(P4), (a, 1/10), 4, 0)
    @info "besseli2x_large_constants: x > 10" E=Float32(E) N=(Float32.(N)...,)
end
