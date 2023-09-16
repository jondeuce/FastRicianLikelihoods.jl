Pkg.activate(@__DIR__)

using Remez
using ArbNumerics

setworkingprecision(ArbFloat, 500)
setextrabits(128)

arbify(f) = x -> convert(typeof(x), f(ArbFloat(x)))

function logbesseli0_small_constants(a=1/1e16)
    # Small x < 1.0:
    #   log(besseli(0, x)) = x^2 * P(x^2)
    #   => P(y) = log(besseli(0, x)) / x^2 where x = √y
    P4(y) = (x = sqrt(y); log(ArbNumerics.besseli(0, x)) / y)
    N, D, E, X = ratfn_minimax(arbify(P4), (a, 1.0), 4, 0)
    @info "logbesseli0_small_constants: x < 3.75" E=Float64(E) N=(Float64.(N)...,)
end

function besseli0x_small_constants()
    # Small x < 3.75:
    #   besselix(0, x) = exp(-|x|) * besseli(0, x)
    #                  = exp(-|x|) * P((|x| / 3.75)^2)
    #   => P(y) = besseli(0, x) where x = 3.75 * √y
    P6(y) = (x = 3.75 * sqrt(y); ArbNumerics.besseli(0, x))
    N, D, E, X = ratfn_minimax(arbify(P6), (0.0, 1.0), 6, 0)
    @info "besseli0x_small_constants: x < 3.75" E=Float64(E) N=(Float64.(N)...,)
end

function besseli0x_large_constants(a=1/1e16)
    # Large x > 3.75:
    #   besselix(0, x) = exp(-|x|) * besseli(0, x)
    #                  = P(3.75 / |x|) / √|x|
    #   => P(y) = exp(-|x|) * √|x| * besseli(0, x) where x = 3.75 / y
    P8(y) = (x = 3.75 / y; exp(-x) * sqrt(x) * ArbNumerics.besseli(0, x))
    N, D, E, X = ratfn_minimax(arbify(P8), (a, 1.0), 8, 0)
    @info "besseli0x_large_constants: x > 3.75" E=Float64(E) N=(Float64.(N)...,)
end

function besseli1x_small_constants(a=1/1e16)
    # Small x < 3.75:
    #   besselix(1, x) = exp(-|x|) * besseli(1, x)
    #                  = exp(-|x|) * |x| * P((|x| / 3.75)^2)
    #   => P(y) = besseli(1, x) / |x| where x = 3.75 * √y
    P6(y) = (x = 3.75 * sqrt(y); ArbNumerics.besseli(1, x) / x)
    N, D, E, X = ratfn_minimax(arbify(P6), (a, 1.0), 6, 0)
    @info "besseli1x_small_constants: x < 3.75" E=Float64(E) N=(Float64.(N)...,)
end

function besseli1x_large_constants(a=1/1e16)
    # Large x > 3.75:
    #   besselix(1, x) = exp(-|x|) * besseli(1, x)
    #                  = P(3.75 / |x|) / √|x|
    #   => P(y) = exp(-|x|) * √|x| * besseli(1, x) where x = 3.75 / y
    P8(y) = (x = 3.75 / y; exp(-x) * sqrt(x) * ArbNumerics.besseli(1, x))
    N, D, E, X = ratfn_minimax(arbify(P8), (a, 1.0), 8, 0)
    @info "besseli1x_large_constants: x > 3.75" E=Float64(E) N=(Float64.(N)...,)
end

function besseli2x_small_constants(a=1/1e16)
    # Small x < 3.75:
    #   besselix(2, x) = exp(-|x|) * besseli(2, x)
    #                  = exp(-|x|) * |x|^2 * P((|x| / 3.75)^2)
    #   => P(y) = besseli(2, x) / |x|^2 where x = 3.75 * √y
    P6(y) = (x = 3.75 * sqrt(y); ArbNumerics.besseli(2, x) / x^2)
    N, D, E, X = ratfn_minimax(arbify(P6), (a, 1.0), 6, 0)
    @info "besseli2x_small_constants: x < 3.75" E=Float64(E) N=(Float64.(N)...,)
end

function besseli2x_large_constants(a=1/1e16)
    # Large x > 3.75:
    #   besselix(2, x) = exp(-|x|) * besseli(2, x)
    #                  = P(3.75 / |x|) / √|x|
    #   => P(y) = exp(-|x|) * √|x| * besseli(2, x) where x = 3.75 / y
    P8(y) = (x = 3.75 / y; exp(-x) * sqrt(x) * ArbNumerics.besseli(2, x))
    N, D, E, X = ratfn_minimax(arbify(P8), (a, 1.0), 8, 0)
    @info "besseli2x_large_constants: x > 3.75" E=Float64(E) N=(Float64.(N)...,)
end

function besseli0_constants(min=1/1e16, low=7.75, branch=50)
    # Small x < low:
    #   besseli(0, x) = (1 + y * P(y)) where y = x^2 / 4
    Plow(y) = (x = 2*sqrt(y); (ArbNumerics.besseli(0, x) - 1) / y)
    N, D, E, X = ratfn_minimax(arbify(Plow), (min, low^2 / 4), 8, 0)
    @info "besseli0: x < $low" E=Float64(E) N=(Float64.(N)...,)

    # Medium x > low:
    #   besseli(0, x) = exp(x) * P(y) / sqrt(x) where y = 1/x
    Pmed(y) = (x = 1/y; exp(-x) * sqrt(x) * ArbNumerics.besseli(0, x))
    N, D, E, X = ratfn_minimax(arbify(Pmed), (min, 1/low), 4, 0)
    @info "besseli0: x > $low" E=Float64(E) N=(Float64.(N)...,)

    # Large x > branch:
    #   besselix(0, x) = exp(-|x|) * besseli(0, x) = P(y) / sqrt(x) where y = 1/x
    Plarge(y) = (x = 1/y; exp(-x) * sqrt(x) * ArbNumerics.besseli(0, x))
    N, D, E, X = ratfn_minimax(arbify(Plarge), (min, 1/branch), 2, 0)
    @info "besseli0x: x > $branch" E=Float64(E) N=(Float64.(N)...,)
end

function besseli2_constants(min=1/1e16, low=7.75, branch=500)
    # Small x < low:
    #   besseli(2, x) = y * P(y) where y = x^2 / 4
    Plow(y) = (x = 2*sqrt(y); ArbNumerics.besseli(2, x) / y)
    N, D, E, X = ratfn_minimax(arbify(Plow), (min, low^2 / 4), 13, 0)
    @info "besseli2: x < $low" E=Float64(E) N=(Float64.(N)...,)

    # Medium x > low:
    #   besseli(2, x) = exp(x) * P(y) / sqrt(x) where y = 1/x
    Pmed(y) = (x = 1/y; exp(-x) * sqrt(x) * ArbNumerics.besseli(2, x))
    N, D, E, X = ratfn_minimax(arbify(Pmed), (min, 1/low), 20, 0)
    @info "besseli2: x > $low" E=Float64(E) N=(Float64.(N)...,)

    # Large x > branch:
    #   besselix(2, x) = exp(-|x|) * besseli(2, x) = P(y) / sqrt(x) where y = 1/x
    Plarge(y) = (x = 1/y; exp(-x) * sqrt(x) * ArbNumerics.besseli(2, x))
    N, D, E, X = ratfn_minimax(arbify(Plarge), (min, 1/branch), 4, 0)
    @info "besseli2x: x > $branch" E=Float64(E) N=(Float64.(N)...,)
end

function ∇neglogpdf_rician_constants()
    T = Float64
    low, med, smalldeg, num, den, largedeg =
        T == Float32 ? (0.5, 15.0, 3, 4, 4, 4) :
                       (0.5, 15.0, 7, 7, 7, 15)

    # Small z < low:
    #   logÎ₀′(z) + 1 - 1/2z = I₁(z) / I₀(z) ≈ z/2 + 𝒪(z^2)
    #                        = z * P(z^2)
    #   => P(y) = (I₁(z) / I₀(z)) / z where z = √y
    Psmall(y) = if y == 0; convert(typeof(y), 1/2); else; z = sqrt(y); (ArbNumerics.besseli(1, z) / ArbNumerics.besseli(0, z)) / z end
    N, D, E, X = ratfn_minimax(arbify(Psmall), (0.0, low^2), smalldeg, 0)
    @info "∇neglogpdf_rician (degree $smalldeg): x < $low" E=T(E) N=(T.(N)...,)

    # # Medium low < z < med:
    # #   I₁(z) / I₀(z) ≈ z * P(z)
    # Pmed(y) = (z = sqrt(y); (ArbNumerics.besseli(1, z) / ArbNumerics.besseli(0, z)) / z)
    # N, D, E, X = ratfn_minimax(arbify(Pmed), (low^2, med^2), num, den)
    # @info "∇neglogpdf_rician: $low < x < $med (degree $num / $den)" E=T(E) N=(T.(N)...,) D=(T.(D)...,)

    # Large z > med:
    #   z * logÎ₀′(z) = 1/2 + z * (I₁(z) / I₀(z) - 1) ≈ -1/8z + 𝒪(1/z^2)
    #                 = -z⁻¹ * P(z⁻¹)
    #   => P(y) = -z * (1/2 + z * (I₁(z) / I₀(z) - 1)) where z = 1 / y
    Plarge(y) = if y == 0; convert(typeof(y), 1/8); else; z = inv(y); z * (-0.5 - z * (ArbNumerics.besseli(1, z) / ArbNumerics.besseli(0, z) - 1)) end
    N, D, E, X = ratfn_minimax(arbify(Plarge), (0.0, 1/med), largedeg, 0)
    @info "∇neglogpdf_rician (degree $largedeg): x > $med" E=T(E) N=(T.(N)...,)
end
