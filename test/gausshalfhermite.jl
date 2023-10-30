module GaussHalfHermiteTests

using Test

using FastRicianLikelihoods: GaussHalfHermite, clenshaw
using QuadGK: quadgk

using .GaussHalfHermite: gausshalfhermite_rec_coeffs, gausshalfhermite_gw
include("gausshalfhermite_tables.jl")

matchingdigits(x, y) = x == y ? oftype(x / y, Inf) : -log10(abs((x - y) / y))

# Quadrature weight functions
weight(t, γ) = t^γ * exp(-t^2)
weight_norm(t, γ) = t^γ * exp(-t^2 / 2) / √(2 * oftype(t, π))

@testset "quadgk" begin
    @testset "polynomial: γ = $γ, N = $N" for γ in [-0.7, -0.3, 0.4, 1.2, 2.1], N in 1:5
        # Quadrature rule should integrate polynomials of degree 2N-1 exactly.
        # Chebyshev polynomials are a nice test case for two reasons:
        #   1.  They are highly oscillatory on their domain, leading to particularly nasty
        #       integrals, especially when γ < 0 due to sign changes near the singularity
        #   2.  They grow maximially fast among polynomials outside of their domain,
        #       though this should be quickly crushed by exp(-x^2)
        p(t, c) = clenshaw(t, 0, 5, c) # Chebyshev polynomial on [0, 5]
        x, w = gausshalfhermite_gw(N, γ)
        atol = rtol = γ < -0.5 ? 1e-13 : 1e-14

        for deg in 1:2N-1
            c = randn(deg + 1) # coefficients for degree `deg` polynomial in Chebyshev basis
            ctup, cbig = (c...,), big.(c)

            I, E = quadgk(t -> weight(t, γ) * p(t, cbig), big"0.0", big"5.0", big"Inf"; order=15, rtol=1e-30)
            Î = sum(w .* p.(x, (ctup,)))
            @test isapprox(Î, I; atol, rtol)
        end
    end

    @testset "generic: γ = $γ, f = $f, nrm = $normalize" for γ in [-0.6, 0.8, 1.7, 2.5], f in [sin, exp], normalize in [false, true]
        knot = γ < 0 ? big"1.0" : √(big(γ) / 2)
        I, E = quadgk(big"0.0", knot, big"Inf"; order=15, rtol=1e-30) do t
            wₜ = !normalize ? weight(t, γ) : weight_norm(t, γ)
            return wₜ * f(t)
        end

        N = [1, 2, 4, 8, 12, 16, 24, 32]
        Î = map(N) do N
            x, w = gausshalfhermite_gw(N, γ; normalize)
            Î = sum(@. w * f(x))
        end
        ΔI = @. Float64(abs(I - Î))

        atol = rtol = 10 * eps()
        last_converged = isapprox(Î[1], I; atol, rtol)
        @test !last_converged # first estimate should not be good enough
        for i in 2:length(N)
            # Either 1) estimate hasn't converged => higher order estimate should be strictly better,
            # or 2) estimate has converged => higher order estimate should remain accurate
            curr_converged = isapprox(Î[i], I; atol, rtol)
            @test (!last_converged && ΔI[i] < ΔI[i-1]) || (last_converged && curr_converged)
            last_converged = curr_converged
        end
        @test last_converged # final estimate should be good
    end
end

# Recurrence coefficients
@testset "Table 1 (Galant 1969)" begin
    N = 20
    tbl = TABLE_1_GALANT_1969()
    for (T, digits) in [(Float64, 15.5), (BigFloat, 19.5)]
        α, β = gausshalfhermite_rec_coeffs(N, T(0.0))
        for n in 1:N
            @test matchingdigits(α[n], tbl[n, 1]) >= digits
            @test matchingdigits(β[n], tbl[n, 2]) >= digits
        end
    end
end

# Quadrature nodes and weights
@testset "Table II(a-c) (Shizgal 1981)" begin
    tbls = Dict{Int, Dict{Int, Matrix}}(
        0 => TABLE_IIa_SHIZGAL_1981(),
        1 => TABLE_IIb_SHIZGAL_1981(),
        2 => TABLE_IIc_SHIZGAL_1981(),
    )
    for (γ, tbl) in tbls, (N, xw) in tbl, (T, digits) in [(Float64, 13.25), (BigFloat, 15.0)]
        x, w = gausshalfhermite_gw(N, T(γ))
        for n in 1:N
            @test matchingdigits(x[n], xw[n, 1]) >= digits
            @test matchingdigits(w[n], xw[n, 2]) >= digits
        end
    end
end

end # GaussHalfHermiteTests

using .GaussHalfHermiteTests
