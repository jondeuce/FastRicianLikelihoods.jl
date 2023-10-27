module GaussHalfHermiteTests

using Test

using FastRicianLikelihoods: GaussHalfHermite
using QuadGK: quadgk

using .GaussHalfHermite: gausshalfhermite_rec_coeffs, gausshalfhermite_gw

@testset "quadgk" begin
    W(t, γ) = t^γ * exp(-t^2)

    @testset "polynomial: γ = $γ, N = $N" for γ in [-0.3, 0.4, 1.2, 2.1], N in 1:5
        # Quadrature rule should integrate polynomials of degree 2N-1 exactly
        θₖ(k, n) = cos((2k + 1) * π / 2n)
        θab(a, b, n) = (a + b) / 2 .+ ((b - a) / 2) .* θₖ.(0:n-1, n)

        x, w = gausshalfhermite_gw(N, γ)
        for deg in 1:2N-1
            tᵢ = θab(0.0, 4.0, deg) # chebychev nodes on [0, 4]
            A₀ = rand((-1, 1)) * (1 + rand()) # leading coefficient
            p = t -> A₀ * prod(t .- tᵢ) # degree `deg <= 2N-1` polynomial

            # This is a particularly nasty integral when γ < 0, since p(t) is high oscillatory near the singularity
            I, E = quadgk(t -> W(t, γ) * p(t), big"0.0", big"1.0", big"4.0", big"Inf"; order=15, rtol=1e-30)
            Î = sum(@. w * p(x))
            @test Î ≈ I
        end
    end

    @testset "generic: γ = $γ, f = $f" for γ in [-0.7, -0.3, 0.6, 1.7], f in [sin, exp]
        knot = γ < 0 ? big"1.0" : √(BigFloat(γ) / 2)
        I, E = quadgk(t -> W(t, γ) * f(t), big"0.0", knot, big"Inf"; order=15, rtol=1e-30)

        N = 2 .^ (0:5)
        Î = map(N) do N
            x, w = gausshalfhermite_gw(N, γ)
            Î = sum(@. w * f(x))
        end
        ΔI = @. Float64(abs(I - Î))

        last_converged = isapprox(Î[1], I; atol = 10 * eps(), rtol = 10 * eps())
        for i in 2:length(N)
            curr_converged = isapprox(Î[i], I; atol = 10 * eps(), rtol = 10 * eps())
            if !last_converged
                @test ΔI[i] < ΔI[i-1] # haven't converged; next estimate should be strictly better
            else
                @test curr_converged # have converged; next estimate should also be accurate
            end
            last_converged = curr_converged
        end
        @test last_converged # final estimate should be good
    end
end

# Table of recurrence coefficients for 2√π ∫_{0}^{∞} exp(-x^2) f(x) dx
#   [1] Galant D. Gauss Quadrature Rules for the Evaluation of 2π-1/2 ∫∞0 exp (-x2)f(x)dx. Mathematics of Computation 1969; 23: 674-s39.
TABLE_1_GALANT_1969() = [
    big"0.5641895835_4775628695"    big"0.0"                       # 1
    big"0.9884253928_4680028549"    big"0.1816901138_1620932846"   # 2
    big"0.1285967619_3639399603e1"  big"0.3413251289_5943919856"   # 3
    big"0.1524720844_0801153035e1"  big"0.5049621529_8800163194"   # 4
    big"0.1730192274_3094392568e1"  big"0.6702641946_3961908568"   # 5
    big"0.1913499843_1431025707e1"  big"0.8361704992_8031101555"   # 6
    big"0.2080620336_4008332248e1"  big"0.1002347851_0110108422e1" # 7
    big"0.2235228380_5046391497e1"  big"0.1168671164_7442727438e1" # 8
    big"0.2379782443_5046374209e1"  big"0.1335082922_2423353580e1" # 9
    big"0.2516025643_4438664098e1"  big"0.1501552599_3447618439e1" # 10
    big"0.2645247925_0569531803e1"  big"0.1668062362_1881161688e1" # 11
    big"0.2768435953_5042559069e1"  big"0.1834601052_7937676420e1" # 12
    big"0.2886364594_0326945693e1"  big"0.2001161318_5512137843e1" # 13
    big"0.2999655653_3536035387e1"  big"0.2167738111_7632644853e1" # 14
    big"0.3108817175_9249201517e1"  big"0.2334327849_5405013980e1" # 15
    big"0.3214270636_0711282274e1"  big"0.2500927917_1337026700e1" # 16
    big"0.3316370297_0830873659e1"  big"0.2667536360_9572020883e1" # 17
    big"0.3415417332_4133389445e1"  big"0.2834151691_6678327579e1" # 18
    big"0.3511670344_6156295154e1"  big"0.3000772753_7827190276e1" # 19
    big"0.3605353345_9055664303e1"  big"0.3167398636_9644268118e1" # 20
]

@testset "Table 1 (Galant 1969)" begin
    N = 20
    tbl = TABLE_1_GALANT_1969()
    for (T, rtol) in [(Float64, 5e-16), (BigFloat, 5e-20)]
        α, β = gausshalfhermite_rec_coeffs(20, T(0.0))
        for n in 1:N
            @test isapprox(α[n], tbl[n, 1]; rtol)
            @test isapprox(β[n], tbl[n, 2]; rtol)
        end
    end
end

end # GaussHalfHermiteTests

using .GaussHalfHermiteTests
