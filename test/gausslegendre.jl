module GaussLegendreTests

using Test

using FastRicianLikelihoods: FastRicianLikelihoods, Distributions, FastGaussQuadrature, GaussLegendre
using FastRicianLikelihoods: f_quadrature, neglogf_quadrature
using .Distributions: Normal, logpdf, cdf
using .GaussLegendre: eval_legpoly

@testset "nodes and weights" begin
    for n in 1:100
        x, w = GaussLegendre.gausslegendre(n, BigFloat; refine = true)
        x_FGQ, w_FGQ = FastGaussQuadrature.gausslegendre(n)
        x_F64, w_F64 = GaussLegendre.gausslegendre(n, Float64; refine = false)

        # Consistency with FastGaussQuadrature.jl
        @test x_F64 ≈ x_FGQ atol = 8 * eps(Float64) rtol = 0
        @test w_F64 ≈ w_FGQ atol = 8 * eps(Float64) rtol = 0

        # Check that root is approximately the same quality as FastGaussQuadrature.jl
        Px_FGQ = first.(eval_legpoly.(n, BigFloat.(x_FGQ)))
        Px_F64 = first.(eval_legpoly.(n, BigFloat.(x_F64)))
        @test maximum(abs, Px_F64) <= 2 * maximum(abs, Px_FGQ)
        @test maximum(abs, Px_F64) <= 10n * eps(Float64)
        @test maximum(abs, Px_FGQ) <= 10n * eps(Float64)

        # Check that BigFloat solution is accurate to high precision
        Px = first.(eval_legpoly.(n, BigFloat.(x)))
        @test maximum(abs, Px) <= 10n * eps(BigFloat)

        # Check accuracy using BigFloat solution as ground truth
        x_F64_rnd, w_F64_rnd = Float64.(x), Float64.(w)
        @test x ≈ x_F64 atol = 8 * eps(Float64) rtol = 0
        @test w ≈ w_F64 atol = 8 * eps(Float64) rtol = 0

        # Rounded BigFloat solution should be at least as accurate as both Float64 and FastGaussQuadrature.jl solutions
        Px_F64_rnd = first.(eval_legpoly.(n, BigFloat.(x_F64_rnd)))
        @test maximum(abs, Px_F64_rnd) <= min(maximum(abs, Px_F64), maximum(abs, Px_FGQ))
    end
end

@testset "gauss legendre quadrature ($T)" for T in (Float32, Float64)
    order = Val(16)
    d = Normal(randn(T) / 5, 1 + rand(T))
    a, δ = randn(T) / 5, (1 + rand(T)) / 10
    Ω = @inferred f_quadrature(x -> exp(logpdf(d, x)), a, δ, order)
    logΩ = @inferred -neglogf_quadrature(x -> -logpdf(d, x), a, δ, order)
    Ωtrue = cdf(d, a + δ) - cdf(d, a)
    @test Ω isa T
    @test logΩ isa T
    @test Ω ≈ Ωtrue atol = 10 * eps(T) rtol = 10 * eps(T)
    @test logΩ ≈ log(Ωtrue) atol = 10 * eps(T) rtol = 10 * eps(T)
end

end # module GaussLegendreTests

using .GaussLegendreTests
