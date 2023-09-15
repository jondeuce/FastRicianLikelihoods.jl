module RicianTests

using Test

using FastRicianLikelihoods: f_quadrature, neglogf_quadrature
using Distributions: Normal, logpdf, cdf

@testset "gauss legendre quadrature" begin
    map((Float32, Float64)) do T
        d = Normal(randn(T) / 2, 1 + rand(T))
        a, δ = randn(T), rand(T) / 10
        Ω = @inferred f_quadrature(x -> exp(logpdf(d, x)), a, δ)
        logΩ = @inferred -neglogf_quadrature(x -> -logpdf(d, x), a, δ)
        Ωtrue = cdf(d, a + δ) - cdf(d, a)
        @test Ω isa T
        @test logΩ isa T
        @test Ω ≈ Ωtrue rtol = sqrt(eps(T))
        @test logΩ ≈ log(Ωtrue) rtol = sqrt(eps(T))
    end
end

end # module RicianTests

import .RicianTests
