using FastRicianLikelihoods
using Test
using Aqua

@testset "FastRicianLikelihoods.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(FastRicianLikelihoods)
    end
    # Write your tests here.
end
