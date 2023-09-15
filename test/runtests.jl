using FastRicianLikelihoods
using Test
using Aqua

@testset "FastRicianLikelihoods.jl" begin
    @testset "dual rules" verbose=true include("forwarddiff.jl")
    @testset "rician" verbose=true include("rician.jl")

    @testset "Code quality (Aqua.jl)" begin
        # Generated methods for `ForwardDiff.Dual` introduce a lot
        # of method ambiguities that will almost surely never be hit
        Aqua.test_all(FastRicianLikelihoods; ambiguities = false)
    end
end
