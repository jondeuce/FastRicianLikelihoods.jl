using FastRicianLikelihoods
using Test
using Aqua

include("utils.jl")

@testset "FastRicianLikelihoods.jl" begin
    @testset "dual rules" verbose=true begin
        include("forwarddiff.jl")
    end

    @testset "bessels" verbose=true begin
        include("bessels.jl")
    end

    @testset "rician" verbose=true begin
        include("rician.jl")
    end

    @testset "Code quality (Aqua.jl)" begin
        # Generated methods for `ForwardDiff.Dual` introduce a lot
        # of method ambiguities that will almost surely never be hit
        Aqua.test_all(FastRicianLikelihoods; ambiguities = false)
    end
end
