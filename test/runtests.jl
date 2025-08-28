using FastRicianLikelihoods
using Test
using Aqua

include("utils.jl")

@testset "FastRicianLikelihoods.jl" begin
    @testset "gauss legendre quadrature" verbose = true begin
        include("gausslegendre.jl")
    end

    @testset "gaussian half-hermite quadrature" verbose = true begin
        include("gausshalfhermite.jl")
    end

    @testset "dual rules" verbose = true begin
        include("forwarddiff.jl")
    end

    @testset "bessels" verbose = true begin
        include("bessels.jl")
    end

    @testset "rician" verbose = true begin
        include("rician.jl")
    end

    @testset "Code quality (Aqua.jl)" begin
        # Typically causes a lot of false positives with ambiguities and/or unbound args checks;
        # unfortunately have to periodically check this manually
        Aqua.test_all(FastRicianLikelihoods; ambiguities = false, unbound_args = false)
    end
end
