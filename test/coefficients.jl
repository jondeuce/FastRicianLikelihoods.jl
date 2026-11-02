module CoefficientsTests

using Test
using DoubleFloats: Double64

include("interactive/coefficients.jl")

# Error bounds for constant/linear interpolation of y², with y = x/(1+x).
function build_search_fixture((a, b), degree, i, n)
    lo = a / (1 + a)
    hi = isinf(b) ? one(b) : b / (1 + b)
    err = degree == 0 ? (hi^2 - lo^2) / 2 : degree == 1 ? (hi - lo)^2 / 4 : zero(a)
    return (; err, degree, domain = (a, b))
end

@testset "approximant search ($T)" for T in (Float32, Float64)
    tolerance = Double64(1e-4)
    for degree in (0, 4)
        result = search_approximants(build_search_fixture, T, (1, 4), degree, tolerance; fixed_left = true, max_degree = 8)
        @test first(result.branches) == one(T)
        @test all(fit -> fit.degree == 2, result.fits)
        @test first(result.fits).domain[1] == 0
        @test last(result.fits).domain[2] == Inf
        @test Tuple(fit.domain[2] for fit in result.fits[1:(end-1)]) == result.branches
    end
end

@testset "domain subdivision and limits" begin
    tolerance = Double64(1e-2)
    result = search_approximants(build_search_fixture, Float64, (1,), 1, tolerance; max_degree = 1, max_branches = 12)
    @test length(result.branches) > 1
    @test all(fit -> fit.degree <= 1 && fit.err <= tolerance, result.fits)
    @test issorted(result.branches)
    @test_throws ErrorException search_approximants(build_search_fixture, Float64, (1, 4), 0, eps(Double64); max_degree = 0)
    @test_throws ErrorException search_approximants(build_search_fixture, Float64, (1, 4), 0, tolerance; max_iterations = 0)
end

end # module CoefficientsTests

using .CoefficientsTests
