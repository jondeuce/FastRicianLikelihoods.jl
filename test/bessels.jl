module BesselsTests

using Test
using ..Utils: arbify

using FastRicianLikelihoods: FastRicianLikelihoods,
    besseli2, besseli2x, logbesseli0, logbesseli0x, logbesseli1, logbesseli1x, logbesseli2, logbesseli2x,
    laguerre½, mean_rician, std_rician, besseli1i0, besseli1i0x, besseli1i0m1, ∂x_laguerre½, ∂x_besseli0x, ∂x_besseli1x

function pos_range_iterator(::Type{T}; scale = 10, step = 0.05) where {T <: Union{Float32, Float64}}
    return exp10.(-T(scale):T(step):T(scale))
end

for T in (Float32, Float64)
    for f̂ in (besseli2, besseli2x, logbesseli0, logbesseli0x, logbesseli1, logbesseli1x, logbesseli2, logbesseli2x)
        rtol = 5 * eps(T)
        atol = 5 * eps(T)
        @testset "$f̂ ($T)" begin
            f = arbify(f̂)
            for x in pos_range_iterator(T)
                @test f̂(x) ≈ f(x) rtol = rtol atol = atol
            end
        end
    end

    @testset "laguerre½ ($T)" begin
        f̂ = laguerre½
        f = arbify(f̂)
        xsmall = reverse(-pos_range_iterator(T))
        xlarge = T(0.0):T(0.1):T(T == Float32 ? 50 : 500)
        # TODO: inaccurate for large positive arguments, but we only use it for negative arguments
        for x in xsmall
            rtol = 3 * eps(T)
            atol = 3 * eps(T)
            @test f̂(x) ≈ f(x) rtol = rtol atol = atol
        end
    end

    for f̂ in (mean_rician, std_rician)
        @testset "$(f̂) ($T)" begin
            f = arbify(f̂)
            νs = T.(exp10.(-10.0:0.5:10.0))
            σs = T.(exp10.(-2.0:0.5:2.0))
            rtol = f̂ === mean_rician ? 2 * eps(T) : T == Float32 ? 8 * eps(T) : 5e-13
            atol = f̂ === mean_rician ? 2 * eps(T) : T == Float32 ? 8 * eps(T) : 5e-13
            for ν in νs, σ in σs
                @test f̂(ν, σ) ≈ f(ν, σ) rtol = rtol atol = atol
            end
        end
    end

    for f̂ in (besseli1i0, besseli1i0x, besseli1i0m1)
        @testset "$(f̂) ($T)" begin
            f = arbify(f̂)
            for x in pos_range_iterator(T)
                rtol = 2 * eps(T)
                atol = 2 * eps(T)
                @test f̂(x) ≈ f(x) rtol = rtol atol = atol
            end
        end
    end
end

end # module BesselsTests

using .BesselsTests
