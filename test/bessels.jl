module BesselsTests

using Test
using ..Utils: arbify
using ArbNumerics: ArbNumerics, ArbReal
using ForwardDiff: ForwardDiff

using FastRicianLikelihoods: FastRicianLikelihoods,
    besseli2, besseli2x, logbesseli0, logbesseli0x, logbesseli1, logbesseli1x, logbesseli2, logbesseli2x,
    laguerre½, mean_rician, std_rician, besseli1i0, besseli1i0x, besseli1i0m1, ∂x_laguerre½, ∂x_besseli0x, ∂x_besseli1x

function pos_range_iterator(::Type{T}; scale = 10, step = 0.05) where {T <: Union{Float32, Float64}}
    return exp10.((-T(scale)):T(step):T(scale))
end

for T in (Float32, Float64)
    @testset "dual rules ($T)" begin
        for (f̂, k, scaled) in ((FastRicianLikelihoods.besseli0, 0, false), (FastRicianLikelihoods.besseli1, 1, false), (FastRicianLikelihoods.besseli0x, 0, true), (FastRicianLikelihoods.besseli1x, 1, true))
            @testset "$f̂" begin
                for x in T.([-20, -8, -7, -1, -0.1, 0.1, 1, 7, 8, 20])
                    a = ArbReal(x)
                    I₀, I₁, I₂ = ArbNumerics.besseli.((0, 1, 2), a)
                    y = k == 0 ? I₀ : I₁
                    dy = k == 0 ? I₁ : (I₀ + I₂) / 2
                    if scaled
                        dy = exp(-abs(a)) * (dy - sign(a) * y)
                        y *= exp(-abs(a))
                    end
                    @test f̂(x) ≈ T(y) rtol = 4 * eps(T)
                    @test ForwardDiff.derivative(f̂, x) ≈ T(dy) rtol = 8 * eps(T) atol = 4 * eps(T) * abs(T(y))
                end
            end
        end
        for x in -pos_range_iterator(T)
            a = ArbReal(x) / 2
            dy = exp(a) * (ArbNumerics.besseli(1, a) - ArbNumerics.besseli(0, a)) / 2
            @test ForwardDiff.derivative(laguerre½, x) ≈ T(dy) rtol = 4 * eps(T)
        end
    end

    for f̂ in (besseli2, besseli2x, logbesseli0, logbesseli0x, logbesseli1, logbesseli1x, logbesseli2, logbesseli2x)
        rtol = T == Float32 ? 2 * eps(T) : 5 * eps(T)
        atol = T == Float32 ? 2 * eps(T) : 5 * eps(T)
        @testset "$f̂ ($T)" begin
            f = arbify(f̂)
            for x in pos_range_iterator(T)
                ŷ, y = f̂(x), f(x)
                @test ŷ ≈ y rtol = rtol atol = atol
            end
        end
    end

    @testset "laguerre½ ($T)" begin
        f̂ = laguerre½
        f = arbify(f̂)

        # Note: inaccurate for large positive arguments, but we only use it for negative arguments
        xsmall = reverse(-pos_range_iterator(T))
        for x in xsmall
            rtol = 2 * eps(T)
            atol = 2 * eps(T)
            ŷ, y = f̂(x), f(x)
            @test ŷ ≈ y rtol = rtol atol = atol
        end
    end

    @testset "var_rician near ν = 0 and branch points ($T)" begin
        @test FastRicianLikelihoods.var_rician(zero(T)) ≈ T(2 - π / 2) atol = 2 * eps(T)

        function L²c_correction(t)
            L = √(ArbReal(π) / 2) * laguerre½(-t^2 / 2)
            return (L - t) * (L + t) - 1
        end
        L²c = arbify(L²c_correction)

        for t in T.([0.0; 2.0 .* (1 .+ [-1, 1] .* eps(T)); √12 .* (1 .+ [-1, 1] .* eps(T)); 6.0 .* (1 .+ [-1, 1] .* eps(T)); range(0.01, 12.0; length = 400)])
            y = t == 0 ? T(π / 2 - 1) : L²c(t)
            @test FastRicianLikelihoods.laguerre½²c(t) ≈ y atol = 2 * eps(T)
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
                ŷ, y = f̂(ν, σ), f(ν, σ)
                @test ŷ ≈ y rtol = rtol atol = atol
            end
        end
    end

    for f̂ in (besseli1i0, besseli1i0x, besseli1i0m1)
        @testset "$(f̂) ($T)" begin
            f = arbify(f̂)
            for x in pos_range_iterator(T)
                rtol = 2 * eps(T)
                atol = 2 * eps(T)
                ŷ, y = f̂(x), f(x)
                @test ŷ ≈ y rtol = rtol atol = atol
            end
        end
    end
end

end # module BesselsTests

using .BesselsTests
