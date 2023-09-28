module BesselsTests

using Test
using ..Utils: arbify

using ArbNumerics: ArbNumerics, ArbFloat
using FastRicianLikelihoods: FastRicianLikelihoods,
    besseli2, besseli2x, logbesseli0, logbesseli0x, logbesseli1, logbesseli1x, logbesseli2, logbesseli2x,
    laguerre½, mean_rician, std_rician, besseli1i0, ∂x_laguerre½, ∂x_besseli0x, ∂x_besseli1x
using FiniteDifferences: FiniteDifferences

FastRicianLikelihoods.besseli2(x::ArbFloat) = ArbNumerics.besseli(2, x)
FastRicianLikelihoods.besseli2x(x::ArbFloat) = exp(-abs(x)) * ArbNumerics.besseli(2, x)
FastRicianLikelihoods.logbesseli0(x::ArbFloat) = log(ArbNumerics.besseli(0, x))
FastRicianLikelihoods.logbesseli0x(x::ArbFloat) = log(ArbNumerics.besseli(0, x)) - abs(x)
FastRicianLikelihoods.logbesseli1(x::ArbFloat) = log(ArbNumerics.besseli(1, x))
FastRicianLikelihoods.logbesseli1x(x::ArbFloat) = log(ArbNumerics.besseli(1, x)) - abs(x)
FastRicianLikelihoods.logbesseli2(x::ArbFloat) = log(ArbNumerics.besseli(2, x))
FastRicianLikelihoods.logbesseli2x(x::ArbFloat) = log(ArbNumerics.besseli(2, x)) - abs(x)
FastRicianLikelihoods.laguerre½(x::ArbFloat) = exp(x / 2) * ((1 - x) * ArbNumerics.besseli(0, -x/2) - x * ArbNumerics.besseli(1, -x/2))
FastRicianLikelihoods.besseli1i0(x::ArbFloat) = ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x)
FastRicianLikelihoods.mean_rician(ν::ArbFloat, σ::ArbFloat) = σ * √(ArbFloat(π) / 2) * laguerre½(-(ν / σ)^2 / 2)
FastRicianLikelihoods.std_rician(ν::ArbFloat, σ::ArbFloat) = sqrt(ν^2 + 2σ^2 - ArbFloat(π) * σ^2 * laguerre½(-(ν / σ)^2 / 2)^2 / 2)
# FastRicianLikelihoods.∂x_laguerre½(x::ArbFloat) = ArbNumerics.∂x_laguerre½(x)
# FastRicianLikelihoods.∂x_besseli0x(x::ArbFloat) = ArbNumerics.∂x_besseli0x(x)
# FastRicianLikelihoods.∂x_besseli1x(x::ArbFloat) = ArbNumerics.∂x_besseli1x(x)

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
                @test f̂(x) ≈ f(x) rtol=rtol atol=atol
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
            @test f̂(x) ≈ f(x) rtol=rtol atol=atol
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
                @test f̂(ν, σ) ≈ f(ν, σ) rtol=rtol atol=atol
            end
        end
    end

    @testset "besseli1i0 ($T)" begin
        f̂ = besseli1i0
        f = arbify(f̂)
        for x in pos_range_iterator(T)
            rtol = T == Float32 ? 2*eps(T) : 5*eps(T)
            atol = T == Float32 ? 2*eps(T) : 15*eps(T)
            @test f̂(x) ≈ f(x) rtol=rtol atol=atol
        end
    end
end

end # module BesselsTests

using .BesselsTests
