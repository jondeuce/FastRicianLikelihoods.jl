module RicianTests

using Test

using ArbNumerics: ArbNumerics, ArbFloat
using FastRicianLikelihoods: FastRicianLikelihoods, neglogpdf_rician, ∇neglogpdf_rician, f_quadrature, neglogf_quadrature
using FiniteDifferences: FiniteDifferences
using Distributions: Normal, logpdf, cdf

ArbNumerics.setworkingprecision(ArbFloat, 500)
ArbNumerics.setextrabits(128)

arbify(f) = function f_arbified(args::T...) where {T <: Union{Float32, Float64}}
    y = f(ArbFloat.(args)...)
    return convert.(T, y)
end

function FastRicianLikelihoods.neglogpdf_rician(x::ArbFloat, ν::ArbFloat)
    return (x^2 + ν^2) / 2 - log(x) - log(ArbNumerics.besseli(0, x * ν))
end

function FastRicianLikelihoods.∇neglogpdf_rician(x::ArbFloat, ν::ArbFloat)
    ϵ = sqrt(eps(one(ArbFloat)))
    ∂x = (neglogpdf_rician(x + ϵ, ν) - neglogpdf_rician(x - ϵ, ν)) / 2ϵ
    ∂ν = (neglogpdf_rician(x, ν + ϵ) - neglogpdf_rician(x, ν - ϵ)) / 2ϵ
    return (∂x, ∂ν)
end

function xν_iterator(z::T) where {T <: Union{Float32, Float64}}
    rmax = T == Float32 ? 6 : 12
    Iterators.map(Iterators.product(-rmax:rmax, (false, true))) do (r, flip)
        δ = exp10(T(r))
        s = flip ? inv(1 + δ) : 1 + δ
        x = √z * s
        ν = z / x
        return (x, ν)
    end
end

@testset "neglogpdf_rician" begin
    for T in (Float32, Float64)
        f̂ = neglogpdf_rician
        f = arbify(f̂)
        @testset "z < 1 ($T)" begin
            rtol = 3*eps(T)
            atol = 3*eps(T)
            zs = range(eps(T), one(T); length = 10)
            for z in zs, (x, ν) in xν_iterator(z)
                @test f̂(x, ν) ≈ f(x, ν) rtol=rtol atol=atol
            end
        end
        @testset "1 <= z ($T)" begin
            rtol = 3*eps(T)
            atol = T == Float32 ? 8*eps(T) : 20*eps(T)
            zs = T.(exp10.(0:10))
            for z in zs, (x, ν) in xν_iterator(z)
                @test f̂(x, ν) ≈ f(x, ν) rtol=rtol atol=atol
            end
        end
    end
end

@testset "∇neglogpdf_rician" begin
    for T in (Float32, Float64)
        f̂ = ∇neglogpdf_rician
        f = arbify(f̂)
        low, high = T(0.5), T(15.0)
        @testset "z < $(low) ($T)" begin
            rtol = 3*eps(T)
            atol = 3*eps(T)
            zs = range(eps(T), low - eps(T); length = 10)
            for z in zs, (x, ν) in xν_iterator(z)
                ∂f̂, ∂f = f̂(x, ν), f(x, ν)
                @test ∂f̂[1] ≈ ∂f[1] rtol=rtol atol=atol
                @test ∂f̂[2] ≈ ∂f[2] rtol=rtol atol=atol
            end
        end
        @testset "$(low) <= z < $(high) ($T)" begin
            rtol = T == Float32 ? 5*eps(T) : 100*eps(T) #2e-11
            atol = T == Float32 ? 20*eps(T) : 250*eps(T) #2e-12
            zs = range(low + 25*eps(T), high - 25*eps(T); length = 10)
            for z in zs, (x, ν) in xν_iterator(z)
                ∂f̂, ∂f = f̂(x, ν), f(x, ν)
                @test ∂f̂[1] ≈ ∂f[1] rtol=rtol atol=atol
                @test ∂f̂[2] ≈ ∂f[2] rtol=rtol atol=atol
            end
        end
        @testset "$(high) <= z ($T)" begin
            rtol = 3*eps(T)
            atol = 3*eps(T)
            zs = (high + 5*eps(T)) .* T.(exp10.(0:10))
            for z in zs, (x, ν) in xν_iterator(z)
                ∂f̂, ∂f = f̂(x, ν), f(x, ν)
                @test ∂f̂[1] ≈ ∂f[1] rtol=rtol atol=atol
                @test ∂f̂[2] ≈ ∂f[2] rtol=rtol atol=atol
            end
        end
    end
end

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
