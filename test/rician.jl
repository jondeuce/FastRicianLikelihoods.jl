module RicianTests

using Test
using ..Utils: arbify

using ArbNumerics: ArbNumerics, ArbFloat
using Distributions: Normal, logpdf, cdf
using FastRicianLikelihoods: FastRicianLikelihoods, neglogpdf_rician, ∇neglogpdf_rician, neglogcdf_rician, ∇neglogcdf_rician, f_quadrature, neglogf_quadrature
using FiniteDifferences: FiniteDifferences
using QuadGK: quadgk

function FastRicianLikelihoods.neglogpdf_rician(x::ArbFloat, ν::ArbFloat)
    return (x^2 + ν^2) / 2 - log(x) - log(ArbNumerics.besseli(0, x * ν))
end

function FastRicianLikelihoods.∇neglogpdf_rician(x::ArbFloat, ν::ArbFloat)
    ϵ = sqrt(eps(one(ArbFloat)))
    ∂x = (neglogpdf_rician(x + ϵ, ν) - neglogpdf_rician(x - ϵ, ν)) / 2ϵ
    ∂ν = (neglogpdf_rician(x, ν + ϵ) - neglogpdf_rician(x, ν - ϵ)) / 2ϵ
    return (∂x, ∂ν)
end

neglogcdf_rician_arbfloat_eps() = ArbFloat(1e-30)

function FastRicianLikelihoods.neglogcdf_rician(x::ArbFloat, ν::ArbFloat, δ::ArbFloat, ::Val{order}) where {order}
    a, b = ArbFloat(x), ArbFloat(x + δ), ArbFloat(1e-30)
    rtol, atol = neglogcdf_rician_arbfloat_eps(), 0
    I, E = quadgk(a, b; rtol, atol, order) do x̃
        return exp(-neglogpdf_rician(x̃, ν))
    end
    return -log(I)
end

function FastRicianLikelihoods.∇neglogcdf_rician(x::ArbFloat, ν::ArbFloat, δ::ArbFloat, order::Val)
    ϵ = sqrt(neglogcdf_rician_arbfloat_eps())
    ∂x = (neglogpdf_rician(x + ϵ, ν, δ, order) - neglogpdf_rician(x - ϵ, ν, δ, order)) / 2ϵ
    ∂ν = (neglogpdf_rician(x, ν + ϵ, δ, order) - neglogpdf_rician(x, ν - ϵ, δ, order)) / 2ϵ
    ∂δ = (neglogpdf_rician(x, ν, δ + ϵ, order) - neglogpdf_rician(x, ν, δ - ϵ, order)) / 2ϵ
    return (∂x, ∂ν, ∂δ)
end

function FastRicianLikelihoods.neglogpdf_rician(x::ArbFloat, ν::ArbFloat, logσ::ArbFloat)
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end

function FastRicianLikelihoods.neglogcdf_rician(x::ArbFloat, ν::ArbFloat, logσ::ArbFloat, δ::ArbFloat, order::Val)
    σ⁻¹ = exp(-logσ)
    return neglogcdf_rician(σ⁻¹ * x, σ⁻¹ * ν, σ⁻¹ * δ, order)
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

@testset "neglogpdf_rician normalization" begin
    νs = exp10.(-1.0:0.25:1.0)
    logσs = -0.5:0.25:0.5
    for T in (Float32, Float64)
        for ν in νs, logσ in logσs
            ν′ = T(ν / exp(logσ))
            I, E = quadgk(zero(T), ν′/4, ν′/2, ν′, T(Inf); rtol = eps(T), atol = eps(T), order = 15) do x
                return exp(-neglogpdf_rician(x, ν′))
            end
            atol = T == Float32 ? 5*eps(T) : 10*eps(T)
            rtol = zero(T)
            @test isapprox(I, one(T); rtol, atol)
        end
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
            atol = T == Float32 ? 10*eps(T) : 20*eps(T)
            zs = T.(exp10.([0.0:0.1:0.5; 0.75; 1.0; 2.0:10.0]))
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
            rtol = T == Float32 ? 5*eps(T) : 100*eps(T)
            atol = T == Float32 ? 20*eps(T) : 250*eps(T)
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
            zs = (high + 5*eps(T)) .* T.(exp10.([0.0:0.1:0.5; 0.75; 1.0; 2.0:10.0]))
            for z in zs, (x, ν) in xν_iterator(z)
                ∂f̂, ∂f = f̂(x, ν), f(x, ν)
                @test ∂f̂[1] ≈ ∂f[1] rtol=rtol atol=atol
                @test ∂f̂[2] ≈ ∂f[2] rtol=rtol atol=atol
            end
        end
    end
end

function xνδ_iterator()
    xs = exp10.([-1.0, -0.1, 0.0, 0.1, 1.0])
    νs = exp10.([-1.0, -0.1, 0.0, 0.1, 1.0])
    δs = exp10.([-2.0, -1.0, 0.0])
    return Iterators.product(xs, νs, δs)
end

@testset "neglogcdf_rician" begin
    # Unlike the density function `neglogpdf_rician`, the integral defining `neglogcdf_rician`
    # is approximated using Gauss-Legendre quadrature of a given `order`, and therefore
    # `neglogcdf_rician` is not expected to be exact in general
    for (x, ν, δ) in xνδ_iterator()
        y = neglogcdf_rician(map(ArbFloat, (x, ν, δ))..., Val(15))
        for T in (Float32, Float64)
            ŷ4 = @inferred neglogcdf_rician(T(x), T(ν), T(δ), Val(4))
            ŷ8 = @inferred neglogcdf_rician(T(x), T(ν), T(δ), Val(8))
            ŷ16 = @inferred neglogcdf_rician(T(x), T(ν), T(δ), Val(16))
            ŷ32 = @inferred neglogcdf_rician(T(x), T(ν), T(δ), Val(32))

            rtol = 5*eps(T)
            atol = zero(T)
            pass4 = isapprox(y, ŷ4; rtol, atol)
            pass8 = isapprox(y, ŷ8; rtol, atol)
            pass16 = isapprox(y, ŷ16; rtol, atol)
            pass32 = isapprox(y, ŷ32; rtol, atol)

            @test pass4 || pass8 || pass16 || pass32

            if pass4
                # done
            elseif pass8
                @test abs(y - ŷ8) < abs(y - ŷ4)
            elseif pass16
                @test abs(y - ŷ16) < abs(y - ŷ8) < abs(y - ŷ4)
            else # pass32
                @test abs(y - ŷ32) < abs(y - ŷ16) < abs(y - ŷ8) < abs(y - ŷ4)
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
