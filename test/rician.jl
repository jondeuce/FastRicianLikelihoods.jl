module RicianTests

using Test
using ..Utils: arbify, ∇Zyg, ∇Fwd

using ArbNumerics: ArbNumerics, ArbFloat
using Distributions: Normal, logpdf, cdf
using FastRicianLikelihoods: FastRicianLikelihoods, neglogpdf_rician, ∇neglogpdf_rician, neglogcdf_rician, ∇neglogcdf_rician, mean_rician, std_rician, f_quadrature, neglogf_quadrature
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
    ∂x = (neglogcdf_rician(x + ϵ, ν, δ, order) - neglogcdf_rician(x - ϵ, ν, δ, order)) / 2ϵ
    ∂ν = (neglogcdf_rician(x, ν + ϵ, δ, order) - neglogcdf_rician(x, ν - ϵ, δ, order)) / 2ϵ
    ∂δ = (neglogcdf_rician(x, ν, δ + ϵ, order) - neglogcdf_rician(x, ν, δ - ϵ, order)) / 2ϵ
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

@testset "neglogpdf_rician properties" begin
    νs = exp10.(-1.0:0.25:1.0)
    logσs = -0.5:0.25:0.5
    for T in (Float32, Float64)
        @testset "normalization ($T)" begin
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
        @testset "scaling ($T)" begin
            for ν in νs, logσ in logσs
                σ = exp(logσ)
                for x in (ν, √(ν^2 + σ^2), √((ν - σ)^2 + σ^2))
                    @test neglogpdf_rician(x, ν, logσ) ≈ logσ + neglogpdf_rician(x / σ, ν / σ)
                end
            end
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
                @test @inferred(f̂(x, ν)) ≈ f(x, ν) rtol=rtol atol=atol
            end
        end
        @testset "1 <= z ($T)" begin
            rtol = 3*eps(T)
            atol = T == Float32 ? 10*eps(T) : 20*eps(T)
            zs = T.(exp10.([0.0:0.1:0.5; 0.75; 1.0; 2.0:10.0]))
            for z in zs, (x, ν) in xν_iterator(z)
                @test @inferred(f̂(x, ν)) ≈ f(x, ν) rtol=rtol atol=atol
            end
        end
    end
end

@testset "∇neglogpdf_rician" begin
    for T in (Float32, Float64)
        f̂ = neglogpdf_rician
        ∇f̂ = ∇neglogpdf_rician
        ∇f = arbify(∇f̂)
        low, high = T(0.5), T(15.0)
        @testset "z < $(low) ($T)" begin
            rtol = 3*eps(T)
            atol = 3*eps(T)
            zs = range(eps(T), low - eps(T); length = 10)
            for z in zs, (x, ν) in xν_iterator(z)
                ∂ŷ, ∂y = @inferred(∇f̂(x, ν)), ∇f(x, ν)
                @test ∂ŷ[1] ≈ ∂y[1] rtol=rtol atol=atol
                @test ∂ŷ[2] ≈ ∂y[2] rtol=rtol atol=atol
                @test ∂ŷ == ∇Fwd(f̂, x, ν)
                @test ∂ŷ == ∇Zyg(f̂, x, ν)
            end
        end
        @testset "$(low) <= z < $(high) ($T)" begin
            rtol = T == Float32 ? 5*eps(T) : 100*eps(T)
            atol = T == Float32 ? 20*eps(T) : 250*eps(T)
            zs = range(low + 25*eps(T), high - 25*eps(T); length = 10)
            for z in zs, (x, ν) in xν_iterator(z)
                ∂ŷ, ∂y = @inferred(∇f̂(x, ν)), ∇f(x, ν)
                @test ∂ŷ[1] ≈ ∂y[1] rtol=rtol atol=atol
                @test ∂ŷ[2] ≈ ∂y[2] rtol=rtol atol=atol
                @test ∂ŷ == ∇Fwd(f̂, x, ν)
                @test ∂ŷ == ∇Zyg(f̂, x, ν)
            end
        end
        @testset "$(high) <= z ($T)" begin
            rtol = 3*eps(T)
            atol = 3*eps(T)
            zs = (high + 5*eps(T)) .* T.(exp10.([0.0:0.1:0.5; 0.75; 1.0; 2.0:10.0]))
            for z in zs, (x, ν) in xν_iterator(z)
                ∂ŷ, ∂y = @inferred(∇f̂(x, ν)), ∇f(x, ν)
                @test ∂ŷ[1] ≈ ∂y[1] rtol=rtol atol=atol
                @test ∂ŷ[2] ≈ ∂y[2] rtol=rtol atol=atol
                @test ∂ŷ == ∇Fwd(f̂, x, ν)
                @test ∂ŷ == ∇Zyg(f̂, x, ν)
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

function neglogcdf_rician_sum(ν::T, δ::T, order::Val) where {T}
    μx = mean_rician(ν, one(T))
    σx = std_rician(ν, one(T))
    rx = √(-2*log(eps(T))) # solve for exp(-x^2/2) = eps(T)
    N = ceil(Int, (μx + rx * σx) / δ)

    cdf(x̃) = exp(-neglogcdf_rician(x̃, ν, δ, order))
    I = sum(cdf, δ .* (0:N)) # pairwise summation for the bulk of the sum for accuracy
    while true
        N += 1
        I, Ilast = I + cdf(δ * N), I
        I == Ilast && break # stop when contribution of last term is negligible
    end

    return I
end

@testset "neglogcdf_rician properties" begin
    for T in (Float32, Float64)
        νs = exp10.(T[-1.0, -0.1, 0.0, 0.1, 1.0])
        δs = exp10.(T[-2.0, -1.0, 0.0])
        logσs = exp10.(T[-2.0, -1.0, 0.0])
        order = Val(32)
        @testset "normalization ($T)" begin
            for ν in νs, δ in δs
                I = neglogcdf_rician_sum(ν, δ, order)
                atol = T == Float32 ? 6*eps(T) : 4*eps(T)
                @test isapprox(I, one(T); rtol = zero(T), atol)
            end
        end
        @testset "additivity ($T)" begin
            for ν in νs, δ in δs
                cdf(x̃, ν̃, δ̃) = exp(-neglogcdf_rician(x̃, ν̃, δ̃, order))
                for x in δ .* (0, 1, round(Int, ν), round(Int, ν/δ))
                    @test cdf(x, ν, δ) + cdf(x + δ, ν, δ) ≈ cdf(x, ν, 2*δ)
                end
            end
        end
        @testset "scale invariance ($T)" begin
            for ν in νs, δ in δs, logσ in logσs
                σ = exp(logσ)
                for x in δ .* (0, 1, round(Int, ν), round(Int, ν/δ))
                    @test neglogcdf_rician(x, ν, logσ, δ, order) ≈ neglogcdf_rician(x / σ, ν / σ, δ / σ, order)
                end
            end
        end
    end
end

@testset "neglogcdf_rician" begin
    # Unlike the density `neglogpdf_rician`, the integral defining `neglogcdf_rician`
    # is approximated using Gauss-Legendre quadrature of a given `order`. Therefore,
    # `neglogcdf_rician` is not exact in general. However, it should monotonically
    # improve in accuracy as `order` increases.
    for (x, ν, δ) in xνδ_iterator()
        f̂ = neglogcdf_rician
        f = arbify(neglogcdf_rician)
        y = f(x, ν, δ, Val(15))

        for T in (Float32, Float64)
            ŷ4 = @inferred f̂(T(x), T(ν), T(δ), Val(4))
            ŷ8 = @inferred f̂(T(x), T(ν), T(δ), Val(8))
            ŷ16 = @inferred f̂(T(x), T(ν), T(δ), Val(16))
            ŷ32 = @inferred f̂(T(x), T(ν), T(δ), Val(32))

            rtol = 5*eps(T)
            atol = zero(T)
            pass4 = isapprox(y, ŷ4; rtol, atol)
            pass8 = isapprox(y, ŷ8; rtol, atol)
            pass16 = isapprox(y, ŷ16; rtol, atol)
            pass32 = isapprox(y, ŷ32; rtol, atol)

            @test pass4 || pass8 || pass16 || pass32

            if pass4
                @test pass8 && pass16 && pass32
            elseif pass8
                @test pass16 && pass32
                @test abs(y - ŷ8) < abs(y - ŷ4)
            elseif pass16
                @test pass32
                @test abs(y - ŷ16) < abs(y - ŷ8) < abs(y - ŷ4)
            else # pass32
                @test abs(y - ŷ32) < abs(y - ŷ16) < abs(y - ŷ8) < abs(y - ŷ4)
            end
        end
    end
end

@testset "∇neglogcdf_rician" begin
    for (x, ν, δ) in xνδ_iterator()
        f̂ = neglogcdf_rician
        ∇f̂ = ∇neglogcdf_rician
        ∇f = arbify(∇f̂)
        ∂y = ∇f(x, ν, δ, Val(15))

        for T in (Float32, Float64)
            order = Val(64)
            rtol = T == Float32 ? 50*eps(T) : 80*eps(T)
            atol = T == Float32 ? 50*eps(T) : 80*eps(T)

            ∂ŷ = @inferred ∇f̂(T(x), T(ν), T(δ), order)
            @test isapprox(∂ŷ[1], ∂y[1]; rtol, atol)
            @test isapprox(∂ŷ[2], ∂y[2]; rtol, atol)
            @test isapprox(∂ŷ[3], ∂y[3]; rtol, atol)

            @test ∂ŷ == ∇Fwd((xνδ...,) -> f̂(xνδ..., order), T(x), T(ν), T(δ))
            @test ∂ŷ == ∇Zyg((xνδ...,) -> f̂(xνδ..., order), T(x), T(ν), T(δ))
        end
    end
end

@testset "gauss legendre quadrature" begin
    for T in (Float32, Float64)
        d = Normal(randn(T) / 5, 1 + rand(T))
        a, δ = randn(T) / 5, (1 + rand(T)) / 10
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
