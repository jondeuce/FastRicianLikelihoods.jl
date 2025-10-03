module RicianTests

using Test
using ..Utils: Utils, arbify, ∇FD_central, ∇FD_forward, ∇Fwd, ∇Zyg

using FastRicianLikelihoods: FastRicianLikelihoods, Distributions, StaticArrays, mean_rician, std_rician,
    neglogpdf_rician, ∇neglogpdf_rician, ∇²neglogpdf_rician, ∇²neglogpdf_rician_with_gradient, ∇³neglogpdf_rician_with_gradient_and_hessian,
    neglogpdf_qrician, ∇neglogpdf_qrician, ∇²neglogpdf_qrician, ∇²neglogpdf_qrician_with_gradient, ∇²neglogpdf_qrician_with_primal_and_gradient
using .StaticArrays: StaticArrays, SVector, SMatrix, @SVector, @SMatrix
using QuadGK: QuadGK, quadgk

DEBUG = Ref(true)

const HIGH_ORDER = Val(32)

function xν_iterator(z::T) where {T <: Union{Float32, Float64}}
    rmax = T == Float32 ? 6 : 12
    return Iterators.flatten((
        [(√z, √z)],
        Iterators.map(Iterators.product(-rmax:rmax, (false, true))) do (r, flip)
            δ = exp10(T(r))
            s = flip ? inv(1 + δ) : 1 + δ
            x = √z * s
            ν = z / x
            return (x, ν)
        end,
    ))
end

@testset "neglogpdf_rician properties" begin
    νs = exp10.(-1.0:0.25:1.0)
    logσs = -0.5:0.25:0.5
    for T in (Float32, Float64)
        @testset "normalization ($T)" begin
            for ν in νs, logσ in logσs
                ν′ = T(ν / exp(logσ))
                I, E = quadgk(zero(T), ν′ / 4, ν′ / 2, ν′, T(Inf); rtol = eps(T), atol = eps(T), order = 15) do x
                    return exp(-neglogpdf_rician(x, ν′))
                end
                atol = T == Float32 ? 15 * eps(T) : 10 * eps(T)
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
            rtol = 3 * eps(T)
            atol = 3 * eps(T)
            zs = range(eps(T), one(T); length = 10)
            for z in zs, (x, ν) in xν_iterator(z)
                ŷ, y = @inferred(f̂(x, ν)), f(x, ν)
                @test ŷ ≈ y rtol = rtol atol = atol
            end
        end
        @testset "1 <= z ($T)" begin
            rtol = 3 * eps(T)
            atol = 3 * eps(T)
            zs = T.(exp10.([0.0:0.1:0.5; 0.75; 1.0; 2.0:10.0]))
            for z in zs, (x, ν) in xν_iterator(z)
                ŷ, y = @inferred(f̂(x, ν)), f(x, ν)
                @test ŷ ≈ y rtol = rtol atol = atol
            end
        end
    end
end

@testset "∇neglogpdf_rician" begin
    for T in (Float32, Float64)
        f̂ = neglogpdf_rician
        ∇f̂ = ∇neglogpdf_rician
        ∇f = arbify(∇f̂)
        low, high = extrema(FastRicianLikelihoods.neglogpdf_rician_parts_branches(T))
        rtol = 3 * eps(T)
        atol = 3 * eps(T)
        @testset "z < $(low) ($T)" begin
            zs = range(eps(T), low - eps(T); length = 10)
            for z in zs, (x, ν) in xν_iterator(z)
                ∂ŷ, ∂y = @inferred(∇f̂(x, ν)), ∇f(x, ν)
                @test ∂ŷ[1] ≈ ∂y[1] rtol = rtol atol = atol
                @test ∂ŷ[2] ≈ ∂y[2] rtol = rtol atol = atol
                @test ∂ŷ == @inferred ∇Fwd(f̂, x, ν)
                @test ∂ŷ == @inferred ∇Zyg(f̂, x, ν)
            end
        end
        @testset "$(low) <= z < $(high) ($T)" begin
            zs = range(low + eps(T), high * (1 - eps(T)); length = 10)
            for z in zs, (x, ν) in xν_iterator(z)
                ∂ŷ, ∂y = @inferred(∇f̂(x, ν)), ∇f(x, ν)
                @test ∂ŷ[1] ≈ ∂y[1] rtol = rtol atol = atol
                @test ∂ŷ[2] ≈ ∂y[2] rtol = rtol atol = atol
                @test ∂ŷ == @inferred ∇Fwd(f̂, x, ν)
                @test ∂ŷ == @inferred ∇Zyg(f̂, x, ν)
            end
        end
        @testset "z >= $(high) ($T)" begin
            zs = high .* (1 + eps(T)) .* T.(exp10.([0.0:0.1:0.5; 0.75; 1.0:10.0]))
            for z in zs, (x, ν) in xν_iterator(z)
                ∂ŷ, ∂y = @inferred(∇f̂(x, ν)), ∇f(x, ν)
                @test ∂ŷ[1] ≈ ∂y[1] rtol = rtol atol = atol
                @test ∂ŷ[2] ≈ ∂y[2] rtol = rtol atol = atol
                @test ∂ŷ == @inferred ∇Fwd(f̂, x, ν)
                @test ∂ŷ == @inferred ∇Zyg(f̂, x, ν)
            end
        end
    end
end

@testset "∇²neglogpdf_rician" begin
    for T in (Float32, Float64)
        ∇²f̂ = ∇²neglogpdf_rician
        ∇²f = arbify(∇²f̂)

        rtol = 3 * eps(T)
        atol = 3 * eps(T)
        zs = T.(exp10.(-3:10))
        for z in zs, (x, ν) in xν_iterator(z)
            ∇²ŷ, ∇²y = @inferred(∇²f̂(x, ν)), ∇²f(x, ν)
            @test ∇²ŷ[1] ≈ ∇²y[1] rtol = rtol atol = atol
            @test ∇²ŷ[2] ≈ ∇²y[2] rtol = rtol atol = atol
            @test ∇²ŷ[3] ≈ ∇²y[3] rtol = rtol atol = atol
        end
    end
end

@testset "∇³neglogpdf_rician_with_gradient_and_hessian" begin
    for T in (Float32, Float64)
        ∇³f̂ = ∇³neglogpdf_rician_with_gradient_and_hessian
        ∇³f = arbify(∇³f̂)

        rtol = T == Float32 ? 4 * eps(T) : 5 * eps(T)
        atol = T == Float32 ? 4 * eps(T) : 5 * eps(T)
        zs = T.(exp10.(-3:10))
        for z in zs, (x, ν) in xν_iterator(z)
            (∇ŷ, ∇²ŷ, ∇³ŷ), (_, _, ∇³y) = @inferred(∇³f̂(x, ν)), ∇³f(x, ν)

            # Third derivative components should be approximately equal to the high precision reference
            @test ∇³ŷ[1] ≈ ∇³y[1] rtol = rtol atol = atol
            @test ∇³ŷ[2] ≈ ∇³y[2] rtol = rtol atol = atol
            @test ∇³ŷ[3] ≈ ∇³y[3] rtol = rtol atol = atol
            @test ∇³ŷ[4] ≈ ∇³y[4] rtol = rtol atol = atol
        end
    end
end

@testset "rician lower-order results consistency" begin
    for T in (Float32, Float64)
        rtol = eps(T)
        atol = eps(T)

        ∇f̂ = ∇neglogpdf_rician
        ∇²f̂ = ∇²neglogpdf_rician
        ∇²f̂_with_grad = ∇²neglogpdf_rician_with_gradient
        ∇³f̂_with_grad_and_hess = ∇³neglogpdf_rician_with_gradient_and_hessian

        zs = T.(exp10.(-3:10))
        for z in zs, (x, ν) in xν_iterator(z)
            ∇ŷ = @inferred ∇f̂(x, ν)
            ∇²ŷ = @inferred ∇²f̂(x, ν)

            ∇ŷ₂, ∇²ŷ₂ = @inferred ∇²f̂_with_grad(x, ν)
            @test isapprox(SVector(∇ŷ₂), SVector(∇ŷ); rtol, atol)
            @test isapprox(SVector(∇²ŷ₂), SVector(∇²ŷ); rtol, atol)

            ∇ŷ₃, ∇²ŷ₃, _ = @inferred ∇³f̂_with_grad_and_hess(x, ν)
            @test isapprox(SVector(∇ŷ₃), SVector(∇ŷ); rtol, atol)
            @test isapprox(SVector(∇²ŷ₃), SVector(∇²ŷ); rtol, atol)

            @test isapprox(SVector(∇ŷ₂), SVector(∇ŷ₃); rtol, atol)
            @test isapprox(SVector(∇²ŷ₂), SVector(∇²ŷ₃); rtol, atol)
        end
    end
end

function sample_xνδ(ν, δ; dequantize = true)
    # We only expect the `neglogpdf_qrician`-related functions to be accurate near the mode of the distribution,
    # so we sample y ~ Rice(ν, 1) for each ν such that y is near the mode by construction.
    y = √(randn()^2 + (ν + randn())^2)

    # Round y down to the nearest multiple of δ, giving us x ~ QRice(ν, 1, δ)
    x = δ * floor(y / δ)

    if dequantize
        # Add uniform noise to dequantize x
        x += δ * rand()
    end

    return (x, ν, δ)
end

function xνδ_iterator(; dequantize = true)
    νs = exp10.(-1:0.25:3)
    δs = exp10.(-2:0.25:0)
    return (sample_xνδ(ν, δ; dequantize) for ν in νs, δ in δs)
end

function neglogpdf_qrician_sum(ν::T, δ::T, order::Val) where {T}
    μx = mean_rician(ν, one(T))
    σx = std_rician(ν, one(T))
    rx = √(-2 * log(eps(T))) # solve for exp(-x^2/2) = eps(T)
    N = ceil(Int, (μx + rx * σx) / δ)

    pdf(x̃) = exp(-neglogpdf_qrician(x̃, ν, δ, order))
    I = sum(pdf, δ .* (0:N)) # pairwise summation for the bulk of the sum for accuracy
    while true
        N += 1
        I, Ilast = I + pdf(δ * N), I
        I == Ilast && break # stop when contribution of last term is negligible
    end

    return I
end

@testset "neglogpdf_qrician properties" begin
    for T in (Float32, Float64)
        ns = [0, 1, 2, 5, 10, 100, 1000]
        νs = exp10.(T[-1.0, -0.1, 0.0, 0.1, 1.0])
        δs = exp10.(T[-2.0, -1.0, 0.0])
        logσs = exp10.(T[-2.0, -1.0, 0.0])
        order = HIGH_ORDER
        @testset "normalization ($T)" begin
            for ν in νs, δ in δs
                I = neglogpdf_qrician_sum(ν, δ, order)
                atol = T == Float32 ? 6 * eps(T) : 5 * eps(T)
                @test isapprox(I, one(T); rtol = zero(T), atol)
            end
        end
        @testset "additivity ($T)" begin
            for ν in νs, δ in δs
                pdf(x̃, ν̃, δ̃) = exp(-neglogpdf_qrician(x̃, ν̃, δ̃, order))
                for x in δ .* (0, 1, round(Int, ν), round(Int, ν / δ))
                    @test pdf(x, ν, δ) + pdf(x + δ, ν, δ) ≈ pdf(x, ν, 2 * δ)
                end
            end
        end
        @testset "scale invariance ($T)" begin
            for ν in νs, δ in δs, logσ in logσs
                σ = exp(logσ)
                for x in δ .* (0, 1, round(Int, ν), round(Int, ν / δ))
                    @test neglogpdf_qrician(x, ν, logσ, δ, order) ≈ neglogpdf_qrician(x / σ, ν / σ, δ / σ, order)
                end
            end
        end
        @testset "discrete input ($T)" begin
            ν = rand(νs)
            δ = rand(δs)
            logσ = rand(logσs)
            for n in ns
                x = n * δ
                @test neglogpdf_qrician(x, ν, logσ, δ, order) == neglogpdf_qrician(n, ν, logσ, δ, order)
            end
        end
        @testset "midpoint rule fastpath" begin
            for (f1, f2) in [
                FastRicianLikelihoods._neglogpdf_qrician_midpoint => FastRicianLikelihoods._neglogpdf_qrician,
                FastRicianLikelihoods._∇neglogpdf_qrician_midpoint => FastRicianLikelihoods._∇neglogpdf_qrician,
                FastRicianLikelihoods._∇²neglogpdf_qrician_midpoint => FastRicianLikelihoods._∇²neglogpdf_qrician,
                FastRicianLikelihoods._∇²neglogpdf_qrician_midpoint_with_gradient => FastRicianLikelihoods._∇²neglogpdf_qrician_with_gradient,
                FastRicianLikelihoods._∇²neglogpdf_qrician_midpoint_with_primal_and_gradient => FastRicianLikelihoods._∇²neglogpdf_qrician_with_primal_and_gradient,
            ]
                for ν in νs, δ in δs, x in δ .* (0, 1, round(Int, ν), round(Int, ν / δ))
                    y1 = f1(x, ν, δ)
                    y2 = f2(Float64.((x, ν, δ))..., Val(1))
                    rtol = T == Float32 ? 5.0f-5 : 5e-11
                    atol = T == Float32 ? 5.0f-5 : 5e-11
                    @test all(map((out1, out2) -> all(isapprox.(out1, out2; atol, rtol)), y1, y2))
                end
            end
        end
    end
end

@testset "neglogpdf_qrician" begin
    f̂ = neglogpdf_qrician
    f = arbify(neglogpdf_qrician)

    @testset "gausslegendre ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator(), order in (Val(2), Val(3), Val(4))
            rtol = T == Float32 ? 3 * eps(T) : 3 * eps(T)
            atol = T == Float32 ? 3 * eps(T) : 3 * eps(T)

            y = f(T(x), T(ν), T(δ), order; method = :gausslegendre)
            ŷ = @inferred f̂(T(x), T(ν), T(δ), order)
            @test isapprox(ŷ, y; rtol, atol)
        end
    end

    @testset "analytic ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator()
            # Compare high-order approximate integral with analytic integral
            rtol = T == Float32 ? 3 * eps(T) : 3 * eps(T)
            atol = T == Float32 ? 3 * eps(T) : 3 * eps(T)

            y = f(T(x), T(ν), T(δ), Val(nothing); method = :analytic)
            ŷ = @inferred f̂(T(x), T(ν), T(δ), HIGH_ORDER)
            @test isapprox(ŷ, y; rtol, atol)
        end
    end
end

@testset "∇neglogpdf_qrician" begin
    f̂ = neglogpdf_qrician
    ∇f̂ = ∇neglogpdf_qrician
    ∇f = arbify(∇f̂)

    @testset "gausslegendre ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator(), order in (Val(2), Val(3), Val(4))
            # Compare gradients of approximate integral at different orders
            rtol = T == Float32 ? 5 * eps(T) : 5 * eps(T)
            atol = T == Float32 ? 5 * eps(T) : 5 * eps(T)

            ∂y = ∇f(T(x), T(ν), T(δ), order; method = :gausslegendre)
            ∂ŷ = @inferred ∇f̂(T(x), T(ν), T(δ), order)
            @test isapprox(∂ŷ[1], ∂y[1]; rtol, atol)
            @test isapprox(∂ŷ[2], ∂y[2]; rtol, atol)
            @test isapprox(∂ŷ[3], ∂y[3]; rtol, atol)
        end
    end

    @testset "autodiff methods ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator(), order in (Val(1), Val(2))
            # Check that autodiff methods are being called
            ∂ŷ = @inferred ∇f̂(T(x), T(ν), T(δ), order)
            @test ∂ŷ == @inferred ∇Fwd((xνδ...,) -> f̂(xνδ..., order), T(x), T(ν), T(δ)) # calling `f` with `Dual` args should call `∇f`
            @test ∂ŷ == ∇Zyg((xνδ...,) -> f̂(xνδ..., order), T(x), T(ν), T(δ)) # check that Zygote calls `ChainRules.rrule` which should call `∇f`
            @test_throws ErrorException @inferred ∇Zyg((xνδ...,) -> f̂(xνδ..., order), T(x), T(ν), T(δ))
        end
    end

    @testset "analytic ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator()
            # Compare gradients of high-order approximate integral with analytic gradient
            rtol = T == Float32 ? 5 * eps(T) : 5 * eps(T)
            atol = T == Float32 ? 5 * eps(T) : 5 * eps(T)

            ∂y = ∇f(T(x), T(ν), T(δ), Val(nothing); method = :analytic)
            ∂ŷ = @inferred ∇f̂(T(x), T(ν), T(δ), HIGH_ORDER)
            @test isapprox(∂ŷ[1], ∂y[1]; rtol, atol)
            @test isapprox(∂ŷ[2], ∂y[2]; rtol, atol)
            @test isapprox(∂ŷ[3], ∂y[3]; rtol, atol)
        end
    end
end

@testset "∇²neglogpdf_qrician" begin
    ∇²f̂ = ∇²neglogpdf_qrician
    ∇²f = arbify(∇²f̂)

    @testset "gausslegendre ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator(), order in (Val(2), Val(3), Val(4))
            rtol = T == Float32 ? 15 * eps(T) : 25 * eps(T)
            atol = T == Float32 ? 15 * eps(T) : 25 * eps(T)

            H = ∇²f(T(x), T(ν), T(δ), order; method = :gausslegendre)
            Ĥ = @inferred ∇²f̂(T(x), T(ν), T(δ), order)

            @test isapprox(Ĥ[1], H[1]; rtol, atol)
            @test isapprox(Ĥ[2], H[2]; rtol, atol)
            @test isapprox(Ĥ[3], H[3]; rtol, atol)
            @test isapprox(Ĥ[4], H[4]; rtol, atol)
            @test isapprox(Ĥ[5], H[5]; rtol, atol)
            @test isapprox(Ĥ[6], H[6]; rtol, atol)
        end
    end

    @testset "analytic ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator()
            rtol = T == Float32 ? 15 * eps(T) : 25 * eps(T)
            atol = T == Float32 ? 15 * eps(T) : 25 * eps(T)

            H = ∇²f(T(x), T(ν), T(δ), Val(nothing); method = :analytic)
            Ĥ = @inferred ∇²f̂(T(x), T(ν), T(δ), HIGH_ORDER)

            @test isapprox(Ĥ[1], H[1]; rtol, atol)
            @test isapprox(Ĥ[2], H[2]; rtol, atol)
            @test isapprox(Ĥ[3], H[3]; rtol, atol)
            @test isapprox(Ĥ[4], H[4]; rtol, atol)
            @test isapprox(Ĥ[5], H[5]; rtol, atol)
            @test isapprox(Ĥ[6], H[6]; rtol, atol)
        end
    end
end

@testset "∇³neglogpdf_qrician jacobian and vjp" begin
    ∇³f̂ = FastRicianLikelihoods.∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian
    ∇³f = arbify(∇³f̂)

    @testset "gausslegendre and analytic ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator(), order in (Val(2), Val(3), Val(4), HIGH_ORDER)
            rtol = T == Float32 ? 15 * eps(T) : 25 * eps(T)
            atol = T == Float32 ? 15 * eps(T) : 25 * eps(T)

            if order === HIGH_ORDER
                # Compare high-order quadrature with analytic integral
                ref_method, ref_order = :analytic, Val(nothing)
            else
                # Compare low-order quadrature with low-order quadrature
                ref_method, ref_order = :gausslegendre, order
            end

            y1, g1, H1, J1 = ∇³f(T(x), T(ν), T(δ), ref_order; method = ref_method)
            y2, g2, H2, J2 = ∇³f̂(T(x), T(ν), T(δ), order)

            @test isapprox(y1, y2; rtol, atol)
            @test all(isapprox.(g1, g2; rtol, atol))
            @test all(isapprox.(H1, H2; rtol, atol))
            @test all(isapprox.(J1, J2; rtol, atol))

            # Vector Jacobian product of flattened Hessian of `neglogpdf_qrician`
            Δ = @SVector randn(T, 6)
            vjp1 = J1' * Δ
            y3, g3, H3, vjp3 = FastRicianLikelihoods.∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian(Δ, T(x), T(ν), T(δ), order)

            @test isapprox(y1, y3; rtol, atol)
            @test all(isapprox.(g1, g3; rtol, atol))
            @test all(isapprox.(H1, H3; rtol, atol))
            @test all(isapprox.(vjp1, vjp3; rtol, atol))
        end
    end
end

@testset "∇³neglogpdf_qrician jacobian and vjp (jet)" begin
    @testset "inner jacobian and vjp vs. AD (jet) ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator()
            Tad = Float64
            rtol = T === Float32 ? sqrt(eps(T)) : eps(T)^(2 // 3)
            atol = T === Float32 ? sqrt(eps(T)) : eps(T)^(2 // 3)

            # Jacobian of flattened jet of `neglogpdf_qrician` integrand
            t = rand(T)
            ϕ1, Jϕ1 = FastRicianLikelihoods._∇³neglogpdf_qrician_inner_jacobian_with_jet_ad(Tad.((T(x), T(ν), T(δ), T(t)))...)
            ϕ2, Jϕ2 = FastRicianLikelihoods._∇³neglogpdf_qrician_inner_jacobian_with_jet(T(x), T(ν), T(δ), T(t))

            @test isapprox(ϕ1, ϕ2; rtol, atol)
            @test isapprox(Jϕ1, Jϕ2; rtol, atol)

            # Vector Jacobian product of flattened jet of `neglogpdf_qrician` integrand
            Δϕ = @SVector randn(T, 9)
            ϕ3, vjpϕ3 = FastRicianLikelihoods._∇³neglogpdf_qrician_inner_vjp_with_jet(Δϕ, T(x), T(ν), T(δ), T(t))

            @test isapprox(ϕ3, ϕ1; rtol, atol)
            @test isapprox(vjpϕ3, Jϕ1' * Δϕ; rtol, atol)
        end
    end

    @testset "full jacobian and vjp vs. AD (jet) ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator(), order in (Val(2), Val(3), Val(4))
            Tad = Float64
            rtol = T === Float32 ? sqrt(eps(T)) : eps(T)^(2 // 3)
            atol = T === Float32 ? sqrt(eps(T)) : eps(T)^(2 // 3)

            # Jacobian of flattened jet of `neglogpdf_qrician`
            Φ1, JΦ1 = FastRicianLikelihoods._∇³neglogpdf_qrician_jacobian_with_jet_ad(Tad.((T(x), T(ν), T(δ)))..., order)
            Φ2, JΦ2 = FastRicianLikelihoods._∇³neglogpdf_qrician_jacobian_with_jet(T(x), T(ν), T(δ), order)

            @test isapprox(Φ1, Φ2; rtol, atol)
            @test isapprox(JΦ1, JΦ2; rtol, atol)

            # Vector Jacobian product of flattened jet of `neglogpdf_qrician`
            Δ = @SVector randn(T, 9)
            Φ4, vjpΦ4 = FastRicianLikelihoods._∇³neglogpdf_qrician_vjp_with_jet_from_parts(Δ, T(x), T(ν), T(δ), order)
            Φ5, vjpΦ5 = FastRicianLikelihoods._∇³neglogpdf_qrician_vjp_with_jet_two_pass(Δ, T(x), T(ν), T(δ), order)
            Φ6, vjpΦ6 = FastRicianLikelihoods._∇³neglogpdf_qrician_vjp_with_jet_one_pass(Δ, T(x), T(ν), T(δ), order)

            @test isapprox(Φ1, Φ4; rtol, atol)
            @test isapprox(Φ1, Φ5; rtol, atol)
            @test isapprox(Φ1, Φ6; rtol, atol)
            @test isapprox(vjpΦ4, vjpΦ5; rtol, atol)
            @test isapprox(vjpΦ4, vjpΦ6; rtol, atol)
            @test isapprox(JΦ1' * Δ, vjpΦ4; rtol, atol)
            @test isapprox(JΦ2' * Δ, vjpΦ5; rtol, atol)
            @test isapprox(JΦ2' * Δ, vjpΦ6; rtol, atol)
        end
    end
end

@testset "quantized rician lower-order results consistency" begin
    for T in (Float32, Float64)
        rtol = 2 * eps(T)
        atol = 2 * eps(T)

        for (x, ν, δ) in xνδ_iterator(), order in (Val(2), Val(3), Val(4))
            ŷ = @inferred neglogpdf_qrician(x, ν, δ, order)
            ∇ŷ = @inferred ∇neglogpdf_qrician(x, ν, δ, order)
            ∇²ŷ = @inferred ∇²neglogpdf_qrician(x, ν, δ, order)

            ∇ŷ₂, ∇²ŷ₂ = @inferred ∇²neglogpdf_qrician_with_gradient(x, ν, δ, order)
            @test isapprox(SVector(∇ŷ₂), SVector(∇ŷ); rtol, atol)
            @test isapprox(SVector(∇²ŷ₂), SVector(∇²ŷ); rtol, atol) #TODO rare failures

            ŷ₃, ∇ŷ₃, ∇²ŷ₃ = @inferred ∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ, order)
            @test isapprox(ŷ₃, ŷ; rtol, atol)
            @test isapprox(SVector(∇ŷ₃), SVector(∇ŷ); rtol, atol)
            @test isapprox(SVector(∇²ŷ₃), SVector(∇²ŷ); rtol, atol) #TODO rare failures
        end
    end
end

end # module RicianTests

using .RicianTests
