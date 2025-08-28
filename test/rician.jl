module RicianTests

using Test
using ..Utils: arbify, ∇FD_central, ∇FD_forward, ∇Fwd, ∇Zyg

using FastRicianLikelihoods: FastRicianLikelihoods, Distributions, StaticArrays
using FastRicianLikelihoods: mean_rician, std_rician,
    neglogpdf_rician, ∇neglogpdf_rician, ∇²neglogpdf_rician, ∇²neglogpdf_rician_with_gradient, ∇³neglogpdf_rician_with_gradient_and_hessian,
    neglogpdf_qrician, ∇neglogpdf_qrician, ∇²neglogpdf_qrician, ∇²neglogpdf_qrician_with_gradient, ∇²neglogpdf_qrician_with_primal_and_gradient
using .StaticArrays: SVector, SMatrix, @SVector, @SMatrix
using QuadGK: quadgk

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
            ∂ŷ, ∂y = @inferred(∇²f̂(x, ν)), ∇²f(x, ν)
            @test ∂ŷ[1] ≈ ∂y[1] rtol = rtol atol = atol
            @test ∂ŷ[2] ≈ ∂y[2] rtol = rtol atol = atol
            @test ∂ŷ[3] ≈ ∂y[3] rtol = rtol atol = atol
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

function xνδ_iterator()
    function sample_xνδ(ν, δ)
        # We only expect the `neglogpdf_qrician`-related functions to be accurate near the mode of the distribution,
        # so we sample y ~ Rice(ν, 1) for each ν such that y is near the mode by construction.
        y = √(randn()^2 + (ν + randn())^2)

        # Integrate over the interval [y - δ/2, y + δ/2] ∩ [0, ∞)
        return y > δ / 2 ? (y - δ / 2, ν, δ) : (0.0, ν, y + δ / 2)
    end

    νs = exp10.(-1:0.25:3)
    δs = exp10.(-2:0.25:0)
    return (sample_xνδ(ν, δ) for ν in νs, δ in δs)
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
        order = Val(16)
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
                    atol = rtol = T == Float32 ? 5.0f-5 : 5e-11
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
            high_order = Val(32)

            y = f(T(x), T(ν), T(δ), Val(nothing); method = :analytic)
            ŷ = @inferred f̂(T(x), T(ν), T(δ), high_order)
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
            high_order = Val(32)

            ∂y = ∇f(T(x), T(ν), T(δ), Val(nothing); method = :finitediff)
            ∂ŷ = @inferred ∇f̂(T(x), T(ν), T(δ), high_order)
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
            rtol = T == Float32 ? 10 * eps(T) : 50 * eps(T)
            atol = T == Float32 ? 10 * eps(T) : 50 * eps(T)

            H = ∇²f(T(x), T(ν), T(δ), order; method=:gausslegendre)
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
            rtol = T == Float32 ? 10 * eps(T) : 50 * eps(T)
            atol = T == Float32 ? 10 * eps(T) : 50 * eps(T)
            high_order = Val(32)

            H = ∇²f(T(x), T(ν), T(δ), Val(nothing); method=:finitediff)
            Ĥ = @inferred ∇²f̂(T(x), T(ν), T(δ), high_order)

            @test isapprox(Ĥ[1], H[1]; rtol, atol)
            @test isapprox(Ĥ[2], H[2]; rtol, atol)
            @test isapprox(Ĥ[3], H[3]; rtol, atol)
            @test isapprox(Ĥ[4], H[4]; rtol, atol)
            @test isapprox(Ĥ[5], H[5]; rtol, atol)
            @test isapprox(Ĥ[6], H[6]; rtol, atol)
        end
    end
end

@testset "∇²neglogpdf_qrician jacobian and jvp" begin
    @testset "inner jacobian and jvp ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator()
            rtol = sqrt(eps(T))
            atol = sqrt(eps(T))

            # Jacobian of `∇²neglogpdf_qrician` integrand
            t = rand(T)
            ϕ1, Jϕ1 = FastRicianLikelihoods._∇²neglogpdf_qrician_inner_jac_ad(T.((x, ν, δ, t))...)
            ϕ2, Jϕ2 = FastRicianLikelihoods._∇²neglogpdf_qrician_inner_jac(T(x), T(ν), T(δ), T(t))

            @test isapprox(ϕ1, ϕ2; rtol, atol)
            @test isapprox(Jϕ1, Jϕ2; rtol, atol)

            # Jacobian vector product of `∇²neglogpdf_qrician` integrand
            Δϕ = @SVector randn(T, 9)
            ϕ3, jvpϕ3 = FastRicianLikelihoods._∇²neglogpdf_qrician_inner_jvp(Δϕ, T(x), T(ν), T(δ), T(t))

            @test isapprox(ϕ3, ϕ1; rtol, atol)
            @test isapprox(jvpϕ3, Jϕ1' * Δϕ; rtol, atol)
        end
    end

    @testset "full jacobian and jvp ($T)" for T in (Float32, Float64)
        for (x, ν, δ) in xνδ_iterator(), order in (Val(2), Val(3), Val(4))
            rtol = sqrt(eps(T))
            atol = sqrt(eps(T))

            # Jacobian of flattened Hessian of `neglogpdf_qrician`
            Φ1, J1 = FastRicianLikelihoods._∇²neglogpdf_qrician_with_jacobian_ad(T(x), T(ν), T(δ), order)
            Φ2, J2 = FastRicianLikelihoods._∇²neglogpdf_qrician_with_jacobian(T(x), T(ν), T(δ), order)

            @test isapprox(Φ1, Φ2; rtol, atol)
            @test isapprox(J1, J2; rtol, atol)

            # Jacobian vector product of flattened Hessian of `neglogpdf_qrician`
            Δ = @SVector randn(T, 9)
            Φ3, jvpΦ3 = FastRicianLikelihoods._∇²neglogpdf_qrician_jvp_ad(Δ, T(x), T(ν), T(δ), order)
            Φ4, jvpΦ4 = FastRicianLikelihoods._∇²neglogpdf_qrician_jvp_via_jac_parts(Δ, T(x), T(ν), T(δ), order)
            Φ5, jvpΦ5 = FastRicianLikelihoods._∇²neglogpdf_qrician_jvp_via_two_pass(Δ, T(x), T(ν), T(δ), order)
            Φ6, jvpΦ6 = FastRicianLikelihoods._∇²neglogpdf_qrician_jvp_via_one_pass(Δ, T(x), T(ν), T(δ), order)

            @test isapprox(Φ1, Φ3; rtol, atol)
            @test isapprox(Φ1, Φ4; rtol, atol)
            @test isapprox(Φ1, Φ5; rtol, atol)
            @test isapprox(Φ1, Φ6; rtol, atol)
            @test isapprox(jvpΦ3, jvpΦ4; rtol, atol)
            @test isapprox(jvpΦ3, jvpΦ5; rtol, atol)
            @test isapprox(jvpΦ3, jvpΦ6; rtol, atol)
            @test isapprox(J1' * Δ, jvpΦ3; rtol, atol)
            @test isapprox(J2' * Δ, jvpΦ3; rtol, atol)
        end
    end
end

@testset "quantized rician lower-order results consistency" begin
    for T in (Float32, Float64)
        rtol = eps(T)
        atol = eps(T)

        for (x, ν, δ) in xνδ_iterator(), order in (Val(2), Val(3), Val(4))
            ŷ = @inferred neglogpdf_qrician(x, ν, δ, order)
            ∇ŷ = @inferred ∇neglogpdf_qrician(x, ν, δ, order)
            ∇²ŷ = @inferred ∇²neglogpdf_qrician(x, ν, δ, order)

            ∇ŷ₂, ∇²ŷ₂ = @inferred ∇²neglogpdf_qrician_with_gradient(x, ν, δ, order)
            @test isapprox(SVector(∇ŷ₂), SVector(∇ŷ); rtol, atol)
            @test isapprox(SVector(∇²ŷ₂), SVector(∇²ŷ); rtol, atol)

            ŷ₃, ∇ŷ₃, ∇²ŷ₃ = @inferred ∇²neglogpdf_qrician_with_primal_and_gradient(x, ν, δ, order)
            @test isapprox(ŷ₃, ŷ; rtol, atol)
            @test isapprox(SVector(∇ŷ₃), SVector(∇ŷ); rtol, atol)
            @test isapprox(SVector(∇²ŷ₃), SVector(∇²ŷ); rtol, atol)
        end
    end
end

end # module RicianTests

using .RicianTests
