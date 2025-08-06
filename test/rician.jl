module RicianTests

using Test
using ..Utils: arbify, ∇FD_central, ∇FD_forward, ∇Fwd, ∇Zyg

using FastRicianLikelihoods: FastRicianLikelihoods, Distributions, StaticArrays
using FastRicianLikelihoods: neglogpdf_rician, ∇neglogpdf_rician, ∇²neglogpdf_rician, ∇²neglogpdf_rician_with_gradient, ∇³neglogpdf_rician_with_gradient_and_hessian, neglogpdf_qrician, ∇neglogpdf_qrician, mean_rician, std_rician, f_quadrature, neglogf_quadrature
using .Distributions: Normal, logpdf, cdf
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
                @test @inferred(f̂(x, ν)) ≈ f(x, ν) rtol = rtol atol = atol
            end
        end
        @testset "1 <= z ($T)" begin
            rtol = 3 * eps(T)
            atol = T == Float32 ? 10 * eps(T) : 20 * eps(T)
            zs = T.(exp10.([0.0:0.1:0.5; 0.75; 1.0; 2.0:10.0]))
            for z in zs, (x, ν) in xν_iterator(z)
                @test @inferred(f̂(x, ν)) ≈ f(x, ν) rtol = rtol atol = atol
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
            rtol = 3 * eps(T)
            atol = 3 * eps(T)
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
            rtol = T == Float32 ? 3 * eps(T) : 20 * eps(T)
            atol = T == Float32 ? 10 * eps(T) : 40 * eps(T)
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
            rtol = 3 * eps(T)
            atol = 3 * eps(T)
            zs = high .* (1 + eps(T)) .* T.(exp10.([0.0:0.1:0.5; 0.75; 1.0; 2.0:10.0]))
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
        f̂ = neglogpdf_rician
        ∇²f̂ = ∇²neglogpdf_rician
        ∇²f = arbify(∇²f̂)

        rtol = T == Float32 ? 5.0f-5 : 5e-12
        atol = T == Float32 ? 5.0f-5 : 5e-12
        zs = T.(exp10.(-5:5))
        for z in zs, (x, ν) in xν_iterator(z)
            ∂ŷ, ∂y = @inferred(∇²f̂(x, ν)), ∇²f(x, ν)
            @test ∂ŷ[1] ≈ ∂y[1] rtol = rtol atol = atol
            @test ∂ŷ[2] ≈ ∂y[2] rtol = rtol atol = atol
            @test ∂ŷ[3] ≈ ∂y[3] rtol = rtol atol = atol
        end
    end
end

@testset "∇³neglogpdf_rician_with_gradient_and_hessian" begin
    # Note: The Float32 tests pass with tolerances around 1f-3 for almost all inputs, but since we don't yet
    #       have a robust implementation for r''(z) where r(z) = I1(z) / I0(z), occasionally there is catestrophic cancellation.
    for T in (Float64,)
        ∇²f̂_with_grad = ∇²neglogpdf_rician_with_gradient
        ∇³f̂ = ∇³neglogpdf_rician_with_gradient_and_hessian
        ∇³f = arbify(∇³f̂)

        rtol = T == Float32 ? 1.0f-3 : 1e-5
        atol = T == Float32 ? 1.0f-3 : 1e-5
        zs = T.(exp10.(-5:5))
        for z in zs, (x, ν) in xν_iterator(z)
            (∇ŷ, ∇²ŷ, ∇³ŷ), (_, _, ∇³y) = @inferred(∇³f̂(x, ν)), ∇³f(x, ν)

            # Lower order results should be almost exactly equal to the standalone functions
            ∇ŷ_with_grad, ∇²ŷ_with_grad = @inferred(∇²f̂_with_grad(x, ν))
            @test all(isapprox.(∇ŷ_with_grad, ∇ŷ; rtol = 2 * eps(T), atol = 2 * eps(T)))
            @test all(isapprox.(∇²ŷ_with_grad, ∇²ŷ; rtol = 2 * eps(T), atol = 2 * eps(T)))

            # Third derivative components should be approximately equal to the high precision reference
            @test ∇³ŷ[1] ≈ ∇³y[1] rtol = rtol atol = atol
            @test ∇³ŷ[2] ≈ ∇³y[2] rtol = rtol atol = atol
            @test ∇³ŷ[3] ≈ ∇³y[3] rtol = rtol atol = atol
            @test ∇³ŷ[4] ≈ ∇³y[4] rtol = rtol atol = atol
        end
    end
end

function xνδ_iterator()
    xs = [0.0; exp10.([-1.0, -0.1, 0.0, 0.1, 1.0])]
    νs = [0.0; exp10.([-1.0, -0.1, 0.0, 0.1, 1.0])]
    δs = exp10.([-2.0, -1.0, 0.0])
    return Iterators.product(xs, νs, δs)
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
        order = Val(32)
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
    # Unlike the density `neglogpdf_rician`, the integral defining `neglogpdf_qrician`
    # is approximated using Gauss-Legendre quadrature of a given `order`. Therefore,
    # `neglogpdf_qrician` is not exact in general. However, it should monotonically
    # improve in accuracy as `order` increases.
    for (x, ν, δ) in xνδ_iterator()
        f̂ = neglogpdf_qrician
        f = arbify(neglogpdf_qrician)
        y = f(x, ν, δ)

        for T in (Float32, Float64)
            ŷ1 = @inferred f̂(T(x), T(ν), T(δ), Val(1))
            ŷ4 = @inferred f̂(T(x), T(ν), T(δ), Val(4))
            ŷ8 = @inferred f̂(T(x), T(ν), T(δ), Val(8))
            ŷ16 = @inferred f̂(T(x), T(ν), T(δ), Val(16))
            ŷ32 = @inferred f̂(T(x), T(ν), T(δ), Val(32))

            rtol = 5 * eps(T)
            atol = zero(T)
            pass1 = isapprox(y, ŷ1; rtol, atol)
            pass4 = isapprox(y, ŷ4; rtol, atol)
            pass8 = isapprox(y, ŷ8; rtol, atol)
            pass16 = isapprox(y, ŷ16; rtol, atol)
            pass32 = isapprox(y, ŷ32; rtol, atol)

            @test pass1 || pass4 || pass8 || pass16 || pass32

            if pass1
                @test pass4 && pass8 && pass16 && pass32
            elseif pass4
                @test pass8 && pass16 && pass32
                @test abs(y - ŷ4) < abs(y - ŷ1)
            elseif pass8
                @test pass16 && pass32
                @test abs(y - ŷ8) < abs(y - ŷ4) < abs(y - ŷ1)
            elseif pass16
                @test pass32
                @test abs(y - ŷ16) < abs(y - ŷ8) < abs(y - ŷ4) < abs(y - ŷ1)
            else # pass32
                @test abs(y - ŷ32) < abs(y - ŷ16) < abs(y - ŷ8) < abs(y - ŷ4) < abs(y - ŷ1)
            end
        end
    end
end

@testset "∇neglogpdf_qrician" begin
    for (x, ν, δ) in xνδ_iterator()
        f̂ = neglogpdf_qrician
        ∇f̂ = ∇neglogpdf_qrician
        ∇f = arbify(∇f̂)
        ∂y = ∇f(x, ν, δ)

        for T in (Float32, Float64)
            # Compare gradient of high-order approximate integral with true gradient of the exact integral
            high_order = Val(64)
            rtol = T == Float32 ? 80 * eps(T) : 80 * eps(T)
            atol = T == Float32 ? 80 * eps(T) : 80 * eps(T)

            ∂ŷ = @inferred ∇f̂(T(x), T(ν), T(δ), high_order)
            @test isapprox(∂ŷ[1], ∂y[1]; rtol, atol)
            @test isapprox(∂ŷ[2], ∂y[2]; rtol, atol)
            @test isapprox(∂ŷ[3], ∂y[3]; rtol, atol)

            # Test gradient of low-order approximate integral
            for order in (Val(1), Val(2), Val(3), Val(4), Val(8))
                ∂ŷ = @inferred ∇f̂(T(x), T(ν), T(δ), order)
                @test ∂ŷ == @inferred ∇Fwd((xνδ...,) -> f̂(xνδ..., order), T(x), T(ν), T(δ))
                @test ∂ŷ == ∇Zyg((xνδ...,) -> f̂(xνδ..., order), T(x), T(ν), T(δ))
                @test_throws ErrorException @inferred ∇Zyg((xνδ...,) -> f̂(xνδ..., order), T(x), T(ν), T(δ))

                fd = ∇FD_forward((args...,) -> f̂(args[1], args[2], exp(args[3]), order), x, ν, log(δ))
                @test isapprox(∂ŷ[1], fd[1]; rtol = √eps(T), atol = √eps(T))
                @test isapprox(∂ŷ[2], fd[2]; rtol = √eps(T), atol = √eps(T))
                @test isapprox(∂ŷ[3], fd[3] / δ; rtol = √eps(T), atol = √eps(T))
            end
        end
    end
end

@testset "∇³neglogpdf_qrician" begin
    Tad, T = Float64, Float64
    rtol = 1e-8
    atol = 1e-8

    for (x, ν, δ) in xνδ_iterator(), order in (Val(1), Val(2), Val(4), Val(8))
        x > 0 && ν > 0 || continue
        t = rand(T)

        # Test inner jacobian computation
        ϕ1, Jϕ1 = FastRicianLikelihoods._∇²neglogpdf_qrician_inner_jac_ad(Tad.((x, ν, δ, t))...)
        ϕ2, Jϕ2 = FastRicianLikelihoods._∇²neglogpdf_qrician_inner_jac(T(x), T(ν), T(δ), T(t))

        @test isapprox(ϕ1, ϕ2; rtol, atol)
        @test isapprox(Jϕ1, Jϕ2; rtol, atol)

        # Test inner JVP computation
        Δϕ = @SVector randn(T, 9)
        ϕ3, jvp3 = FastRicianLikelihoods._∇²neglogpdf_qrician_inner_jvp(Δϕ, T(x), T(ν), T(δ), T(t))

        @test isapprox(ϕ3, ϕ1; rtol, atol)
        @test isapprox(jvp3, Jϕ1' * Δϕ; rtol, atol)

        # Test different JVP implementations
        Δ = @SVector randn(T, 9)
        Φ1, jvp1 = FastRicianLikelihoods._∇²neglogpdf_qrician_jvp_ad(Tad.(Δ), Tad.((x, ν, δ))..., order)
        Φ2, jvp2 = FastRicianLikelihoods._∇²neglogpdf_qrician_jvp_via_jac_parts(Δ, T(x), T(ν), T(δ), order)
        Φ3, jvp3 = FastRicianLikelihoods._∇²neglogpdf_qrician_jvp_via_two_pass(Δ, T(x), T(ν), T(δ), order)
        Φ4, jvp4 = FastRicianLikelihoods._∇²neglogpdf_qrician_jvp_via_one_pass(Δ, T(x), T(ν), T(δ), order)

        @test isapprox(Φ1, Φ2; rtol, atol)
        @test isapprox(Φ1, Φ3; rtol, atol)
        @test isapprox(jvp1, jvp2; rtol, atol)
        @test isapprox(jvp1, jvp3; rtol, atol)
        @test isapprox(jvp1, jvp4; rtol, atol)

        # Test jacobian computation
        Φ5, J5 = FastRicianLikelihoods._∇²neglogpdf_qrician_with_jacobian_ad(Tad.((x, ν, δ))..., order)
        Φ6, J6 = FastRicianLikelihoods._∇²neglogpdf_qrician_with_jacobian(T(x), T(ν), T(δ), order)

        @test isapprox(Φ1, Φ5; rtol, atol)
        @test isapprox(Φ1, Φ6; rtol, atol)
        @test isapprox(J5' * Δ, jvp1; rtol, atol)
        @test isapprox(J6' * Δ, jvp1; rtol, atol)
        @test isapprox(J5, J6; rtol, atol)
    end
end

@testset "gauss legendre quadrature" begin
    for T in (Float32, Float64)
        order = Val(16)
        d = Normal(randn(T) / 5, 1 + rand(T))
        a, δ = randn(T) / 5, (1 + rand(T)) / 10
        Ω = @inferred f_quadrature(x -> exp(logpdf(d, x)), a, δ, order)
        logΩ = @inferred -neglogf_quadrature(x -> -logpdf(d, x), a, δ, order)
        Ωtrue = cdf(d, a + δ) - cdf(d, a)
        @test Ω isa T
        @test logΩ isa T
        @test Ω ≈ Ωtrue atol = 10 * eps(T) rtol = 10 * eps(T)
        @test logΩ ≈ log(Ωtrue) atol = 10 * eps(T) rtol = 10 * eps(T)
    end
end

end # module RicianTests

using .RicianTests
