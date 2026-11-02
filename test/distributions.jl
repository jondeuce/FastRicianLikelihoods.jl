module DistributionsTests

using FastRicianLikelihoods: FastRicianLikelihoods, Rice, neglogpdf_rician, ∇neglogpdf_rician, mean_rician, mode_rician
using ..Utils: ∇Fwd, ∇Zyg
using Test

using ArbNumerics: ArbNumerics, ArbReal
using Distributions: Distributions, logpdf, pdf, insupport, mode
using QuadGK: quadgk

const F = FastRicianLikelihoods

@testset "constructors ($T)" for T in (Float32, Float64)
    @test Rice(T(2), T(3)) isa Rice{T}
    @test Rice(T(2)) === Rice(T(2), one(T))
    @test Rice(2, 3) isa Rice{Float64}
    @test Rice() === Rice(0.0, 1.0)
    @test Rice(zero(T), one(T)).ν == 0
    @test_throws DomainError Rice(-one(T), one(T))
    @test_throws DomainError Rice(one(T), zero(T))
    @test_throws DomainError Rice(one(T), -one(T))
    @test Rice(-one(T), one(T); check_args = false).ν == -1
end

@testset "support ($T)" for T in (Float32, Float64)
    d = Rice(T(2), T(3))
    @test logpdf(d, -one(T)) === T(-Inf)
    @test logpdf(d, zero(T)) === T(-Inf)
    @test logpdf(d, T(Inf)) === T(-Inf)
    @test pdf(d, -one(T)) === zero(T)
    @test pdf(d, T(Inf)) === zero(T)
    @test isnan(logpdf(d, T(NaN)))
    @test !insupport(d, -one(T))
    @test logpdf(d, 1) == logpdf(d, one(T))
    @test @inferred(logpdf(d, one(T))) isa T
end

@testset "logpdf scaling ($T)" for T in (Float32, Float64)
    for ν in T[0, 0.1, 1, 10, 100], σ in T[0.25, 1, 3], x in T[0.5, 2, 12]
        ref = -(log(σ) + neglogpdf_rician(x / σ, ν / σ))
        @test logpdf(Rice(ν, σ), x) ≈ ref rtol = 4 * eps(T) atol = 4 * eps(T)
    end
end

@testset "mode is a stationary point ($T)" for T in (Float32, Float64)
    for ν in T[0.1, 1, 3, 10, 100], σ in T[0.25, 1, 3]
        m = mode(Rice(ν, σ))
        @test m ≈ T(ArbReal(σ) * F.mode_rician(ArbReal(ν) / ArbReal(σ))) rtol = 8 * eps(T)
        @test abs(first(∇neglogpdf_rician(m / σ, ν / σ))) <= 64 * eps(T)
    end
end

@testset "mode and curvature fits ($T)" for T in (Float32, Float64)
    for (f, branches) in ((F.mode_rician, F.mode_rician_branches(T)), (F.var_mode_rician, F.var_mode_rician_branches(T)))
        νs = sort!(unique!([zero(T); T.(exp10.(-8:0.25:8)); [ν for b in branches for ν in (prevfloat(b), b, nextfloat(b))]]))
        for ν in νs
            ref = f(ArbReal(ν))
            @test f(ν) ≈ T(ref) rtol = 8 * eps(T) atol = 8 * eps(T)
        end
    end
end

@testset "deterministic quantized mean ($T)" for T in (Float32, Float64)
    function mean_qrician_ref(ν, δ)
        A = ArbReal
        ν, δ = A(ν), A(δ)
        hi = ν + 16
        total = zero(ν)
        for n in 1:ceil(Int, hi/δ)
            a = n * δ
            b = min((n + 1) * δ, hi)
            b > a || continue
            p, _ = quadgk(y -> exp(-F.neglogpdf_rician(y, ν)), a, b; rtol = A(1e-30))
            total += a * p
        end
        return T(total)
    end

    for (ν, δ) in ((0, 0.25), (1, 0.5), (1, 2), (1, 6), (9.5, 10), (1000, 50), (5, 10))
        ν, δ = T(ν), T(δ)
        val = @inferred F.mean_qrician(ν, δ)
        ref = mean_qrician_ref(ν, δ)
        @test val isa T
        @test val ≈ ref rtol = 8 * eps(T) atol = 8 * eps(T) * max(one(T), abs(ref))
        @test F.mean_qrician(3ν, T(3), 3δ) ≈ 3 * val rtol = 8 * eps(T) atol = 8 * eps(T)
    end

    @test F.mean_qrician(1, 2) isa Float64
    @test F.mean_qrician(T(1), T(0)) == F.mean_rician(T(1))
end

end # module DistributionsTests

using .DistributionsTests
