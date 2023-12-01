####
#### Rician negative log-pdf
####

#### Utilities

@inline promote_float(x...) = promote(map(float, x)...)

####
#### Rician negative log-likelihood
####

@inline function neglogpdf_rician(x::T, ν::T, logσ::T) where {T <: Real}
    σ⁻¹ = exp(-logσ)
    return logσ + neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν)
end
@inline neglogpdf_rician(x::Real, ν::Real, logσ::Real) = neglogpdf_rician(promote_float(x, ν, logσ)...)

@inline neglogpdf_rician(x::T, ν::T) where {T <: Union{Float32, Float64}} = (x - ν)^2 / 2 - log(x) - logbesseli0x(x * ν) # negative Rician log-likelihood `-logp(x | ν, σ = 1)`

@inline function ∇neglogpdf_rician(x::T, ν::T) where {T <: Union{Float32, Float64}}
    # Define the univariate normalized Bessel function `Î₀` as
    #
    #   Î₀(z) = I₀(z) / (exp(z) / √2πz).
    #
    # The negative likelihood is then be written as
    #
    #        -logp(x | ν, σ = 1) = (x - ν)^2 / 2 - log(x / ν) / 2 - logÎ₀(x * ν) + log√2π.
    #   ∂/∂x -logp(x | ν, σ = 1) = x - ν - 1 / 2x - ∂/∂x logÎ₀(x * ν).
    #   ∂/∂ν -logp(x | ν, σ = 1) = ν - x + 1 / 2ν - ∂/∂ν logÎ₀(x * ν).
    #
    # All that must be approximated then is `d/dz logÎ₀(z)` where `z = x * ν`:
    #
    #   d/dz logÎ₀(z) =  1/2z + (I₁(z) / I₀(z) - 1)
    #                 ≈ -1/8z^2 - 1/8z^3 - 25/128z^4 - 13/32z^5 - 1073/1024z^6 - 103/32z^7 + 𝒪(1/z^8)   (z >> 1)
    #                 ≈  1/2z - 1 + z/2 - z^3/16 + z^5/96 - 11*z^7/6144 + 𝒪(z^9)                        (z << 1)
    #   ∂/∂x logÎ₀(z) = ν * d/dz logÎ₀(z)
    #   ∂/∂ν logÎ₀(z) = x * d/dz logÎ₀(z)

    # Note: there are really three relevant limits: z << 1, z >> 1, and x ≈ ν.
    # Could plausibly better account for the latter case, though it is tested quite robustly
    z = x * ν
    if z < besseli1i0_low_cutoff(T)
        z² = z^2
        r = z * evalpoly(z², besseli1i0_low_coefs(T)) # r = logÎ₀′(z) + 1 - 1/2z = I₁(z) / I₀(z) ≈ z/2 + 𝒪(z^3)
        ∂x = x - ν * r - inv(x)
        ∂ν = ν - x * r
    elseif z < besseli1i0_mid_cutoff(T)
        z² = z^2
        r = z * evalpoly(z², besseli1i0_mid_num_coefs(T)) / evalpoly(z², besseli1i0_mid_den_coefs(T)) # r = I₁(z) / I₀(z)
        ∂x = x - ν * r - inv(x)
        ∂ν = ν - x * r
    elseif z < besseli1i0_high_cutoff(T)
        z² = z^2
        r = z * evalpoly(z², besseli1i0_high_num_coefs(T)) / evalpoly(z², besseli1i0_high_den_coefs(T)) # r = I₁(z) / I₀(z)
        ∂x = x - ν * r - inv(x)
        ∂ν = ν - x * r
    else
        z⁻¹ = inv(z)
        tmp = z⁻¹ * evalpoly(z⁻¹, besseli1i0c_tail_coefs(T)) # -z * logÎ₀′(z) = -1/2 - z * (I₁(z) / I₀(z) - 1) ≈ 1/8z + 𝒪(1/z^2)
        ∂x = x - ν + (T(-0.5) + tmp) / x
        ∂ν = ν - x + (T(+0.5) + tmp) / ν
    end

    return (∂x, ∂ν)
end

@inline pdf_rician(args...) = exp(-neglogpdf_rician(args...))
@inline ∇pdf_rician(x::T, ν::T) where {T <: Union{Float32, Float64}} = -exp(-neglogpdf_rician(x, ν)) .* ∇neglogpdf_rician(x, ν)

@scalar_rule neglogpdf_rician(x, ν) (∇neglogpdf_rician(x, ν)...,)
@dual_rule_from_frule neglogpdf_rician(x, ν)

####
#### Quantized Rician negative log-pdf
####

# Quantized Rician PDF is the integral of the Rician PDF over `(x, x+δ)`.
# This integral is approximated using Gauss-Legendre quadrature.
# Note: Rician PDF is never evaluated at the interval endpoints.
@inline function neglogpdf_qrician(x::T, ν::T, logσ::T, δ::T, order::Val) where {T <: Real}
    σ⁻¹ = exp(-logσ)
    return neglogpdf_qrician(σ⁻¹ * x, σ⁻¹ * ν, σ⁻¹ * δ, order)
end
@inline neglogpdf_qrician(x::Real, ν::Real, logσ::Real, δ::Real, order::Val) = neglogpdf_qrician(promote_float(x, ν, logσ, δ)..., order)
@inline neglogpdf_qrician(n::Int, ν::Real, logσ::Real, δ::Real, order::Val) = neglogpdf_qrician(n * δ, ν, logσ, δ, order)

function neglogpdf_qrician_direct(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δ = x - ν
    I0 = Δ^2 / 2
    I1 = neglogf_quadrature(zero(T), δ, order) do t
        return t * (Δ + t / 2) - log(t + x) - logbesseli0x((t + x) * ν)
    end
    I = I0 + I1
    return I
end

function neglogpdf_qrician_taylor(x::T, ν::T, δ::T) where {T <: Union{Float32, Float64}}
    I = cdf_qrician_taylor_scaled(x + δ, ν)
    if x > 0
        I -= exp(-δ * ν) * cdf_qrician_taylor_scaled(x, ν)
    end
    return ν * (ν / 2 - (x + δ)) - log(I)
end

function cdf_qrician_taylor_scaled(a::T, ν::T) where {T <: Union{Float32, Float64}}
    # I = exp(-a * ν) * ∫_{0}^{a} [x * exp(-x^2/2) * I₀(x * ν)] dx
    if a < 3e-8 # nterms == 1
        return (a / ν) * besseli1x(a * ν)
    end

    ν⁻², a², aν, a⁻¹ν = inv(ν^2), a^2, a * ν, a / ν
    if a < 3e-4 # nterms == 2
        c₀ = T(1)
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2))), T(-2)))
    elseif a < 5e-3 # nterms == 3
        c₀ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2))), T(-4)))
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8))), evalpoly(a², (T(-2), T(2))), T(8)))
    elseif a < 3e-2 # nterms == 4
        c₀ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8))), evalpoly(a², (T(-4), T(3))), T(24)))
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48))), evalpoly(a², (T(-2), T(2), T(-3 // 4))), evalpoly(a², (T(8), T(-12))), T(-48)))
    elseif a < 8e-2 # nterms == 5
        c₀ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48))), evalpoly(a², (T(-4), T(3), T(-1))), evalpoly(a², (T(24), T(-24))), T(-192)))
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384))), evalpoly(a², (T(-2), T(2), T(-3 // 4), T(1 // 6))), evalpoly(a², (T(8), T(-12), T(6))), evalpoly(a², (T(-48), T(96))), T(384)))
    elseif a < 0.15 # nterms == 6
        c₀ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384))), evalpoly(a², (T(-4), T(3), T(-1), T(5 // 24))), evalpoly(a², (T(24), T(-24), T(10))), evalpoly(a², (T(-192), T(240))), T(1920)))
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840))), evalpoly(a², (T(-2), T(2), T(-3 // 4), T(1 // 6), T(-5 // 192))), evalpoly(a², (T(8), T(-12), T(6), T(-5 // 3))), evalpoly(a², (T(-48), T(96), T(-60))), evalpoly(a², (T(384), T(-960))), T(-3840)))
    elseif a < 0.24 # nterms == 7
        c₀ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840))), evalpoly(a², (T(-4), T(3), T(-1), T(5 // 24), T(-1 // 32))), evalpoly(a², (T(24), T(-24), T(10), T(-5 // 2))), evalpoly(a², (T(-192), T(240), T(-120))), evalpoly(a², (T(1920), T(-2880))), T(-23040)))
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840), T(1 // 46080))), evalpoly(a², (T(-2), T(2), T(-3 // 4), T(1 // 6), T(-5 // 192), T(1 // 320))), evalpoly(a², (T(8), T(-12), T(6), T(-5 // 3), T(5 // 16))), evalpoly(a², (T(-48), T(96), T(-60), T(20))), evalpoly(a², (T(384), T(-960), T(720))), evalpoly(a², (T(-3840), T(11520))), T(46080)))
    elseif a < 0.33 # nterms == 8
        c₀ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840), T(1 // 46080))), evalpoly(a², (T(-4), T(3), T(-1), T(5 // 24), T(-1 // 32), T(7 // 1920))), evalpoly(a², (T(24), T(-24), T(10), T(-5 // 2), T(7 // 16))), evalpoly(a², (T(-192), T(240), T(-120), T(35))), evalpoly(a², (T(1920), T(-2880), T(1680))), evalpoly(a², (T(-23040), T(40320))), T(322560)))
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840), T(1 // 46080), T(-1 // 645120))), evalpoly(a², (T(-2), T(2), T(-3 // 4), T(1 // 6), T(-5 // 192), T(1 // 320), T(-7 // 23040))), evalpoly(a², (T(8), T(-12), T(6), T(-5 // 3), T(5 // 16), T(-7 // 160))), evalpoly(a², (T(-48), T(96), T(-60), T(20), T(-35 // 8))), evalpoly(a², (T(384), T(-960), T(720), T(-280))), evalpoly(a², (T(-3840), T(11520), T(-10080))), evalpoly(a², (T(46080), T(-161280))), T(-645120)))
    elseif a < 0.42 # nterms == 9
        c₀ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840), T(1 // 46080), T(-1 // 645120))), evalpoly(a², (T(-4), T(3), T(-1), T(5 // 24), T(-1 // 32), T(7 // 1920), T(-1 // 2880))), evalpoly(a², (T(24), T(-24), T(10), T(-5 // 2), T(7 // 16), T(-7 // 120))), evalpoly(a², (T(-192), T(240), T(-120), T(35), T(-7))), evalpoly(a², (T(1920), T(-2880), T(1680), T(-560))), evalpoly(a², (T(-23040), T(40320), T(-26880))), evalpoly(a², (T(322560), T(-645120))), T(-5160960)))
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840), T(1 // 46080), T(-1 // 645120), T(1 // 10321920))), evalpoly(a², (T(-2), T(2), T(-3 // 4), T(1 // 6), T(-5 // 192), T(1 // 320), T(-7 // 23040), T(1 // 40320))), evalpoly(a², (T(8), T(-12), T(6), T(-5 // 3), T(5 // 16), T(-7 // 160), T(7 // 1440))), evalpoly(a², (T(-48), T(96), T(-60), T(20), T(-35 // 8), T(7 // 10))), evalpoly(a², (T(384), T(-960), T(720), T(-280), T(70))), evalpoly(a², (T(-3840), T(11520), T(-10080), T(4480))), evalpoly(a², (T(46080), T(-161280), T(161280))), evalpoly(a², (T(-645120), T(2580480))), T(10321920)))
    elseif a < 0.53 # nterms == 10
        c₀ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840), T(1 // 46080), T(-1 // 645120), T(1 // 10321920))), evalpoly(a², (T(-4), T(3), T(-1), T(5 // 24), T(-1 // 32), T(7 // 1920), T(-1 // 2880), T(1 // 35840))), evalpoly(a², (T(24), T(-24), T(10), T(-5 // 2), T(7 // 16), T(-7 // 120), T(1 // 160))), evalpoly(a², (T(-192), T(240), T(-120), T(35), T(-7), T(21 // 20))), evalpoly(a², (T(1920), T(-2880), T(1680), T(-560), T(126))), evalpoly(a², (T(-23040), T(40320), T(-26880), T(10080))), evalpoly(a², (T(322560), T(-645120), T(483840))), evalpoly(a², (T(-5160960), T(11612160))), T(92897280)))
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840), T(1 // 46080), T(-1 // 645120), T(1 // 10321920), T(-1 // 185794560))), evalpoly(a², (T(-2), T(2), T(-3 // 4), T(1 // 6), T(-5 // 192), T(1 // 320), T(-7 // 23040), T(1 // 40320), T(-1 // 573440))), evalpoly(a², (T(8), T(-12), T(6), T(-5 // 3), T(5 // 16), T(-7 // 160), T(7 // 1440), T(-1 // 2240))), evalpoly(a², (T(-48), T(96), T(-60), T(20), T(-35 // 8), T(7 // 10), T(-7 // 80))), evalpoly(a², (T(384), T(-960), T(720), T(-280), T(70), T(-63 // 5))), evalpoly(a², (T(-3840), T(11520), T(-10080), T(4480), T(-1260))), evalpoly(a², (T(46080), T(-161280), T(161280), T(-80640))), evalpoly(a², (T(-645120), T(2580480), T(-2903040))), evalpoly(a², (T(10321920), T(-46448640))), T(-185794560)))
    else # nterms == 11
        c₀ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840), T(1 // 46080), T(-1 // 645120), T(1 // 10321920), T(-1 // 185794560))), evalpoly(a², (T(-4), T(3), T(-1), T(5 // 24), T(-1 // 32), T(7 // 1920), T(-1 // 2880), T(1 // 35840), T(-1 // 516096))), evalpoly(a², (T(24), T(-24), T(10), T(-5 // 2), T(7 // 16), T(-7 // 120), T(1 // 160), T(-1 // 1792))), evalpoly(a², (T(-192), T(240), T(-120), T(35), T(-7), T(21 // 20), T(-1 // 8))), evalpoly(a², (T(1920), T(-2880), T(1680), T(-560), T(126), T(-21))), evalpoly(a², (T(-23040), T(40320), T(-26880), T(10080), T(-2520))), evalpoly(a², (T(322560), T(-645120), T(483840), T(-201600))), evalpoly(a², (T(-5160960), T(11612160), T(-9676800))), evalpoly(a², (T(92897280), T(-232243200))), T(-1857945600)))
        c₁ = evalpoly(ν⁻², (evalpoly(a², (T(1), T(-1 // 2), T(1 // 8), T(-1 // 48), T(1 // 384), T(-1 // 3840), T(1 // 46080), T(-1 // 645120), T(1 // 10321920), T(-1 // 185794560), T(1 // 3715891200))), evalpoly(a², (T(-2), T(2), T(-3 // 4), T(1 // 6), T(-5 // 192), T(1 // 320), T(-7 // 23040), T(1 // 40320), T(-1 // 573440), T(1 // 9289728))), evalpoly(a², (T(8), T(-12), T(6), T(-5 // 3), T(5 // 16), T(-7 // 160), T(7 // 1440), T(-1 // 2240), T(1 // 28672))), evalpoly(a², (T(-48), T(96), T(-60), T(20), T(-35 // 8), T(7 // 10), T(-7 // 80), T(1 // 112))), evalpoly(a², (T(384), T(-960), T(720), T(-280), T(70), T(-63 // 5), T(7 // 4))), evalpoly(a², (T(-3840), T(11520), T(-10080), T(4480), T(-1260), T(252))), evalpoly(a², (T(46080), T(-161280), T(161280), T(-80640), T(25200))), evalpoly(a², (T(-645120), T(2580480), T(-2903040), T(1612800))), evalpoly(a², (T(10321920), T(-46448640), T(58060800))), evalpoly(a², (T(-185794560), T(928972800))), T(3715891200)))
    end

    I₀, I₁ = besseli0x(aν), besseli1x(aν)
    return a⁻¹ν * muladd(a⁻¹ν, c₀ * I₀, c₁ * I₁) # Note: c₀ ~ O(1), c₁ ~ O(1)
end

function neglogpdf_qrician_right_laguerre_tail(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δ = x - ν
    I0 = Δ^2 / 2

    if δ * (Δ + δ / 2) > -log(eps(T))
        I1 = f_laguerre_tail_quadrature(Δ, order) do t̂
            t = x + t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν)
        end
        I1 = -log(I1)
    else
        I1⁺ = f_laguerre_tail_quadrature(Δ, order) do t̂
            t = x + t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν)
        end
        I1⁻ = f_laguerre_tail_quadrature(Δ + δ, order) do t̂
            t = x + δ + t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν)
        end
        I1 = -log(I1⁺ - exp(-δ * (Δ + δ / 2)) * I1⁻)

        #TODO merging into one call worth it?
        # I1 = f_laguerre_tail_quadrature(Δ, order) do t̂
        #     t = x + t̂
        #     f1 = t * besseli0x(t * ν)
        #     f2 = (t + δ) * besseli0x((t + δ) * ν)
        #     return exp(-t̂^2 / 2) * (f1 - exp(-δ * (t̂ + Δ + δ / 2)) * f2)
        # end
        # I1 = -log(I1)
    end
    I = I0 + I1

    return I
end

function neglogpdf_qrician_right_halfhermite_tail(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δ = x - ν
    I0 = Δ^2 / 2

    if δ * (Δ + δ / 2) > -log(eps(T))
        I1 = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + t̂
            return exp(-Δ * t̂) * t * besseli0x(t * ν)
        end
        I1 = -log(I1) - T(log2π) / 2
    else
        I1⁺ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + t̂
            return exp(-Δ * t̂) * t * besseli0x(t * ν)
        end
        I1⁻ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + δ + t̂
            return exp(-(Δ + δ) * t̂) * t * besseli0x(t * ν)
        end
        I1 = -log(I1⁺ - exp(-δ * (Δ + δ / 2)) * I1⁻) - T(log2π) / 2

        #TODO merging into one call worth it?
        # I1 = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
        #     t = x + t̂
        #     f1 = t * besseli0x(t * ν)
        #     f2 = (t + δ) * besseli0x((t + δ) * ν)
        #     return exp(-Δ * t̂) * (f1 - exp(-δ * (t̂ + Δ + δ / 2)) * f2)
        # end
        # I1 = -log(I1) - T(log2π) / 2
    end
    I = I0 + I1

    return I
end

function neglogpdf_qrician_left_laguerre_tail(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δ = ν - (x + δ)
    I0 = Δ^2 / 2

    if δ * (Δ + δ / 2) > -log(eps(T))
        I1 = f_laguerre_tail_quadrature(Δ, order) do t̂
            t = x + δ - t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1 = -log(I1)
    else
        I1⁺ = f_laguerre_tail_quadrature(Δ, order) do t̂
            t = x + δ - t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1⁻ = f_laguerre_tail_quadrature(Δ + δ, order) do t̂
            t = x - t̂
            return exp(-t̂^2 / 2) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1 = -log(I1⁺ - exp(-δ * (Δ + δ / 2)) * I1⁻)

        #TODO merging into one call worth it?
        # I1 = f_laguerre_tail_quadrature(Δ, order) do t̂
        #     t = x - t̂
        #     I1⁺ = exp(-t̂^2 / 2) * (t + δ) * besseli0x((t + δ) * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        #     I1⁻ = exp(-t̂^2 / 2) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        #     return I1⁺ - exp(-δ * (Δ + δ / 2)) * I1⁻
        # end
        # I1 = -log(I1)
    end
    I = I0 + I1

    return I
end

function neglogpdf_qrician_left_halfhermite_tail(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Δ = ν - (x + δ)
    I0 = Δ^2 / 2

    if δ * (Δ + δ / 2) > -log(eps(T))
        I1 = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + δ - t̂
            return exp(-Δ * t̂) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1 = -log(I1) - T(log2π) / 2
    else
        I1⁺ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x + δ - t̂
            return exp(-Δ * t̂) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1⁻ = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
            t = x - t̂
            return exp(-(Δ + δ) * t̂) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        end
        I1 = -log(I1⁺ - exp(-δ * (Δ + δ / 2)) * I1⁻) - T(log2π) / 2

        #TODO merging into one call worth it?
        # I1 = f_halfhermite_tail_quadrature(Val(zero(T)), order) do t̂
        #     t = x - t̂
        #     I1⁺ = exp(-Δ * t̂) * (t + δ) * besseli0x((t + δ) * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        #     I1⁻ = exp(-(Δ + δ) * t̂) * t * besseli0x(t * ν) # odd extension of `t * besseli0x(t * ν)` to `t < 0`
        #     return I1⁺ - exp(-δ * (Δ + δ / 2)) * I1⁻
        # end
        # I1 = -log(I1) - T(log2π) / 2
    end
    I = I0 + I1

    return I
end

@inline neglogpdf_qrician(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}} = neglogf_quadrature(Base.Fix2(neglogpdf_rician, ν), x, δ, order)
@inline ∇neglogpdf_qrician(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}} = ∇neglogpdf_qrician_with_primal(x, ν, δ, order)[2]

@inline function ∇neglogpdf_qrician_with_primal(Ω::T, x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    ∂x, ∂ν = f_quadrature(x, δ, order) do y
        ∇ = ∇neglogpdf_rician(y, ν) # differentiate the integrand
        ∇ = SVector{2, T}(∇)
        return exp(Ω - neglogpdf_rician(y, ν)) * ∇
    end
    ∂δ = -exp(Ω - neglogpdf_rician(x + δ, ν)) # by fundamental theorem of calculus
    return Ω, (∂x, ∂ν, ∂δ)
end
@inline ∇neglogpdf_qrician_with_primal(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}} = ∇neglogpdf_qrician_with_primal(neglogpdf_qrician(x, ν, δ, order), x, ν, δ, order)

@scalar_rule neglogpdf_qrician(x, ν, δ, order::Val) (∇neglogpdf_qrician_with_primal(Ω, x, ν, δ, order)[2]..., NoTangent())
@dual_rule_from_frule neglogpdf_qrician(x, ν, δ, !order)

#### Gaussian quadrature

const DEFAULT_GAUSSLEGENDRE_ORDER = 16
const DEFAULT_GAUSSLAGUERRE_ORDER = 16

@generated function gausslegendre_unit_interval(::Val{order}, ::Type{T}) where {order, T <: AbstractFloat}
    x, w = gausslegendre(order)
    x = SVector{order, T}(@. T((1 + x) / 2)) # rescale from [-1, 1] to [0, 1]
    w = SVector{order, T}(@. T(w / 2)) # adjust weights to account for rescaling
    return :($x, $w)
end

@generated function gausslaguerre_positive_real_axis(::Val{order}, ::Type{T}) where {order, T <: AbstractFloat}
    x, w = gausslaguerre(order)
    x = SVector{order, T}(T.(x)) # nodes lie in [0, ∞)
    w = SVector{order, T}(T.(w)) # exponentially decreasing weights
    return :($x, $w)
end

@generated function gausshalfhermite_positive_real_axis(::Val{order}, ::Type{T}, ::Val{γ}) where {order, T <: AbstractFloat, γ}
    @assert γ > -1 "γ must be greater than -1"
    x, w = gausshalfhermite_gw(order, BigFloat(γ); normalize = true)
    x = SVector{order, T}(T.(x)) # nodes lie in [0, ∞)
    w = SVector{order, T}(T.(w)) # exponentially decreasing weights
    return :($x, $w)
end

@inline function f_quadrature(f::F, x₀::T, δ::T, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order, T <: AbstractFloat}
    # I = ∫_{0}^{δ} [f(t)] dt
    x, w = gausslegendre_unit_interval(Val(order), T)
    y = @. f(x₀ + δ * x)
    return vecdot(w, y) * δ
end

@inline function neglogf_quadrature(neglogf::F, x₀::T, δ::T, ::Val{order} = Val(DEFAULT_GAUSSLEGENDRE_ORDER)) where {F, order, T <: AbstractFloat}
    # I = ∫_{0}^{δ} [f(t)] dt, where f(t) = exp(-neglogf(t))
    x, w = gausslegendre_unit_interval(Val(order), T)
    logy = @. -neglogf(x₀ + δ * x)
    return -weighted_logsumexp(w, logy) .- log(δ)
end

@inline function f_laguerre_tail_quadrature(f::F, λ::T, ::Val{order} = Val(DEFAULT_GAUSSLAGUERRE_ORDER)) where {F, order, T <: AbstractFloat}
    # I = ∫_{0}^{∞} [exp(-λt) f(t)] dt
    x, w = gausslaguerre_positive_real_axis(Val(order), T)
    y = @. f(x / λ)
    return vecdot(w, y) / λ
end

@inline function f_halfhermite_tail_quadrature(f::F, ::Val{γ}, ::Val{order} = Val(DEFAULT_GAUSSLAGUERRE_ORDER)) where {F, order, γ}
    # I = ∫_{0}^{∞} [x^γ exp(-t^2/2) f(t)] / √(2π) dt
    T = typeof(float(γ))
    x, w = gausshalfhermite_positive_real_axis(Val(order), T, Val(γ))
    y = @. f(x)
    return vecdot(w, y)
end

@inline function weighted_logsumexp(w::SVector{N, T}, logy::SVector{N, T}) where {N, T <: AbstractFloat}
    max_ = maximum(logy)
    ȳ = exp.(logy .- max_)
    return log(vecdot(w, ȳ)) + max_
end

@inline function weighted_logsumexp(w::SVector{N, T}, logy::SVector{N, SVector{M, T}}) where {N, M, T <: AbstractFloat}
    max_ = reduce(BroadcastFunction(max), logy) # elementwise maximum
    logy = reduce(hcat, logy) # stack as columns
    ȳ = exp.(logy .- max_)
    return log.(vecdot(w, ȳ)) .+ max_
end

@inline vecdot(w::SVector{N, T}, y::SVector{N, T}) where {N, T <: AbstractFloat} = dot(w, y)
@inline vecdot(w::SVector{N, T}, y::SVector{N, SVector{M, T}}) where {N, M, T <: AbstractFloat} = vecdot(w, reduce(hcat, y))
@inline vecdot(w::SVector{N, T}, y::SMatrix{M, N, T}) where {N, M, T <: AbstractFloat} = y * w
