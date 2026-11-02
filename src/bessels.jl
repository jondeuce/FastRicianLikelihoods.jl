####
#### Fast + differentiable Bessel functions
####

# Bessels.jl provides `besseli0x` and `besseli1x`, but not `besseli2x`

"""
    besseli0(x::T) where {T <: Union{Float32, Float64}}

Modified Bessel function of the first kind of order zero, ``I_0(x)``.
"""
@inline besseli0(x) = Bessels.besseli0(x)

"""
    besseli0x(x::T) where {T <: Union{Float32, Float64}}

Scaled modified Bessel function of the first kind of order zero, ``I_0(x) e^{-|x|}``.
"""
@inline besseli0x(x) = Bessels.besseli0x(x)

"""
    besseli1(x::T) where {T <: Union{Float32, Float64}}

Modified Bessel function of the first kind of order one, ``I_1(x)``.
"""
@inline besseli1(x) = Bessels.besseli1(x)

"""
    besseli1x(x::T) where {T <: Union{Float32, Float64}}

Scaled modified Bessel function of the first kind of order one, ``I_1(x) e^{-|x|}``.
"""
@inline besseli1x(x) = Bessels.besseli1x(x)

"""
    logbesseli0x(x::T) where T <: Union{Float32, Float64}

Log of scaled modified Bessel function of the first kind of order zero, ``\\log(I_0(x) e^{-|x|})``.
"""
@inline function logbesseli0x(x::Real)
    T = checkedfloattype(x)
    low, _, _, _, high = logbesseli0x_branches(T)
    x = abs(x)
    if x < low
        return logbesseli0_taylor(x) - x
    elseif x < high
        return logbesseli0x_middle(x)
    else
        return logbesseli0x_scaled_tail(x) - (log(x) + T(log2π)) / 2
    end
end

"""
    logbesseli0(x::T) where {T <: Union{Float32, Float64}}

Log of modified Bessel function of the first kind of order zero, ``\\log I_0(x)``.
"""
@inline function logbesseli0(x::Real)
    T = checkedfloattype(x)
    low, _, _, _, _ = logbesseli0x_branches(T)
    x = abs(x)
    return x < low ? logbesseli0_taylor(x) : logbesseli0x(x) + x # log(I0(x)) = log(besselix(0, x)) + |x|
end

@inline function logbesseli0_taylor(x::Real)
    T = checkedfloattype(x)
    x² = abs2(x)
    return x² * (T(0.25) + x² * estrin(x², logbesseli0_taylor_coefs(T))) # log(I0x(x)) = log(exp(-x) * I0(x)) loses accuracy near x = 0 since I0x(x) -> 1 as x -> 0
end

@inline function logbesseli0x_middle(x::Real)
    T = checkedfloattype(x)
    _, mid1, mid2, mid3, _ = logbesseli0x_branches(T)
    x = abs(x)
    if x < mid1
        return x * estrin(muladd(x, logbesseli0x_mid1_transplant(T)...), logbesseli0x_mid1_coefs(T))
    elseif x < mid2
        return x * estrin(muladd(x, logbesseli0x_mid2_transplant(T)...), logbesseli0x_mid2_coefs(T))
    elseif x < mid3
        return x * estrin(muladd(x, logbesseli0x_mid3_transplant(T)...), logbesseli0x_mid3_coefs(T))
    else # x < high
        return x * estrin(muladd(x, logbesseli0x_mid4_transplant(T)...), logbesseli0x_mid4_coefs(T))
    end
end

@inline function logbesseli0x_scaled_tail(x::Real)
    T = checkedfloattype(x)
    x⁻¹ = inv(x)
    return x⁻¹ * (x < inv(sqrt(eps(T))) ? estrin(x⁻¹, logbesseli0x_tail_coefs(T)) : muladd(x⁻¹, T(1 / 16), T(1 / 8))) # log(sqrt(2πx) * exp(-x) * I0(x)) ~ 1/8x⁻¹ + 1/16x⁻² + 25/384x⁻³ + 𝒪(x⁻⁴)
end

"""
    besseli2(x::T) where {T <: Union{Float32, Float64}}

Modified Bessel function of the first kind of order two, ``I_2(x)``.
"""
@inline function besseli2(x::Real)
    T = checkedfloattype(x)
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return a * estrin(a, besseli2_small_coefs(T))
    else
        a = exp(x / 2)
        s = a * estrin(inv(x), besseli2_med_coefs(T)) / sqrt(x)
        return a * s
    end
end

"""
    besseli2x(x::T) where {T <: Union{Float32, Float64}}

Scaled modified Bessel function of the first kind of order two, ``I_2(x) e^{-|x|}``.
"""
@inline function besseli2x(x::Real)
    T = checkedfloattype(x)
    T == Float32 ? branch = 50 : branch = 500
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return a * estrin(a, besseli2_small_coefs(T)) * exp(-x)
    elseif x < branch
        return estrin(inv(x), besseli2_med_coefs(T)) / sqrt(x)
    else
        return estrin(inv(x), besseli2_large_coefs(T)) / sqrt(x)
    end
end

#### Derived special functions

"""
    logbesseli1(x::T) where {T <: Union{Float32, Float64}}

Log of modified Bessel function of the first kind of order one, ``\\log I_1(x)``.
"""
@inline logbesseli1(x::Real) = logbesseli1x(x) + abs(x) # log(besselix(1, x)) = log(I1(x)) - |x|

"""
    logbesseli1x(x::T) where {T <: Union{Float32, Float64}}

Log of scaled modified Bessel function of the first kind of order one, ``\\log(I_1(x) e^{-|x|})``.
"""
@inline logbesseli1x(x::Real) = log(besseli1x(x))

"""
    logbesseli2(x::T) where {T <: Union{Float32, Float64}}

Log of modified Bessel function of the first kind of order two, ``\\log I_2(x)``.
"""
@inline logbesseli2(x::Real) = logbesseli2x(x) + abs(x) # log(besselix(2, x)) = log(I2(x)) - |x|

"""
    logbesseli2x(x::T) where {T <: Union{Float32, Float64}}

Log of scaled modified Bessel function of the first kind of order two, ``\\log(I_2(x) e^{-|x|})``.
"""
@inline logbesseli2x(x::Real) = log(besseli2x(x))

@inline laguerre½(x::Real) = (x < zero(x) ? one(x) : exp(x)) * ((1 - x) * besseli0x(-x / 2) - x * besseli1x(-x / 2)) # besselix(ν, ±x/2) = Iν(±x/2) * exp(-|±x/2|) = Iν(-x/2) * exp(∓x/2)

@inline function laguerre½²c(t::Real)
    # laguerre½²c(t) = L^2 - t^2 - 1 where L = √(π/2) * laguerre½(-t^2/2)
    return _laguerre½²c(t, checkedfloattype(t))
end

@inline function _laguerre½²c(t::Real, ::Type{T}) where {T}
    y = t * t
    if y < T(4)
        return estrin(y / T(4), laguerre½²c_small1_coefs(T))
    elseif y < T(12)
        return estrin(muladd(y, laguerre½²c_small2_transplant(T)...), laguerre½²c_small2_coefs(T))
    elseif y < T(36)
        return estrin(muladd(y, laguerre½²c_small3_transplant(T)...), laguerre½²c_small3_coefs(T))
    else
        y⁻¹ = inv(t)^2
        return y⁻¹ * estrin(muladd(y⁻¹, laguerre½²c_large_transplant(T)...), laguerre½²c_large_coefs(T))
    end
end

"""
    besseli1i0(x::T) where {T <: Union{Float32, Float64}}

Ratio of modified Bessel functions of the first kind of orders one and zero, ``I_1(x) / I_0(x)``.
"""
@inline besseli1i0(x::Real) = _besseli1i0_parts(x)[1]

"""
    besseli1i0x(x::T) where {T <: Union{Float32, Float64}}

Ratio of modified Bessel functions divided by argument, ``(I_1(x) / I_0(x)) / x``.
"""
@inline besseli1i0x(x::Real) = _besseli1i0_parts(x)[2]

"""
    besseli1i0m1(x::T) where {T <: Union{Float32, Float64}}

Ratio of modified Bessel functions minus one, ``I_1(x) / I_0(x) - 1``.
"""
@inline besseli1i0m1(x::Real) = _besseli1i0_parts(x)[3]

@inline function _besseli1i0_parts(x::Real)
    T = checkedfloattype(x)
    low, mid1, mid2, mid3, tail = besseli1i0_branches(T)
    if x < tail
        x² = x^2
        if x < low
            rx = estrin(x², besseli1i0_low_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) = 1/2 + 𝒪(x^2)
        elseif x < mid1
            rx = estrin(x², besseli1i0_mid1_num_coefs(T)) / estrin(x², besseli1i0_mid1_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2) = 1/2 + 𝒪(x^2)
        elseif x < mid2
            rx = estrin(x², besseli1i0_mid2_num_coefs(T)) / estrin(x², besseli1i0_mid2_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2) = 1/2 + 𝒪(x^2)
        elseif x < mid3
            rx = estrin(x², besseli1i0_mid3_num_coefs(T)) / estrin(x², besseli1i0_mid3_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2) = 1/2 + 𝒪(x^2)
        else # x < tail
            rx = estrin(x², besseli1i0_mid4_num_coefs(T)) / estrin(x², besseli1i0_mid4_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2) = 1/2 + 𝒪(x^2)
        end
        r = x * rx # I₁(x) / I₀(x) = x * R(x^2) = x/2 - x^3/16 + 𝒪(x^5) where R(x^2) = P(x^2) or P(x^2) / Q(x^2)
        rm1 = r - one(T) # I₁(x) / I₀(x) - 1 = -1 + x/2 - x^3/16 + 𝒪(x^5)
    else # x >= tail
        x⁻¹ = inv(x)
        rm1x²m½x = estrin(x⁻¹, besseli1i0c_tail_coefs(T)) # P(1/x) = x^2 * (1 - I₁(x) / I₀(x)) - x/2 = 1/8 + 1/8x + 25/128x^2 + 𝒪(1/x^3)
        rm1 = x⁻¹ * (T(-0.5) - x⁻¹ * rm1x²m½x) # I₁(x) / I₀(x) - 1 = -1/2x - P(1/x) / x^2 = -1/2x - 1/8x^2 - 1/8x^3 + 𝒪(1/x^4)
        r = rm1 + one(T) # I₁(x) / I₀(x) = 1 - 1/2x - 1/8x^2 - 1/8x^3 + 𝒪(1/x^4)
        rx = x⁻¹ * r
    end
    return r, rx, rm1
end

@inline function _neglogpdf_rician_parts(z::Real, degree::Val, scaled::Val = Val(false))
    T = checkedfloattype(z)
    if z < neglogpdf_rician_parts_taylor_branch(T)
        return _neglogpdf_rician_parts_taylor(z, degree)
    else
        # If scaled=true, return z²r′, z³r′′ and z³(2r′ + zr′′)
        return _neglogpdf_rician_parts_asymptotic(z, degree, scaled)
    end
end

@inline function _neglogpdf_rician_parts_taylor(z::Real, ::Val{degree}) where {degree}
    T = checkedfloattype(z)
    r_tail = r′ = r′′ = one_minus_r_minus_z_r′ = two_r′_plus_z_r′′ = T(NaN)
    z² = z^2
    a1 = estrin(z², a1_taylor_coefs(T))
    a0 = T(0.5) + z² * a1
    r = z * a0
    r² = r * r
    if degree >= 1
        r′ = (one(T) - a0) - r²
        one_minus_r_minus_z_r′ = one(T) - (r + z * r′)
    end
    if degree >= 2
        r′′ = T(2) * (z * a1 + r * ((T(1.5) * a0 - one(T)) + r²))
        two_r′_plus_z_r′′ = 2 * r′ + z * r′′
    end
    u = r_tail = T(NaN)
    return (; u, r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′)
end

@inline function _neglogpdf_rician_parts_asymptotic(z::Real, ::Val{degree}, ::Val{scaled} = Val(false)) where {degree, scaled}
    T = checkedfloattype(z)
    r_tail = r′ = r′′ = one_minus_r_minus_z_r′ = two_r′_plus_z_r′′ = T(NaN)
    u = inv(z)
    b3 = b3_asymptotic(u, z)
    b2 = one(T) + u * b3
    b1 = T(0.25) * (one(T) + u * b2)
    b0 = T(0.5) * (one(T) + u * b1)
    b0², b1² = b0 * b0, b1 * b1
    u² = u^2
    halfu² = T(0.5) * u²
    r = one(T) - u * b0
    r_tail = b0
    if degree >= 1
        r′ = b1 + b0 * (one(T) - b0)
        if !scaled
            r′ *= u²
        end
        one_minus_r_minus_z_r′ = halfu² * (b1 * (T(0.5) + b0) + T(-0.5) * b2)
    end
    if degree >= 2
        r′′ = T(-2) * ((b0² * (T(0.5) + b0) + T(0.25) * b2) - u * b1 * (T(0.75) * b1 + b0))
        two_r′_plus_z_r′′ = T(0.5) * (b1² - b3 + (b2 - u * b1²) * (T(0.5) + b0))
        if !scaled
            u³ = u² * u
            r′′ *= u³
            two_r′_plus_z_r′′ *= u³
        end
    end
    return (; u, r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′)
end

@inline function b3_asymptotic(u::Real, z::Real)
    T = checkedfloattype(u, z)
    _, mid1, mid2, mid3, mid4, mid5, high, tail = neglogpdf_rician_parts_branches(T)
    if z < mid1
        return estrin(muladd(u, b3_mid1_transplant(T)...), b3_mid1_coefs(T))
    elseif z < mid2
        return estrin(muladd(u, b3_mid2_transplant(T)...), b3_mid2_coefs(T))
    elseif z < mid3
        return estrin(muladd(u, b3_mid3_transplant(T)...), b3_mid3_coefs(T))
    elseif z < mid4
        return estrin(muladd(u, b3_mid4_transplant(T)...), b3_mid4_coefs(T))
    elseif z < mid5
        return estrin(muladd(u, b3_mid5_transplant(T)...), b3_mid5_coefs(T))
    elseif z < high
        return estrin(muladd(u, b3_mid6_transplant(T)...), b3_mid6_coefs(T))
    elseif z < tail
        return estrin(u, b3_high_coefs(T))
    elseif z < inv(sqrt(eps(T)))
        return estrin(u, b3_tail_coefs(T))
    else
        return evalpoly(u, (T(25 / 16), T(13 / 4), T(1073 / 128)))
    end
end

#### ChainRules and ForwardDiff

@inline ∂x_laguerre½(x::Real) = (x < zero(x) ? one(x) : exp(x)) * (besseli1x(x / 2) - besseli0x(x / 2)) / 2
@scalar_rule laguerre½(x) ∂x_laguerre½(x)
@dual_rule_from_frule laguerre½(x)

@scalar_rule besseli0(x) besseli1(x)
@dual_rule_from_frule besseli0(x)

@inline ∂x_besseli1(x::Real) = (besseli0(x) + besseli2(x)) / 2
@scalar_rule besseli1(x) ∂x_besseli1(x)
@dual_rule_from_frule besseli1(x)

@inline ∂x_besseli0x(Ω::Real, x::Real) = besseli1x(x) - sign(x) * Ω
@inline f_∂x_besseli0x(x::Real) = (Ω = besseli0x(x); return (Ω, ∂x_besseli0x(Ω, x)))
@scalar_rule besseli0x(x) ∂x_besseli0x(Ω, x)
@dual_rule_from_frule besseli0x(x)

@inline ∂x_besseli1x(Ω::Real, x::Real) = (besseli0x(x) + besseli2x(x)) / 2 - sign(x) * Ω
@inline f_∂x_besseli1x(x::Real) = (Ω = besseli1x(x); return (Ω, ∂x_besseli1x(Ω, x)))
@scalar_rule besseli1x(x) ∂x_besseli1x(Ω, x)
@dual_rule_from_frule besseli1x(x)
