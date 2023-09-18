####
#### Fast + differentiable Bessel functions
####

# Bessels.jl provides `besseli0x` and `besseli1x`, but not `besseli2x`
@inline besseli0(x) = Bessels.besseli0(x)
@inline besseli0x(x) = Bessels.besseli0x(x)

@inline besseli1(x) = Bessels.besseli1(x)
@inline besseli1x(x) = Bessels.besseli1x(x)

"""
    besseli2(x::T) where {T <: Union{Float32, Float64}}

Modified Bessel function of the first kind of order two, ``I_2(x)``.
"""
@inline function besseli2(x::T) where {T <: Union{Float32, Float64}}
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return a * evalpoly(a, besseli2_small_coefs(T))
    else
        a = exp(x / 2)
        s = a * evalpoly(inv(x), besseli2_med_coefs(T)) / sqrt(x)
        return a * s
    end
end

"""
    besseli2x(x::T) where {T <: Union{Float32, Float64}}

Scaled modified Bessel function of the first kind of order two, ``I_2(x)*e^{-|x|}``.
"""
@inline function besseli2x(x::T) where {T <: Union{Float32, Float64}}
    T == Float32 ? branch = 50 : branch = 500
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return a * evalpoly(a, besseli2_small_coefs(T)) * exp(-x)
    elseif x < branch
        return evalpoly(inv(x), besseli2_med_coefs(T)) / sqrt(x)
    else
        return evalpoly(inv(x), besseli2_large_coefs(T)) / sqrt(x)
    end
end

@inline besseli2_small_coefs(::Type{Float32}) = (0.5000000380180327f0, 0.1666662618114895f0, 0.02083404267763115f0, 0.0013884111223694947f0, 5.80310930509866f-5, 1.623226061096556f-6, 3.775086797193045f-8, 3.403189301562835f-10, 1.3439158558362364f-11)
@inline besseli2_small_coefs(::Type{Float64}) = (0.49999999999999983, 0.16666666666667154, 0.020833333333312287, 0.0013888888889246186, 5.787037033873442e-5, 1.6534391702073172e-6, 3.444664327750705e-8, 5.467735445704131e-10, 6.834436191839501e-12, 6.906148607507894e-14, 5.733547826165566e-16, 4.1288365068666296e-18, 2.0258796870216445e-20, 1.958474603154919e-22)
@inline besseli2_med_coefs(::Type{Float32}) = (0.3989423335005962f0, -0.7480354669272465f0, 0.3283003548501443f0, 0.10303896764745404f0, 0.2639897680554983f0)
@inline besseli2_med_coefs(::Type{Float64}) = (0.3989422804014328, -0.7480167757536108, 0.32725734026661746, 0.12272117712518572, 0.1266192433674273, 0.1985002457586728, 0.9652668510022477, -22.515059045641358, 650.0016469706677, -5111.504274365941, -366285.0422528223, 1.866344682244392e7, -4.870990420065428e8, 8.501718935762064e9, -1.0580797337687886e11, 9.548560528330159e11, -6.211912811203429e12, 2.8410346605629793e13, -8.670581882322688e13, 1.586062121825567e14, -1.3160619165647164e14)
@inline besseli2_large_coefs(::Type{Float32}) =  (0.398942312409439f0, -0.7480453792346994f0, 0.3310342475515811f0)
@inline besseli2_large_coefs(::Type{Float64}) = (0.3989422804014327, -0.7480167757530116, 0.3272573406917582, 0.12271968486936348, 0.12759241505245672)

#### Derived special functions

@inline logbesseli0_small(x::T) where {T <: Union{Float32, Float64}} = (x² = abs2(x); return x² * evalpoly(x², logbesseli0_small_coefs(T))) # log(besselix(0, x)) loses accuracy near x = 0 since besselix(0, x) -> 1 as x -> 0
@inline logbesseli0(x::T) where {T <: Union{Float32, Float64}} = abs(x) < one(T) ? logbesseli0_small(x) : log(besseli0x(x)) + abs(x) # log(besselix(0, x)) = log(I0(x)) - |x|
@inline logbesseli0x(x::T) where {T <: Union{Float32, Float64}} = abs(x) < one(T) ? logbesseli0_small(x) - abs(x) : log(besseli0x(x))

@inline logbesseli0_small_coefs(::Type{Float32}) = (0.24999999426684533f0, -0.015624705149866972f0, 0.0017336759629143878f0, -0.00021666015596172704f0, 2.2059316402289948f-5)
@inline logbesseli0_small_coefs(::Type{Float64}) = (0.25, -0.015624999999997167, 0.0017361111109961576, -0.0002237955710956064, 3.092446434101836e-5, -4.455118991727041e-6, 6.600804196191383e-7, -9.949296105322181e-8, 1.483764672332753e-8, -1.968806398401359e-9, 1.6562710526172217e-10)

@inline logbesseli1(x::Union{Float32, Float64}) = logbesseli1x(x) + abs(x) # log(besselix(1, x)) = log(I1(x)) - |x|
@inline logbesseli1x(x::Union{Float32, Float64}) = log(besseli1x(x))

@inline logbesseli2(x::Union{Float32, Float64}) = logbesseli2x(x) + abs(x) # log(besselix(2, x)) = log(I2(x)) - |x|
@inline logbesseli2x(x::Union{Float32, Float64}) = log(besseli2x(x))

@inline laguerre½(x::T) where {T <: Union{Float32, Float64}} = ifelse(x < zero(T), one(x), exp(x)) * ((1 - x) * besseli0x(-x/2) - x * besseli1x(-x/2)) # besselix(ν, ±x/2) = Iν(±x/2) * exp(-|±x/2|) = Iν(-x/2) * exp(∓x/2)

"""
    besseli1i0(x::T) where {T <: Union{Float32, Float64}}

Ratio of modified Bessel functions of the first kind of orders one and zero, ``I_1(x) / I_0(x)``.
"""
@inline function besseli1i0(x::T) where {T <: Union{Float32, Float64}}
    if x < besseli1i0_low_cutoff(T)
        return x * evalpoly(x^2, besseli1i0_low_coefs(T)) # I₁(x) / I₀(x) ≈ x/2 + 𝒪(x^3)
    elseif x < besseli1i0_mid_cutoff(T)
        x² = x^2
        return x * evalpoly(x², besseli1i0_mid_num_coefs(T)) / evalpoly(x², besseli1i0_mid_den_coefs(T)) # I₁(x) / I₀(x)
    elseif x < besseli1i0_high_cutoff(T)
        x² = x^2
        return x * evalpoly(x², besseli1i0_high_num_coefs(T)) / evalpoly(x², besseli1i0_high_den_coefs(T)) # I₁(x) / I₀(x)
    else
        x⁻¹ = inv(x)
        tmp = evalpoly(x⁻¹, besseli1i0c_tail_coefs(T)) # P(1/x) = -x * (1/2 + x * (I₁(x) / I₀(x) - 1)) ≈ 1/8 + 1/8x + 𝒪(1/x^2)
        tmp = muladd(x⁻¹, tmp, T(0.5))
        return muladd(x⁻¹, -tmp, T(1.0)) # I₁(x) / I₀(x) = 1 - 1/2x - P(1/x)/x^2
    end
end

@inline besseli1i0_low_cutoff(::Type{T}) where {T} = T(0.5)
@inline besseli1i0_mid_cutoff(::Type{T}) where {T} = T(7.75)
@inline besseli1i0_high_cutoff(::Type{T}) where {T} = T(15.0)

@inline besseli1i0_low_coefs(::Type{Float32}) = (0.5f0, -0.0624989f0, 0.010394423f0, -0.0016448505f0)
@inline besseli1i0_mid_num_coefs(::Type{Float32}) = (0.49999964f0, 0.042507876f0, 0.0005986794f0, 8.2039816f-7)
@inline besseli1i0_mid_den_coefs(::Type{Float32}) = (1.0f0, 0.21001345f0, 0.0066175098f0, 3.363584f-5)
@inline besseli1i0_high_num_coefs(::Type{Float32}) = (0.4427933f0, 0.018132959f0, 9.000428f-5, 3.4805463f-8)
@inline besseli1i0_high_den_coefs(::Type{Float32}) = (1.0f0, 0.12933768f0, 0.0016975396f0, 2.7292274f-6)
@inline besseli1i0c_tail_coefs(::Type{Float32}) = (0.12500001f0, 0.12498587f0, 0.19689824f0, 0.34546292f0, 1.9343305f0)

# TODO: these coefficients may be suboptimal, but it's very tricky to choose good branch points and polynomial degrees to get a good fit in the middle region because Remez.jl keeps failing to converge
@inline besseli1i0_low_coefs(::Type{Float64}) = (0.4999999999999999, -0.06249999999994528, 0.010416666662044488, -0.001790364434468454, 0.0003092424332731733, -5.344192059352683e-5, 9.146096503297768e-6, -1.3486016148802727e-6)
@inline besseli1i0_mid_num_coefs(::Type{Float64}) = (0.49999999999999883, 0.052470559275092726, 0.0014603579674395833, 1.4865655828999244e-5, 5.7490818341880375e-8, 7.017125206341158e-11, 1.2004842788927884e-14)
@inline besseli1i0_mid_den_coefs(::Type{Float64}) = (1.0, 0.22994111855013522, 0.010830022420376213, 0.00017377331103972547, 1.0784649140018225e-6, 2.4131976302659546e-9, 1.328735170767829e-12)
@inline besseli1i0_high_num_coefs(::Type{Float64}) = (0.4998040084026318, 0.04956314429268099, 0.001200034937816074, 9.523479731806683e-6, 2.515523506038697e-8, 1.8096428999138172e-11, 1.5667018049770046e-15)
@inline besseli1i0_high_den_coefs(::Type{Float64}) = (1.0, 0.2239897265935517, 0.009595105452405184, 0.00012722326485112647, 5.780836248954148e-7, 8.23139782906151e-10, 2.480850142412075e-13)
@inline besseli1i0c_tail_coefs(::Type{Float64}) = (0.12500000000000017, 0.12499999999879169, 0.19531250150899987, 0.4062492562129355, 1.0480435526081948, 3.1889066971543234, 14.493937314937872, -164.07408273123662, 10554.06604261355, -363473.6613975362, 9.257867756487811e6, -1.6750893375624812e8, 2.1100222176196077e9, -1.752346161183495e10, 8.611676733884451e10, -1.88444663825226e11)

#### ChainRules and ForwardDiff

@inline ∂x_laguerre½(x::T) where {T <: Union{Float32, Float64}} = ifelse(x < zero(T), one(x), exp(x)) * (besseli1x(x/2) - besseli0x(x/2)) / 2
@scalar_rule laguerre½(x::Union{Float32, Float64}) ∂x_laguerre½(x)
@define_unary_dual_scalar_rule laguerre½ (laguerre½, ∂x_laguerre½)

@inline ∂x_besseli0x(Ω::T, x::T) where {T <: Union{Float32, Float64}} = besseli1x(x) - sign(x) * Ω
@inline f_∂x_besseli0x(x::Union{Float32, Float64}) = (Ω = besseli0x(x); return (Ω, ∂x_besseli0x(Ω, x)))
@scalar_rule besseli0x(x::Union{Float32, Float64}) ∂x_besseli0x(Ω, x)
@define_unary_dual_scalar_rule besseli0x f_∂x_besseli0x

@inline ∂x_besseli1x(Ω::T, x::T) where {T <: Union{Float32, Float64}} = (besseli0x(x) + besseli2x(x)) / 2 - sign(x) * Ω
@inline f_∂x_besseli1x(x::Union{Float32, Float64}) = (Ω = besseli1x(x); return (Ω, ∂x_besseli1x(Ω, x)))
@scalar_rule besseli1x(x::Union{Float32, Float64}) ∂x_besseli1x(Ω, x)
@define_unary_dual_scalar_rule besseli1x f_∂x_besseli1x
