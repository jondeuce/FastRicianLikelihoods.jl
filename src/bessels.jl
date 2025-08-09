####
#### Fast + differentiable Bessel functions
####

# Bessels.jl provides `besseli0x` and `besseli1x`, but not `besseli2x`
@inline besseli0(x) = Bessels.besseli0(x)
@inline besseli0x(x) = Bessels.besseli0x(x)

@inline besseli1(x) = Bessels.besseli1(x)
@inline besseli1x(x) = Bessels.besseli1x(x)

"""
    logbesseli0x(x::T) where T <: Union{Float32, Float64}

Log of scaled modified Bessel function of the first kind of order zero, ``log(I_0(x)*e^{-x})``.
"""
@inline function logbesseli0x(x::Float32)
    T = Float32
    branch1, branch2, branch3, branch4 = logbesseli0x_branches(T)
    x = abs(x)
    if x < 1
        return logbesseli0_taylor(x) - x
    elseif x < branch1
        return x * evalpoly(x, logbesseli0x_branch1_coefs(T))
    elseif x < branch2
        return x * evalpoly(x, logbesseli0x_branch2_coefs(T))
    elseif x < branch3
        return x * evalpoly(x, logbesseli0x_branch3_coefs(T))
    elseif x < branch4
        return x * evalpoly(x, logbesseli0x_branch4_coefs(T))
    else
        return evalpoly(inv(x), logbesseli0x_tail_coefs(T)) - log(x) / 2
    end
end
@inline function logbesseli0x(x::Float64)
    T = Float64
    branch1, branch2, branch3, branch4 = logbesseli0x_branches(T)
    x = abs(x)
    if x < 1
        return logbesseli0_taylor(x) - x
    elseif x < branch1
        return x * evalpoly(x, logbesseli0x_branch1_num_coefs(T)) / evalpoly(x, logbesseli0x_branch1_den_coefs(T))
    elseif x < branch2
        return x * evalpoly(x, logbesseli0x_branch2_num_coefs(T)) / evalpoly(x, logbesseli0x_branch2_den_coefs(T))
    elseif x < branch3
        return x * evalpoly(x, logbesseli0x_branch3_num_coefs(T)) / evalpoly(x, logbesseli0x_branch3_den_coefs(T))
    elseif x < branch4
        return x * evalpoly(x, logbesseli0x_branch4_num_coefs(T)) / evalpoly(x, logbesseli0x_branch4_den_coefs(T))
    else
        return evalpoly(inv(x), logbesseli0x_tail_coefs(T)) - log(x) / 2
    end
end
@inline logbesseli0(x::Real) = abs(x) < one(x) ? logbesseli0_taylor(x) : logbesseli0x(x) + abs(x) # log(I0(x)) = log(besselix(0, x)) + |x|

@inline logbesseli0x_branches(::Type{Float32}) = (2.0f0, 3.0f0, 4.5f0, 6.25f0)
@inline logbesseli0x_branch1_coefs(::Type{Float32}) = (-0.9994487f0, 0.24674983f0, 0.007880564f0, -0.02573334f0, 0.007163849f0, -0.0006978541f0)
@inline logbesseli0x_branch2_coefs(::Type{Float32}) = (-1.0187888f0, 0.29313418f0, -0.037066877f0, -0.003718994f0, 0.001711131f0, -0.00015129537f0)
@inline logbesseli0x_branch3_coefs(::Type{Float32}) = (-1.0552132f0, 0.35522184f0, -0.07971522f0, 0.011032459f0, -0.000856916f0, 2.8618902f-5)
@inline logbesseli0x_branch4_coefs(::Type{Float32}) = (-1.0204555f0, 0.31935093f0, -0.06491181f0, 0.007980039f0, -0.0005426071f0, 1.5698264f-5)
@inline logbesseli0x_tail_coefs(::Type{Float32}) = (-0.9189385f0, 0.12500004f0, 0.06251328f0, 0.06340634f0, 0.15677911f0, -0.4903953f0, 3.8373053f0)

@inline logbesseli0x_branches(::Type{Float64}) = (2.0, 3.25, 5.0, 9.0)
@inline logbesseli0x_branch1_num_coefs(::Type{Float64}) = (-0.999999583120092, -0.29014047213447114, -0.39736271732758155, -0.08588147980086669, -0.03237626537907129, -0.004974271652449432, -5.556627549008132e-5)
@inline logbesseli0x_branch1_den_coefs(::Type{Float64}) = (1.0, 0.5401364144817467, 0.5324149799563537, 0.20331083501027733, 0.07485514008307156, 0.01698675965644882, 0.0021746610380462936)
@inline logbesseli0x_branch2_num_coefs(::Type{Float64}) = (-0.9999353256135443, -0.2722412308582146, -0.11292177835689929, -0.032528368383928814, 0.00030382545025762017, -0.0010141994682156873, -6.4711206283102365e-6)
@inline logbesseli0x_branch2_den_coefs(::Type{Float64}) = (1.0, 0.5218789555967246, 0.24432070766138303, 0.07654991556809776, 0.012142788791123238, 0.0009534245109375208, 0.00041679630031355927)
@inline logbesseli0x_branch3_num_coefs(::Type{Float64}) = (-1.0009452748298309, -0.011327281044590451, -0.12146320544136592, 0.006024767096478824, -0.002434365653126344, 0.00024240550222782585, 1.0119682492473649e-6)
@inline logbesseli0x_branch3_den_coefs(::Type{Float64}) = (1.0, 0.26483858319922615, 0.18176906263003073, 0.029622874557209883, 0.002076350050709619, 0.0005369824734025731, -9.515200972041569e-5)
@inline logbesseli0x_branch4_num_coefs(::Type{Float64}) = (-1.0134548793484943, -0.04809351298992965, -0.16990476185557626, 0.000854632825008215, -0.005080277098134271, -0.0002441769568189205, -2.42738849558008e-7)
@inline logbesseli0x_branch4_den_coefs(::Type{Float64}) = (1.0, 0.3330697949534713, 0.2119569922180127, 0.06485846738742465, 0.004126388294682845, 0.0024780499456405454, 7.310909443124124e-5)
@inline logbesseli0x_tail_coefs(::Type{Float64}) = (-0.9189385332046728, 0.12500000000000208, 0.06249999998635532, 0.06510418148686051, 0.10155612551537994, 0.21101443760192773, 0.3377647017909049, 19.689505081763162, -1136.2923646828697, 52138.794249036684, -1.7545055923722375e6, 4.418531972967025e7, -8.37239151644251e8, 1.1923572636252838e10, -1.2649126846844588e11, 9.811859382815109e11, -5.386620434336986e12, 1.9769950390805406e13, -4.3437377427437695e13, 4.3162096163039664e13)

@inline logbesseli0_taylor(x::Real) = (T = checkedfloattype(x); x² = abs2(x); return x² * evalpoly(x², logbesseli0_taylor_coefs(T))) # log(besselix(0, x)) loses accuracy near x = 0 since besselix(0, x) -> 1 as x -> 0
@inline logbesseli0_taylor_coefs(::Type{Float32}) = (0.25f0, -0.015624976f0, 0.0017352961f0, -0.00021954361f0, 2.360602f-5)
@inline logbesseli0_taylor_coefs(::Type{Float64}) = (0.25, -0.015624999999998638, 0.0017361111108868327, -0.00022379556683046643, 3.092441544406814e-5, -4.4548540542767594e-6, 6.592913431306256e-7, -9.812670153985689e-8, 1.3466859972860875e-8, -1.229771795305333e-9)

"""
    besseli2(x::T) where {T <: Union{Float32, Float64}}

Modified Bessel function of the first kind of order two, ``I_2(x)``.
"""
@inline function besseli2(x::Real)
    T = checkedfloattype(x)
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
@inline function besseli2x(x::Real)
    T = checkedfloattype(x)
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

@inline besseli2_small_coefs(::Type{Float32}) = (0.50000006f0, 0.16666625f0, 0.020834042f0, 0.0013884112f0, 5.803109f-5, 1.623226f-6, 3.775087f-8, 3.4031894f-10, 1.3439159f-11)
@inline besseli2_small_coefs(::Type{Float64}) = (0.49999999999999983, 0.16666666666667154, 0.020833333333312287, 0.0013888888889246186, 5.787037033873442e-5, 1.6534391702073172e-6, 3.444664327750705e-8, 5.467735445704131e-10, 6.834436191839501e-12, 6.906148607507894e-14, 5.733547826165566e-16, 4.1288365068666296e-18, 2.0258796870216445e-20, 1.958474603154919e-22)
@inline besseli2_med_coefs(::Type{Float32}) = (0.39894232f0, -0.7480355f0, 0.32830036f0, 0.10303897f0, 0.26398978f0)
@inline besseli2_med_coefs(::Type{Float64}) = (0.3989422804014328, -0.7480167757536108, 0.32725734026661746, 0.12272117712518572, 0.12661924336742728, 0.19850024575867328, 0.9652668510022121, -22.51505904563947, 650.0016469705932, -5111.50427436369, -366285.04225287505, 1.8663446822444886e7, -4.8709904200655675e8, 8.501718935762221e9, -1.0580797337688026e11, 9.548560528330254e11, -6.211912811203479e12, 2.8410346605629977e13, -8.670581882322736e13, 1.5860621218255747e14, -1.3160619165647222e14)
@inline besseli2_large_coefs(::Type{Float32}) = (0.39894232f0, -0.7480454f0, 0.33103424f0)
@inline besseli2_large_coefs(::Type{Float64}) = (0.3989422804014327, -0.7480167757530116, 0.3272573406917582, 0.12271968486936348, 0.12759241505245666)

#### Derived special functions

@inline logbesseli1(x::Real) = logbesseli1x(x) + abs(x) # log(besselix(1, x)) = log(I1(x)) - |x|
@inline logbesseli1x(x::Real) = log(besseli1x(x))

@inline logbesseli2(x::Real) = logbesseli2x(x) + abs(x) # log(besselix(2, x)) = log(I2(x)) - |x|
@inline logbesseli2x(x::Real) = log(besseli2x(x))

@inline laguerre½(x::Real) = (x < zero(x) ? one(x) : exp(x)) * ((1 - x) * besseli0x(-x / 2) - x * besseli1x(-x / 2)) # besselix(ν, ±x/2) = Iν(±x/2) * exp(-|±x/2|) = Iν(-x/2) * exp(∓x/2)

@inline function laguerre½²c(t::Real)
    # laguerre½²c(t) = L^2 - t^2 - 1 where L = √(π/2) * laguerre½(-t^2/2)
    T = checkedfloattype(t)
    branch = T == Float32 ? 3.4f0 : 4.3e0
    if t < branch
        y = t^2
        return evalpoly(y, laguerre½²c_small_coefs(T))
    else
        y = inv(t)^2
        return y * evalpoly(y, laguerre½²c_large_coefs(T))
    end
end

@inline laguerre½²c_small_coefs(::Type{Float32}) = (0.5707962f0, -0.21459818f0, 0.049073838f0, -0.0081615625f0, 0.0010717527f0, -0.0001148001f0, 9.935807f-6, -6.625331f-7, 3.134115f-8, -9.190797f-10, 1.2392542f-11)
@inline laguerre½²c_small_coefs(::Type{Float64}) = (0.5707963267944944, -0.21460183658265033, 0.049087385048506654, -0.008181230332361096, 0.0010865687931868046, -0.00012143915271414923, 1.1850357912302052e-5, -1.0363714628793908e-6, 8.272717280746797e-8, -6.095692789362868e-9, 4.1612417953794933e-10, -2.617973804424017e-11, 1.4957198539034854e-12, -7.583260826274592e-14, 3.3138533832359202e-15, -1.2069999116679684e-16, 3.524893658912788e-18, -7.863203684119532e-20, 1.2499337199986914e-21, -1.254956407795466e-23, 5.96116567823201e-26)
@inline laguerre½²c_large_coefs(::Type{Float32}) = (0.5000001f0, 0.49966717f0, 1.5056641f0, -12.993781f0, 1426.22f0, -51440.547f0, 991048.25f0, -7.3582105f6, -2.198495f7, 5.8008864f8, -2.2002685f9)
@inline laguerre½²c_large_coefs(::Type{Float64}) = (0.5000000000007456, 0.49999998836832427, 1.375030137769133, 6.344086335703835, 58.50578586624112, -5115.534157744382, 1.1866205173287583e6, -1.779111923731936e8, 1.9293241453016853e10, -1.5425475363829568e12, 9.230350664567888e13, -4.16652806576365e15, 1.4220659445965458e17, -3.658456330858182e18, 7.027759437240608e19, -9.895548988767412e20, 9.874128025778949e21, -6.5532901003608015e22, 2.5238431384199616e23, -3.597153866972595e23, -4.337934371146472e23)

"""
    besseli1i0(x::T) where {T <: Union{Float32, Float64}}

Ratio of modified Bessel functions of the first kind of orders one and zero, ``I_1(x) / I_0(x)``.
"""
@inline besseli1i0(x::Real) = _besseli1i0_parts(x)[1]
@inline besseli1i0x(x::Real) = _besseli1i0_parts(x)[2]
@inline besseli1i0m1(x::Real) = _besseli1i0_parts(x)[3]

@inline function _besseli1i0_parts(x::Real)
    T = checkedfloattype(x)
    if x < besseli1i0_low_cutoff(T)
        x² = x^2
        rx = evalpoly(x², besseli1i0_low_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) = 1/2 + 𝒪(x^2)
        r = x * rx # I₁(x) / I₀(x) = x * P(x^2) = x/2 + 𝒪(x^3)
        rm1 = r - one(T) # I₁(x) / I₀(x) - 1 = -1 + x/2 + 𝒪(x^3)
        rm1_tail = T(NaN) # unused
        r²m1 = rm1 * (one(T) + r) # (I₁(x) / I₀(x))^2 - 1
        r²m1prx = r²m1 + rx
        return r, rx, rm1, rm1_tail, r²m1, r²m1prx
    elseif x < besseli1i0_mid_cutoff(T)
        x² = x^2
        rx = evalpoly(x², besseli1i0_mid_num_coefs(T)) / evalpoly(x², besseli1i0_mid_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2)
        r = x * rx # I₁(x) / I₀(x) = x * P(x^2) / Q(x^2)
        rm1 = r - one(T) # I₁(x) / I₀(x) - 1 = -1 + x * P(x^2) / Q(x^2)
        rm1_tail = T(NaN) # unused
        r²m1 = rm1 * (one(T) + r) # (I₁(x) / I₀(x))^2 - 1
        r²m1prx = r²m1 + rx
        return r, rx, rm1, rm1_tail, r²m1, r²m1prx
    elseif x < besseli1i0_high_cutoff(T)
        x² = x^2
        rx = evalpoly(x², besseli1i0_high_num_coefs(T)) / evalpoly(x², besseli1i0_high_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2)
        r = x * rx # I₁(x) / I₀(x) = x * P(x^2) / Q(x^2)
        rm1 = r - one(T) # I₁(x) / I₀(x) - 1 = -1 + x * P(x^2) / Q(x^2)
        rm1_tail = T(NaN) # unused
        r²m1 = rm1 * (one(T) + r) # (I₁(x) / I₀(x))^2 - 1
        r²m1prx = r²m1 + rx
        return r, rx, rm1, rm1_tail, r²m1, r²m1prx
    else
        x⁻¹ = inv(x)
        rm1_tail = x⁻¹ * evalpoly(x⁻¹, besseli1i0c_tail_coefs(T)) # P(1/x) / x = -1/2 + x * (1 - I₁(x) / I₀(x)) = 1/8x + 1/8x^2 + 25/128x^3 + 𝒪(1/x^4)
        rm1 = x⁻¹ * (T(-0.5) - rm1_tail) # I₁(x) / I₀(x) - 1 = -1/2x - P(1/x)/x^2
        r = rm1 + one(T)
        rx = x⁻¹ * r
        r²m1 = rm1 * (one(T) + r) # (I₁(x) / I₀(x))^2 - 1
        r²m1prx = -x⁻¹^2 * evalpoly(x⁻¹, besseli1i0sqm1pi1i0x_tail_coefs(T)) # -P(1/x) / x^2 = -((I₁(x) / I₀(x))^2 - 1 + I₁(x) / I₀(x) / x) = 1/2x^2 + 1/4x^3 + 3/8x^4 + 𝒪(1/x^5)
        return r, rx, rm1, rm1_tail, r²m1, r²m1prx
    end
end

@inline besseli1i0_low_cutoff(::Type{T}) where {T} = T(0.5)
@inline besseli1i0_mid_cutoff(::Type{T}) where {T} = T(7.75)
@inline besseli1i0_high_cutoff(::Type{T}) where {T} = T(15.0)

@inline besseli1i0_low_coefs(::Type{Float32}) = (0.5f0, -0.0624989f0, 0.010394423f0, -0.0016448505f0)
@inline besseli1i0_mid_num_coefs(::Type{Float32}) = (0.49999997f0, 0.045409925f0, 0.00081378774f0, 2.6481487f-6)
@inline besseli1i0_mid_den_coefs(::Type{Float32}) = (1.0f0, 0.21581972f0, 0.0077718287f0, 6.1209554f-5, 5.342459f-8)
@inline besseli1i0_high_num_coefs(::Type{Float32}) = (0.4427933f0, 0.018132959f0, 9.000428f-5, 3.4805463f-8)
@inline besseli1i0_high_den_coefs(::Type{Float32}) = (1.0f0, 0.12933768f0, 0.0016975396f0, 2.7292274f-6)
@inline besseli1i0c_tail_coefs(::Type{Float32}) = (0.12500001f0, 0.12498587f0, 0.19689824f0, 0.34546292f0, 1.9343305f0)
@inline besseli1i0sqm1pi1i0x_tail_coefs(::Type{Float32}) = (0.50000006f0, 0.24997225f0, 0.3781122f0, 0.66200954f0, 3.7685757f0)

@inline besseli1i0_low_coefs(::Type{Float64}) = (0.4999999999999999, -0.06249999999994528, 0.010416666662044488, -0.001790364434468454, 0.0003092424332731733, -5.344192059352683e-5, 9.146096503297768e-6, -1.3486016148802727e-6)
@inline besseli1i0_mid_num_coefs(::Type{Float64}) = (0.5, 0.05310262646313703, 0.0015251216891410731, 1.6554273263234744e-5, 7.23879268360211e-8, 1.1223461073961568e-10, 3.2770305625089195e-14, -4.187086265904127e-18)
@inline besseli1i0_mid_den_coefs(::Type{Float64}) = (1.0, 0.23120525292627392, 0.011117566660733302, 0.00018675744315389288, 1.267293455710655e-6, 3.3622476790203076e-9, 2.6172247682003815e-12)
@inline besseli1i0_high_num_coefs(::Type{Float64}) = (0.49997216621078, 0.051199025817031056, 0.0013405622969567084, 1.2254189207681713e-5, 4.069839887346411e-8, 4.231106956361109e-11, 7.197568971764241e-15, -4.641694561673169e-19)
@inline besseli1i0_high_den_coefs(::Type{Float64}) = (1.0, 0.22737775033918348, 0.010274538971299216, 0.00015166980814467748, 8.257914214448346e-7, 1.5687317520073357e-9, 7.658428486442891e-13)
@inline besseli1i0c_tail_coefs(::Type{Float64}) = (0.12500000000000017, 0.12499999999879169, 0.19531250150899987, 0.4062492562129355, 1.0480435526081948, 3.1889066971543234, 14.493937314937872, -164.07408273123662, 10554.06604261355, -363473.6613975362, 9.257867756487811e6, -1.6750893375624812e8, 2.1100222176196077e9, -1.752346161183495e10, 8.611676733884451e10, -1.88444663825226e11)
@inline besseli1i0sqm1pi1i0x_tail_coefs(::Type{Float64}) = (0.5000000000000003, 0.2499999999975866, 0.3750000030140739, 0.7812485143050504, 2.031633509653731, 6.227493643819341, 28.57924498983948, -328.90603776073607, 21081.741880506986, -726212.8856050207, 1.8497370163125448e7, -3.3470137500038964e8, 4.21624304589359e9, -3.5017062058175514e10, 1.7209582910664124e11, -3.766118906296537e11)

#### ChainRules and ForwardDiff

@inline ∂x_laguerre½(x::Real) = (x < zero(x) ? one(x) : exp(x)) * (besseli1x(x / 2) - besseli0x(x / 2)) / 2
@scalar_rule laguerre½(x) ∂x_laguerre½(x)
@dual_rule_from_frule laguerre½(x)

@inline ∂x_besseli0x(Ω::Real, x::Real) = besseli1x(x) - sign(x) * Ω
@inline f_∂x_besseli0x(x::Real) = (Ω = besseli0x(x); return (Ω, ∂x_besseli0x(Ω, x)))
@scalar_rule besseli0x(x) ∂x_besseli0x(Ω, x)
@dual_rule_from_frule besseli0x(x)

@inline ∂x_besseli1x(Ω::Real, x::Real) = (besseli0x(x) + besseli2x(x)) / 2 - sign(x) * Ω
@inline f_∂x_besseli1x(x::Real) = (Ω = besseli1x(x); return (Ω, ∂x_besseli1x(Ω, x)))
@scalar_rule besseli1x(x) ∂x_besseli1x(Ω, x)
@dual_rule_from_frule besseli1x(x)
