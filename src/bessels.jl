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

@inline logbesseli0x_branches(::Type{Float32}) = (1.0f0, 2.012587f0, 2.9916978f0, 5.218942f0, 7.3152285f0)
@inline logbesseli0_taylor_coefs(::Type{Float32}) = (-0.015624976f0, 0.0017352961f0, -0.00021954361f0, 2.360602f-5)
@inline logbesseli0x_mid1_coefs(::Type{Float32}) = (-0.6663704f0, 0.088754125f0, -0.008902577f0, 0.00020709152f0, 0.00012540427f0, -2.313366f-5)
@inline logbesseli0x_mid1_transplant(::Type{Float32}) = (1.9751388f0, -2.975139f0)
@inline logbesseli0x_mid2_coefs(::Type{Float32}) = (-0.52341723f0, 0.056475032f0, -0.005849333f0, 0.0004616325f0, -1.0493147f-5, -4.2228025f-6)
@inline logbesseli0x_mid2_transplant(::Type{Float32}) = (2.04267f0, -5.111051f0)
@inline logbesseli0x_mid3_coefs(::Type{Float32}) = (-0.38710552f0, 0.06905217f0, -0.013259217f0, 0.0024660034f0, -0.00041200756f0, 4.962626f-5)
@inline logbesseli0x_mid3_transplant(::Type{Float32}) = (0.8979706f0, -3.6864567f0)
@inline logbesseli0x_mid4_coefs(::Type{Float32}) = (-0.28955814f0, 0.03443216f0, -0.004517907f0, 0.0006069093f0, -8.3052546f-5, 1.0799444f-5)
@inline logbesseli0x_mid4_transplant(::Type{Float32}) = (0.9540682f0, -5.9792266f0)
@inline logbesseli0x_tail_coefs(::Type{Float32}) = (0.12499995f0, 0.062525846f0, 0.06303961f0, 0.16020738f0, -0.49523312f0, 3.7963736f0)

@inline logbesseli0x_branches(::Type{Float64}) = (1.0, 2.4273313920504247, 4.368422375132864, 7.804751908461284, 13.115758945221774)
@inline logbesseli0_taylor_coefs(::Type{Float64}) = (-0.01562499999999995, 0.0017361111111010013, -0.00022379557257775202, 3.092447475294501e-5, -4.455160371599049e-6, 6.601794380225889e-7, -9.963839425018547e-8, 1.496590273686032e-8, -2.0311101682813397e-9, 1.784377125069117e-10)
@inline logbesseli0x_mid1_coefs(::Type{Float64}) = (-0.6314937374457755, 0.11501811747051277, -0.016964792253668346, 0.0010428006147571967, 0.0003096745255119058, -0.00011260405860526542, 1.4197326426499429e-5, 1.4817997937189319e-6, -1.0159423988239176e-6, 1.8542342958162965e-7, 1.6913322557800307e-9, -9.480097797535782e-9, 2.359654865642401e-9, -1.302358772144236e-10, -8.69022995592349e-11, 2.5713301162051096e-11, -1.7342796618966495e-12)
@inline logbesseli0x_mid1_transplant(::Type{Float64}) = (1.4012162915627544, -2.401216291562754)
@inline logbesseli0x_mid2_coefs(::Type{Float64}) = (-0.43702710599956285, 0.07786739972887159, -0.014495828275497476, 0.002460697007573363, -0.00032789433716279265, 1.795530460077842e-5, 7.104561708708271e-6, -3.172618163848316e-6, 7.758128274682523e-7, -1.256981784280769e-7, 8.415711530138176e-9, 2.819995831513679e-9, -1.3763063676487773e-9, 3.5864320608290456e-10, -6.155471302065167e-11, 1.2666722360704097e-12, 2.1595297700656985e-12)
@inline logbesseli0x_mid2_transplant(::Type{Float64}) = (1.0303484058351626, -3.500997030232801)
@inline logbesseli0x_mid3_coefs(::Type{Float64}) = (-0.29562474551688944, 0.059085218329401945, -0.013025193402085546, 0.0029345192192710553, -0.0006553570128493334, 0.0001412852524305638, -2.836580652872031e-5, 4.960058377886241e-6, -6.20290283257456e-7, -1.0894670349200352e-8, 4.259980984442922e-8, -1.8902339757216067e-8, 6.057649433517548e-9, -1.628234397068111e-9, 3.791867796750589e-10, -6.748341056153555e-11, 5.675981263485359e-12)
@inline logbesseli0x_mid3_transplant(::Type{Float64}) = (0.5820163580361878, -3.542493281138623)
@inline logbesseli0x_mid4_coefs(::Type{Float64}) = (-0.19886140852329215, 0.038026354208407886, -0.00802623574986359, 0.0017533187066348472, -0.0003890453861945658, 8.691902817038615e-5, -1.944322911623319e-5, 4.3343353371717545e-6, -9.58120274121212e-7, 2.0871660587257014e-7, -4.441203642100071e-8, 9.096637063856574e-9, -1.753361935382705e-9, 3.140720929793921e-10, -4.5848625092234583e-11, -2.8668536796293043e-12, 3.425379518957655e-12)
@inline logbesseli0x_mid4_transplant(::Type{Float64}) = (0.37657641689360727, -3.9390855084318934)
@inline logbesseli0x_tail_coefs(::Type{Float64}) = (0.12500000000000006, 0.06249999999967958, 0.06510416704462363, 0.10156232389156852, 0.20961334668207743, 0.5301120787106338, 2.250696664936657, -34.902839834523085, 1945.3838946369242, -65509.539033617555, 1.6320871386107295e6, -2.9438009334150616e7, 3.795109032267882e8, -3.3816374005861516e9, 1.9602149973462765e10, -6.555322646176011e10, 9.411821247781784e10)

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

@inline besseli2_small_coefs(::Type{Float32}) = (0.50000006f0, 0.16666625f0, 0.020834042f0, 0.0013884112f0, 5.803109f-5, 1.623226f-6, 3.775087f-8, 3.4031894f-10, 1.3439159f-11)
@inline besseli2_med_coefs(::Type{Float32}) = (0.39894232f0, -0.7480355f0, 0.32830036f0, 0.10303897f0, 0.26398978f0)
@inline besseli2_large_coefs(::Type{Float32}) = (0.39894232f0, -0.7480454f0, 0.33103424f0)

@inline besseli2_small_coefs(::Type{Float64}) = (0.49999999999999983, 0.16666666666667154, 0.020833333333312287, 0.0013888888889246186, 5.787037033873442e-5, 1.6534391702073172e-6, 3.444664327750705e-8, 5.467735445704131e-10, 6.834436191839501e-12, 6.906148607507894e-14, 5.733547826165566e-16, 4.1288365068666296e-18, 2.0258796870216445e-20, 1.958474603154919e-22)
@inline besseli2_med_coefs(::Type{Float64}) = (0.3989422804014328, -0.7480167757536108, 0.32725734026661746, 0.12272117712518572, 0.12661924336742728, 0.19850024575867328, 0.9652668510022131, -22.515059045639532, 650.001646970596, -5111.504274363786, -366285.0422528725, 1.8663446822444834e7, -4.870990420065559e8, 8.501718935762211e9, -1.0580797337688016e11, 9.548560528330247e11, -6.211912811203474e12, 2.841034660562996e13, -8.670581882322731e13, 1.5860621218255738e14, -1.3160619165647214e14)
@inline besseli2_large_coefs(::Type{Float64}) = (0.3989422804014327, -0.7480167757530116, 0.3272573406917582, 0.12271968486936348, 0.12759241505245666)

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

@inline laguerre½²c_small1_coefs(::Type{Float32}) = (0.5707963f0, -0.8584068f0, 0.7853828f0, -0.52343524f0, 0.2772832f0, -0.12166526f0, 0.043580707f0, -0.011369183f0, 0.0015599251f0)
@inline laguerre½²c_small2_coefs(::Type{Float32}) = (0.07509229f0, -0.04485518f0, 0.024552079f0, -0.011766576f0, 0.0048901024f0, -0.0017767233f0, 0.0005716923f0, -0.00016371648f0, 4.3126685f-5, -1.2159225f-5, 2.791931f-6)
@inline laguerre½²c_small2_transplant(::Type{Float32}) = (0.25f0, -2.0f0)
@inline laguerre½²c_small3_coefs(::Type{Float32}) = (0.021829125f0, -0.011498634f0, 0.006102043f0, -0.0032706815f0, 0.0017730005f0, -0.0009646032f0, 0.0005276015f0, -0.00030712632f0, 0.00016790304f0, -5.49553f-5, 2.5261792f-5, -3.8417656f-5, 1.837507f-5)
@inline laguerre½²c_small3_transplant(::Type{Float32}) = (0.083333336f0, -2.0f0)
@inline laguerre½²c_large_coefs(::Type{Float32}) = (0.50722855f0, 0.007533506f0, 0.00032818364f0, 2.5914162f-5, 3.1417717f-6, 5.391343f-7, 1.4190516f-7, 6.10655f-8, 1.523176f-8, -2.2704474f-9)
@inline laguerre½²c_large_transplant(::Type{Float32}) = (72.0f0, -1.0f0)

@inline laguerre½²c_small1_coefs(::Type{Float64}) = (0.5707963267948967, -0.8584073464102066, 0.7853981633974321, -0.5235987755976764, 0.278161849523994, -0.12435470904965956, 0.04854196856341194, -0.016985786503682567, 0.005430384746837412, -0.0016078871741822703, 0.00044501651519756414, -0.00011568705864588991, 2.8157619704925723e-5, -6.27245130459232e-6, 1.1988133148520703e-6, -1.7060604809840622e-7, 1.3038066466819481e-8)
@inline laguerre½²c_small2_coefs(::Type{Float64}) = (0.0750922939077032, -0.044855175550378656, 0.02455207206532838, -0.011766701500643757, 0.004890175170994685, -0.0017760270733748061, 0.0005714184721117123, -0.00016528947258048134, 4.359761938122429e-5, -1.0622713584690922e-5, 2.4187399906158563e-6, -5.198566828225391e-7, 1.0635744262248545e-7, -2.0857540914158283e-8, 3.937573527891655e-9, -7.118544089915365e-10, 1.255707237845092e-10, -2.4611454675903543e-11, 4.101243460992632e-12)
@inline laguerre½²c_small2_transplant(::Type{Float64}) = (0.25, -2.0)
@inline laguerre½²c_small3_coefs(::Type{Float64}) = (0.021829123227749478, -0.011498656624258922, 0.006102109294456777, -0.0032700434324974947, 0.0017720825242225386, -0.0009695987238590336, 0.0005324406798493483, -0.0002905439625996708, 0.0001557402734793816, -8.114063699385476e-5, 4.0759459211195254e-5, -1.964098868871727e-5, 9.056935949377753e-6, -3.994781626215287e-6, 1.6875742027698553e-6, -6.850068517395359e-7, 2.675006593737042e-7, -9.986721055142827e-8, 3.643165300348487e-8, -1.3966998410461667e-8, 4.9187263485043116e-9, -1.0492390879382305e-9, 3.2779939123533993e-10, -3.3654383743781266e-10, 1.1361941228163363e-10)
@inline laguerre½²c_small3_transplant(::Type{Float64}) = (0.08333333333333333, -2.0)
@inline laguerre½²c_large_coefs(::Type{Float64}) = (0.5072285353870277, 0.007533505907967427, 0.0003281831578511096, 2.591272457517579e-5, 3.1455208701366716e-6, 5.464308372395623e-7, 1.3253491762073007e-7, 4.535156161604887e-8, 2.2068828349851417e-8, 1.3167567843892215e-8, 4.442567019688479e-9, -5.624736376140391e-9, -8.01704904322673e-9, -7.911299629805929e-10, 3.6710262514452458e-9, 9.588970809096078e-10, -9.756504977745483e-10, -2.0974516065267094e-10, 1.2540677182453326e-10)
@inline laguerre½²c_large_transplant(::Type{Float64}) = (72.0, -1.0)

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

@inline besseli1i0_branches(::Type{Float32}) = (0.5f0, 5.2539897f0, 10.244816f0, 19.181583f0, 38.592796f0)
@inline besseli1i0_low_coefs(::Type{Float32}) = (0.5f0, -0.0624989f0, 0.010394423f0, -0.0016448505f0)
@inline besseli1i0_mid1_num_coefs(::Type{Float32}) = (0.49999997f0, 0.04353572f0, 0.000664983f0, 1.0970086f-6)
@inline besseli1i0_mid1_den_coefs(::Type{Float32}) = (1.0f0, 0.2120713f0, 0.007005701f0, 4.0410923f-5)
@inline besseli1i0_mid2_num_coefs(::Type{Float32}) = (0.49287614f0, 0.03344882f0, 0.00030526333f0, 2.2926868f-7)
@inline besseli1i0_mid2_den_coefs(::Type{Float32}) = (1.0f0, 0.18673065f0, 0.004251321f0, 1.2868143f-5)
@inline besseli1i0_mid3_num_coefs(::Type{Float32}) = (0.3814032f0, 0.0096941255f0, 2.9229745f-5, 6.8207853f-9)
@inline besseli1i0_mid3_den_coefs(::Type{Float32}) = (1.0f0, 0.08665983f0, 0.00070395553f0, 6.8782975f-7)
@inline besseli1i0_mid4_num_coefs(::Type{Float32}) = (0.23153244f0, 0.0016276878f0, 1.3243184f-6, 8.2235885f-11)
@inline besseli1i0_mid4_den_coefs(::Type{Float32}) = (1.0f0, 0.026988983f0, 6.0895945f-5, 1.6060818f-8)
@inline besseli1i0c_tail_coefs(::Type{Float32}) = (0.12499999f0, 0.12500551f0, 0.19427566f0, 0.46759263f0)

@inline besseli1i0_branches(::Type{Float64}) = (0.5, 7.515971274223541, 12.995005502686293, 20.456767637632435, 32.23120437631026)
@inline besseli1i0_low_coefs(::Type{Float64}) = (0.5, -0.06249999999999999, 0.0104166666666654, -0.0017903645832681267, 0.00030924478996177326, -5.346225011561096e-5, 9.243845992068521e-6, -1.597088340378288e-6, 2.7173295526886746e-7, -3.86430832554434e-8)
@inline besseli1i0_mid1_num_coefs(::Type{Float64}) = (0.5, 0.05051257275549395, 0.0012611738462395938, 9.788747912878983e-6, 1.5777297769408054e-8, -1.6912776880190478e-11, 3.9157264469423244e-14, -1.1131493430228754e-16, 3.2423010302481025e-19, -8.493282510185491e-22, 1.7011724320890625e-24, -1.8824009065016947e-27)
@inline besseli1i0_mid1_den_coefs(::Type{Float64}) = (1.0, 0.2260251455109879, 0.009942157548019366, 0.0001342191578492413, 5.259146742416428e-7)
@inline besseli1i0_mid2_num_coefs(::Type{Float64}) = (0.4999959419330061, 0.049730915009281514, 0.001189978099945009, 8.479918876568225e-6, 1.1776977719687032e-8, -1.0024897612610195e-11, 1.6613677903552577e-14, -2.9975348671507035e-17, 4.8666415934125773e-20, -6.237589566817162e-23, 5.414840689481311e-26, -2.3386499201170585e-29)
@inline besseli1i0_mid2_den_coefs(::Type{Float64}) = (1.0, 0.22445856510880843, 0.00960471825036602, 0.00012190896402176206, 4.2640656363050543e-7)
@inline besseli1i0_mid3_num_coefs(::Type{Float64}) = (0.4893835515722947, 0.03947303528081722, 0.000673447634678756, 3.0531643081541053e-6, 2.4394093737493614e-9, -1.1059389659786812e-12, 9.125651909755847e-16, -7.772328741930979e-19, 5.714793200998154e-22, -3.213101634787341e-25, 1.1940832849865227e-28, -2.1670094850761264e-32)
@inline besseli1i0_mid3_den_coefs(::Type{Float64}) = (1.0, 0.19780038684494777, 0.006456332433511402, 5.5327495093660825e-5, 1.1743775976109534e-7)
@inline besseli1i0_mid4_num_coefs(::Type{Float64}) = (0.4018154544324209, 0.015319062735257509, 0.00011549534088144715, 2.2421905564927978e-7, 7.524438683982764e-11, -1.424969873544215e-14, 4.872327888929994e-18, -1.7099878558448984e-21, 5.160003375939312e-25, -1.1870317989011305e-28, 1.8007236155905433e-32, -1.331531532197771e-36)
@inline besseli1i0_mid4_den_coefs(::Type{Float64}) = (1.0, 0.1086273760146177, 0.0016417116904751817, 6.164327133597336e-6, 5.5812466487748526e-9)
@inline besseli1i0c_tail_coefs(::Type{Float64}) = (0.125, 0.1249999999999999, 0.19531250000016376, 0.4062499998866536, 1.047851603178, 3.218741339509917, 11.467639207339296, 46.371873585506364, 217.8164210629866, 793.715143893347, 13264.204676401096, -86990.15182730988, 1.2614025408069761e6)

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

@inline neglogpdf_rician_parts_branches(::Type{Float32}) = (1.25f0, 1.7736309f0, 2.4552262f0, 3.368769f0, 4.736905f0, 6.865108f0, 11.346635f0, 30.0f0)
@inline neglogpdf_rician_parts_taylor_branch(::Type{Float32}) = first(neglogpdf_rician_parts_branches(Float32))
@inline a1_taylor_coefs(::Type{Float32}) = (-0.0625f0, 0.010416648f0, -0.001790165f0, 0.00030844187f0, -5.189695f-5, 7.617877f-6, -6.7984206f-7)
@inline b3_mid1_coefs(::Type{Float32}) = (-0.9786126f0, -0.628833f0, 0.23862931f0, -0.039429966f0, 0.0010880656f0, 0.001399136f0, -0.0005035984f0, 0.000111060166f0, -1.5574844f-5)
@inline b3_mid1_transplant(::Type{Float32}) = (8.467945f0, -5.774356f0)
@inline b3_mid2_coefs(::Type{Float32}) = (0.8829879f0, -1.0979296f0, 0.12853086f0, 0.022795105f0, -0.010818173f0, 0.0019171516f0, -0.00012398159f0, -3.2030835f-5, 1.3861369f-5, -2.6926102f-6)
@inline b3_mid2_transplant(::Type{Float32}) = (12.777861f0, -6.2043514f0)
@inline b3_mid3_coefs(::Type{Float32}) = (2.896896f0, -0.73595107f0, -0.13142815f0, 0.04830029f0, -0.0031885663f0, -0.0010850336f0, 0.00034242112f0, -4.383489f-5, -9.420497f-7, 1.4751752f-6)
@inline b3_mid3_transplant(::Type{Float32}) = (18.107723f0, -6.3751755f0)
@inline b3_mid4_coefs(::Type{Float32}) = (3.5190217f0, 0.1255342f0, -0.19634402f0, -0.0047439034f0, 0.009102384f0, -0.001065171f0, -0.00016898877f0, 7.0777445f-5, -8.131132f-6)
@inline b3_mid4_transplant(::Type{Float32}) = (23.327414f0, -5.9246106f0)
@inline b3_mid5_coefs(::Type{Float32}) = (2.8098564f0, 0.42554978f0, 0.0014337796f0, -0.024722438f0, -0.0014616672f0, 0.0012273814f0, -3.6492107f-5, -4.775254f-5, 7.9661195f-6)
@inline b3_mid5_transplant(::Type{Float32}) = (30.560398f0, -5.451554f0)
@inline b3_mid6_coefs(::Type{Float32}) = (2.1356993f0, 0.22855726f0, 0.039798003f0, 0.005707509f0, -0.0012225008f0, -0.00080096425f0, -3.433441f-6, 6.214366f-5, -2.2442087f-7, -3.887005f-6)
@inline b3_mid6_transplant(::Type{Float32}) = (34.763096f0, -4.063736f0)
@inline b3_high_coefs(::Type{Float32}) = (1.5630388f0, 3.1534708f0, 15.986863f0, -319.89398f0, 10080.004f0, -189825.39f0, 2.3859912f6, -1.89319f7, 8.625075f7, -1.6949213f8)
@inline b3_tail_coefs(::Type{Float32}) = (1.5625f0, 3.2500045f0, 8.381256f0, 25.940771f0, 81.37432f0, 617.13983f0)

@inline neglogpdf_rician_parts_branches(::Type{Float64}) = (1.25, 1.8591389603935717, 2.74690836511996, 4.14578464932015, 6.2763334877759585, 10.033473692870817, 18.13555275269315, 100.0)
@inline neglogpdf_rician_parts_taylor_branch(::Type{Float64}) = first(neglogpdf_rician_parts_branches(Float64))
@inline a1_taylor_coefs(::Type{Float64}) = (-0.0625, 0.010416666666666652, -0.0017903645833326689, 0.00030924479165391074, -5.346227562391155e-5, 9.244080424735487e-6, -1.5984263283747596e-6, 2.763836785633796e-7, -4.777587188953313e-8, 8.24080691209807e-9, -1.4050898275433108e-9, 2.2912333392066555e-10, -3.291104278817345e-11, 3.5409731367567005e-12, -2.0285970507152215e-13)
@inline b3_mid1_coefs(::Type{Float64}) = (-0.9066413097635835, -0.7576166768987637, 0.3099752233041559, -0.05429950799927307, 0.0003377171930970658, 0.0029644190876579298, -0.001106126822657037, 0.0002558871579646217, -3.988749239329138e-5, 2.470752511931632e-6, 9.599678752277343e-7, -4.7257379710467814e-7, 1.3323835889587424e-7, -2.880940733416941e-8, 4.956490794786943e-9, -6.398775838228047e-10, 2.6043808329355625e-11, 3.014949433628068e-11, -1.0624538418735766e-11)
@inline b3_mid1_transplant(::Type{Float64}) = (7.630192293037538, -5.104153834430031)
@inline b3_mid2_coefs(::Type{Float64}) = (1.390956666335002, -1.3261856780715606, 0.1035081495610392, 0.06277677233954594, -0.023247239418819742, 0.003439358223932491, 0.00013000771801100055, -0.00020941167052518795, 6.190030052072196e-5, -1.0890878614540761e-5, 8.937006850991466e-7, 1.7147135870814256e-7, -9.690206109012347e-8, 2.6201786000440738e-8, -5.083126254923448e-9, 7.22055998135966e-10, -4.4019035579894205e-11, -2.479387676836325e-11, 9.181282086005465e-12)
@inline b3_mid2_transplant(::Type{Float64}) = (11.50497941252994, -5.188337535616157)
@inline b3_mid3_coefs(::Type{Float64}) = (3.4145482574172124, -0.4227721531138229, -0.3276628400826416, 0.0622001208463868, 0.01029482222421844, -0.005926467561233111, 0.0008970678024529246, 7.397868180176319e-5, -6.746192988764895e-5, 1.6511729219022485e-5, -1.8099459239964246e-6, -2.2097591087340852e-7, 1.5887283071487396e-7, -4.253054437842285e-8, 6.977568068658435e-9, -4.4886761292700706e-10, -1.767380549512039e-10, 9.969010891883516e-11, -2.2581765036901864e-11)
@inline b3_mid3_transplant(::Type{Float64}) = (16.281769391372023, -4.927307076609009)
@inline b3_mid4_coefs(::Type{Float64}) = (3.0875520554759373, 0.49260384416832537, -0.07613928136229342, -0.046292528807978225, 0.0049246550911270625, 0.002359863485903642, -0.0005983474246044642, -2.3060304387410408e-5, 3.610106592013356e-5, -7.0212987531816784e-6, -9.058549482842997e-8, 3.713155981398159e-7, -9.703792866133195e-8, 9.573925526822241e-9, 1.8996970008309483e-9, -1.036284938942767e-9, 2.0196754718939483e-10, -1.088604629000335e-11)
@inline b3_mid4_transplant(::Type{Float64}) = (24.425938103812463, -4.89175274885973)
@inline b3_mid5_coefs(::Type{Float64}) = (2.243863900499485, 0.2760032053911868, 0.04864767722460813, 0.0023431551414816983, -0.0032518621851148886, -0.0006949504496972516, 0.00019428110870631094, 4.433139897135638e-5, -1.6231383985795364e-5, -9.984193290415918e-7, 1.1889505982856505e-6, -1.5023399528174868e-7, -4.340344904513636e-8, 1.8924462506820824e-8, -1.893520873907179e-9, -6.975658533746278e-10, 2.974572754191339e-10, -2.217255760245679e-11, -9.262190816287589e-12)
@inline b3_mid5_transplant(::Type{Float64}) = (33.521999978541814, -4.3410163822286725)
@inline b3_mid6_coefs(::Type{Float64}) = (1.8811827581502512, 0.11807211467479745, 0.01071928532567251, 0.0014457869330265793, 0.00028050300529725903, 5.26750373422372e-5, -1.4607343764699284e-6, -6.249048440457558e-6, -1.6095447378919227e-6, 3.35068396617675e-7, 1.9753540621921212e-7, -2.3357185643437115e-8, -1.9038363482639613e-8, 3.1061572212835498e-9, 1.5395444954325444e-9, -4.3869037019394584e-10, -7.930629319060699e-11, 3.738869437164717e-11)
@inline b3_mid6_transplant(::Type{Float64}) = (44.9175057676635, -3.47676519046232)
@inline b3_high_coefs(::Type{Float64}) = (1.562499999848177, 3.2500001099174156, 8.382776029784255, 25.75736138498818, 90.71989653431152, 472.250342375048, -5755.586780614253, 429203.2029954704, -1.8253063358811587e7, 6.14837763229502e8, -1.5853769529729286e10, 3.100397939020194e11, -4.474257254144003e12, 4.53759864819417e13, -2.918569899908291e14, 8.72925274832046e14, 1.2022856782957382e15, -1.203682272130206e16)
@inline b3_tail_coefs(::Type{Float64}) = (1.5625, 3.249999999999951, 8.382812500129667, 25.749999868034124, 91.73175661609746, 371.80898186259145, 1693.3965569777363, 8211.619321835962, 62507.00709166558)

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
