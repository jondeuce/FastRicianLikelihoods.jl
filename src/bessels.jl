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
@inline function logbesseli0x(x::Real)
    T = checkedfloattype(x)
    low, mid1, mid2, mid3, high = logbesseli0x_branches(T)
    x = abs(x)
    if x < low
        return logbesseli0_taylor(x) - x
    elseif x < high
        return logbesseli0x_middle(x)
    else
        return logbesseli0x_scaled_tail(x) - (log(x) + T(log2π)) / 2
    end
end

@inline function logbesseli0(x::Real)
    T = checkedfloattype(x)
    low, mid1, mid2, mid3, high = logbesseli0x_branches(T)
    x = abs(x)
    return x < low ? logbesseli0_taylor(x) : logbesseli0x(x) + x # log(I0(x)) = log(besselix(0, x)) + |x|
end

@inline function logbesseli0_taylor(x::Real)
    T = checkedfloattype(x)
    x² = abs2(x)
    return x² * (T(0.25) + x² * evalpoly(x², logbesseli0_taylor_coefs(T))) # log(I0x(x)) = log(exp(-x) * I0(x)) loses accuracy near x = 0 since I0x(x) -> 1 as x -> 0
end

@inline function logbesseli0x_middle(x::Real)
    T = checkedfloattype(x)
    low, mid1, mid2, mid3, high = logbesseli0x_branches(T)
    x = abs(x)
    if x < mid1
        return x * evalpoly(muladd(x, logbesseli0x_mid1_transplant(T)...), logbesseli0x_mid1_coefs(T))
    elseif x < mid2
        return x * evalpoly(muladd(x, logbesseli0x_mid2_transplant(T)...), logbesseli0x_mid2_coefs(T))
    elseif x < mid3
        return x * evalpoly(muladd(x, logbesseli0x_mid3_transplant(T)...), logbesseli0x_mid3_coefs(T))
    else # x < high
        return x * evalpoly(muladd(x, logbesseli0x_mid4_transplant(T)...), logbesseli0x_mid4_coefs(T))
    end
end

@inline function logbesseli0x_scaled_tail(x::Real)
    T = checkedfloattype(x)
    x⁻¹ = inv(x)
    return x⁻¹ * evalpoly(x⁻¹, logbesseli0x_tail_coefs(T)) # log(sqrt(2πx) * exp(-x) * I0(x)) ~ 1/8x⁻¹ + 1/16x⁻² + 25/384x⁻³ + 𝒪(x⁻⁴)
end

@inline logbesseli0x_branches(::Type{Float32}) = (1.0f0, 2.0262144f0, 3.032815f0, 5.18301f0, 7.1340613f0)
@inline logbesseli0_taylor_coefs(::Type{Float32}) = (-0.015624976f0, 0.0017352961f0, -0.00021954361f0, 2.360602f-5)
@inline logbesseli0x_mid1_transplant(::Type{Float32}) = (1.9489106f0, -2.9489107f0)
@inline logbesseli0x_mid1_coefs(::Type{Float32}) = (-0.6651776f0, 0.08970585f0, -0.009135113f0, 0.00022245155f0, 0.00013069467f0, -2.4635681f-5)
@inline logbesseli0x_mid2_transplant(::Type{Float32}) = (1.9868851f0, -5.025855f0)
@inline logbesseli0x_mid2_coefs(::Type{Float32}) = (-0.5202778f0, 0.057392687f0, -0.0061008586f0, 0.00049862283f0, -1.2916373f-5, -4.5256024f-6)
@inline logbesseli0x_mid3_transplant(::Type{Float32}) = (0.93014824f0, -3.8209674f0)
@inline logbesseli0x_mid3_coefs(::Type{Float32}) = (-0.38694486f0, 0.06660385f0, -0.012341707f0, 0.0022151936f0, -0.0003573263f0, 4.17854f-5)
@inline logbesseli0x_mid4_transplant(::Type{Float32}) = (1.0250883f0, -6.3130426f0)
@inline logbesseli0x_mid4_coefs(::Type{Float32}) = (-0.2931732f0, 0.032936238f0, -0.0040817023f0, 0.0005173513f0, -6.647147f-5, 8.107054f-6)
@inline logbesseli0x_tail_coefs(::Type{Float32}) = (0.12499994f0, 0.06252784f0, 0.06290423f0, 0.16337825f0, -0.52516294f0, 3.8941476f0)

@inline logbesseli0x_branches(::Type{Float64}) = (1.0, 2.4539954370828134, 4.426675225425466, 8.113004453118227, 13.79718132864979)
@inline logbesseli0_taylor_coefs(::Type{Float64}) = (-0.01562499999999995, 0.0017361111111010013, -0.00022379557257775202, 3.092447475294501e-5, -4.455160371599049e-6, 6.601794380225889e-7, -9.963839425018547e-8, 1.496590273686032e-8, -2.0311101682813397e-9, 1.784377125069117e-10)
@inline logbesseli0x_mid1_transplant(::Type{Float64}) = (1.3755201350649688, -2.375520135064969)
@inline logbesseli0x_mid1_coefs(::Type{Float64}) = (-0.6293509921334998, 0.1165222156862489, -0.017543241808282393, 0.0011263889520766045, 0.00032222558333416774, -0.00012176511385428864, 1.6070398691462904e-5, 1.5165939776545204e-6, -1.1419110868276816e-6, 2.191954526212264e-7, -2.4282770550872775e-10, -1.0977853470173317e-8, 2.902257499240661e-9, -1.9267719751651386e-10, -1.023777707120882e-10, 3.3047056495605973e-11, -2.612196667913543e-12)
@inline logbesseli0x_mid2_transplant(::Type{Float64}) = (1.0138492885762773, -3.4879815280558413)
@inline logbesseli0x_mid2_coefs(::Type{Float64}) = (-0.433648178631133, 0.07785990208975836, -0.014641801812046053, 0.0025229449612280725, -0.0003453680388341347, 2.135186866647759e-5, 6.801589942277674e-6, -3.2577624319146623e-6, 8.273124612471309e-7, -1.407801084204267e-7, 1.1292965383300882e-8, 2.565188338696305e-9, -1.4394021269523833e-9, 3.975416943967199e-10, -7.30324481618524e-11, 3.163082922739749e-12, 2.098918916266162e-12)
@inline logbesseli0x_mid3_transplant(::Type{Float64}) = (0.5425451381215293, -3.4016711215976114)
@inline logbesseli0x_mid3_coefs(::Type{Float64}) = (-0.2894676393211177, 0.06050732983792175, -0.013958370849695728, 0.0032965788519068446, -0.0007742680839766047, 0.00017655604402086076, -3.7888051852072746e-5, 7.245606514336846e-6, -1.07400018711158e-6, 4.550555059341293e-8, 4.934991263707759e-8, -2.6916330419166043e-8, 9.66295495228346e-9, -2.850929991938722e-9, 7.368411765561862e-10, -1.6070835311259337e-10, 2.1091869151328898e-11)
@inline logbesseli0x_mid4_transplant(::Type{Float64}) = (0.35185393484311084, -3.854592540229329)
@inline logbesseli0x_mid4_coefs(::Type{Float64}) = (-0.19204323036278093, 0.03768199446511169, -0.008157807879284613, 0.0018280979828959336, -0.000416301900904904, 9.552336542286533e-5, -2.1970050940049594e-5, 5.044041118490876e-6, -1.1512652564047615e-6, 2.5996853766320234e-7, -5.7700172869363094e-8, 1.24572056976531e-8, -2.5769860973226465e-9, 5.080843549259749e-10, -8.967154903917677e-11, 6.013660636164252e-12, 2.3047924090477283e-12)
@inline logbesseli0x_tail_coefs(::Type{Float64}) = (0.125, 0.06249999999990704, 0.06510416677685975, 0.1015624484769859, 0.20958291517669933, 0.534604986182686, 1.8155149103752943, -5.79677657036416, 556.8677969367772, -17339.18845642846, 406001.5911443486, -6.564842028194815e6, 7.056595146213631e7, -4.4273395883670574e8, 9.37272527534153e8, 5.461197809026326e9, -2.8269544002663994e10)

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
    low, mid1, mid2, mid3, tail = besseli1i0_branches(T)
    if x < tail
        x² = x^2
        if x < low
            rx = evalpoly(x², besseli1i0_low_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) = 1/2 + 𝒪(x^2)
        elseif x < mid1
            rx = evalpoly(x², besseli1i0_mid1_num_coefs(T)) / evalpoly(x², besseli1i0_mid1_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2) = 1/2 + 𝒪(x^2)
        elseif x < mid2
            rx = evalpoly(x², besseli1i0_mid2_num_coefs(T)) / evalpoly(x², besseli1i0_mid2_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2) = 1/2 + 𝒪(x^2)
        elseif x < mid3
            rx = evalpoly(x², besseli1i0_mid3_num_coefs(T)) / evalpoly(x², besseli1i0_mid3_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2) = 1/2 + 𝒪(x^2)
        else # x < tail
            rx = evalpoly(x², besseli1i0_mid4_num_coefs(T)) / evalpoly(x², besseli1i0_mid4_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2) = 1/2 + 𝒪(x^2)
        end
        r = x * rx # I₁(x) / I₀(x) = x * R(x^2) = x/2 - x^3/16 + 𝒪(x^5) where R(x^2) = P(x^2) or P(x^2) / Q(x^2)
        rm1 = r - one(T) # I₁(x) / I₀(x) - 1 = -1 + x/2 - x^3/16 + 𝒪(x^5)
    else # x >= tail
        x⁻¹ = inv(x)
        rm1x²m½x = evalpoly(x⁻¹, besseli1i0c_tail_coefs(T)) # P(1/x) = x^2 * (1 - I₁(x) / I₀(x)) - x/2 = 1/8 + 1/8x + 25/128x^2 + 𝒪(1/x^3)
        rm1 = x⁻¹ * (T(-0.5) - x⁻¹ * rm1x²m½x) # I₁(x) / I₀(x) - 1 = -1/2x - P(1/x) / x^2 = -1/2x - 1/8x^2 - 1/8x^3 + 𝒪(1/x^4)
        r = rm1 + one(T) # I₁(x) / I₀(x) = 1 - 1/2x - 1/8x^2 - 1/8x^3 + 𝒪(1/x^4)
        rx = x⁻¹ * r
    end
    return r, rx, rm1
end

@inline besseli1i0_branches(::Type{Float32}) = (0.5f0, 5.197075f0, 9.929153f0, 18.494247f0, 36.094307f0)
@inline besseli1i0_low_coefs(::Type{Float32}) = (0.5f0, -0.0624989f0, 0.010394423f0, -0.0016448505f0)
@inline besseli1i0_mid1_num_coefs(::Type{Float32}) = (0.49999997f0, 0.043556698f0, 0.00066640606f0, 1.1037953f-6)
@inline besseli1i0_mid1_den_coefs(::Type{Float32}) = (1.0f0, 0.21211326f0, 0.007013781f0, 4.0564966f-5)
@inline besseli1i0_mid2_num_coefs(::Type{Float32}) = (0.49386856f0, 0.034112554f0, 0.0003199398f0, 2.4867086f-7)
@inline besseli1i0_mid2_den_coefs(::Type{Float32}) = (1.0f0, 0.18871176f0, 0.004393305f0, 1.3712045f-5)
@inline besseli1i0_mid3_num_coefs(::Type{Float32}) = (0.38958442f0, 0.010547966f0, 3.396568f-5, 8.475613f-9)
@inline besseli1i0_mid3_den_coefs(::Type{Float32}) = (1.0f0, 0.09150172f0, 0.00079196814f0, 8.2656f-7)
@inline besseli1i0_mid4_num_coefs(::Type{Float32}) = (0.24115495f0, 0.0018708268f0, 1.6862418f-6, 1.1641679f-10)
@inline besseli1i0_mid4_den_coefs(::Type{Float32}) = (1.0f0, 0.029562477f0, 7.367802f-5, 2.155656f-8)
@inline besseli1i0c_tail_coefs(::Type{Float32}) = (0.12499999f0, 0.12500682f0, 0.19411263f0, 0.47243553f0)

@inline besseli1i0_branches(::Type{Float64}) = (0.5, 7.52740354489351, 13.090559031850686, 20.80952798408195, 33.4745971087716)
@inline besseli1i0_low_coefs(::Type{Float64}) = (0.5, -0.06249999999999999, 0.0104166666666654, -0.0017903645832681267, 0.00030924478996177326, -5.346225011561096e-5, 9.243845992068521e-6, -1.597088340378288e-6, 2.7173295526886746e-7, -3.86430832554434e-8)
@inline besseli1i0_mid1_num_coefs(::Type{Float64}) = (0.5, 0.05051232213180785, 0.001261150320831201, 9.788283800870312e-6, 1.5775685868303345e-8, -1.6909410781453963e-11, 3.91429085414582e-14, -1.1124286855644106e-16, 3.2387399247670293e-19, -8.478147054904883e-22, 1.6964956996840654e-24, -1.8748316429974648e-27)
@inline besseli1i0_mid1_den_coefs(::Type{Float64}) = (1.0, 0.22602464426361568, 0.009942047841281054, 0.0001342149589365221, 5.258773123126486e-7)
@inline besseli1i0_mid2_num_coefs(::Type{Float64}) = (0.49999554772921173, 0.04971418890122518, 0.0011885222737792341, 8.455342928370799e-6, 1.1711502412689139e-8, -9.932208927352689e-12, 1.638151591206671e-14, -2.938614891171288e-17, 4.739302412725169e-20, -6.029444407699415e-23, 5.192050899871583e-26, -2.223157457100178e-29)
@inline besseli1i0_mid2_den_coefs(::Type{Float64}) = (1.0, 0.22442480753717464, 0.00959765754083271, 0.00012166703966703814, 4.2465187089283047e-7)
@inline besseli1i0_mid3_num_coefs(::Type{Float64}) = (0.48821543235627424, 0.038831575688716057, 0.0006509361968343166, 2.8932601080870305e-6, 2.2628763609937953e-9, -1.0033474895681845e-12, 8.0901283743250495e-16, -6.728383536147867e-19, 4.828050185413794e-22, -2.647792728844039e-25, 9.593633715211663e-29, -1.6967481567167612e-32)
@inline besseli1i0_mid3_den_coefs(::Type{Float64}) = (1.0, 0.19587834025191733, 0.006294445466707653, 5.294738466311112e-5, 1.1011346235778047e-7)
@inline besseli1i0_mid4_num_coefs(::Type{Float64}) = (0.3953099776294151, 0.0143303499269297, 0.00010250856948462276, 1.885851249584892e-7, 5.991135040216946e-11, -1.0735229148812402e-14, 3.470381388869812e-18, -1.1506814269246478e-21, 3.278190375527831e-25, -7.115086603209037e-29, 1.0176985081949667e-32, -7.090984812661566e-37)
@inline besseli1i0_mid4_den_coefs(::Type{Float64}) = (1.0, 0.10403308063121751, 0.0014947027941623618, 5.324010820694674e-6, 4.567138270589379e-9)
@inline besseli1i0c_tail_coefs(::Type{Float64}) = (0.125, 0.12500000000000067, 0.19531249999896427, 0.40625000063246103, 1.047851363431197, 3.218786836985941, 11.46216325763734, 46.805884323072455, 194.8624724456196, 1595.7616954693328, -4511.937268947673, 139379.87983903964)

@inline function _neglogpdf_rician_parts(z::Real, ::Val{degree}) where {degree}
    T = checkedfloattype(z)
    low, mid1, mid2, mid3, mid4, high, tail = neglogpdf_rician_parts_branches(T)
    r_tail = r′ = r′′ = one_minus_r_minus_z_r′ = two_r′_plus_z_r′′ = T(NaN)
    if z < low
        z² = z^2
        a1 = evalpoly(z², a1_taylor_coefs(T))
        a0 = T(0.5) + z² * a1
        a0² = a0 * a0
        r = z * a0
        if degree >= 1
            r′ = one(T) - z² * a0² - a0
            one_minus_r_minus_z_r′ = one(T) - (r + z * r′)
        end
        if degree >= 2
            r′′ = z * ((T(2) * a1 + a0 * (T(3) * a0 - T(2))) + T(2) * z² * a0 * a0²)
            two_r′_plus_z_r′′ = 2 * r′ + z * r′′
        end
    else
        u = inv(z)
        if z < mid1
            b3 = evalpoly(muladd(u, b3_mid1_transplant(T)...), b3_mid1_coefs(T))
        elseif z < mid2
            b3 = evalpoly(muladd(u, b3_mid2_transplant(T)...), b3_mid2_coefs(T))
        elseif z < mid3
            b3 = evalpoly(muladd(u, b3_mid3_transplant(T)...), b3_mid3_coefs(T))
        elseif z < mid4
            b3 = evalpoly(muladd(u, b3_mid4_transplant(T)...), b3_mid4_coefs(T))
        elseif z < high
            b3 = evalpoly(muladd(u, b3_mid5_transplant(T)...), b3_mid5_coefs(T))
        elseif z < tail
            b3 = evalpoly(u, b3_high_coefs(T))
        else # z >= tail
            b3 = evalpoly(u, b3_tail_coefs(T))
        end
        b2 = one(T) + u * b3
        b1 = T(0.25) * (one(T) + u * b2)
        b0 = T(0.5) * (one(T) + u * b1)
        b0², b1² = b0 * b0, b1 * b1
        u² = u^2
        halfu² = T(0.5) * u²
        halfu³ = halfu² * u
        r = one(T) - u * b0
        r_tail = b0
        if degree >= 1
            r′ = u² * (b1 + b0 * (one(T) - b0))
            one_minus_r_minus_z_r′ = halfu² * (b1 + T(-0.5) * (b2 - u * b1²))
        end
        if degree >= 2
            r′′ = halfu³ * (b0² * (T(-2) + T(-4) * b0) - b2 + u * b1 * (T(3) * b1 + T(4) * b0))
            two_r′_plus_z_r′′ = halfu³ * (b1² - b3 + (b2 - u * b1²) * (T(0.5) + b0))
        end
    end
    return r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′
end

@inline neglogpdf_rician_parts_branches(::Type{Float32}) = (2.0f0, 2.6060996f0, 3.5124323f0, 4.822016f0, 6.7038217f0, 11.265795f0, 30.0f0)
@inline a1_taylor_coefs(::Type{Float32}) = (-0.062499993f0, 0.010416392f0, -0.001788809f0, 0.00030583347f0, -4.968289f-5, 6.8428612f-6, -6.5504764f-7, 3.043636f-8)
@inline b3_mid1_transplant(::Type{Float32}) = (17.19915f0, -7.599575f0)
@inline b3_mid1_coefs(::Type{Float32}) = (1.530969f0, -0.9001749f0, 0.03678463f0, 0.02181827f0, -0.005012985f0, 0.00044332922f0, 3.3053548f-5, -1.6691927f-5)
@inline b3_mid2_transplant(::Type{Float32}) = (20.199532f0, -6.750867f0)
@inline b3_mid2_coefs(::Type{Float32}) = (3.1194985f0, -0.5696182f0, -0.14454196f0, 0.036770962f0, -0.00054329867f0, -0.0010684185f0, 0.00022074053f0, -1.45563345f-5)
@inline b3_mid3_transplant(::Type{Float32}) = (25.86625f0, -6.364198f0)
@inline b3_mid3_coefs(::Type{Float32}) = (3.4891043f0, 0.17810252f0, -0.15597716f0, -0.008672251f0, 0.0066092443f0, -0.0004907746f0, -0.00015136568f0, 3.9740677f-5)
@inline b3_mid4_transplant(::Type{Float32}) = (34.35629f0, -6.124881f0)
@inline b3_mid4_coefs(::Type{Float32}) = (2.8084152f0, 0.37852308f0, 0.0013338927f0, -0.01738603f0, -0.0009319629f0, 0.0006840263f0, -1.1296704f-5, -2.1204842f-5)
@inline b3_mid5_transplant(::Type{Float32}) = (33.110184f0, -3.9390013f0)
@inline b3_mid5_coefs(::Type{Float32}) = (2.1523418f0, 0.24606755f0, 0.04518204f0, 0.0061474717f0, -0.0018303139f0, -0.0010039372f0, 2.945304f-5, 7.2044975f-5)
@inline b3_high_coefs(::Type{Float32}) = (1.5627546f0, 3.2188735f0, 9.927305f0, -13.473833f0, 605.299f0, -2241.9482f0, -7155.295f0, 125444.33f0)
@inline b3_tail_coefs(::Type{Float32}) = (1.5625f0, 3.249928f0, 8.399429f0, 24.425596f0, 133.03055f0)

@inline neglogpdf_rician_parts_branches(::Type{Float64}) = (2.0, 2.897974423546584, 4.234741944354931, 6.327475220473357, 9.919317887108292, 18.211327630457376, 100.0)
@inline a1_taylor_coefs(::Type{Float64}) = (-0.062499999999999986, 0.010416666666664732, -0.0017903645832794987, 0.00030924479107213686, -5.3462272282636315e-5, 9.244068846364443e-6, -1.5984001634629747e-6, 2.763434489361867e-7, -4.773315324974686e-8, 8.210027641665284e-9, -1.3912968740700294e-9, 2.2667154318714788e-10, -3.4025047639873384e-11, 4.437131733996034e-12, -4.678903548144084e-13, 3.647065518389165e-14, -1.834972296113143e-15, 4.41929493021651e-17)
@inline b3_mid1_transplant(::Type{Float64}) = (12.908939709444839, -5.4544698547224195)
@inline b3_mid1_coefs(::Type{Float64}) = (1.8333360040083098, -1.2212709889386149, 0.020510617991516333, 0.06844415625792045, -0.017888162803863922, 0.0014485117897190813, 0.00041528099374805116, -0.00018584591254295761, 3.844488376327165e-5, -4.187095878501654e-6, -2.32831817173261e-7, 2.3861642291181444e-7, -6.762260636756006e-8, 1.2854412352025097e-8, -1.6976836196566833e-9, 7.54282968424866e-11, 5.84126703926284e-11, -1.966763264811086e-11)
@inline b3_mid2_transplant(::Type{Float64}) = (18.360969508953218, -5.3357941877495225)
@inline b3_mid2_coefs(::Type{Float64}) = (3.484297761731685, -0.2551239552705028, -0.2841510027244296, 0.036077789515001, 0.010254236421247872, -0.0037773315469154943, 0.00034639064170243053, 8.790311526127584e-5, -3.795269375972866e-5, 6.5360544436602935e-6, -2.497135435932903e-7, -1.975368249079707e-7, 6.906638565772812e-8, -1.3049277620198214e-8, 1.3075927915089944e-9, 1.163802222539146e-10, -1.0770075458981495e-10, 2.5834526328094403e-11)
@inline b3_mid3_transplant(::Type{Float64}) = (25.607873706394546, -5.047091899078007)
@inline b3_mid3_coefs(::Type{Float64}) = (3.048884170715356, 0.4803342605034443, -0.05931511455925456, -0.04137247668861298, 0.003274103301017207, 0.0020801890676649362, -0.0004364023766479829, -3.3754132696338e-5, 2.805951429750254e-5, -4.4559513324027475e-6, -2.7888645251837413e-7, 2.7696222185744057e-7, -5.980178088896274e-8, 3.71020320173379e-9, 1.6848368087526513e-9, -6.563011224814749e-10, 1.0019399635152762e-10, -3.7669280872671637e-13)
@inline b3_mid4_transplant(::Type{Float64}) = (34.94821124416195, -4.523247429098187)
@inline b3_mid4_coefs(::Type{Float64}) = (2.24321296529402, 0.2645195596758001, 0.04474278255429483, 0.002094885055201418, -0.0027457029006763543, -0.0005664881461932309, 0.00015073528733716736, 3.334535364978677e-5, -1.1615020008392194e-5, -7.055236530656481e-7, 7.863967082425159e-7, -9.421585939386579e-8, -2.6715808346143148e-8, 1.1044202650587625e-8, -1.005934317101632e-9, -3.7821484487969754e-10, 1.3350407521240528e-10, -1.1019894451816739e-11)
@inline b3_mid5_transplant(::Type{Float64}) = (43.57060676579103, -3.392500297063557)
@inline b3_mid5_coefs(::Type{Float64}) = (1.883620498051352, 0.12217944258749858, 0.011488018148597796, 0.0016096333444384071, 0.0003229472101132312, 6.105982115392979e-5, -2.8582669278089307e-6, -8.055139505280397e-6, -1.969373569570641e-6, 4.934597607685526e-7, 2.5994248282541115e-7, -3.907383319348374e-8, -2.6152438333666675e-8, 5.238821841928624e-9, 2.144522215094479e-9, -7.26952190362273e-10, -1.0783621213766397e-10, 6.184459128914788e-11)
@inline b3_high_coefs(::Type{Float64}) = (1.5624999998704658, 3.2500000935922175, 8.382781524169124, 25.756232733875247, 90.8784242279162, 456.0874252818104, -4516.344124943383, 356179.8462126366, -1.490298072425562e7, 4.944164164364225e8, -1.2459762406128853e10, 2.3540773520365933e11, -3.208612359334424e12, 2.9158438167785562e13, -1.4011294640981267e14, -1.047514827982499e14, 5.077726267776098e15, -1.915785968925904e16)
@inline b3_tail_coefs(::Type{Float64}) = (1.5625, 3.249999999999951, 8.382812500129667, 25.749999868034124, 91.73175661609746, 371.80898186259145, 1693.3965569777363, 8211.619321835962, 62507.00709166558)

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
