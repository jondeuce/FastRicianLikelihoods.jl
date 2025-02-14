####
#### Fast + differentiable Bessel functions
####

# Bessels.jl provides `besseli0x` and `besseli1x`, but not `besseli2x`
@inline besseli0(x) = Bessels.besseli0(x)
@inline besseli0x(x) = Bessels.besseli0x(x)

@inline besseli1(x) = Bessels.besseli1(x)
@inline besseli1x(x) = Bessels.besseli1x(x)

"""
    besseli0m1x(x::T) where {T <: Union{Float32, Float64}}

Scaled modified Bessel function of the first kind of order zero minus one, ``(I_0(x) - 1)*e^{-x}``.
"""
function besseli0m1x(x::Real)
    T = checkedfloattype(x)
    T == Float32 ? low = 6.75 : low = 16.0
    T == Float32 ? branch = 50 : branch = 500
    x = abs(x)
    if x < low
        a = x * x / 4
        return a * evalpoly(a, besseli0m1x_small_coefs(T)) * exp(-x)
    elseif x < branch
        return evalpoly(inv(x), besseli0m1x_med_coefs(T)) / sqrt(x)
    else
        return evalpoly(inv(x), besseli0m1x_large_coefs(T)) / sqrt(x)
    end
end

function logbesseli0m1x(x::Real)
    T = checkedfloattype(x)
    T == Float32 ? low = 6.75 : low = 16.0
    T == Float32 ? branch = 50 : branch = 500
    x = abs(x)
    if x < low
        a = x * x / 4
        return log(a * evalpoly(a, besseli0m1x_small_coefs(T))) - x
    elseif x < branch
        return log(evalpoly(inv(x), besseli0m1x_med_coefs(T)) / sqrt(x))
    else
        return log(evalpoly(inv(x), besseli0m1x_large_coefs(T)) / sqrt(x))
    end
end

@inline besseli0m1x_small_coefs(::Type{Float32}) = (0.9999999f0, 0.25000122f0, 0.027775574f0, 0.0017376335f0, 6.892991f-5, 2.0227067f-6, 3.0070932f-8, 1.0751665f-9)
@inline besseli0m1x_med_coefs(::Type{Float32}) = (0.39887866f0, 0.06057018f0, -0.70964724f0, 27.146845f0, -578.4168f0, 7264.5205f0, -51609.934f0, 185786.06f0, -265738.56f0)
@inline besseli0m1x_med_num_coefs(::Type{Float32}) = (0.39894706f0, -7.777536f0, 68.3436f0, -268.09674f0, 438.21545f0)
@inline besseli0m1x_med_den_coefs(::Type{Float32}) = (1.0f0, -19.61879f0, 173.612f0, -690.39886f0, 1150.0469f0)
@inline besseli0m1x_large_coefs(::Type{Float32}) = (0.3989423f0, 0.049860857f0, 0.028961208f0)

@inline besseli0m1x_small_coefs(::Type{Float64}) = (1.0, 0.25000000000000006, 0.027777777777777655, 0.0017361111111112342, 6.944444444437954e-5, 1.929012345699996e-6, 3.93675988868729e-8, 6.151187333719624e-10, 7.594058350037932e-12, 7.594059093379885e-14, 6.276076969892409e-16, 4.358412357436062e-18, 2.5788371910145114e-20, 1.3160763622513412e-22, 5.840301033998216e-25, 2.299821101721548e-27, 7.657011478962633e-30, 2.7450102221441183e-32, 3.914976964195122e-35, 3.6000158386208433e-37, -4.742187837043056e-40, 2.95126297137588e-42)
@inline besseli0m1x_med_coefs(::Type{Float64}) = (0.3989422804012607, 0.04986778540587323, 0.028050307495006976, 0.029388810402809357, -0.013798992100663144, 14.269520676696201, -2511.3666124068586, 335129.1941264714, -3.440937920417942e7, 2.761040774786925e9, -1.7506342771577005e11, 8.8359431835763e12, -3.564976706320503e14, 1.1510754413445302e16, -2.968668369343783e17, 6.083314332002715e18, -9.81294687877789e19, 1.2279434474327913e21, -1.1654678488341868e22, 8.098203967282104e22, -3.882057548943311e23, 1.1468205083405064e24, -1.5730545601072846e24)
@inline besseli0m1x_med_num_coefs(::Type{Float64}) = (0.3989422804014235, -12.161834141246917, 180.3499525703925, 21.901634823723352, 11.935046173406183, 12.353455075204916, -31.290613372311775, 5975.026295933406, -563683.1575798317, 4.242633067816853e7, -2.5401108147247005e9, 1.2080841731102097e11, -4.539283010575132e12, 1.335213616835021e14, -3.0341880553731475e15, 5.229312867669351e16, -6.660539624080351e17, 6.037427451929038e18, -3.668380959994813e19, 1.33631431219921e20, -2.204320347434442e20)
@inline besseli0m1x_med_den_coefs(::Type{Float64}) = (1.0, -30.61019732986138, 455.82625263684065)
@inline besseli0m1x_large_coefs(::Type{Float64}) = (0.3989422804014327, 0.049867785050035494, 0.02805062966442321, 0.02921860341712994, 0.04519908777162243)

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

@inline besseli2_small_coefs(::Type{Float32}) = (0.5000000380180327f0, 0.1666662618114895f0, 0.02083404267763115f0, 0.0013884111223694947f0, 5.80310930509866f-5, 1.623226061096556f-6, 3.775086797193045f-8, 3.403189301562835f-10, 1.3439158558362364f-11)
@inline besseli2_small_coefs(::Type{Float64}) = (0.49999999999999983, 0.16666666666667154, 0.020833333333312287, 0.0013888888889246186, 5.787037033873442e-5, 1.6534391702073172e-6, 3.444664327750705e-8, 5.467735445704131e-10, 6.834436191839501e-12, 6.906148607507894e-14, 5.733547826165566e-16, 4.1288365068666296e-18, 2.0258796870216445e-20, 1.958474603154919e-22)
@inline besseli2_med_coefs(::Type{Float32}) = (0.3989423335005962f0, -0.7480354669272465f0, 0.3283003548501443f0, 0.10303896764745404f0, 0.2639897680554983f0)
@inline besseli2_med_coefs(::Type{Float64}) = (0.3989422804014328, -0.7480167757536108, 0.32725734026661746, 0.12272117712518572, 0.1266192433674273, 0.1985002457586728, 0.9652668510022477, -22.515059045641358, 650.0016469706677, -5111.504274365941, -366285.0422528223, 1.866344682244392e7, -4.870990420065428e8, 8.501718935762064e9, -1.0580797337687886e11, 9.548560528330159e11, -6.211912811203429e12, 2.8410346605629793e13, -8.670581882322688e13, 1.586062121825567e14, -1.3160619165647164e14)
@inline besseli2_large_coefs(::Type{Float32}) = (0.398942312409439f0, -0.7480453792346994f0, 0.3310342475515811f0)
@inline besseli2_large_coefs(::Type{Float64}) = (0.3989422804014327, -0.7480167757530116, 0.3272573406917582, 0.12271968486936348, 0.12759241505245672)

#### Derived special functions

@inline logbesseli0(x::Real) = abs(x) < one(x) ? logbesseli0_small(x) : log(besseli0x(x)) + abs(x) # log(besselix(0, x)) = log(I0(x)) - |x|
@inline logbesseli0x(x::Real) = abs(x) < one(x) ? logbesseli0_small(x) - abs(x) : log(besseli0x(x))

@inline logbesseli0_small(x::Real) = (T = checkedfloattype(x); x² = abs2(x); return x² * evalpoly(x², logbesseli0_small_coefs(T))) # log(besselix(0, x)) loses accuracy near x = 0 since besselix(0, x) -> 1 as x -> 0
@inline logbesseli0_small_coefs(::Type{Float32}) = (0.24999999426684533f0, -0.015624705149866972f0, 0.0017336759629143878f0, -0.00021666015596172704f0, 2.2059316402289948f-5)
@inline logbesseli0_small_coefs(::Type{Float64}) = (0.25, -0.015624999999997167, 0.0017361111109961576, -0.0002237955710956064, 3.092446434101836e-5, -4.455118991727041e-6, 6.600804196191383e-7, -9.949296105322181e-8, 1.483764672332753e-8, -1.968806398401359e-9, 1.6562710526172217e-10)

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
@inline laguerre½²c_large_coefs(::Type{Float32}) = (0.5000001f0, 0.49966717f0, 1.5056641f0, -12.99378f0, 1426.22f0, -51440.543f0, 991048.2f0, -7.35821f6, -2.1984954f7, 5.800887f8, -2.2002685f9)
@inline laguerre½²c_large_coefs(::Type{Float64}) = (0.5000000000007115, 0.49999998853939026, 1.3750298806013577, 6.344269530038998, 58.43045968676148, -5095.59086727239, 1.1829801320283161e6, -1.7743162867297137e8, 1.924618208549089e10, -1.5390306113432373e12, 9.210028517590392e13, -4.15736157803401e15, 1.4188235458026496e17, -3.649463648908279e18, 7.008308295117078e19, -9.863118766945332e20, 9.833293950803035e21, -6.515761189852251e22, 2.5001002794213213e23, -3.504756044111116e23, -4.504604424324448e23)

"""
    besseli1i0(x::T) where {T <: Union{Float32, Float64}}

Ratio of modified Bessel functions of the first kind of orders one and zero, ``I_1(x) / I_0(x)``.
"""
@inline function besseli1i0(x::Real)
    T = checkedfloattype(x)
    if x < besseli1i0_low_cutoff(T)
        x² = x^2
        return x * evalpoly(x², besseli1i0_low_coefs(T)) # I₁(x) / I₀(x) = x * P(x^2) = x/2 + 𝒪(x^3)
    elseif x < besseli1i0_mid_cutoff(T)
        x² = x^2
        return x * evalpoly(x², besseli1i0_mid_num_coefs(T)) / evalpoly(x², besseli1i0_mid_den_coefs(T)) # I₁(x) / I₀(x) = x * P(x^2) / Q(x^2)
    elseif x < besseli1i0_high_cutoff(T)
        x² = x^2
        return x * evalpoly(x², besseli1i0_high_num_coefs(T)) / evalpoly(x², besseli1i0_high_den_coefs(T)) # I₁(x) / I₀(x) = x * P(x^2) / Q(x^2)
    else
        x⁻¹ = inv(x)
        P = evalpoly(x⁻¹, besseli1i0c_tail_coefs(T)) # P(1/x) = x * (-1/2 + x * (1 - I₁(x) / I₀(x))) = 1/8 + 1/8x + 𝒪(1/x^2)
        return evalpoly(x⁻¹, (T(1.0), T(-0.5), -P)) # I₁(x) / I₀(x) = 1 - 1/2x - P(1/x)/x^2
    end
end

"""
    besseli1i0x(x::T) where {T <: Union{Float32, Float64}}

Ratio of modified Bessel functions of the first kind of orders one and zero divided by x, ``I_1(x) / I_0(x) / x``.
"""
@inline function besseli1i0x(x::Real)
    T = checkedfloattype(x)
    if x < besseli1i0_low_cutoff(T)
        x² = x^2
        return evalpoly(x², besseli1i0_low_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) = x/2 + 𝒪(x^3)
    elseif x < besseli1i0_mid_cutoff(T)
        x² = x^2
        return evalpoly(x², besseli1i0_mid_num_coefs(T)) / evalpoly(x², besseli1i0_mid_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2)
    elseif x < besseli1i0_high_cutoff(T)
        x² = x^2
        return evalpoly(x², besseli1i0_high_num_coefs(T)) / evalpoly(x², besseli1i0_high_den_coefs(T)) # I₁(x) / I₀(x) / x = P(x^2) / Q(x^2)
    else
        x⁻¹ = inv(x)
        P = evalpoly(x⁻¹, besseli1i0c_tail_coefs(T)) # P(1/x) = x * (-1/2 + x * (1 - I₁(x) / I₀(x))) = 1/8 + 1/8x + 𝒪(1/x^2)
        return x⁻¹ * evalpoly(x⁻¹, (T(1.0), T(-0.5), -P)) # I₁(x) / I₀(x) / x = 1/x - 1/2x^2 - P(1/x)/x^3
    end
end

@inline besseli1i0_low_cutoff(::Type{T}) where {T} = T(0.5)
@inline besseli1i0_mid_cutoff(::Type{T}) where {T} = T(7.75)
@inline besseli1i0_high_cutoff(::Type{T}) where {T} = T(15.0)

@inline besseli1i0_low_coefs(::Type{Float32}) = (0.5f0, -0.0624989f0, 0.010394423f0, -0.0016448505f0)
@inline besseli1i0_mid_num_coefs(::Type{Float32}) = (0.49999985f0, 0.045098994f0, 0.00079124485f0, 2.4653918f-6)
@inline besseli1i0_mid_den_coefs(::Type{Float32}) = (1.0f0, 0.21519727f0, 0.0076493085f0, 5.8382207f-5, 4.8237073f-8)
@inline besseli1i0_high_num_coefs(::Type{Float32}) = (0.4427933f0, 0.018132959f0, 9.000428f-5, 3.4805463f-8)
@inline besseli1i0_high_den_coefs(::Type{Float32}) = (1.0f0, 0.12933768f0, 0.0016975396f0, 2.7292274f-6)
@inline besseli1i0c_tail_coefs(::Type{Float32}) = (0.12500001f0, 0.12498587f0, 0.19689824f0, 0.34546292f0, 1.9343305f0)

@inline besseli1i0_low_coefs(::Type{Float64}) = (0.4999999999999999, -0.06249999999994528, 0.010416666662044488, -0.001790364434468454, 0.0003092424332731733, -5.344192059352683e-5, 9.146096503297768e-6, -1.3486016148802727e-6)
@inline besseli1i0_mid_num_coefs(::Type{Float64}) = (0.4999999999999992, 0.044100170014814616, 0.0005862073670888185, -9.126630084614697e-6, -1.78058166465628e-7, -7.469543568561091e-10, -5.864802850333419e-13, 2.722655649153202e-16, -2.161520376487376e-19, 1.2603329375211949e-22)
@inline besseli1i0_mid_den_coefs(::Type{Float64}) = (1.0, 0.21320034002962768, 0.006989123904549891, -5.557356052544936e-6, -1.734441296325505e-6, -1.3998972195398574e-8, -2.8240024128156755e-11)
@inline besseli1i0_high_num_coefs(::Type{Float64}) = (0.49995309847544056, 0.05097481736469462, 0.0013209911878074828, 1.1867021758543425e-5, 3.844970338974902e-8, 3.8730062593663346e-11, 6.345341208204658e-15, -3.9223192697507965e-19)
@inline besseli1i0_high_den_coefs(::Type{Float64}) = (1.0, 0.22691601767357894, 0.010180573965603021, 0.0001482335760070065, 7.903111440196504e-7, 1.4596724371417906e-9, 6.883865347029874e-13)
@inline besseli1i0c_tail_coefs(::Type{Float64}) = (0.12500000000000017, 0.12499999999879169, 0.19531250150899987, 0.4062492562129355, 1.0480435526081948, 3.1889066971543234, 14.493937314937872, -164.07408273123662, 10554.06604261355, -363473.6613975362, 9.257867756487811e6, -1.6750893375624812e8, 2.1100222176196077e9, -1.752346161183495e10, 8.611676733884451e10, -1.88444663825226e11)

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
