####
#### Rice distribution
####

@doc raw"""
    Rice(ν, σ)

The Rice distribution with shape parameter $\nu$ and scale parameter $\sigma$.

The probability density function is
```math
p(x \mid \nu, \sigma) = \frac{x}{\sigma^2} \exp\!\left(-\frac{x^2 + \nu^2}{2\sigma^2}\right) I_0\!\left(\frac{x\nu}{\sigma^2}\right)
```
where $I_0$ is the modified Bessel function of the first kind of order zero, and $x \ge 0$.

# External links

- [Rice distribution Wiki](https://en.wikipedia.org/wiki/Rice_distribution)
"""
struct Rice{T <: Real} <: Distributions.ContinuousUnivariateDistribution
    ν::T
    σ::T
end

#### Outer constructors

@inline Rice(ν::Real, σ::Real) = Rice(promote(ν, σ)...)
@inline Rice(ν::Integer, σ::Integer) = Rice(float(ν), float(σ))
@inline Rice(ν::Real) = Rice(ν, one(typeof(ν)))
@inline Rice() = Rice(0.0, 1.0)

#### Conversions

@inline Base.convert(::Type{Rice{T}}, ν::Real, σ::Real) where {T <: Real} = Rice(T(ν), T(σ))
@inline Base.convert(::Type{Rice{T}}, d::Rice{<:Real}) where {T <: Real} = Rice(T(d.ν), T(d.σ))

# Distributions.@distr_support Rice 0 Inf

@inline Base.minimum(::Union{Rice, Type{<:Rice}}) = 0
@inline Base.maximum(::Union{Rice, Type{<:Rice}}) = Inf

#### Parameters

@inline Distributions.params(d::Rice) = (d.ν, d.σ)
@inline Distributions.partype(::Rice{T}) where {T} = T

@inline Distributions.location(d::Rice) = d.ν
@inline Distributions.scale(d::Rice) = d.σ

@inline Base.eltype(::Type{Rice{T}}) where {T} = T

#### Statistics

@inline mean_rician(ν, σ) = σ * mean_rician(ν / σ)
@inline var_rician(ν, σ) = σ^2 * var_rician(ν / σ)
@inline std_rician(ν, σ) = σ * std_rician(ν / σ)
@inline mode_rician(ν, σ) = σ * mode_rician(ν / σ)
@inline var_mode_rician(ν, σ) = σ^2 * var_mode_rician(ν / σ)

@inline mean_rician(ν) = _mean_rician(float(ν))
@inline var_rician(ν) = _var_rician(float(ν))
@inline std_rician(ν) = sqrt(_var_rician(float(ν)))
@inline mode_rician(ν) = _mode_rician(float(ν))
@inline var_mode_rician(ν) = _var_mode_rician(float(ν))

@inline Distributions.mean(d::Rice) = mean_rician(d.ν, d.σ)
# @inline Distributions.mode(d::Rice) = ?
# @inline Distributions.median(d::Rice) = ?

@inline Distributions.var(d::Rice) = var_rician(d.ν, d.σ)
@inline Distributions.std(d::Rice) = sqrt(Distributions.var(d))
# @inline Distributions.skewness(d::Rice{T}) where {T <: Real} = ?
# @inline Distributions.kurtosis(d::Rice{T}) where {T <: Real} = ?
# @inline Distributions.entropy(d::Rice) = ?

#### Evaluation

# p(x | ν, σ) = x * I₀(x * ν / σ^2) * exp(-(x^2 + ν^2) / 2σ^2) / σ^2
@inline Distributions.logpdf(d::Rice, x::Real) = -neglogpdf_rician(x, d.ν, log(d.σ))
@inline Distributions.pdf(d::Rice, x::Real) = exp(Distributions.logpdf(d, x))

#### Sampling

@inline Distributions.rand(rng::Random.AbstractRNG, d::Rice{T}) where {T} = hypot(d.ν + d.σ * randn(rng, T), d.σ * randn(rng, T))

#### Utils

@inline _mean_rician(ν) = ν > 1 / √eps(one(ν)) ? ν : sqrthalfπ * laguerre½(-ν^2 / 2)

@inline _var_rician(ν) = 1 - laguerre½²c(ν) # equivalent to: ν^2 + 2σ^2 - π * σ^2 * laguerre½(-(ν / σ)^2 / 2)^2 / 2 where σ = 1

@inline function _mode_rician_crude(ν::Real)
    ν² = ν^2
    twoν⁴ = 2 * ν²^2
    return √(1 + ν² * ((1 + twoν⁴) / (2 + twoν⁴)))
end

@inline function _var_mode_rician_crude(v::Real)
    twoν² = 2 * v^2
    return (1 + twoν²) / (2 + twoν²)
end

@inline function _mode_rician(ν::Real)
    T = checkedfloattype(ν)
    tay, low, med1, med2, med3, med4, med5, tail = mode_rician_branches(T)
    if ν < tay
        return evalpoly(ν^2, mode_rician_coeff_taylor(T))
    elseif ν < low
        return mode_rician_kernel_small_approx(T, ν)
    elseif ν < med1
        return ν + mode_rician_kernel_med_1_approx(T, ν)
    elseif ν < med2
        return ν + mode_rician_kernel_med_2_approx(T, ν)
    elseif ν < med3
        return ν + mode_rician_kernel_med_3_approx(T, ν)
    elseif ν < med4
        return ν + mode_rician_kernel_med_4_approx(T, ν)
    elseif ν < med5
        return ν + mode_rician_kernel_med_5_approx(T, ν)
    elseif ν < tail
        return inv(ν) * mode_rician_kernel_tail_approx(T, ν) + ν
    else
        return inv(ν) * mode_rician_kernel_tail_long_approx(T, ν) + ν
    end
end

function _var_mode_rician(ν::Real)
    T = checkedfloattype(ν)
    tay, low, med1, med2, med3, med4, med5, med6, tail = var_mode_rician_branches(T)
    if ν < tay
        return evalpoly(ν^2, var_mode_rician_coeff_taylor(T))
    elseif ν < low
        return var_mode_rician_kernel_small_approx(T, ν)
    elseif ν < med1
        return var_mode_rician_kernel_med_1_approx(T, ν)
    elseif ν < med2
        return var_mode_rician_kernel_med_2_approx(T, ν)
    elseif ν < med3
        return var_mode_rician_kernel_med_3_approx(T, ν)
    elseif ν < med4
        return var_mode_rician_kernel_med_4_approx(T, ν)
    elseif ν < med5
        return var_mode_rician_kernel_med_5_approx(T, ν)
    elseif ν < med6
        return var_mode_rician_kernel_med_6_approx(T, ν)
    elseif ν < tail
        return var_mode_rician_kernel_tail_approx(T, ν)
    else
        return var_mode_rician_kernel_tail_long_approx(T, ν)
    end
end

mode_rician_coeff_taylor(::Type{T}) where {T} = T.((1 // 1, 1 // 4, 1 // 16, 1 // 192, -67 // 12288, -817 // 245760, -10109 // 17694720, 210407 // 495452160, 45860611 // 126835752960, 185952481 // 1956894474240))
mode_rician_coeff_taylor(::Type{Float64}) = (1.0, 0.25, 0.0625, 0.005208333333333333, -0.005452473958333333, -0.0033243815104166668, -0.000571300365306713, 0.0004246767235811425)
mode_rician_coeff_taylor(::Type{Float32}) = (1.0f0, 0.25f0, 0.0625f0, 0.0052083335f0)

mode_rician_coeff_taylor_cutoff(::Type{Float64}) = 0.18
mode_rician_coeff_taylor_cutoff(::Type{Float32}) = 0.26f0
mode_rician_coeff_long_tail_cutoff(::Type{Float64}) = 50.0
mode_rician_coeff_long_tail_cutoff(::Type{Float32}) = 10.0f0

var_mode_rician_coeff_taylor(::Type{T}) where {T} = T.((1 // 2, 1 // 4, 1 // 16, -1 // 64, -71 // 3072, -217 // 24576, 4357 // 2949120, 25063 // 7077888, 27151871 // 15854469120, -40160509 // 761014517760))
var_mode_rician_coeff_taylor(::Type{Float64}) = (0.5, 0.25, 0.0625, -0.015625, -0.023111979166666668, -0.008829752604166666, 0.001477389865451389, 0.003541028058087384)
var_mode_rician_coeff_taylor(::Type{Float32}) = (0.5f0, 0.25f0, 0.0625f0, -0.015625f0)

var_mode_rician_coeff_taylor_cutoff(::Type{Float64}) = 0.15
var_mode_rician_coeff_taylor_cutoff(::Type{Float32}) = 0.19f0
var_mode_rician_coeff_long_tail_cutoff(::Type{Float64}) = 25.0
var_mode_rician_coeff_long_tail_cutoff(::Type{Float32}) = 12.0f0

mode_rician_branches(::Type{Float64}) = (0.18, 0.7038881878839673, 1.1085997811823203, 1.4760924042951453, 1.9747764589232897, 2.6693085898036144, 3.718791486999396, 50.0)
mode_rician_kernel_small_approx(::Type{Float64}, x::Real) = (y = abs2(x); evalpoly(y, mode_rician_kernel_small_coeffs(Float64)))
mode_rician_kernel_small_coeffs(::Type{Float64}) = (1.0000000000000162, 0.24999999999839906, 0.0625000000674915, 0.005208331726299668, -0.005452449664432777, -0.003324628711704546, -0.0005695541214170501, 0.0004159862138196782, 0.0003921007617442605, 2.028018246388088e-5, 8.403285899724668e-5, -0.00017478850771093533, 4.950272377911742e-5)
mode_rician_kernel_med_1_approx(::Type{Float64}, x::Real) = (t = muladd(4.941790729789157, x, -4.478468121693078); evalpoly(t, mode_rician_kernel_med_1_num_coeffs(Float64)) / evalpoly(t, mode_rician_kernel_med_1_den_coeffs(Float64)))
mode_rician_kernel_med_1_num_coeffs(::Type{Float64}) = (0.3404050657921232, -0.19468901689454896, 0.09719511356858829, -0.025300457651436053, 0.005134177428254747, -0.0005634104904842026, 3.0938695287789366e-5, 3.1944491827689693e-7, -8.475453374488945e-9, -1.4531193005662301e-8)
mode_rician_kernel_med_1_den_coeffs(::Type{Float64}) = (1.0, -0.3480561493218156, 0.15145840552165973, -0.020165581925577034, 0.0033656430624079406)
mode_rician_kernel_med_2_approx(::Type{Float64}, x::Real) = (t = muladd(5.442286114641202, x, -7.033317195822817); evalpoly(t, mode_rician_kernel_med_2_num_coeffs(Float64)) / evalpoly(t, mode_rician_kernel_med_2_den_coeffs(Float64)))
mode_rician_kernel_med_2_num_coeffs(::Type{Float64}) = (0.255547412095773, 0.03688071977990919, 0.03289831444661936, -0.00010435620441703232, 0.0009518901375806413, -0.00012274943812814535, 1.3214508769746367e-5, -7.747078373933349e-7, -4.136090702937435e-9, 2.420201657967001e-9)
mode_rician_kernel_med_2_den_coeffs(::Type{Float64}) = (1.0, 0.21787179914420726, 0.12117867389316689, 0.011319233879161568, 0.002740746503283847)
mode_rician_kernel_med_3_approx(::Type{Float64}, x::Real) = (t = muladd(4.010555343485661, x, -6.919950279524493); evalpoly(t, mode_rician_kernel_med_3_num_coeffs(Float64)) / evalpoly(t, mode_rician_kernel_med_3_den_coeffs(Float64)))
mode_rician_kernel_med_3_num_coeffs(::Type{Float64}) = (0.22577817409408293, 0.09484937493804667, 0.028088053106505178, 0.0027542921243359476, 0.00042629719786964126, -4.8796551555640405e-5, 5.951035528319935e-6, -1.0505482089767883e-6, 1.660892555479546e-7, -1.2736959437753702e-8)
mode_rician_kernel_med_3_den_coeffs(::Type{Float64}) = (1.0, 0.4876243827769599, 0.15966897926593765, 0.024527913692630663, 0.0029485300916166036)
mode_rician_kernel_med_4_approx(::Type{Float64}, x::Real) = (t = muladd(2.8792452747036545, x, -6.685865787950897); evalpoly(t, mode_rician_kernel_med_4_num_coeffs(Float64)) / evalpoly(t, mode_rician_kernel_med_4_den_coeffs(Float64)))
mode_rician_kernel_med_4_num_coeffs(::Type{Float64}) = (0.18846938795233165, 0.09410568938503082, 0.026233903314686596, 0.0036562935156273816, 0.00016244510171902716, -2.4093739964999205e-5, 4.049468229510188e-6, -6.87709879173897e-7, 8.873290687957068e-8, -5.975250418125374e-9)
mode_rician_kernel_med_4_den_coeffs(::Type{Float64}) = (1.0000000000000002, 0.6085625113764729, 0.19854506418246262, 0.03575202811743427, 0.003305098327068669)
mode_rician_kernel_med_5_approx(::Type{Float64}, x::Real) = (t = muladd(1.905767999546853, x, -6.087262692129057); evalpoly(t, mode_rician_kernel_med_5_num_coeffs(Float64)) / evalpoly(t, mode_rician_kernel_med_5_den_coeffs(Float64)))
mode_rician_kernel_med_5_num_coeffs(::Type{Float64}) = (0.1458057760646754, 0.11051478872314598, 0.03766960711235546, 0.005389361209814875, 0.00015447813430908913, -2.4298686381293057e-5, 3.4755241762798936e-6, -4.3333247422117455e-7, 4.2447473066335906e-8, -2.4564197543204727e-9)
mode_rician_kernel_med_5_den_coeffs(::Type{Float64}) = (1.0, 0.8994490665131367, 0.3676469936234835, 0.07466760215645728, 0.0066045037846699135)
mode_rician_kernel_tail_approx(::Type{Float64}, x::Real) = (y = abs2(3.7188486184122316 / x); evalpoly(y, mode_rician_kernel_tail_coeffs(Float64)))
mode_rician_kernel_tail_coeffs(::Type{Float64}) = (0.4999999999999983, -0.027115290871554492, 0.0016338644284122224, -0.0002274202805685834, 1.740183445519612e-5, -3.5878853613012627e-6, 4.911501229197889e-8, 2.2930964680320706e-7, -6.831343961933456e-7, 9.567852549341278e-7, -9.204405149757446e-7, 5.580451903126323e-7, -1.9340618902672559e-7, 2.7944306781279216e-8)
mode_rician_kernel_tail_long_approx(::Type{Float64}, x::Real) = (y = abs2(50.0 / x); evalpoly(y, mode_rician_kernel_tail_long_coeffs(Float64)))
mode_rician_kernel_tail_long_coeffs(::Type{Float64}) = (0.49999999999999994, -0.00014999999999618354, 4.999998017026792e-8, -3.846773052006147e-11)

mode_rician_branches(::Type{Float32}) = (0.26f0, 0.8690923f0, 1.1862338f0, 1.5172783f0, 1.8364475f0, 2.3214612f0, 3.752015f0, 10.0f0)
mode_rician_kernel_small_approx(::Type{Float32}, x::Real) = (y = abs2(x); evalpoly(y, mode_rician_kernel_small_coeffs(Float32)))
mode_rician_kernel_small_coeffs(::Type{Float32}) = (0.9999977f0, 0.2500582f0, 0.06200089f0, 0.0070985244f0, -0.0087182345f0, -0.0013475548f0)
mode_rician_kernel_med_1_approx(::Type{Float32}, x::Real) = (t = muladd(6.3156257f0, x, -6.5201635f0); evalpoly(t, mode_rician_kernel_med_1_num_coeffs(Float32)) / evalpoly(t, mode_rician_kernel_med_1_den_coeffs(Float32)))
mode_rician_kernel_med_1_num_coeffs(::Type{Float32}) = (0.3001829f0, -0.043586366f0, 0.016705895f0, -0.0016924157f0)
mode_rician_kernel_med_1_den_coeffs(::Type{Float32}) = (1.0f0, -0.006497911f0, 0.01943058f0)
mode_rician_kernel_med_2_approx(::Type{Float32}, x::Real) = (t = muladd(6.1483426f0, x, -8.320976f0); evalpoly(t, mode_rician_kernel_med_2_num_coeffs(Float32)) / evalpoly(t, mode_rician_kernel_med_2_den_coeffs(Float32)))
mode_rician_kernel_med_2_num_coeffs(::Type{Float32}) = (0.249898f0, 0.05394396f0, 0.011779146f0, -0.000992551f0)
mode_rician_kernel_med_2_den_coeffs(::Type{Float32}) = (1.0f0, 0.27054062f0, 0.048954926f0)
mode_rician_kernel_med_3_approx(::Type{Float32}, x::Real) = (t = muladd(6.393863f0, x, -10.69319f0); evalpoly(t, mode_rician_kernel_med_3_num_coeffs(Float32)) / evalpoly(t, mode_rician_kernel_med_3_den_coeffs(Float32)))
mode_rician_kernel_med_3_num_coeffs(::Type{Float32}) = (0.2289977f0, 0.10069596f0, 0.028447324f0, -0.0015269173f0)
mode_rician_kernel_med_3_den_coeffs(::Type{Float32}) = (1.0f0, 0.48093697f0, 0.14471795f0)
mode_rician_kernel_med_4_approx(::Type{Float32}, x::Real) = (t = muladd(4.0923085f0, x, -8.484072f0); evalpoly(t, mode_rician_kernel_med_4_num_coeffs(Float32)) / evalpoly(t, mode_rician_kernel_med_4_den_coeffs(Float32)))
mode_rician_kernel_med_4_num_coeffs(::Type{Float32}) = (0.20381519f0, 0.0037126273f0, 0.008300254f0, -0.0004947217f0)
mode_rician_kernel_med_4_den_coeffs(::Type{Float32}) = (1.0f0, 0.09431294f0, 0.04658943f0)
mode_rician_kernel_med_5_approx(::Type{Float32}, x::Real) = (t = muladd(1.3952801f0, x, -4.233612f0); evalpoly(t, mode_rician_kernel_med_5_num_coeffs(Float32)) / evalpoly(t, mode_rician_kernel_med_5_den_coeffs(Float32)))
mode_rician_kernel_med_5_num_coeffs(::Type{Float32}) = (0.1523429f0, 0.058295216f0, -0.00048590277f0, 3.6453876f-5)
mode_rician_kernel_med_5_den_coeffs(::Type{Float32}) = (1.0f0, 0.5826229f0, 0.078148834f0)
mode_rician_kernel_tail_approx(::Type{Float32}, x::Real) = (y = abs2(3.75094f0 / x); evalpoly(y, mode_rician_kernel_tail_coeffs(Float32)))
mode_rician_kernel_tail_coeffs(::Type{Float32}) = (0.49999958f0, -0.026648765f0, 0.0015620231f0, -0.0001905378f0)
mode_rician_kernel_tail_long_approx(::Type{Float32}, x::Real) = (y = abs2(10.0f0 / x); evalpoly(y, mode_rician_kernel_tail_long_coeffs(Float32)))
mode_rician_kernel_tail_long_coeffs(::Type{Float32}) = (0.49999997f0, -0.0037496674f0, 3.0358828f-5)

var_mode_rician_branches(::Type{Float64}) = (0.15, 0.8261641732390133, 1.209525924947686, 1.6215052011735214, 2.053976580997712, 2.5608584568898918, 3.0938222980264993, 3.699206978772744, 25.0)
var_mode_rician_kernel_small_approx(::Type{Float64}, x::Real) = (y = abs2(x); evalpoly(y, var_mode_rician_kernel_small_coeffs(Float64)))
var_mode_rician_kernel_small_coeffs(::Type{Float64}) = (0.49999999999996164, 0.2500000000046394, 0.06249999977258744, -0.015624993932978384, -0.02311207912720615, -0.008828659294346022, 0.0014690807831744485, 0.0035862100506892226, 0.0015337263890343367, 0.0004660957729374347, -0.0016785882452499648, 0.0013219880460022325, -0.0017761904712600843, 0.001241308880185041, -0.0002933966846260878)
var_mode_rician_kernel_med_1_approx(::Type{Float64}, x::Real) = (t = muladd(5.217004542278532, x, -5.31010224445572); evalpoly(t, var_mode_rician_kernel_med_1_num_coeffs(Float64)) / evalpoly(t, var_mode_rician_kernel_med_1_den_coeffs(Float64)))
var_mode_rician_kernel_med_1_num_coeffs(::Type{Float64}) = (0.778963442406643, -0.07114508801768872, 0.09872376749681282, -0.0010945031812689345, 0.002988797632863837, 0.0001113598060486983, 1.679282731829208e-5, -1.6065866996527098e-6, -1.0090503154453772e-7, 2.227893226204727e-9, 1.9881370892172553e-9)
var_mode_rician_kernel_med_1_den_coeffs(::Type{Float64}) = (1.0, -0.20290338231172575, 0.16005710908624501, -0.014726470188845845, 0.005386780346590474)
var_mode_rician_kernel_med_2_approx(::Type{Float64}, x::Real) = (t = muladd(4.854613121131017, x, -6.871780425599165); evalpoly(t, var_mode_rician_kernel_med_2_num_coeffs(Float64)) / evalpoly(t, var_mode_rician_kernel_med_2_den_coeffs(Float64)))
var_mode_rician_kernel_med_2_num_coeffs(::Type{Float64}) = (0.8956148828707347, 0.3785008541581573, 0.1548349521806669, 0.02629843931223097, 0.003961700448490816, 0.00012900454673619264, -1.0068844995122768e-5, -1.3963084304637865e-6, 2.3668834008773003e-7, 1.663193660745471e-8, -3.692021766654372e-9)
var_mode_rician_kernel_med_2_den_coeffs(::Type{Float64}) = (1.0, 0.3942776075607666, 0.1769041088158316, 0.025847761290737355, 0.0045489693295449095)
var_mode_rician_kernel_med_3_approx(::Type{Float64}, x::Real) = (t = muladd(4.62458348298804, x, -8.498786170926266); evalpoly(t, var_mode_rician_kernel_med_3_num_coeffs(Float64)) / evalpoly(t, var_mode_rician_kernel_med_3_den_coeffs(Float64)))
var_mode_rician_kernel_med_3_num_coeffs(::Type{Float64}) = (0.9186695415555708, 0.36452534816288384, 0.0913794643372544, 0.01234124461059357, 0.0012767710105878543, 1.9946977199576125e-5, -1.127434669890247e-6, 4.708587606178836e-8, -1.4266836748035072e-8, 1.6953686865210236e-9)
var_mode_rician_kernel_med_3_den_coeffs(::Type{Float64}) = (1.0, 0.3894797063133625, 0.09653795311142714, 0.01220317886156112, 0.0014255976888216768)
var_mode_rician_kernel_med_4_approx(::Type{Float64}, x::Real) = (t = muladd(3.945692468249597, x, -9.104359925603731); evalpoly(t, var_mode_rician_kernel_med_4_num_coeffs(Float64)) / evalpoly(t, var_mode_rician_kernel_med_4_den_coeffs(Float64)))
var_mode_rician_kernel_med_4_num_coeffs(::Type{Float64}) = (0.9345392281404478, 0.44900090867366277, 0.11697522064196408, 0.0170559297793005, 0.0013117666184442238, 6.327069378776402e-6, -1.2980701521408995e-6, 1.515625295701692e-7, -8.48665212654084e-9)
var_mode_rician_kernel_med_4_den_coeffs(::Type{Float64}) = (1.0000000000000002, 0.4710239103934375, 0.1209888144906178, 0.017418396761636642, 0.001315915619113063)
var_mode_rician_kernel_med_5_approx(::Type{Float64}, x::Real) = (t = muladd(3.7525997931393746, x, -10.609876915584227); evalpoly(t, var_mode_rician_kernel_med_5_num_coeffs(Float64)) / evalpoly(t, var_mode_rician_kernel_med_5_den_coeffs(Float64)))
var_mode_rician_kernel_med_5_num_coeffs(::Type{Float64}) = (0.9507086276251446, 0.2694329388029369, 0.030856260586612845, 6.704504765973214e-5, -6.78848839662894e-6, 2.2503607846412826e-7, 3.686249834922993e-8, -1.680754936029291e-9, -2.0315421558626575e-9, 5.674101352078217e-10, -6.099407432733905e-11)
var_mode_rician_kernel_med_5_den_coeffs(::Type{Float64}) = (1.0, 0.2759285314627861, 0.031050058014680717)
var_mode_rician_kernel_med_6_approx(::Type{Float64}, x::Real) = (t = muladd(3.3036845225991565, x, -11.221012841662303); evalpoly(t, var_mode_rician_kernel_med_6_num_coeffs(Float64)) / evalpoly(t, var_mode_rician_kernel_med_6_den_coeffs(Float64)))
var_mode_rician_kernel_med_6_num_coeffs(::Type{Float64}) = (0.9633018115092351, 0.3780314701366718, 0.04154166791323024, 5.383068323511419e-5, -8.39533162679189e-6, 8.948531544435543e-7, -7.505175914307447e-8, 4.64842647462138e-9, -1.3001664160365715e-10, -7.844844363300677e-12)
var_mode_rician_kernel_med_6_den_coeffs(::Type{Float64}) = (1.0, 0.3867425135479653, 0.04150596153441399)
var_mode_rician_kernel_tail_approx(::Type{Float64}, x::Real) = (y = abs2(3.699206978772744 / x); evalpoly(y, var_mode_rician_kernel_tail_coeffs(Float64)))
var_mode_rician_kernel_tail_coeffs(::Type{Float64}) = (0.9999999999999921, -0.0365386704876737, 0.005340297733352079, -0.0006341633681125274, 0.00010693810653693547, -1.3034947083001783e-5, 2.2937235383135756e-6, 7.766273157172993e-7, -2.5181640785223333e-6, 4.706376020212144e-6, -6.024002397659921e-6, 5.3321280170546485e-6, -3.087693185537301e-6, 1.0513289696723317e-6, -1.579982538857172e-7)
var_mode_rician_kernel_tail_long_approx(::Type{Float64}, x::Real) = (y = abs2(25.0 / x); evalpoly(y, var_mode_rician_kernel_tail_long_coeffs(Float64)))
var_mode_rician_kernel_tail_long_coeffs(::Type{Float64}) = (1.0, -0.0007999999999940112, 2.559999948819128e-6, -6.655855356355297e-9, 2.4411138863215767e-11)

var_mode_rician_branches(::Type{Float32}) = (0.19f0, 0.7077095f0, 0.9824105f0, 1.3322446f0, 1.6206616f0, 1.9067316f0, 2.2263358f0, 2.920515f0, 12.0f0)
var_mode_rician_kernel_small_approx(::Type{Float32}, x::Real) = (y = abs2(x); evalpoly(y, var_mode_rician_kernel_small_coeffs(Float32)))
var_mode_rician_kernel_small_coeffs(::Type{Float32}) = (0.4999999f0, 0.25001055f0, 0.06226728f0, -0.013776709f0, -0.029471133f0)
var_mode_rician_kernel_med_1_approx(::Type{Float32}, x::Real) = (t = muladd(7.2806435f0, x, -6.1525803f0); evalpoly(t, var_mode_rician_kernel_med_1_num_coeffs(Float32)) / evalpoly(t, var_mode_rician_kernel_med_1_den_coeffs(Float32)))
var_mode_rician_kernel_med_1_num_coeffs(::Type{Float32}) = (0.6976813f0, -0.013499326f0, 0.013602491f0, 8.531515f-6)
var_mode_rician_kernel_med_1_den_coeffs(::Type{Float32}) = (1.0f0, -0.111189544f0, 0.026152609f0)
var_mode_rician_kernel_med_2_approx(::Type{Float32}, x::Real) = (t = muladd(5.716995f0, x, -6.6164355f0); evalpoly(t, var_mode_rician_kernel_med_2_num_coeffs(Float32)) / evalpoly(t, var_mode_rician_kernel_med_2_den_coeffs(Float32)))
var_mode_rician_kernel_med_2_num_coeffs(::Type{Float32}) = (0.8359698f0, 0.097335964f0, 0.047540665f0, 0.0020982553f0)
var_mode_rician_kernel_med_2_den_coeffs(::Type{Float32}) = (1.0f0, 0.042625416f0, 0.07072521f0)
var_mode_rician_kernel_med_3_approx(::Type{Float32}, x::Real) = (t = muladd(6.9344044f0, x, -10.238322f0); evalpoly(t, var_mode_rician_kernel_med_3_num_coeffs(Float32)) / evalpoly(t, var_mode_rician_kernel_med_3_den_coeffs(Float32)))
var_mode_rician_kernel_med_3_num_coeffs(::Type{Float32}) = (0.9020357f0, 0.22288358f0, 0.036535215f0, 0.000703692f0)
var_mode_rician_kernel_med_3_den_coeffs(::Type{Float32}) = (1.0f0, 0.23282693f0, 0.042666364f0)
var_mode_rician_kernel_med_4_approx(::Type{Float32}, x::Real) = (t = muladd(6.9912963f0, x, -12.330525f0); evalpoly(t, var_mode_rician_kernel_med_4_num_coeffs(Float32)) / evalpoly(t, var_mode_rician_kernel_med_4_den_coeffs(Float32)))
var_mode_rician_kernel_med_4_num_coeffs(::Type{Float32}) = (0.91635376f0, 0.3654001f0, 0.06841858f0, 0.0005011232f0)
var_mode_rician_kernel_med_4_den_coeffs(::Type{Float32}) = (1.0f0, 0.39378494f0, 0.073009096f0)
var_mode_rician_kernel_med_5_approx(::Type{Float32}, x::Real) = (t = muladd(6.257741f0, x, -12.931831f0); evalpoly(t, var_mode_rician_kernel_med_5_num_coeffs(Float32)) / evalpoly(t, var_mode_rician_kernel_med_5_den_coeffs(Float32)))
var_mode_rician_kernel_med_5_num_coeffs(::Type{Float32}) = (0.92612207f0, 0.08753001f0, 0.098030165f0, 0.000535869f0)
var_mode_rician_kernel_med_5_den_coeffs(::Type{Float32}) = (1.0f0, 0.08859264f0, 0.10515651f0)
var_mode_rician_kernel_med_6_approx(::Type{Float32}, x::Real) = (t = muladd(2.8811f0, x, -7.4142957f0); evalpoly(t, var_mode_rician_kernel_med_6_num_coeffs(Float32)) / evalpoly(t, var_mode_rician_kernel_med_6_den_coeffs(Float32)))
var_mode_rician_kernel_med_6_num_coeffs(::Type{Float32}) = (0.9433643f0, 0.30217266f0, 0.045004528f0, 7.840921f-5)
var_mode_rician_kernel_med_6_den_coeffs(::Type{Float32}) = (1.0f0, 0.30884498f0, 0.045258533f0)
var_mode_rician_kernel_tail_approx(::Type{Float32}, x::Real) = (y = abs2(2.920515f0 / x); evalpoly(y, var_mode_rician_kernel_tail_coeffs(Float32)))
var_mode_rician_kernel_tail_coeffs(::Type{Float32}) = (1.0000001f0, -0.058620427f0, 0.013730596f0, -0.0025382978f0, 0.00054561417f0)
var_mode_rician_kernel_tail_long_approx(::Type{Float32}, x::Real) = (y = abs2(12.0f0 / x); evalpoly(y, var_mode_rician_kernel_tail_long_coeffs(Float32)))
var_mode_rician_kernel_tail_long_coeffs(::Type{Float32}) = (1.0f0, -0.0034719242f0, 4.74246f-5)
