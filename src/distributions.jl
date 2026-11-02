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
    Rice{T}(ν::Real, σ::Real) where {T <: Real} = new{T}(ν, σ)
end

#### Outer constructors

@inline function Rice(ν::T, σ::T; check_args::Bool = true) where {T <: Real}
    Distributions.@check_args Rice (ν, ν >= zero(ν)) (σ, σ > zero(σ))
    return Rice{T}(ν, σ)
end
@inline Rice(ν::Real, σ::Real; check_args::Bool = true) = Rice(promote(ν, σ)...; check_args)
@inline Rice(ν::Integer, σ::Integer; check_args::Bool = true) = Rice(float(ν), float(σ); check_args)
@inline Rice(ν::Real; check_args::Bool = true) = Rice(ν, one(ν); check_args)
@inline Rice() = Rice{Float64}(0.0, 1.0)

#### Conversions

@inline Base.convert(::Type{Rice{T}}, ν::Real, σ::Real) where {T <: Real} = Rice(T(ν), T(σ))
@inline Base.convert(::Type{Rice{T}}, d::Rice{<:Real}) where {T <: Real} = Rice{T}(T(d.ν), T(d.σ))

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
@inline Distributions.mode(d::Rice) = mode_rician(d.ν, d.σ)
# @inline Distributions.median(d::Rice) = ?

@inline Distributions.var(d::Rice) = var_rician(d.ν, d.σ)
@inline Distributions.std(d::Rice) = sqrt(Distributions.var(d))
# @inline Distributions.skewness(d::Rice{T}) where {T <: Real} = ?
# @inline Distributions.kurtosis(d::Rice{T}) where {T <: Real} = ?
# @inline Distributions.entropy(d::Rice) = ?

#### Evaluation

# p(x | ν, σ) = x * I₀(x * ν / σ^2) * exp(-(x^2 + ν^2) / 2σ^2) / σ^2
@inline Distributions.logpdf(d::Rice, x::Real) = _logpdf_rician(promote(x, d.ν, d.σ)...)
@inline Distributions.pdf(d::Rice, x::Real) = exp(Distributions.logpdf(d, x))

@inline function _logpdf_rician(x::D, ν::D, σ::D) where {D}
    (x < zero(x) || isinf(x)) && return oftype(x, -Inf)
    σ⁻¹ = inv(σ)
    return -(_neglogpdf_rician(σ⁻¹ * x, σ⁻¹ * ν) + log(σ))
end

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

@inline mode_rician_coeff_taylor(::Type{T}) where {T} = T.((1 // 1, 1 // 4, 1 // 16, 1 // 192, -67 // 12288, -817 // 245760, -10109 // 17694720, 210407 // 495452160, 45860611 // 126835752960, 185952481 // 1956894474240))

@inline mode_rician_branches(::Type{Float32}) = (0.21909316f0, 0.76683325f0, 1.1996887f0, 1.5219874f0, 1.8953346f0, 2.464339f0, 3.742762f0, 10.0f0)
@inline mode_rician_coeff_taylor(::Type{Float32}) = (1.0f0, 0.25f0, 0.0625f0, 0.0052083335f0)
@inline mode_rician_coeff_taylor_cutoff(::Type{Float32}) = first(mode_rician_branches(Float32))
@inline mode_rician_coeff_long_tail_cutoff(::Type{Float32}) = last(mode_rician_branches(Float32))
@inline mode_rician_kernel_small_coeffs(::Type{Float32}) = (1.0f0, 0.25000295f0, 0.062448014f0, 0.00552988f0, -0.006250926f0, -0.0027746593f0)
@inline mode_rician_kernel_small_approx(::Type{Float32}, x::Real) = (y = abs2(x); evalpoly(y, mode_rician_kernel_small_coeffs(Float32)))
@inline mode_rician_kernel_med_1_num_coeffs(::Type{Float32}) = (0.31414363f0, -0.10599814f0, 0.05013841f0, -0.008402054f0, 0.0007690017f0)
@inline mode_rician_kernel_med_1_den_coeffs(::Type{Float32}) = (1.0f0, -0.1267164f0, 0.06609025f0)
@inline mode_rician_kernel_med_1_approx(::Type{Float32}, x::Real) = (y = muladd(4.6204805f0, x, -4.543138f0); evalpoly(y, mode_rician_kernel_med_1_num_coeffs(Float32)) / evalpoly(y, mode_rician_kernel_med_1_den_coeffs(Float32)))
@inline mode_rician_kernel_med_2_num_coeffs(::Type{Float32}) = (0.24927716f0, 0.025209881f0, 0.009496439f0, -0.00112718f0, 8.542719f-5)
@inline mode_rician_kernel_med_2_den_coeffs(::Type{Float32}) = (1.0f0, 0.15428491f0, 0.034165204f0)
@inline mode_rician_kernel_med_2_approx(::Type{Float32}, x::Real) = (y = muladd(6.2054224f0, x, -8.444575f0); evalpoly(y, mode_rician_kernel_med_2_num_coeffs(Float32)) / evalpoly(y, mode_rician_kernel_med_2_den_coeffs(Float32)))
@inline mode_rician_kernel_med_3_num_coeffs(::Type{Float32}) = (0.2268014f0, 0.06240589f0, 0.008346785f0, -0.0007790067f0, 5.5923956f-5)
@inline mode_rician_kernel_med_3_den_coeffs(::Type{Float32}) = (1.0f0, 0.32525274f0, 0.0543353f0)
@inline mode_rician_kernel_med_3_approx(::Type{Float32}, x::Real) = (y = muladd(5.3569446f0, x, -9.153202f0); evalpoly(y, mode_rician_kernel_med_3_num_coeffs(Float32)) / evalpoly(y, mode_rician_kernel_med_3_den_coeffs(Float32)))
@inline mode_rician_kernel_med_4_num_coeffs(::Type{Float32}) = (0.19711192f0, 0.045578193f0, 0.010445554f0, -0.00095244986f0, 5.624061f-5)
@inline mode_rician_kernel_med_4_den_coeffs(::Type{Float32}) = (1.0f0, 0.3208459f0, 0.078368746f0)
@inline mode_rician_kernel_med_4_approx(::Type{Float32}, x::Real) = (y = muladd(3.5149112f0, x, -7.661933f0); evalpoly(y, mode_rician_kernel_med_4_num_coeffs(Float32)) / evalpoly(y, mode_rician_kernel_med_4_den_coeffs(Float32)))
@inline mode_rician_kernel_med_5_num_coeffs(::Type{Float32}) = (0.14944631f0, 0.05022709f0, -0.0004446034f0, 3.534088f-5)
@inline mode_rician_kernel_med_5_den_coeffs(::Type{Float32}) = (1.0f0, 0.5118227f0, 0.05954602f0)
@inline mode_rician_kernel_med_5_approx(::Type{Float32}, x::Real) = (y = muladd(1.5644274f0, x, -4.8552794f0); evalpoly(y, mode_rician_kernel_med_5_num_coeffs(Float32)) / evalpoly(y, mode_rician_kernel_med_5_den_coeffs(Float32)))
@inline mode_rician_kernel_tail_coeffs(::Type{Float32}) = (0.5f0, -0.026769556f0, 0.0015897594f0, -0.00021126815f0, 8.031598f-6)
@inline mode_rician_kernel_tail_approx(::Type{Float32}, x::Real) = (y = abs2(3.742762f0 / x); evalpoly(y, mode_rician_kernel_tail_coeffs(Float32)))
@inline mode_rician_kernel_tail_long_coeffs(::Type{Float32}) = (0.49999997f0, -0.0037496674f0, 3.0358828f-5)
@inline mode_rician_kernel_tail_long_approx(::Type{Float32}, x::Real) = (y = abs2(10.0f0 / x); evalpoly(y, mode_rician_kernel_tail_long_coeffs(Float32)))

@inline mode_rician_branches(::Type{Float64}) = (0.1581104311481667, 0.7510063627960759, 1.1343836365763194, 1.5289450901141906, 2.027443908548148, 2.7606126450657427, 3.907559893103011, 50.0)
@inline mode_rician_coeff_taylor(::Type{Float64}) = (1.0, 0.25, 0.0625, 0.005208333333333333, -0.005452473958333333, -0.0033243815104166668, -0.000571300365306713, 0.0004246767235811425)
@inline mode_rician_coeff_taylor_cutoff(::Type{Float64}) = first(mode_rician_branches(Float64))
@inline mode_rician_coeff_long_tail_cutoff(::Type{Float64}) = last(mode_rician_branches(Float64))
@inline mode_rician_kernel_small_coeffs(::Type{Float64}) = (1.0, 0.2499999999999908, 0.0625000000009455, 0.005208333295333425, -0.005452473175503219, -0.003324390880141168, -0.0005712314852858641, 0.00042436679004512483, 0.00036232514895452806, 9.488254937550234e-5, -4.461313500799741e-5, -2.9628598550838144e-5, -4.6915632837258244e-5, 2.8568461176477305e-5)
@inline mode_rician_kernel_small_approx(::Type{Float64}, x::Real) = (y = abs2(x); evalpoly(y, mode_rician_kernel_small_coeffs(Float64)))
@inline mode_rician_kernel_med_1_num_coeffs(::Type{Float64}) = (0.32729549791193246, -0.163230200014426, 0.08090886636869171, -0.019812936321745578, 0.00402062553999065, -0.0004366227582117239, 2.4657601609045375e-5, 2.2184532233919444e-7, -2.528608207168162e-8, -1.002624308659088e-8)
@inline mode_rician_kernel_med_1_den_coeffs(::Type{Float64}) = (1.0, -0.297978856232195, 0.13563311709500633, -0.01622203787096988, 0.0029359922997909313)
@inline mode_rician_kernel_med_1_approx(::Type{Float64}, x::Real) = (y = muladd(5.216793317661351, x, -4.917844974955725); evalpoly(y, mode_rician_kernel_med_1_num_coeffs(Float64)) / evalpoly(y, mode_rician_kernel_med_1_den_coeffs(Float64)))
@inline mode_rician_kernel_med_2_num_coeffs(::Type{Float64}) = (0.2517818801272677, 0.053781538532873714, 0.03646473775518504, 0.0009190283793326705, 0.001034920691783027, -0.0001428777776956786, 1.7348171277266628e-5, -1.1998909903168355e-6, 3.881803438611392e-9, 3.5191794810275246e-9)
@inline mode_rician_kernel_med_2_den_coeffs(::Type{Float64}) = (1.0, 0.2839137634255219, 0.14291984591539852, 0.016598978581115535, 0.003482253336272397)
@inline mode_rician_kernel_med_2_approx(::Type{Float64}, x::Real) = (y = muladd(5.068918877064188, x, -6.750098629274427); evalpoly(y, mode_rician_kernel_med_2_num_coeffs(Float64)) / evalpoly(y, mode_rician_kernel_med_2_den_coeffs(Float64)))
@inline mode_rician_kernel_med_3_num_coeffs(::Type{Float64}) = (0.22252824596760715, 0.08698812193397697, 0.022226413083846756, 0.0019394435517558213, 0.0003205282515249593, -3.6100160069023306e-5, 3.35606077110166e-6, -4.792819868888264e-7, 1.08519471271668e-7, -1.6119734561732295e-8, 1.0151793829161166e-9)
@inline mode_rician_kernel_med_3_den_coeffs(::Type{Float64}) = (1.0, 0.4604116590178067, 0.13423710054176133, 0.018736123181585275, 0.0021354553095323005)
@inline mode_rician_kernel_med_3_approx(::Type{Float64}, x::Real) = (y = muladd(4.012045617847267, x, -7.134197448721733); evalpoly(y, mode_rician_kernel_med_3_num_coeffs(Float64)) / evalpoly(y, mode_rician_kernel_med_3_den_coeffs(Float64)))
@inline mode_rician_kernel_med_4_num_coeffs(::Type{Float64}) = (0.1842638258710463, 0.09172665100680209, 0.026196349607808374, 0.003669391170069685, 0.0001513768762860866, -2.1634511189349246e-5, 3.691102824298998e-6, -7.457653505951346e-7, 1.3342479912680237e-7, -1.6188803832165498e-8, 9.614275277544787e-10)
@inline mode_rician_kernel_med_4_den_coeffs(::Type{Float64}) = (1.0, 0.6124327620630727, 0.20369011691071567, 0.03724328388208659, 0.00344789362618887)
@inline mode_rician_kernel_med_4_approx(::Type{Float64}, x::Real) = (y = muladd(2.727885001615865, x, -6.530633829745941); evalpoly(y, mode_rician_kernel_med_4_num_coeffs(Float64)) / evalpoly(y, mode_rician_kernel_med_4_den_coeffs(Float64)))
@inline mode_rician_kernel_med_5_num_coeffs(::Type{Float64}) = (0.14048443307462774, 0.12132773829810738, 0.04602493627032977, 0.007430574580775336, 0.000208809783857791, -3.554087920419999e-5, 5.616280338485666e-6, -8.098842851840951e-7, 1.017964671820133e-7, -1.0025340471791867e-8, 5.806982150850504e-10)
@inline mode_rician_kernel_med_5_den_coeffs(::Type{Float64}) = (1.0, 1.0136974104439707, 0.45923975312206555, 0.1034023316578973, 0.009829568870713339)
@inline mode_rician_kernel_med_5_approx(::Type{Float64}, x::Real) = (y = muladd(1.7437593607051514, x, -5.813844141114397); evalpoly(y, mode_rician_kernel_med_5_num_coeffs(Float64)) / evalpoly(y, mode_rician_kernel_med_5_den_coeffs(Float64)))
@inline mode_rician_kernel_tail_coeffs(::Type{Float64}) = (0.4999999999999999, -0.02455952601718606, 0.0013403784832819391, -0.0001689844747714708, 1.1713535022871865e-5, -2.198876007410005e-6, 7.49010670527706e-8, -1.621659130303743e-8, -6.277995580139565e-8, 7.24197582296137e-8, -6.095419450148664e-8, 2.838596595473899e-8, -6.242751335848708e-9)
@inline mode_rician_kernel_tail_approx(::Type{Float64}, x::Real) = (y = abs2(3.907559893103011 / x); evalpoly(y, mode_rician_kernel_tail_coeffs(Float64)))
@inline mode_rician_kernel_tail_long_coeffs(::Type{Float64}) = (0.5, -0.00014999999999999817, 4.999999998539174e-8, -3.849995909380669e-11, 1.625324361225775e-14)
@inline mode_rician_kernel_tail_long_approx(::Type{Float64}, x::Real) = (y = abs2(50.0 / x); evalpoly(y, mode_rician_kernel_tail_long_coeffs(Float64)))

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

@inline var_mode_rician_coeff_taylor(::Type{T}) where {T} = T.((1 // 2, 1 // 4, 1 // 16, -1 // 64, -71 // 3072, -217 // 24576, 4357 // 2949120, 25063 // 7077888, 27151871 // 15854469120, -40160509 // 761014517760))

@inline var_mode_rician_branches(::Type{Float32}) = (0.18327937f0, 0.72472477f0, 1.054741f0, 1.3505462f0, 1.7769837f0, 2.1602983f0, 2.7716742f0, 4.314869f0, 12.0f0)
@inline var_mode_rician_coeff_taylor(::Type{Float32}) = (0.5f0, 0.25f0, 0.0625f0, -0.015625f0)
@inline var_mode_rician_coeff_taylor_cutoff(::Type{Float32}) = first(var_mode_rician_branches(Float32))
@inline var_mode_rician_coeff_long_tail_cutoff(::Type{Float32}) = last(var_mode_rician_branches(Float32))
@inline var_mode_rician_kernel_small_coeffs(::Type{Float32}) = (0.5f0, 0.24999876f0, 0.062537976f0, -0.016062262f0, -0.02072267f0, -0.015461736f0, 0.010259702f0)
@inline var_mode_rician_kernel_small_approx(::Type{Float32}, x::Real) = (y = abs2(x); evalpoly(y, var_mode_rician_kernel_small_coeffs(Float32)))
@inline var_mode_rician_kernel_med_1_num_coeffs(::Type{Float32}) = (0.7187339f0, -0.046266977f0, 0.024197998f0, 0.0008334461f0, 0.0001778671f0)
@inline var_mode_rician_kernel_med_1_den_coeffs(::Type{Float32}) = (1.0f0, -0.17338623f0, 0.05058311f0)
@inline var_mode_rician_kernel_med_1_approx(::Type{Float32}, x::Real) = (y = muladd(6.0603075f0, x, -5.392055f0); evalpoly(y, var_mode_rician_kernel_med_1_num_coeffs(Float32)) / evalpoly(y, var_mode_rician_kernel_med_1_den_coeffs(Float32)))
@inline var_mode_rician_kernel_med_2_num_coeffs(::Type{Float32}) = (0.8509773f0, 0.10070966f0, 0.03464075f0, 0.0012678452f0)
@inline var_mode_rician_kernel_med_2_den_coeffs(::Type{Float32}) = (1.0f0, 0.06461062f0, 0.049891472f0)
@inline var_mode_rician_kernel_med_2_approx(::Type{Float32}, x::Real) = (y = muladd(6.761206f0, x, -8.131321f0); evalpoly(y, var_mode_rician_kernel_med_2_num_coeffs(Float32)) / evalpoly(y, var_mode_rician_kernel_med_2_den_coeffs(Float32)))
@inline var_mode_rician_kernel_med_3_num_coeffs(::Type{Float32}) = (0.908287f0, 0.28509128f0, 0.067601725f0, 0.0023089407f0, -0.00021616057f0)
@inline var_mode_rician_kernel_med_3_den_coeffs(::Type{Float32}) = (1.0f0, 0.30050978f0, 0.07722548f0)
@inline var_mode_rician_kernel_med_3_approx(::Type{Float32}, x::Real) = (y = muladd(4.690019f0, x, -7.334088f0); evalpoly(y, var_mode_rician_kernel_med_3_num_coeffs(Float32)) / evalpoly(y, var_mode_rician_kernel_med_3_den_coeffs(Float32)))
@inline var_mode_rician_kernel_med_4_num_coeffs(::Type{Float32}) = (0.9228328f0, 0.3477858f0, 0.06264925f0, 0.0005146497f0, -4.9227354f-5)
@inline var_mode_rician_kernel_med_4_den_coeffs(::Type{Float32}) = (1.0f0, 0.37005273f0, 0.06502434f0)
@inline var_mode_rician_kernel_med_4_approx(::Type{Float32}, x::Real) = (y = muladd(5.217646f0, x, -10.271672f0); evalpoly(y, var_mode_rician_kernel_med_4_num_coeffs(Float32)) / evalpoly(y, var_mode_rician_kernel_med_4_den_coeffs(Float32)))
@inline var_mode_rician_kernel_med_5_num_coeffs(::Type{Float32}) = (0.9399199f0, 0.24138974f0, 0.042731486f0, 0.00012449107f0)
@inline var_mode_rician_kernel_med_5_den_coeffs(::Type{Float32}) = (1.0f0, 0.24611649f0, 0.043568913f0)
@inline var_mode_rician_kernel_med_5_approx(::Type{Float32}, x::Real) = (y = muladd(3.2713103f0, x, -8.067006f0); evalpoly(y, var_mode_rician_kernel_med_5_num_coeffs(Float32)) / evalpoly(y, var_mode_rician_kernel_med_5_den_coeffs(Float32)))
@inline var_mode_rician_kernel_med_6_num_coeffs(::Type{Float32}) = (0.9658325f0, 0.5415705f0, 0.083717555f0, -7.873431f-5)
@inline var_mode_rician_kernel_med_6_den_coeffs(::Type{Float32}) = (1.0f0, 0.5476194f0, 0.082870066f0)
@inline var_mode_rician_kernel_med_6_approx(::Type{Float32}, x::Real) = (y = muladd(1.2960126f0, x, -4.592125f0); evalpoly(y, var_mode_rician_kernel_med_6_num_coeffs(Float32)) / evalpoly(y, var_mode_rician_kernel_med_6_den_coeffs(Float32)))
@inline var_mode_rician_kernel_tail_coeffs(::Type{Float32}) = (1.0f0, -0.026855424f0, 0.0028834539f0, -0.00024751498f0, 2.5779698f-5)
@inline var_mode_rician_kernel_tail_approx(::Type{Float32}, x::Real) = (y = abs2(4.314869f0 / x); evalpoly(y, var_mode_rician_kernel_tail_coeffs(Float32)))
@inline var_mode_rician_kernel_tail_long_coeffs(::Type{Float32}) = (1.0f0, -0.0034719242f0, 4.74246f-5)
@inline var_mode_rician_kernel_tail_long_approx(::Type{Float32}, x::Real) = (y = abs2(12.0f0 / x); evalpoly(y, var_mode_rician_kernel_tail_long_coeffs(Float32)))

@inline var_mode_rician_branches(::Type{Float64}) = (0.14352975602050932, 0.6598030301174764, 1.0222302053076529, 1.4351303133613367, 1.811785070453181, 2.3026670515934327, 2.967406142636103, 4.007963073769858, 25.0)
@inline var_mode_rician_coeff_taylor(::Type{Float64}) = (0.5, 0.25, 0.0625, -0.015625, -0.023111979166666668, -0.008829752604166666, 0.001477389865451389, 0.003541028058087384)
@inline var_mode_rician_coeff_taylor_cutoff(::Type{Float64}) = first(var_mode_rician_branches(Float64))
@inline var_mode_rician_coeff_long_tail_cutoff(::Type{Float64}) = last(var_mode_rician_branches(Float64))
@inline var_mode_rician_kernel_small_coeffs(::Type{Float64}) = (0.5, 0.2499999999999955, 0.06250000000063673, -0.015625000035477024, -0.023111978140787328, -0.008829770221755791, 0.0014775833086811188, 0.0035396153886338875, 0.0017195252299548993, -7.554135769982454e-5, -0.0005331384875690511, -0.0003932779104940293, -4.79586025256334e-5, 0.0001884728639980545)
@inline var_mode_rician_kernel_small_approx(::Type{Float64}, x::Real) = (y = abs2(x); evalpoly(y, var_mode_rician_kernel_small_coeffs(Float64)))
@inline var_mode_rician_kernel_med_1_num_coeffs(::Type{Float64}) = (0.6957971438876385, -0.1963460686908271, 0.07454904976165508, -0.0072521372198291285, 0.0014307338284965689, -1.5116282902558834e-5, 1.1750682391409532e-5, -3.456533377361415e-7, 5.727001846623604e-9, -6.7629994493220996e-9)
@inline var_mode_rician_kernel_med_1_den_coeffs(::Type{Float64}) = (1.0, -0.4034028412731433, 0.1495262168864651, -0.02141651687427548, 0.002839063241851068)
@inline var_mode_rician_kernel_med_1_approx(::Type{Float64}, x::Real) = (y = muladd(5.518349993900264, x, -4.6410240472241515); evalpoly(y, var_mode_rician_kernel_med_1_num_coeffs(Float64)) / evalpoly(y, var_mode_rician_kernel_med_1_den_coeffs(Float64)))
@inline var_mode_rician_kernel_med_2_num_coeffs(::Type{Float64}) = (0.8586921448117373, 0.20387532395078117, 0.14955921210291603, 0.020385782758009247, 0.005669807369085333, 0.00027119029050807976, 1.2110468580658361e-6, -3.380597597752153e-6, 8.18747832862691e-8, 2.337490462348879e-8, 2.0395448161617922e-10)
@inline var_mode_rician_kernel_med_2_den_coeffs(::Type{Float64}) = (1.0, 0.16928534412042154, 0.18722305407334133, 0.014935700527979249, 0.007813364736581743)
@inline var_mode_rician_kernel_med_2_approx(::Type{Float64}, x::Real) = (y = muladd(4.84378657449992, x, -5.951464944517506); evalpoly(y, var_mode_rician_kernel_med_2_num_coeffs(Float64)) / evalpoly(y, var_mode_rician_kernel_med_2_den_coeffs(Float64)))
@inline var_mode_rician_kernel_med_3_num_coeffs(::Type{Float64}) = (0.9112676335491374, 0.33903521481338406, 0.08567357731721935, 0.009817809357607381, 0.0008845710957540601, 9.433343665858811e-6, -1.8308067930136521e-6, 2.6845471120504946e-7, 1.0461951341442418e-8, -7.011396639282682e-9, 5.69181356239323e-10)
@inline var_mode_rician_kernel_med_3_den_coeffs(::Type{Float64}) = (1.0, 0.3629627338299818, 0.09398177818847686, 0.009263791357433371, 0.0009898141464419713)
@inline var_mode_rician_kernel_med_3_approx(::Type{Float64}, x::Real) = (y = muladd(5.309902403575154, x, -8.620401900360925); evalpoly(y, var_mode_rician_kernel_med_3_num_coeffs(Float64)) / evalpoly(y, var_mode_rician_kernel_med_3_den_coeffs(Float64)))
@inline var_mode_rician_kernel_med_4_num_coeffs(::Type{Float64}) = (0.9258033736492656, 0.4183226960574212, 0.10894392862369848, 0.01648762338316787, 0.0015512310390474937, 2.0496239422608778e-5, -2.3053049480402145e-6, 4.645556733022444e-8, 1.9359046594983133e-8, -1.7879505475603567e-9)
@inline var_mode_rician_kernel_med_4_den_coeffs(::Type{Float64}) = (1.0, 0.44278423523586524, 0.11324124954720827, 0.016797347743675698, 0.001629270305481974)
@inline var_mode_rician_kernel_med_4_approx(::Type{Float64}, x::Real) = (y = muladd(4.074299071549284, x, -8.381754230394248); evalpoly(y, var_mode_rician_kernel_med_4_num_coeffs(Float64)) / evalpoly(y, var_mode_rician_kernel_med_4_den_coeffs(Float64)))
@inline var_mode_rician_kernel_med_5_num_coeffs(::Type{Float64}) = (0.9452518731908671, 0.5477385562759473, 0.16340247969631022, 0.02607561276388251, 0.0019823215533398392, 7.203173059628684e-7, -7.648304960998264e-7, 2.4946619981981545e-7, -5.378670343364823e-8, 7.3151357804397005e-9, -4.890689022300746e-10)
@inline var_mode_rician_kernel_med_5_den_coeffs(::Type{Float64}) = (1.0, 0.5688812790589279, 0.1678828869666718, 0.026438888573370276, 0.0019617320236703)
@inline var_mode_rician_kernel_med_5_approx(::Type{Float64}, x::Real) = (y = muladd(3.00869924298106, x, -7.9280326149665905); evalpoly(y, var_mode_rician_kernel_med_5_num_coeffs(Float64)) / evalpoly(y, var_mode_rician_kernel_med_5_den_coeffs(Float64)))
@inline var_mode_rician_kernel_med_6_num_coeffs(::Type{Float64}) = (0.9649031482456069, 0.9078322507597031, 0.381020948062324, 0.07961906547078143, 0.007105786922176211, 1.4620110534926965e-6, -1.0813164819152643e-6, 2.3438771160356622e-7, -3.537099362920164e-8, 3.796851929648128e-9, -2.2832667112116746e-10)
@inline var_mode_rician_kernel_med_6_den_coeffs(::Type{Float64}) = (1.0, 0.9316698560092325, 0.38792554294146936, 0.08022914281809435, 0.007066487373050203)
@inline var_mode_rician_kernel_med_6_approx(::Type{Float64}, x::Real) = (y = muladd(1.922047645985952, x, -6.7034959911379755); evalpoly(y, var_mode_rician_kernel_med_6_num_coeffs(Float64)) / evalpoly(y, var_mode_rician_kernel_med_6_den_coeffs(Float64)))
@inline var_mode_rician_kernel_tail_coeffs(::Type{Float64}) = (1.0, -0.031125947534768647, 0.00387529844165871, -0.0003920226325096266, 5.6317715993757044e-5, -5.875227582291128e-6, 9.978208143127467e-7, -1.1332775962008893e-7, 8.301157501223511e-8, -7.418009895265052e-8, 5.9444871538910984e-8, -2.6756177938308316e-8, 5.737982432089239e-9)
@inline var_mode_rician_kernel_tail_approx(::Type{Float64}, x::Real) = (y = abs2(4.007963073769858 / x); evalpoly(y, var_mode_rician_kernel_tail_coeffs(Float64)))
@inline var_mode_rician_kernel_tail_long_coeffs(::Type{Float64}) = (1.0, -0.0007999999999999901, 2.5599999998842064e-6, -6.655999505712713e-9, 2.457504604783536e-11, -6.501456937485586e-14)
@inline var_mode_rician_kernel_tail_long_approx(::Type{Float64}, x::Real) = (y = abs2(25.0 / x); evalpoly(y, var_mode_rician_kernel_tail_long_coeffs(Float64)))

function _mean_qrician(ν::T, δ::T, order = Val(16); H = one(T), K = basefloattype(T) === Float64 ? 9.0 : 6.5) where {T}
    K = oftype(ν, K)
    μ = mean_rician(ν)
    δ <= std_rician(ν) / 2 && return μ - δ / 2
    lo, hi = max(zero(T), μ - K), μ + K
    total, mass = zero(T), zero(T)
    for n in floor(Int, lo/δ):floor(Int, hi/δ)
        a, b = max(lo, n * δ), min(hi, (n + 1) * δ)
        b > a || continue
        panels = ceil(Int, (b - a) / H)
        h = (b - a) / panels
        p = sum(j -> f_quadrature(y -> exp(-neglogpdf_rician(y, ν)), a + j * h, h, order), 0:(panels-1))
        total += n * δ * p
        mass += p
    end
    return total / mass
end

@inline mean_qrician(ν::Real, δ::Real) = _mean_qrician(promote_float(ν, δ)...)
@inline mean_qrician(ν::Real, σ::Real, δ::Real) = σ * _mean_qrician(promote_float(ν / σ, δ / σ)...)
