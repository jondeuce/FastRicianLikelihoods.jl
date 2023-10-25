####
#### Rice distribution
####

"""
    Rice(ν, σ)

The *Rice distribution* with parameters `ν` and `σ` has probability density function:

```math
f(x; \\nu, \\sigma) = \\frac{x}{\\sigma^2} \\exp\\left( \\frac{-(x^2 + \\nu^2)}{2\\sigma^2} \\right) I_0\\left( \\frac{x\\nu}{\\sigma^2} \\right).
```

External links:

* [Rice distribution on Wikipedia](https://en.wikipedia.org/wiki/Rice_distribution)

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
@inline Distributions.std(d::Rice) = sqrt(var(d))
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

@inline _mean_rician(ν) = ν > 1/√eps(one(ν)) ? ν : sqrthalfπ * laguerre½(-ν^2 / 2)

@inline _var_rician(ν) = 1 - laguerre½²c(ν) # equivalent to: ν^2 + 2σ^2 - π * σ^2 * laguerre½(-(ν / σ)^2 / 2)^2 / 2 where σ = 1

function _mode_rician(ν::T) where {T <: Union{Float32, Float64}}
    low, med1, med2, med3, med4, med5, tail = T.((0.9, 1.2, 1.5, 1.9, 2.4, 3.7, 50.0))
    if ν < low
        return evalpoly(ν^2, mode_rician_coeff_small(T))
    elseif ν < med1
        return ν + evalpoly(ν, mode_rician_coeff_med1_num(T)) / evalpoly(ν, mode_rician_coeff_med1_den(T))
    elseif ν < med2
        return ν + evalpoly(ν, mode_rician_coeff_med2_num(T)) / evalpoly(ν, mode_rician_coeff_med2_den(T))
    elseif ν < med3
        return ν + evalpoly(ν, mode_rician_coeff_med3_num(T)) / evalpoly(ν, mode_rician_coeff_med3_den(T))
    elseif ν < med4
        return ν + evalpoly(ν, mode_rician_coeff_med4_num(T)) / evalpoly(ν, mode_rician_coeff_med4_den(T))
    elseif ν < med5
        return ν + evalpoly(ν, mode_rician_coeff_med5_num(T)) / evalpoly(ν, mode_rician_coeff_med5_den(T))
    elseif ν < tail
        return ν * evalpoly(1 / ν^2, mode_rician_coeff_tail(T))
    else
        return ν * evalpoly(1 / ν^2, mode_rician_coeff_long_tail(T))
    end
end

function _var_mode_rician(ν::T) where {T <: Union{Float32, Float64}}
    low, med1, med2, med3, med4, med5, med6, tail = T.((0.8, 1.1, 1.5, 1.8, 2.45, 3.2, 4.0, 50.0))
    if ν < low
        return evalpoly(ν^2, var_mode_rician_coeff_small(T))
    elseif ν < med1
        return evalpoly(ν, var_mode_rician_coeff_med1_num(T)) / evalpoly(ν, var_mode_rician_coeff_med1_den(T))
    elseif ν < med2
        return evalpoly(ν, var_mode_rician_coeff_med2_num(T)) / evalpoly(ν, var_mode_rician_coeff_med2_den(T))
    elseif ν < med3
        return evalpoly(ν, var_mode_rician_coeff_med3_num(T)) / evalpoly(ν, var_mode_rician_coeff_med3_den(T))
    elseif ν < med4
        return evalpoly(ν, var_mode_rician_coeff_med4_num(T)) / evalpoly(ν, var_mode_rician_coeff_med4_den(T))
    elseif ν < med5
        return evalpoly(ν, var_mode_rician_coeff_med5_num(T)) / evalpoly(ν, var_mode_rician_coeff_med5_den(T))
    elseif ν < med6
        return evalpoly(ν, var_mode_rician_coeff_med6_num(T)) / evalpoly(ν, var_mode_rician_coeff_med6_den(T))
    elseif ν < tail
        return evalpoly(1 / ν^2, var_mode_rician_coeff_tail(T))
    else
        return evalpoly(1 / ν^2, var_mode_rician_coeff_long_tail(T))
    end
end

mode_rician_coeff_small(::Type{Float64}) = (1.0, 0.24999999999999983, 0.06250000000004943, 0.005208333328153905, -0.005452473741104163, -0.0033243862764034894, -0.0005712375613787041, 0.00042413748492215375, 0.00036474830466622473, 8.181692412922012e-5, 2.2765516530717507e-7, -0.0001332065886118482, 0.00011517103870107813, -0.000138717972958229, 0.00010619522319742027, -3.539577878713359e-5, 3.9325828224333946e-6)
mode_rician_coeff_med1_num(::Type{Float64}) = (1.0036501542798688, -2.6141004053929526, 2.8850891040721804, -1.5438153836598543, 0.35156754778068067)
mode_rician_coeff_med1_den(::Type{Float64}) = (1.0, -1.5724834531262466, 0.8492166790147374, 0.3456469745132729, -0.8606271510640903, 0.8851680473625081, -0.5731161613886224, 0.221845868025135, -0.02914184326805634)
mode_rician_coeff_med2_num(::Type{Float64}) = (1.0217049822030149, -3.4828552045076573, 5.33144720202483, -4.619800821903168, 2.4280472132388495, -0.7700608585474804, 0.14002018652049505, -0.011035747261516828)
mode_rician_coeff_med2_den(::Type{Float64}) = (1.0, -2.3316015361095723, 2.299977219243335, -1.006049623348924, 0.10712086888504588, 0.051746838135892696)
mode_rician_coeff_med3_num(::Type{Float64}) = (4.573008333272072, -10.10060738810654, 9.287028516810604, -4.012594132371049, 0.7549789712321409)
mode_rician_coeff_med3_den(::Type{Float64}) = (1.0, 11.363639299176645, -30.527271554036634, 34.79978239905035, -21.46187735076052, 8.030047912614453, -1.8514660127287854, 0.28929003282228444, -0.019273087780015002)
mode_rician_coeff_med4_num(::Type{Float64}) = (0.7820649715687427, -2.1792616941321734, 2.7066146902020325, -1.8918814698890896, 0.8209389269920119, -0.22675814995521404, 0.0399819339929118, -0.0040884822288673045, 0.00018495988374135573)
mode_rician_coeff_med4_den(::Type{Float64}) = (1.0, -2.062741985197654, 1.8532218171207933, -0.7805216805190053, 0.1452557166581288)
mode_rician_coeff_med5_num(::Type{Float64}) = (0.4914136552697067, -1.224854396901276, 1.448905768241859, -1.0650238164284158, 0.5556449364604901, -0.22039436378799412, 0.06802306790789782, -0.016067123641295485, 0.0028361995283187474, -0.00036120961878139584, 3.1312094857623656e-5, -1.6528396199446517e-6, 4.009180530747741e-8)
mode_rician_coeff_med5_den(::Type{Float64}) = (1.0, -1.78150221203171, 1.3191280547211228, -0.45892266504053547, 0.06637012485882755)
mode_rician_coeff_tail(::Type{Float64}) = (1.0, 0.4999999999999998, -0.37499999999970596, 0.3124999997382041, -0.60156239502447, 0.6366954951674759, -1.8239572701282578, 0.7996801220203986, 9.14300534787131, -834.639526308576, 26620.79609158913, -627118.6175470988, 1.0528187274768809e7, -1.234956151893735e8, 9.582414307496653e8, -4.413710064329928e9, 9.080077189499544e9)
mode_rician_coeff_long_tail(::Type{Float64}) = (1.0, 0.49999999999999994, -0.37499999999045885, 0.3124998760641745, -0.6010582893759605)

mode_rician_coeff_small(::Type{Float32}) = (1.0f0, 0.25f0, 0.06249987f0, 0.005212568f0, -0.0054999627f0, -0.0030814086f0, -0.0011982243f0, 0.0012189724f0)
mode_rician_coeff_med1_num(::Type{Float32}) = (1.0010357f0, -2.1447766f0, 1.8740907f0, -0.7425226f0, 0.11735056f0)
mode_rician_coeff_med1_den(::Type{Float32}) = (1.0f0, -1.1391562f0, 0.479373f0)
mode_rician_coeff_med2_num(::Type{Float32}) = (2.8454518f0, -3.941992f0, 1.7829679f0)
mode_rician_coeff_med2_den(::Type{Float32}) = (1.0f0, 5.1823807f0, -7.594482f0, 3.6317074f0)
mode_rician_coeff_med3_num(::Type{Float32}) = (1.2849327f0, -1.7210481f0, 0.7297914f0)
mode_rician_coeff_med3_den(::Type{Float32}) = (1.0f0, 1.4269043f0, -2.869467f0, 1.4098948f0)
mode_rician_coeff_med4_num(::Type{Float32}) = (1.462152f0, -2.0267386f0, 0.80248296f0)
mode_rician_coeff_med4_den(::Type{Float32}) = (1.0f0, 1.4824599f0, -3.2993996f0, 1.5249913f0)
mode_rician_coeff_med5_num(::Type{Float32}) = (1.3615276f0, -1.1199051f0, 0.028932165f0, -0.0016617135f0)
mode_rician_coeff_med5_den(::Type{Float32}) = (1.0f0, 1.3079536f0, -1.834874f0)
mode_rician_coeff_tail(::Type{Float32}) = (1.0f0, 0.5f0, -0.3749947f0, 0.31192484f0, -0.5797993f0, 0.30167985f0)
mode_rician_coeff_long_tail(::Type{Float32}) = (1.0, 0.5f0, -0.3748751f0)

var_mode_rician_coeff_small(::Type{Float64}) = (0.5, 0.25000000000000144, 0.06249999999904324, -0.01562499990166115, -0.02311198313661696, -0.008829669116827891, 0.001476339859924175, 0.003549566532828328, 0.0016655707808988823, 0.00012672227302081484, -0.001058910505103033, 0.00053883530417269, -0.0011251866563915759, 0.0009212644926788002, -0.00022291910144633337)
var_mode_rician_coeff_med1_num(::Type{Float64}) = (0.4999439772226663, -1.2358720539593997, 1.5337347813500488, -1.2555388691333846, 0.8267330399315418, -0.46232078763245477, 0.21763037776815466, -0.07876616017894338, 0.02088481027285555, -0.0026757344211095816)
var_mode_rician_coeff_med1_den(::Type{Float64}) = (1.0, -2.473042985011394, 2.5741155022757733, -1.2939733356619358, 0.27561100003170313)
var_mode_rician_coeff_med2_num(::Type{Float64}) = (0.4985140555919787, -1.4752187253799824, 2.1554138650244417, -1.9961438634555653, 1.3180371342249235, -0.629349025533249, 0.2066831108228229, -0.03902189200570773, 0.0034820152925163547)
var_mode_rician_coeff_med2_den(::Type{Float64}) = (1.0, -2.9806742237331685, 3.950326026047539, -2.885161458630239, 1.22712327628481, -0.2914798883475255, 0.0348694162649842)
var_mode_rician_coeff_med3_num(::Type{Float64}) = (0.37269219482515936, -0.4605437313851376, -0.053019489313923196, 0.4907332551990692, -0.375856656795812, 0.12717288821330822, -0.017491743993905517, 0.0010613248472684466)
var_mode_rician_coeff_med3_den(::Type{Float64}) = (1.0, -2.1657745248635423, 1.9548738830793133, -0.8294297068125464, 0.1502916624055565)
var_mode_rician_coeff_med4_num(::Type{Float64}) = (0.5004725170473115, -1.197227039195089, 1.3037143619406055, -0.7545261404443518, 0.2328108375413002, -0.019813841986902298, -0.008914088950204718, 0.0028405074484364246, -0.00012580842548162702)
var_mode_rician_coeff_med4_den(::Type{Float64}) = (1.0, -2.6161683101095083, 3.0693901990554964, -2.0056341391343495, 0.7873094953716606, -0.1773099354035607, 0.019133348823846243)
var_mode_rician_coeff_med5_num(::Type{Float64}) = (0.47632084764057286, -0.33534969251156177, -0.594482295996336, 1.0474886100576601, -0.7092112250812861, 0.2723264805018142, -0.06667972504085426, 0.011286991420930582, -0.0012625595527062995, 8.417771861420662e-5, -2.5378083807169782e-6)
var_mode_rician_coeff_med5_den(::Type{Float64}) = (1.0, -1.85491477468255, 1.3927385521071558, -0.48781419341001664, 0.069767281480521)
var_mode_rician_coeff_med6_num(::Type{Float64}) = (0.7080171061125257, -0.9246311484804044, 0.9443197493994299, -0.7823588447905289, 0.3129512103148035, -0.04556860946541361, -3.240172106402777e-7)
var_mode_rician_coeff_med6_den(::Type{Float64}) = (1.0, -1.304535360709752, 1.1073114254399665, -0.8063777265249159, 0.31310049934508744, -0.045579085378644295)
var_mode_rician_coeff_tail(::Type{Float64}) = (1.0, -0.49999999999999917, 0.99999999999605, -1.6249999965175812, 3.7499987975905342, -6.281034638039942, 16.914632696654618, -20.609000269788844, 29.361882919947277, 2102.8573585196664, -40808.71894464553, 543148.5678300181, -4.040760226801702e6, 1.4065882596719136e7)
var_mode_rician_coeff_long_tail(::Type{Float64}) = (1.0, -0.4999999999999987, 0.9999999998998206, -1.6249987462940019, 3.744982182164996)

var_mode_rician_coeff_small(::Type{Float32}) = (0.5f0, 0.24999674f0, 0.06258493f0, -0.016452594f0, -0.019272337f0, -0.017924484f0, 0.011817813f0)
var_mode_rician_coeff_med1_num(::Type{Float32}) = (0.49341157f0, -0.59734076f0, 0.41812313f0, -0.1650894f0, 0.061340135f0)
var_mode_rician_coeff_med1_den(::Type{Float32}) = (1.0f0, -1.2764804f0, 0.5495f0)
var_mode_rician_coeff_med2_num(::Type{Float32}) = (0.42575645f0, -0.5385677f0, 0.16767637f0, 0.107873715f0, -0.05022238f0)
var_mode_rician_coeff_med2_den(::Type{Float32}) = (1.0f0, -1.7818083f0, 1.193247f0, -0.26546502f0)
var_mode_rician_coeff_med3_num(::Type{Float32}) = (0.15950882f0, 0.5474597f0, -0.89737535f0, 0.47839922f0, -0.039324768f0)
var_mode_rician_coeff_med3_den(::Type{Float32}) = (1.0f0, -1.0470282f0, 0.234852f0, 0.13608545f0)
var_mode_rician_coeff_med4_num(::Type{Float32}) = (1.1370257f0, -2.5723681f0, 1.8557117f0, -0.48524946f0)
var_mode_rician_coeff_med4_den(::Type{Float32}) = (1.0f0, -2.3911107f0, 1.7793211f0, -0.48004913f0)
var_mode_rician_coeff_med5_num(::Type{Float32}) = (1.0542713f0, -1.2146388f0, 0.5223497f0)
var_mode_rician_coeff_med5_den(::Type{Float32}) = (1.0f0, -1.1475852f0, 0.5170691f0)
var_mode_rician_coeff_med6_num(::Type{Float32}) = (0.92249256f0, -0.55931264f0, 0.36800492f0)
var_mode_rician_coeff_med6_den(::Type{Float32}) = (1.0f0, -0.5417841f0, 0.36691815f0)
var_mode_rician_coeff_tail(::Type{Float32}) = (1.0f0, -0.49999997f0, 0.9999772f0, -1.6220785f0, 3.6185923f0, -3.8391333f0)
var_mode_rician_coeff_long_tail(::Type{Float32}) = (1.0f0, -0.49999997f0, 0.9993506f0)
