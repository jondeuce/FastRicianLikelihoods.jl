####
#### Fast + differentiable Bessel functions
####

#### CUDA-friendly native julia Besselix functions

"Approximation of besselix(0, x) = exp(-|x|) * besseli(0, x)"
function besseli0x end

"Approximation of besselix(1, x) = exp(-|x|) * besseli(1, x)"
function besseli1x end

"Approximation of besselix(2, x) = exp(-|x|) * besseli(2, x)"
function besseli2x end

@inline function besseli0x(x::Real)
    ax = abs(x)
    if ax < 3.75f0
        y = (ax / 3.75f0)^2
        y = exp(-ax) * evalpoly(y, (1.0f0, 3.5156229f0, 3.0899424f0, 1.2067492f0, 0.2659732f0, 0.360768f-1, 0.45813f-2))
    else
        y = 3.75f0 / ax
        y = evalpoly(y, (0.39894228f0, 0.1328592f-1, 0.225319f-2, -0.157565f-2, 0.916281f-2, -0.2057706f-1, 0.2635537f-1, -0.1647633f-1, 0.392377f-2))
        y /= sqrt(ax)
    end
    return y
end

@inline function besseli1x(x::Real)
    ax = abs(x)
    if ax < 3.75f0
        y = (ax / 3.75f0)^2
        y = exp(-ax) * ax * evalpoly(y, (0.5f0, 0.87890594f0, 0.51498869f0, 0.15084934f0, 0.2658733f-1, 0.301532f-2, 0.32411f-3))
    else
        y = 3.75f0 / ax
        y = evalpoly(y, (0.39894228f0, -0.3988024f-1, -0.362018f-2, 0.163801f-2, -0.1031555f-1, 0.2282967f-1, -0.2895312f-1, 0.1787654f-1, -0.420059f-2))
        y /= sqrt(ax)
    end
    return x < 0 ? -y : y
end

@inline function besseli2x(x::Real)
    ax = abs(x)
    if ax < 3.75f0
        y = (ax / 3.75f0)^2
        y = exp(-ax) * ax^2 * evalpoly(y, (0.125f0, 0.14648436f0, 0.06437322f0, 0.015086209f0, 0.0022135116f0, 0.0002170313f0, 1.9755797f-5))
    else
        y = 3.75f0 / ax
        y = evalpoly(y, (0.39894232f0, -0.19947755f0, 0.02343734f0, 0.0007118358f0, 0.008240167f0, -0.018552817f0, 0.0246264f0, -0.014846447f0, 0.0032440408f0))
        y /= sqrt(ax)
    end
    return y
end

#### Derived special functions

@inline besseli1i0m1(x::Real) = besseli1x(x) / besseli0x(x) - 1
@inline besseli2i0(x::Real) = besseli2x(x) / besseli0x(x)

# log(besselix(0, x)) loses accuracy near zero since besselix(0, x) -> 1 as x -> 0; replace with Taylor series
@inline logbesseli0_taylor(x::Real) = (x² = abs2(x); return x² * evalpoly(x², (0.25f0, -0.015625f0, 0.0017361111f0, -0.00022379558f0, 3.092448f-5, -4.45519f-6)))
@inline logbesseli0(x::Real) = abs(x) < 1 ? logbesseli0_taylor(x) : log(besseli0x(x)) + abs(x) # since log(besselix(0, x)) = log(I0(x)) - |x|
@inline logbesseli0x(x::Real) = abs(x) < 1 ? logbesseli0_taylor(x) - abs(x) : log(besseli0x(x)) # since log(besselix(0, x)) = log(I0(x)) - |x|

@inline logbesseli1(x::Real) = logbesseli1x(x) + abs(x) # since log(besselix(1, x)) = log(I1(x)) - |x|
@inline logbesseli1x(x::Real) = log(besseli1x(x)) # since log(besselix(1, x)) = log(I1(x)) - |x|

@inline logbesseli2(x::Real) = logbesseli2x(x) + abs(x) # since log(besselix(2, x)) = log(I2(x)) - |x|
@inline logbesseli2x(x::Real) = log(besseli2x(x)) # since log(besselix(2, x)) = log(I2(x)) - |x|

@inline laguerre½(x::Real) = ifelse(x < 0, one(x), exp(x)) * ((1 - x) * besseli0x(-x/2) - x * besseli1x(-x/2)) # besselix(ν, ±x/2) = Iν(±x/2) * exp(-|±x/2|) = Iν(-x/2) * exp(∓x/2)
@inline ∂x_laguerre½(x::Real) = ifelse(x < 0, one(x), exp(x)) * (besseli1x(x/2) - besseli0x(x/2)) / 2
@scalar_rule laguerre½(x::Real) ∂x_laguerre½(x)
@define_unary_dual_scalar_rule laguerre½ (laguerre½, ∂x_laguerre½)

#### ChainRules and ForwardDiff

@inline ∂x_besseli0x(Ω::Real, x::Real) = besseli1x(x) - sign(x) * Ω
@inline f_∂x_besseli0x(x::Real) = (Ω = besseli0x(x); return (Ω, ∂x_besseli0x(Ω, x)))
@scalar_rule besseli0x(x::Real) ∂x_besseli0x(Ω, x)
@define_unary_dual_scalar_rule besseli0x f_∂x_besseli0x

@inline ∂x_besseli1x(Ω::Real, x::Real) = (besseli0x(x) + besseli2x(x)) / 2 - sign(x) * Ω
@inline f_∂x_besseli1x(x::Real) = (Ω = besseli1x(x); return (Ω, ∂x_besseli1x(Ω, x)))
@scalar_rule besseli1x(x::Real) ∂x_besseli1x(Ω, x)
@define_unary_dual_scalar_rule besseli1x f_∂x_besseli1x
