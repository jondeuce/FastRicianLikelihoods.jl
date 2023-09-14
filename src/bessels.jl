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
        y = exp(-ax) * evalpoly(y, (1.0f0, 3.5156212f0, 3.089966f0, 1.20663f0, 0.26623538f0, 0.035819437f0, 0.0046737944f0))
    else
        y = 3.75f0 / ax
        y = evalpoly(y, (0.3989423f0, 0.013294099f0, 0.002100603f0, -0.00051136187f0, 0.005409987f0, -0.013205506f0, 0.018180212f0, -0.01168953f0, 0.002773462f0))
        y /= sqrt(ax)
    end
    return y
end

@inline function besseli1x(x::Real)
    ax = abs(x)
    if ax < 3.75f0
        y = (ax / 3.75f0)^2
        y = exp(-ax) * ax * evalpoly(y, (0.5f0, 0.878906f0, 0.5149879f0, 0.15085198f0, 0.026583772f0, 0.0030173112f0, 0.00032376772f0))
    else
        y = 3.75f0 / ax
        y = evalpoly(y, (0.39894226f0, -0.039889723f0, -0.0034435112f0, 0.00041189327f0, -0.0060203597f0, 0.014447087f0, -0.01970894f0, 0.012488588f0, -0.0029104343f0))
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

# log(besselix(0, x)) loses accuracy near x = 0 since besselix(0, x) -> 1 as x -> 0; replace with minimax polynomial approximation
@inline logbesseli0_remez(x::Real) = (x² = abs2(x); return x² * evalpoly(x², (0.25f0, -0.015624705f0, 0.001733676f0, -0.00021666016f0, 2.2059316f-5)))
@inline logbesseli0(x::Real) = abs(x) < 1 ? logbesseli0_remez(x) : log(besseli0x(x)) + abs(x) # log(besselix(0, x)) = log(I0(x)) - |x|
@inline logbesseli0x(x::Real) = abs(x) < 1 ? logbesseli0_remez(x) - abs(x) : log(besseli0x(x))

@inline logbesseli1(x::Real) = logbesseli1x(x) + abs(x) # log(besselix(1, x)) = log(I1(x)) - |x|
@inline logbesseli1x(x::Real) = log(besseli1x(x))

@inline logbesseli2(x::Real) = logbesseli2x(x) + abs(x) # log(besselix(2, x)) = log(I2(x)) - |x|
@inline logbesseli2x(x::Real) = log(besseli2x(x))

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
