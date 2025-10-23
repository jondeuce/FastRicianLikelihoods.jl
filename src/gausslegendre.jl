#=
The MIT License (MIT)

Copyright (c) 2014 Alex Townsend

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=#

@doc raw"""
    GaussLegendre

Compute nodes `x` and weights `w` for Gauss--Legendre quadrature on `[-1, 1]`.

```math
\int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{n} w_i f(x_i)
```

Numerical method:
- For `n ≤ 60`, roots are found by Newton’s method; Legendre polynomials are evaluated via a three‑term recurrence.
- For `n > 60`, an `O(n)` asymptotic expansion provides nodes and weights; optional refinement by Newton is available via `refine=true`.

Public API:
- `gausslegendre(n::Integer, ::Type{T} = Float64; refine = true) -> x, w`

Porting notes:
- Adapted from `FastGaussQuadrature.jl` ([`gausslegendre.jl`](https://github.com/JuliaApproximation/FastGaussQuadrature.jl/blob/b654654677bc254e8f936c54bd2128a6dda57bba/src/gausslegendre.jl))
- Generalized from `Float64` to arbitrary `T`.
- Added `refine` keyword to control Newton refinement for large `n`.
"""
module GaussLegendre

using FastGaussQuadrature: besselZeroRoots, besselJ1
using LinearAlgebra: rmul!

@doc raw"""
    gausslegendre(n::Integer) -> x, w  # nodes, weights

Return nodes `x` and weights `w` of [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).

```math
\int_{-1}^{1} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)
```

# Example
```jldoctest; setup = :(using FastRicianLikelihoods, LinearAlgebra; using FastRicianLikelihoods.GaussLegendre: gausslegendre)
julia> x, w = gausslegendre(3);

julia> f(x) = x^4;

julia> I = dot(w, f.(x));

julia> I ≈ 2/5 # ∫_{-1}^{1} x^4 dx
true
```
"""
function gausslegendre(n::Integer, ::Type{T} = Float64; refine = true) where {T}
    # GAUSSLEGENDRE(n) COMPUTE THE GAUSS-LEGENDRE NODES AND WEIGHTS IN O(n) time.

    if n < 0
        throw(DomainError(n, "Input n must be a non-negative integer"))
    elseif n == 0
        return T[], T[]
    elseif n == 1
        return T[0], T[2]
    elseif n == 2
        return [-inv(sqrt(T(3))), inv(sqrt(T(3)))], T[1, 1]
    elseif n == 3
        return [-sqrt(T(3 // 5)), zero(T), sqrt(T(3 // 5))], T[5//9, 8//9, 5//9]
    elseif n == 4
        a = T(2 // 7) * sqrt(T(6 // 5))
        return ([-sqrt(T(3 // 7) + a), -sqrt(T(3 // 7) - a), sqrt(T(3 // 7) - a), sqrt(T(3 // 7) + a)],
            [(18 - sqrt(T(30))) / 36, (18 + sqrt(T(30))) / 36, (18 + sqrt(T(30))) / 36, (18 - sqrt(T(30))) / 36])
    elseif n == 5
        b = T(2) * sqrt(T(10 // 7))
        return ([-sqrt(T(5) + b) / 3, -sqrt(T(5) - b) / 3, zero(T), sqrt(T(5) - b) / 3, sqrt(T(5) + b) / 3],
            [(322 - 13 * sqrt(T(70))) / 900, (322 + 13 * sqrt(T(70))) / 900, T(128 // 225), (322 + 13 * sqrt(T(70))) / 900, (322 - 13 * sqrt(T(70))) / 900])
    elseif n ≤ 60
        # NEWTON'S METHOD WITH THREE-TERM RECURRENCE:
        return rec(n, T; tol = refine ? sqrt(eps(T)) : T(Inf))
    else
        # USE ASYMPTOTIC EXPANSIONS:
        x, w = asy(n, T)
        return refine ? rec!(n, x, w; tol = sqrt(eps(T))) : (x, w)
    end
end

function asy(n, ::Type{T} = Float64) where {T}
    # COMPUTE GAUSS-LEGENDRE NODES AND WEIGHTS USING ASYMPTOTIC EXPANSIONS.
    # COMPLEXITY O(n).

    # Nodes and weights:
    m = (n + 1) >> 1
    a = T.(besselZeroRoots(m))
    rmul!(a, 2 / T(2n + 1))
    x = legpts_nodes(n, a)
    w = legpts_weights(n, m, a)
    # Use symmetry to get the others:
    resize!(x, n)
    resize!(w, n)
    @inbounds for i in 1:m
        x[n+1-i] = x[i]
        w[n+1-i] = w[i]
    end
    @inbounds for i in 1:m
        x[i] = -x[i]
    end
    @inbounds isodd(n) && (x[m] = zero(T))

    return x, w
end

function legpts_nodes(n, a::AbstractVector{T}) where {T}
    # ASYMPTOTIC EXPANSION FOR THE GAUSS-LEGENDRE NODES.
    vn = 2 / T(2n + 1)
    m = length(a)
    nodes = cot.(a)
    vn² = vn * vn
    vn⁴ = vn² * vn²
    @inbounds if n ≤ 255
        vn⁶ = vn⁴ * vn²
        for i in 1:m
            u = nodes[i]
            u² = u^2
            ai = a[i]
            ai² = ai * ai
            ai³ = ai² * ai
            ai⁵ = ai² * ai³
            node = ai + (u - inv(ai)) / 8 * vn²
            v1 = (6 * (T(1) + u²) / ai + 25 / ai³ - u * muladd(31, u², 33)) / 384
            v2 = u * evalpoly(u², (T(2595 // 15360), T(6350 // 15360), T(3779 // 15360)))
            v3 = (T(1) + u²) * (-muladd(T(31 // 1024), u², T(11 // 1024)) / ai +
                                u / 512 / ai² + T(-25 // 3072) / ai³)
            v4 = (v2 - T(1073 // 5120) / ai⁵ + v3)
            node = muladd(v1, vn⁴, node)
            node = muladd(v4, vn⁶, node)
            nodes[i] = node
        end
    elseif n ≤ 3950
        for i in 1:m
            u = nodes[i]
            u² = u^2
            ai = a[i]
            ai² = ai * ai
            ai³ = ai² * ai
            node = ai + (u - inv(ai)) / 8 * vn²
            v1 = (6 * (T(1) + u²) / ai + 25 / ai³ - u * muladd(31, u², 33)) / 384
            node = muladd(v1, vn⁴, node)
            nodes[i] = node
        end
    else
        for i in 1:m
            u = nodes[i]
            ai = a[i]
            node = ai + (u - inv(ai)) / 8 * vn²
            nodes[i] = node
        end
    end
    @inbounds for j in 1:m
        nodes[j] = cos(nodes[j])
    end

    return nodes
end

function legpts_weights(n, m, a::AbstractVector{T}) where {T}
    # ASYMPTOTIC EXPANSION FOR THE GAUSS-LEGENDRE WEIGHTS.
    vn = 2 / T(2n + 1)
    vn² = vn^2
    weights = similar(a, m)
    if n ≤ 850000
        @inbounds for i in eachindex(weights)
            weights[i] = cot(a[i])
        end
    end
    # Split out the part that can be vectorized by llvm
    @inbounds if n ≤ 170
        for i in eachindex(weights)
            u = weights[i]
            u² = u^2
            ai = a[i]
            ai⁻¹ = inv(ai)
            ai² = ai^2
            ai⁻² = inv(ai²)
            ua = u * ai
            W1 = muladd(ua - T(1), ai⁻², T(1)) / 8
            W2 = evalpoly(ai⁻², (evalpoly(u², (T(-27), T(-84), T(-56))),
                muladd(T(-3), muladd(u², T(-2), T(1)), T(6) * ua),
                muladd(ua, T(-31), T(81)))) / 384
            W3 = evalpoly(ai⁻¹, (evalpoly(u², (T(153 // 1024), T(295 // 256), T(187 // 96), T(151 // 160))),
                evalpoly(u², (T(-65 // 1024), T(-119 // 768), T(-35 // 384))) * u,
                evalpoly(u², (T(5 // 512), T(15 // 512), T(7 // 384))),
                muladd(u², T(1 // 512), T(-13 // 1536)) * u,
                muladd(u², T(-7 // 384), T(53 // 3072)),
                T(3749 // 15360) * u, T(-1125 // 1024)))
            weights[i] = evalpoly(vn², (inv(vn²) + W1, W2, W3))
        end
    elseif n ≤ 1500
        for i in eachindex(weights)
            u = weights[i]
            u² = u^2
            ai = a[i]
            ai² = ai^2
            ai⁻² = inv(ai²)
            ua = u * ai
            W1 = muladd(ua - T(1), ai⁻², T(1)) / 8
            W2 = evalpoly(ai⁻², (evalpoly(u², (T(-27), T(-84), T(-56))),
                muladd(T(-3), muladd(u², T(-2), T(1)), T(6) * ua),
                muladd(ua, T(-31), T(81)))) / 384
            weights[i] = muladd(vn², W2, inv(vn²) + W1)
        end
    elseif n ≤ 850000
        for i in eachindex(weights)
            u = weights[i]
            u² = u^2
            ai = a[i]
            ai² = ai^2
            ai⁻² = inv(ai²)
            ua = u * ai
            W1 = muladd(ua - T(1), ai⁻², T(1)) / 8
            weights[i] = inv(vn²) + W1
        end
    else
        for i in eachindex(weights)
            weights[i] = inv(vn²)
        end
    end
    bJ1 = T.(besselJ1(m))
    @inbounds for i in eachindex(weights)
        weights[i] = T(2) / (bJ1[i] * (a[i] / sin(a[i])) * weights[i])
    end

    return weights
end

function rec(n, ::Type{T} = Float64; tol::T) where {T}
    # Initial guesses:
    x, w = asy(n, T)
    return rec!(n, x, w; tol)
end

function rec!(n, x::AbstractVector{T}, w::AbstractVector{T}; tol::T) where {T}
    # COMPUTE GAUSS-LEGENDRE NODES AND WEIGHTS USING NEWTON'S METHOD.
    # THREE-TERM RECURENCE IS USED FOR EVALUATION. COMPLEXITY O(n^2).

    # Newton iteration to find zeros of Legendre polynomials:
    xhalf = @view x[1:n÷2+1]
    whalf = @view w[1:n÷2+1]
    leg_newton_rec!(n, xhalf, whalf; tol)

    # Use symmetry to get the other Legendre nodes and weights:
    @inbounds for i in 1:n÷2
        x[n+1-i] = -x[i]
        w[n+1-i] = w[i]
    end

    return x, w
end

function eval_legpoly(n, x::T) where {T}
    # EVALUATE LEGENDRE AND ITS DERIVATIVE USING THREE-TERM RECURRENCE RELATION.
    Pm2 = one(T)
    Pm1 = x
    PPm1 = one(T)
    PPm2 = zero(T)
    for k in 1:(n-1)
        Pm2, Pm1 = Pm1, muladd((2 * k + 1) * Pm1, x, -k * Pm2) / (k + 1)
        PPm2, PPm1 = PPm1, ((2 * k + 1) * muladd(x, PPm1, Pm2) -
                            k * PPm2) / (k + 1)
    end
    return Pm1, PPm1
end

function leg_newton_rec!(n, x::AbstractVector{T}, w::AbstractVector{T}; tol::T) where {T}
    # Find zeros of Legendre polynomials via Newton iteration using three-term recurrence relation to evaluate Legendre polynomials and their derivatives.
    @inbounds for j in eachindex(x, w)
        # Newton iteration to find zeros of Legendre polynomials
        xj = x[j]
        while true
            # Newton squares residual with each iteration; continue until residual is below `tol`
            PP1, PP2 = eval_legpoly(n, xj)
            xj -= PP1 / PP2
            abs(PP1) < tol && break
        end
        PP1, PP2 = eval_legpoly(n, xj) # evaluate derivative at converged point
        xj -= PP1 / PP2
        x[j] = xj
        w[j] = T(2) / ((one(T) - xj^2) * PP2^2)
    end
    return x, w
end

end # module GaussLegendre
