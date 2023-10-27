#### Gaussian quadrature, positive real line

module GaussHalfHermite

using LinearAlgebra: SymTridiagonal, Tridiagonal, eigen, ldiv!
using SpecialFunctions: gamma

# Recursion base cases
alpha₀(γ) = gamma(γ / 2 + 1) / gamma((γ + 1) / 2)
beta₀(γ) = zero(γ)

function g₀₁(γ)
    g₀ = -γ / 12
    g₁ = (5γ + 4) / 12 - alpha₀(γ)^2
    return g₀, g₁
end

# Equation 3.2: α_{n}^2 = (2n + γ + 1)/3 - g_{n+1} - g_{n} = (Y + 1)/3 - g_{n+1} - g_{n}
alphaₙ²(n, γ, gₙ, gₙ₊₁) = (Yₙ(n, γ) + 1) / 3 - gₙ₊₁ - gₙ

# Equation 3.3: β_{n} = (n + γ/2) / 6 + g_{n} = Y / 12 + g_{n}
betaₙ(n, γ, gₙ) = Yₙ(n, γ) / 12 + gₙ

# Equation 3.4
Yₙ(n, γ) = 2 * n + γ

# Equation 3.7, asymptotic limit (valid for all `γ` for suitably large `n`)
gₙasy_large_γ(n, γ) = (2 - 9γ^2) / (72 * Yₙ(n, γ))

# Equation 3.9-3.13, asymptotic limit (valid for small `γ` and large `n`)
function gₙasy(n, γ)
    Y = Yₙ(n, γ)
    Y⁻², γ² = 1 / Y^2, γ^2
    T = typeof(Y⁻²)
    C₀ = evalpoly(γ², (T(1 // 36), T(-1 // 8)))
    C₁ = evalpoly(γ², (T(23 // 432), T(-11 // 48), T(3 // 32)))
    C₂ = evalpoly(γ², (T(1189 // 2592), T(-409 // 192), T(75 // 64), T(9 // 64)))
    C₃ = evalpoly(γ², (T(196057 // 20736), T(-153559 // 3456), T(7111 // 256), T(639 // 128), T(135 // 512)))
    return evalpoly(Y⁻², (C₀, C₁, C₂, C₃)) / Y
end

function g_init!(g, γ)
    N = length(g) - 1
    g₀, g₁ = g₀₁(γ)
    N >= 1 && (g[1] = g₀)
    N >= 2 && (g[2] = g₁)
    return g₀, g₁, g
end
g_init(N, γ) = g_init!(zeros(typeof(float(γ)), N + 1), γ)

# Returns length `n+1` vector `g` containing `[g₀, g₁, ..., gₙ]`
function g_rec!(g, γ; asymptotic)
    N = length(g) - 1
    gₙ₋₁, gₙ, _ = g_init!(g, γ)

    for i in 3:N+1
        n = i - 2
        Y = Yₙ(n, γ)
        gₙ₊₁ = if n >= asymptotic
            γ < 5 ? gₙasy(n + 1, γ) : gₙasy_large_γ(n + 1, γ)
        else
            (16 * (Y + 12 * gₙ)^2 * (-Y^2 + Y * (6 * gₙ + 3 * gₙ₋₁) - 9 * gₙ * (gₙ + gₙ₋₁) + 3 * gₙ₋₁ + 1) + (9 * γ^2 - 4 * (Y - 6 * gₙ)^2)^2) / (48 * (Y + 12 * gₙ)^2 * (-Y + 3 * gₙ + 3 * gₙ₋₁ + 1))
        end
        gₙ₋₁, gₙ, g[i] = gₙ, gₙ₊₁, gₙ₊₁
    end

    return g
end
g_rec(N, γ; kwargs...) = g_rec!(zeros(typeof(float(γ)), N + 1), γ; kwargs...)

# Equation 3.5: The nonlinear equation to solve for g_{n+1} and g_{n}
function Fₙ(n, γ, gₙ₋₁, gₙ, gₙ₊₁)
    # Equivalent version, but *much* less accurate, since gₙ^4 and Y^4 terms cancel out catestrophically
    #   F = ((Y + 1) / 3 - gₙ₊₁ - gₙ) * ((Y - 1) / 3 - gₙ - gₙ₋₁) * (Y / 12 + gₙ)^2 - ((Y / 6 - gₙ)^2 - γ^2 / 16)^2
    Y = Yₙ(n, γ)
    F = Y^3 * (14 * gₙ - (gₙ₊₁ + gₙ₋₁)) / 432 +
        Y^2 * (-414 * gₙ^2 - 126 * gₙ * (gₙ₊₁ + gₙ₋₁) + 18 * gₙ₊₁ * gₙ₋₁ + 6 * (gₙ₊₁ - gₙ₋₁) + 9 * γ^2 - 2) / 2592 -
        Y * gₙ * (-36 * gₙ^2 + 36 * gₙ * (gₙ₊₁ + gₙ₋₁) - 36 * gₙ₊₁ * gₙ₋₁ - 12 * (gₙ₊₁ - gₙ₋₁) + 9 * γ^2 + 4) / 216 +
        gₙ^3 * (gₙ₊₁ + gₙ₋₁) + gₙ^2 * (72 * gₙ₊₁ * gₙ₋₁ + 24 * (gₙ₊₁ - gₙ₋₁) + 9 * γ^2 - 8) / 72 - γ^4 / 256
    return F
end

# Derivative of Equation 3.5 w.r.t. g_{n-1}, g_{n}, and g_{n+1}
function ∇Fₙ(n, γ, gₙ₋₁, gₙ, gₙ₊₁)
    Y = Yₙ(n, γ)
    ∂gₙ₋₁ = (Y + 12 * gₙ)^2 * (-Y + 3 * gₙ + 3 * gₙ₊₁ - 1) / 432
    ∂gₙ = -(Y - 6 * gₙ) * (9 * γ^2 - 4 * (Y - 6 * gₙ)^2) / 216 + (Y + 12 * gₙ)^2 * (-2 * Y + 6 * gₙ + 3 * gₙ₊₁ + 3 * gₙ₋₁) / 432 - (Y + 12 * gₙ) * (-Y + 3 * gₙ + 3 * gₙ₋₁ + 1) * (Y - 3 * gₙ - 3 * gₙ₊₁ + 1) / 54
    ∂gₙ₊₁ = (Y + 12 * gₙ)^2 * (-Y + 3 * gₙ + 3 * gₙ₋₁ + 1) / 432
    return (∂gₙ₋₁, ∂gₙ, ∂gₙ₊₁)
end

function F!(F, g, γ)
    N = length(g) - 1
    @assert length(F) == N - 2

    for i in 1:N-2
        n = i + 1
        gₙ₋₁, gₙ, gₙ₊₁ = g[n], g[n+1], g[n+2] # note: g[n] = gₙ₋₁
        F[i] = Fₙ(n, γ, gₙ₋₁, gₙ, gₙ₊₁)
    end

    return F
end

function J!(Jdiags, g, γ)
    N = length(g) - 1
    (; Jlow, Jdiag, Jhigh) = Jdiags
    @assert length.((Jlow, Jdiag, Jhigh)) == (N - 3, N - 2, N - 3)

    for i in 1:N-2
        n = i + 1
        gₙ₋₁, gₙ, gₙ₊₁ = g[n], g[n+1], g[n+2] # note: g[n] = gₙ₋₁
        ∂gₙ₋₁, ∂gₙ, ∂gₙ₊₁ = ∇Fₙ(n, γ, gₙ₋₁, gₙ, gₙ₊₁)
        Jdiag[i] = ∂gₙ
        i > 1 && (Jlow[i-1] = ∂gₙ₋₁)
        i < N - 2 && (Jhigh[i] = ∂gₙ₊₁)
    end

    return Tridiagonal(Jlow, Jdiag, Jhigh)
end

function g_newton!(Jdiags, F, g, γ; maxiter=50, verbose=false)
    # `g::AbstractVector` has length `N + 1`, representing `gₙ` where `n ∈ 0:N`
    # `F::AbstractVector` has length `N - 2`, representing the `N - 2` equations `Fₙ` used to determine `gₙ` where `n ∈ 2:N-1`
    # `J::Tridiagonal` has size `(N - 2) × (N - 2)`, representing `∂Fᵢ/∂gⱼ` where `i,j ∈ 2:N-1`
    Δg_norm_last = eltype(g)(Inf)
    Δg_norm_decrease = zero(eltype(g))

    for i in 1:maxiter
        F = F!(F, g, γ) # Recurrence equation Fᵢ where `i ∈ 2:N-1`
        J = J!(Jdiags, g, γ) # Tridiagonal Jacobian ∂Fᵢ/∂gⱼ where `i,j ∈ 2:N-1`
        Δg = J \ F # TODO: in-place Tridiagonal solve?
        @views g[3:end-1] .-= Δg # Update estimates for {g₂, g₃, ..., g_{N-1}}

        Δg_norm = maximum(abs, Δg)
        (i > 1) && (Δg_norm_decrease = Δg_norm / Δg_norm_last)
        verbose && @info "iter $i:" F_norm = maximum(abs, F) Δg_norm_decrease Δg_norm

        (Δg_norm_decrease >= 0.95 && Δg_norm <= √eps(eltype(g))) && break
        Δg_norm_last = Δg_norm
    end

    return g
end

function g_newton(N, γ; asymptotic, kwargs...)
    # Initial guess using recurrence relation, switching to asymptotic approximation when `n ≥ asymptotic`
    g = g_rec(N, γ; asymptotic)
    F = similar(g, N - 2)
    Jdiags = (; Jlow=similar(g, N - 3), Jdiag=similar(g, N - 2), Jhigh=similar(g, N - 3))
    return g_newton!(Jdiags, F, g, γ; kwargs...)
end

function g_heuristic(N, γ)
    if N <= 1
        g₀, g₁ = g₀₁(γ)
        return N <= 0 ? [g₀] : [g₀, g₁]
    end

    # Two considerations here:
    #   1. Error ϵₙ in estimate gₙ decreases by factors of ~14 as we get away from the fixed g_{N}, i.e. |ϵₙ₋₁ / ϵₙ| ~ 14
    #   2. Asymptotic estimate for g_{N} is accurate to ~8 digits above N=9 in Float64
    # So if we compute ~10 extra gₙ, error in g_{N} should be less than 10^-8 / 14^10 ~ 10^-20.
    # In practice we get about 10^-15 absolute error and 10^-14 - 10^-11 relative error for N <= 50,
    # both of which slowly increase with N.
    Nnewt = N + 10
    g = g_newton(Nnewt, γ; asymptotic=9)
    g = g[1:N+1] # note: g[n] = gₙ₋₁

    return g
end

function gausshalfhermite_rec_coeffs(N, γ)
    g = g_heuristic(N, γ) # g₀, g₁, ..., g_{N}
    α = [alpha₀(γ); zeros(eltype(g), N - 1)] # β₀, β₁, ..., β_{N-1}
    β = [beta₀(γ); zeros(eltype(g), N - 1)] # β₀, β₁, ..., β_{N-1}
    gₙ, gₙ₊₁ = g[1], g[2]
    for n in 1:N-1
        gₙ, gₙ₊₁ = gₙ₊₁, g[n+2]
        α[n+1] = √alphaₙ²(n, γ, gₙ, gₙ₊₁) # note: α[n+1] = αₙ
        β[n+1] = betaₙ(n, γ, gₙ) # note: β[n+1] = βₙ
    end
    return α, β
end

function gausshalfhermite_gw(n::Integer, γ; normalize=false)
    # Golub-Welsch algorithm
    α, β = gausshalfhermite_rec_coeffs(n, γ)
    T = SymTridiagonal(α, sqrt.(β[2:end]))
    x, Ψ = eigen(T) # eigenvalue decomposition
    w = abs2.(Ψ[1, :]) # quadrature weights
    if normalize
        w ./= sum(w) # ensure weights sum to 1
    else
        Iγ = gamma((γ + 1) / 2) / 2 # Iγ = ∫_{0}^{∞} x^γ exp(-x^2) dx
        w .*= (Iγ / sum(w)) # ensure weights sum to `Iγ`
    end
    return x, w
end

end # module GaussHalfHermite
