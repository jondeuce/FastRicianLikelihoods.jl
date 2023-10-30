#### Gaussian quadrature, positive real line

module GaussHalfHermite

using LinearAlgebra: SymTridiagonal, Tridiagonal, eigen
using SpecialFunctions: gamma, loggamma
using IrrationalConstants: sqrtπ, invsqrtπ

# Recursion base cases
function alpha₀(γ)
    T = typeof(float(γ))
    α₀ = γ == 0 ? T(invsqrtπ) :
         γ == 1 ? T(sqrtπ) / 2 :
         γ == 2 ? 2 * T(invsqrtπ) :
         γ < 25 ? gamma(γ / 2 + 1) / gamma((γ + 1) / 2) : exp(loggamma(γ / 2 + 1) - loggamma((γ + 1) / 2))
    return α₀
end

beta₀(γ) = zero(γ)

function g₀₁(γ)
    T = typeof(float(γ))
    g₀ = -γ / 12
    g₁ = γ == 0 ? (T(π) - 3) / (3 * T(π)) :
         γ == 1 ? (3 - T(π)) / 4 :
         γ == 2 ? (7 * T(π) - 24) / (6 * T(π)) :
         (5γ + 4) / 12 - alpha₀(γ)^2
    return g₀, g₁
end

# Equation 3.2: α_{n}^2 = (2n + γ + 1)/3 - g_{n+1} - g_{n} = (Y + 1)/3 - g_{n+1} - g_{n}
alphaₙ²(n, γ, gₙ, gₙ₊₁) = (Yₙ(n, γ) + 1) / 3 - gₙ₊₁ - gₙ

# Equation 3.3: β_{n} = (n + γ/2) / 6 + g_{n} = Y / 12 + g_{n}
betaₙ(n, γ, gₙ) = Yₙ(n, γ) / 12 + gₙ

# Equation 3.4
Yₙ(n, γ) = 2n + γ

# Equation 3.7, asymptotic limit (valid for all `γ` for suitably large `n`)
gₙ_limit(n, γ) = (2 - 9γ^2) / (72 * Yₙ(n, γ))

# Empirical improvement to Equation 3.7 (found via symbolic regression); better for all `n ≥ 2` for `γ ⪆ 2.5`, and good enough for initializing Newton for `γ ⪅ 2.5`
function gₙ_heuristic(n, γ)
    Y = Yₙ(n, γ)
    return (2 - 9γ^2) / (72 * (Y + γ^2 / (Y + γ)))
end

# Equation 3.9-3.13, asymptotic limit (valid for large `n ≫ 1`, empirically around `n ⪆ 2γ`)
function gₙ_asy(n, γ)
    T = typeof(float(γ))
    Y, γ² = Yₙ(n, γ), γ^2
    C₀ = evalpoly(γ², (T(1 // 36), T(-1 // 8)))
    C₁ = evalpoly(γ², (T(23 // 432), T(-11 // 48), T(3 // 32)))
    C₂ = evalpoly(γ², (T(1189 // 2592), T(-409 // 192), T(75 // 64), T(9 // 64)))
    C₃ = evalpoly(γ², (T(196057 // 20736), T(-153559 // 3456), T(7111 // 256), T(639 // 128), T(135 // 512)))
    return evalpoly(inv(Y^2), (C₀, C₁, C₂, C₃)) / Y
end

# Asymptotic estimate for large γ
function gₙ_asy_large_γ(n, γ)
    m = 2 * floor(Int, γ)
    return n >= m ?
           gₙ_asy(n, γ) : # Use asymptotic limit for `n >= 2γ` (i.e. when `γ / Y = γ / (2n + γ) <= 1/5`)
           gₙ_heuristic(n, γ) # Empirically never worse than Equation 3.7 for any `n` for `γ ⪆ 2.5`, and much better for `2.5 <= γ <= n/2`)
end

# Solving Equation 3.5 for gₙ₊₁
function gₙ₊₁_rec(n, γ, gₙ₋₁, gₙ)
    Y = Yₙ(n, γ)
    gₙ₊₁ = (
        24gₙ * (Y^2 + 18gₙ^2) * (2Y - 3gₙ) -
        (Y + 12gₙ)^2 * (3Y * (2gₙ + gₙ₋₁) - 9gₙ * (gₙ + gₙ₋₁) + 3gₙ₋₁ + 1) +
        (3γ / 2)^2 * (2 * (Y - 6gₙ)^2 - (3γ / 2)^2)
    ) / (
        3 * (Y + 12gₙ)^2 * (Y - 3 * (gₙ + gₙ₋₁) - 1)
    )
    return gₙ₊₁
end

# Equation 3.5: The nonlinear equation to solve for g_{n+1} and g_{n}
function Fₙ(n, γ, gₙ₋₁, gₙ, gₙ₊₁)
    # Equivalent version, but *much* less accurate, since gₙ^4 and Y^4 terms cancel out catestrophically
    #   F = ((Y + 1) / 3 - gₙ₊₁ - gₙ) * ((Y - 1) / 3 - gₙ - gₙ₋₁) * (Y / 12 + gₙ)^2 - ((Y / 6 - gₙ)^2 - γ^2 / 16)^2
    Y = Yₙ(n, γ)
    F = (Y / 12 + gₙ)^2 * (
        gₙ₊₁ * (-Y + 3gₙ + 1 + 3gₙ₋₁ / 2) / 3 +
        gₙ₋₁ * (-Y + 3gₙ - 1 + 3gₙ₊₁ / 2) / 3
    ) - gₙ * (
        2Y * (-7Y^2 + 9γ^2 + 4) + 3gₙ * (23Y^2 - 24Y * gₙ - 18γ^2 + 16)
    ) / 432 +
        Y^2 * (9γ^2 - 2) / 2592 - γ^4 / 256
    return F
end

# Derivative of Equation 3.5 w.r.t. g_{n-1}, g_{n}, and g_{n+1}
function ∇Fₙ(n, γ, gₙ₋₁, gₙ, gₙ₊₁)
    Y = Yₙ(n, γ)
    ∂gₙ₋₁ = (Y + 12gₙ)^2 * (-Y + 3gₙ + 3gₙ₊₁ - 1) / 432
    ∂gₙ = (Y - 6gₙ) * (2Y - 12gₙ - 3γ) * (2Y - 12gₙ + 3γ) / 216 + (Y + 12gₙ) * ((Y + 12gₙ) * (-2Y + 6gₙ + 3gₙ₊₁ + 3gₙ₋₁) - 8 * (-Y + 3gₙ + 3gₙ₋₁ + 1) * (Y - 3gₙ - 3gₙ₊₁ + 1)) / 432
    ∂gₙ₊₁ = (Y + 12gₙ)^2 * (-Y + 3gₙ + 3gₙ₋₁ + 1) / 432
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
    @assert length.(Jdiags) == (N - 3, N - 2, N - 3)

    J₋, J₀, J₊ = Jdiags
    for i in 1:N-2
        n = i + 1
        gₙ₋₁, gₙ, gₙ₊₁ = g[n], g[n+1], g[n+2] # note: g[n] = gₙ₋₁
        ∂gₙ₋₁, ∂gₙ, ∂gₙ₊₁ = ∇Fₙ(n, γ, gₙ₋₁, gₙ, gₙ₊₁)
        J₀[i] = ∂gₙ
        i > 1 && (J₋[i-1] = ∂gₙ₋₁)
        i < N - 2 && (J₊[i] = ∂gₙ₊₁)
    end

    return Tridiagonal(J₋, J₀, J₊)
end

# Returns length `n+1` vector `g` containing `[g₀, g₁, ..., gₙ]`
function g_init!(g, γ; asymptotic)
    N = length(g) - 1
    gₙ₋₁, gₙ = g₀₁(γ)
    N >= 0 && (g[1] = gₙ₋₁)
    N >= 1 && (g[2] = gₙ)

    for i in 3:N+1
        n = i - 2

        if n < asymptotic
            gₙ₊₁ = gₙ₊₁_rec(n, γ, gₙ₋₁, gₙ)
            if γ >= 2.5
                # Detect if estimates have diverged from heuristic; recurrence equation can fail quickly for large `γ`, and asymptotic limit can be inaccurate for moderate `n`
                ĝₙ₊₁ = gₙ_heuristic(n + 1, γ)
                if !isfinite(gₙ₊₁) || !isapprox(gₙ₊₁, ĝₙ₊₁; rtol = 0.05, atol = 0.05) # only switch if estimate is very bad, otherwise Newton handles it fine
                    gₙ₊₁ = ĝₙ₊₁
                    asymptotic = n
                end
            end
        else
            gₙ₊₁ = γ < 2.5 ? gₙ_asy(n + 1, γ) : gₙ_asy_large_γ(n + 1, γ)
        end

        g[i] = gₙ₊₁
        gₙ₋₁, gₙ = gₙ, gₙ₊₁
    end

    return g
end
g_init(N, γ; kwargs...) = g_init!(zeros(typeof(float(γ)), N + 1), γ; kwargs...)

function g_newton!(Jdiags, F, g, γ; maxiter = 50, verbose = false)
    # `J` is tri-diagonal with size `(N - 2) × (N - 2)`, representing `∂Fᵢ/∂gⱼ` where `i,j ∈ 2:N-1`
    # `F` has length `N - 2`, representing the `N - 2` equations `Fₙ` used to determine `gₙ` where `n ∈ 2:N-1`
    # `g` has length `N + 1`, representing `gₙ` where `n ∈ 0:N`
    Δg_norm_last = eltype(g)(Inf)
    Δg_norm_decrease = zero(eltype(g))

    for i in 1:maxiter
        F = F!(F, g, γ) # Recurrence equation Fᵢ where `i ∈ 2:N-1`
        J = J!(Jdiags, g, γ) # Tridiagonal Jacobian ∂Fᵢ/∂gⱼ where `i,j ∈ 2:N-1`
        Δg = J \ F # TODO: in-place Tridiagonal solve?
        @views g[3:end-1] .-= Δg # Update estimates for {g₂, g₃, ..., g_{N-1}}

        g_norm = @views maximum(abs, g[3:end-1])
        Δg_norm = maximum(abs, Δg)
        (i > 1) && (Δg_norm_decrease = Δg_norm_last == 0 ? one(eltype(g)) : Δg_norm / Δg_norm_last)
        verbose && @info "iter $i:" F_norm = maximum(abs, F) Δg_norm Δg_norm_last Δg_norm_decrease

        (Δg_norm_decrease >= 0.95 && Δg_norm <= cbrt(eps(eltype(g)))^2 * g_norm) && break
        Δg_norm_last = Δg_norm

        if i == maxiter
            @warn "Newton's method failed to converge in $maxiter iterations" γ F_norm = maximum(abs, F) Δg_norm Δg_norm_last Δg_norm_decrease
        end
    end

    return g
end

function g_newton(N, γ; asymptotic, kwargs...)
    # Initial guess using recurrence relation, switching to asymptotic approximation when `n ≥ asymptotic`
    g = g_init(N, γ; asymptotic)
    F = similar(g, N - 2)
    Jdiags = (similar(g, N - 3), similar(g, N - 2), similar(g, N - 3))
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
    # In practice we get ~machine precision for all `N` when `γ ⪅ 2`, with slow loss of accuracy for large `γ` when `N ⪅ γ`.
    # For large γ, it seems that computing another ~10 terms is necessary, but this has not been investigated thoroughly.
    Nnewt = N + (γ < 2.5 ? 10 : 20)
    g = g_newton(Nnewt, γ; asymptotic = 9)
    g = g[1:N+1] # note: g[n] = gₙ₋₁

    return g
end

function gausshalfhermite_rec_coeffs(N, γ)
    γ = float(γ)
    T = typeof(γ)
    g = g_heuristic(N, γ) # [g₀, g₁, ..., g_{N}]
    α, β = zeros(T, N), zeros(T, N) # [α₀, ..., α_{N-1}] and [β₀, ..., β_{N-1}]
    α[1], β[1] = alpha₀(γ), beta₀(γ)
    gₙ, gₙ₊₁ = g[1], g[2]

    for n in 1:N-1
        gₙ, gₙ₊₁ = gₙ₊₁, g[n+2]
        α[n+1] = √alphaₙ²(n, γ, gₙ, gₙ₊₁) # note: α[n+1] = αₙ
        β[n+1] = betaₙ(n, γ, gₙ) # note: β[n+1] = βₙ
    end

    return α, β
end

function gausshalfhermite_gw(N, γ; normalize = false)
    # Golub-Welsch algorithm for computing nodes and weights from recurrence coefficients
    #   see: https://en.wikipedia.org/wiki/Gaussian_quadrature#The_Golub-Welsch_algorithm
    γ = float(γ)
    T = typeof(γ)
    α, β = gausshalfhermite_rec_coeffs(N, γ)
    𝒥 = SymTridiagonal(α, sqrt.(β[2:end]))
    x, ϕ = eigen(𝒥) # eigenvalue decomposition
    w = abs2.(ϕ[1, :]) # quadrature weights

    Iγ = gamma((γ + 1) / 2) / 2 # Iγ = ∫_{0}^{∞} [x^γ exp(-x^2)] dx
    if normalize
        Iγ *= exp2(γ / 2) * T(invsqrtπ) # Iγ′ = ∫_{0}^{∞} [x^γ exp(-x^2 / 2) / √(2π)] dx = (2^(γ/2) / √π) * Iγ
        x .*= √(T(2))
    end
    w .*= (Iγ / sum(w)) # ensure weights sum to `Iγ`

    return x, w
end

end # module GaussHalfHermite
