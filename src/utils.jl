# Floating point utilities

@inline promote_float(x...) = promote(map(float, x)...)

@inline basefloattype(::Type{T}) where {T} = error("Argument is not a floating-point type: $T.")
@inline basefloattype(::Type{T}) where {T <: AbstractFloat} = T
@inline basefloattype(::Type{D}) where {T, D <: ForwardDiff.Dual{<:Any, T}} = basefloattype(T)

@generated function checkedfloattype(::Ts) where {Ts <: Tuple}
    Tbases = map(basefloattype, Ts.types)
    Tunique = unique(Tbases)
    if length(Tunique) == 1 && (T = Tunique[1]) <: Union{Float32, Float64}
        return T
    else
        error("Incompatible types: $Ts.\nBase float types must all be Float32 or Float64, but found the following types: $Tunique.")
    end
end
checkedfloattype(xs::Number...) = checkedfloattype(xs)

# Clenshaw scheme for evaluating scalar-valued Chebyshev polynomials
#   See: https://github.com/chebfun/chebfun/blob/18f759287b6b88e3c3e0cf7885f559791a483127/%40chebtech/clenshaw.m#L94

function clenshaw(x::T1, c::AbstractVector{T2}) where {T1, T2}
    n = length(c)
    n == 0 && return zero(T1)
    n == 1 && return convert(promote_type(T1, T2), c[1])
    n == 2 && return muladd(x, c[2], c[1])
    x2 = 2x
    @inbounds begin
        bₖ₊₂ = c[n]
        bₖ₊₁ = muladd(x2, bₖ₊₂, c[n-1])
        for k in n-2:-2:3
            bₖ₊₂ = muladd(x2, bₖ₊₁, c[k] - bₖ₊₂)
            bₖ₊₁ = muladd(x2, bₖ₊₂, c[k-1] - bₖ₊₁)
        end
        if iseven(n)
            bₖ₊₂, bₖ₊₁ = bₖ₊₁, muladd(x2, bₖ₊₁, c[2] - bₖ₊₂)
        end
        y = muladd(x, bₖ₊₁, c[1] - bₖ₊₂)
    end
    return y
end

@generated function clenshaw(x::T1, c::NTuple{n, T2}) where {n, T1, T2}
    n == 0 && return :($(zero(T1)))
    n == 1 && return :(convert($(promote_type(T1, T2)), c[1]))
    n == 2 && return :(muladd(x, c[2], c[1]))
    return quote
        x2 = 2x
        bₖ₊₂ = c[$(n)]
        bₖ₊₁ = muladd(x2, bₖ₊₂, c[$(n - 1)])
        $(
            map(n-2:-2:3) do k
                quote
                    bₖ₊₂ = muladd(x2, bₖ₊₁, c[$(k)] - bₖ₊₂)
                    bₖ₊₁ = muladd(x2, bₖ₊₂, c[$(k - 1)] - bₖ₊₁)
                end
            end...
        )
        $(
            if iseven(n)
                :((bₖ₊₂, bₖ₊₁) = (bₖ₊₁, muladd(x2, bₖ₊₁, c[2] - bₖ₊₂)))
            end
        )
        y = muladd(x, bₖ₊₁, c[1] - bₖ₊₂)
        return y
    end
end

# Rescale x from interval [a, b] to [-1, 1]
clenshaw(x, a, b, c) = clenshaw((2x - (a + b)) / (b - a), c)
