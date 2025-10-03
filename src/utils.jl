#### Floating point utilities

@inline promote_float(x...) = promote(map(float, x)...)

@inline basefloattype(::Type{T}) where {T} = error("Argument is not a floating-point type: T = $T.")
@inline basefloattype(::Type{T}) where {T <: AbstractFloat} = T
@inline basefloattype(::Type{D}) where {T, D <: ForwardDiff.Dual{<:Any, T}} = basefloattype(T)
@inline basefloattype(x::Number) = basefloattype(typeof(x))
@inline basefloattype(x1::Number, x2::Number, xs::Number...) = promote_type(basefloattype(x1), basefloattype(x2), map(basefloattype, xs)...)

@inline function checkedfloattype(::Type{T}) where {T}
    TF = basefloattype(T)
    if TF <: Union{Float32, Float64}
        return TF
    else
        error("Base float type is not Float32 or Float64; found T = $T.")
    end
end
@inline function checkedfloattype(::Type{T1}, ::Type{T2}) where {T1, T2}
    TF1, TF2 = basefloattype(T1), basefloattype(T2)
    if TF1 === TF2
        return TF1
    else
        error("Incompatible types: $T1 and $T2.\nBase float types must all be Float32 or Float64, but found the following types: $(TF1), $(TF2).")
    end
end
@inline checkedfloattype(T1::Type, T2::Type, T3::Type, Ts::Type...) = checkedfloattype(checkedfloattype(T1, T2), T3, Ts...)
@inline checkedfloattype(xs::Number...) = checkedfloattype(map(typeof, xs)...)

@inline promote_eltypes(xs::Tuple) = convert_eltype.(promote_type(map(eltype, xs)...), xs)
@inline promote_eltypes(xs::Union{Number, SVector, SMatrix}...) = promote_eltypes(xs)

@inline convert_eltype(::Type{T}, x::Number) where {T} = convert(T, x)
@inline convert_eltype(::Type{T}, x::SVector{N, <:Number}) where {T, N} = SVector{N, T}(x)
@inline convert_eltype(::Type{T}, x::SMatrix{N, M, <:Number, L}) where {T, N, M, L} = SMatrix{N, M, T, L}(x)

#### Tuple utilities

@inline second(x::Tuple) = x[2]
@inline third(x::Tuple) = x[3]
@inline fourth(x::Tuple) = x[4]

#### Math utilities

@inline logratio(y, x) = y > x ? log1p((y - x) / x) : -log1p((x - y) / y) # robust log(y / x)
@inline logexpplus(x, y) = ((lo, hi) = minmax(x, y); return hi + log1p(exp(lo - hi))) # robust log(exp(x) + exp(y))
@inline logexpminus(x, y) = x + log1p(-exp(y - x)) # robust log(exp(x) - exp(y)) (note: requires x > y)
@inline logabsexpm1(x) = ifelse(x > 0, x, zero(x)) + log(-expm1(-abs(x))) # robust log|exp(x) - 1|

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
