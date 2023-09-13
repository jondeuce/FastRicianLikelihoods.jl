####
#### Macros for defining custom ForwardDiff rules
####

# Automatically defines dispatch overloads for `Dual` arguments.
# For unary functions, this is trivial and requires only one method.
# For binary and ternary functions, this requires three and seven methods
# respectively, as well as many additional methods to disambiguate
# ambiguous argument types. For these cases, we defer to the macros
# `@define_binary_dual_op` and `@define_ternary_dual_op` from `ForwardDiff`.

# Note that there is a package `ForwardDiffChainRules` that provides similar
# functionality, but it's very inefficient for scalar functions of one to three
# variables; internally it stacks everything into vectors and matrices and
# then does multivariable calculus as usual and finally unpacks the results.

@inline unpack_dual(x) = (ForwardDiff.value(x), ForwardDiff.partials(x))

@inline primal_and_partials(fdf, args...) = fdf(args...)
@inline primal_and_partials((f, df)::Tuple, args...) = f(args...), df(args...)

@inline untuple_scalar(x::Tuple) = only(x)
@inline untuple_scalar(x) = x

@inline function unary_dual_pushforward(fdf::F, x::ForwardDiff.Dual{T}) where {F, T}
    vx, px = unpack_dual(x)
    Ω, dΩ_dx = primal_and_partials(fdf, vx)
    dΩ = untuple_scalar(dΩ_dx) * px
    return ForwardDiff.Dual{T}(Ω, dΩ)
end

@inline function binary_dual_pushforward(fdf::F, x, y, ::Type{T}) where {F, T}
    vx, px = unpack_dual(x)
    vy, py = unpack_dual(y)
    Ω, (dΩ_dx, dΩ_dy) = primal_and_partials(fdf, vx, vy)
    dΩ = dΩ_dx * px + dΩ_dy * py
    return ForwardDiff.Dual{T}(Ω, dΩ)
end

@inline function ternary_dual_pushforward(fdf::F, x, y, z, ::Type{T}) where {F, T}
    vx, px = unpack_dual(x)
    vy, py = unpack_dual(y)
    vz, pz = unpack_dual(z)
    Ω, (dΩ_dx, dΩ_dy, dΩ_dz) = primal_and_partials(fdf, vx, vy, vz)
    dΩ = dΩ_dx * px + dΩ_dy * py + dΩ_dz * pz
    return ForwardDiff.Dual{T}(Ω, dΩ)
end

macro define_unary_dual_scalar_rule(f, fdf)
    local M = @__MODULE__
    local FD = ForwardDiff
    local _f, _fdf = esc(f), esc(fdf)
    quote
        $(_f)(x::$(FD).Dual) = $(M).unary_dual_pushforward($(_fdf), x)
    end
end

macro define_binary_dual_scalar_rule(f, df)
    local M = @__MODULE__
    local FD = ForwardDiff
    local _f, _df = esc(f), esc(df)
    quote
        # See: https://github.com/JuliaDiff/ForwardDiff.jl/blob/2ff680824249ad71f55615467bd570c6c29fa673/src/dual.jl#L136
        $(FD).@define_binary_dual_op(
            $(_f),
            $(M).binary_dual_pushforward($(_df), x, y, Txy),
            $(M).binary_dual_pushforward($(_df), x, y, Tx),
            $(M).binary_dual_pushforward($(_df), x, y, Ty),
        )
    end
end

macro define_ternary_dual_scalar_rule(f, df)
    local M = @__MODULE__
    local FD = ForwardDiff
    local _f, _df = esc(f), esc(df)
    quote
        # See: https://github.com/JuliaDiff/ForwardDiff.jl/blob/2ff680824249ad71f55615467bd570c6c29fa673/src/dual.jl#L152
        $(FD).@define_ternary_dual_op(
            $(_f),
            $(M).ternary_dual_pushforward($(_df), x, y, z, Txyz),
            $(M).ternary_dual_pushforward($(_df), x, y, z, Txy),
            $(M).ternary_dual_pushforward($(_df), x, y, z, Txz),
            $(M).ternary_dual_pushforward($(_df), x, y, z, Tyz),
            $(M).ternary_dual_pushforward($(_df), x, y, z, Tx),
            $(M).ternary_dual_pushforward($(_df), x, y, z, Ty),
            $(M).ternary_dual_pushforward($(_df), x, y, z, Tz),
        )
    end
end
