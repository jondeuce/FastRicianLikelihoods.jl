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

#### Utilities

@inline unpack_dual(x) = (ForwardDiff.value(x), ForwardDiff.partials(x))

@inline primal_and_partials(fdf::F, args::TA) where {F, TA <: Tuple} = fdf(args...)
@inline primal_and_partials(fdf::F, args::TA) where {F <: Tuple{Any, Any}, TA <: Tuple} = primal_and_partials(fdf[1], fdf[2], args)
@inline primal_and_partials(f::F, df::DF, args::TA) where {F, DF, TA <: Tuple} = f(args...), df(args...)

@inline untuple_scalar(x::Tuple) = only(x)
@inline untuple_scalar(x) = x

#### Pushforwards

@inline function unary_dual_pushforward(fdf::F, x::Dual{T}) where {F, T}
    vx, px = unpack_dual(x)
    Ω, dΩ_dx = primal_and_partials(fdf, (vx,))
    dΩ = untuple_scalar(dΩ_dx) * px
    return Dual{T}(Ω, dΩ)
end

@inline function binary_dual_pushforward(fdf::F, x, y, ::Type{T}) where {F, T}
    vx, px = unpack_dual(x)
    vy, py = unpack_dual(y)
    Ω, (dΩ_dx, dΩ_dy) = primal_and_partials(fdf, (vx, vy))
    dΩ = dΩ_dx * px + dΩ_dy * py
    return Dual{T}(Ω, dΩ)
end

@inline function ternary_dual_pushforward(fdf::F, x, y, z, ::Type{T}) where {F, T}
    vx, px = unpack_dual(x)
    vy, py = unpack_dual(y)
    vz, pz = unpack_dual(z)
    Ω, (dΩ_dx, dΩ_dy, dΩ_dz) = fdf(vx, vy, vz)
    dΩ = dΩ_dx * px + dΩ_dy * py + dΩ_dz * pz
    return Dual{T}(Ω, dΩ)
end

@inline function ternary_dual_pushforward(f::F, df::DF, x, y, z, ::Type{T}) where {F, DF, T}
    vx, px = unpack_dual(x)
    vy, py = unpack_dual(y)
    vz, pz = unpack_dual(z)
    Ω, (dΩ_dx, dΩ_dy, dΩ_dz) = f(vx, vy, vz), df(vx, vy, vz)
    dΩ = dΩ_dx * px + dΩ_dy * py + dΩ_dz * pz
    return Dual{T}(Ω, dΩ)
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

macro define_ternary_dual_scalar_rule(ex...)
    args, kwargs = split_args_kwargs(ex...)
    return define_ternary_dual_scalar_rule(args...; kwargs...)
end

function define_ternary_dual_scalar_rule(f, df; fused = false)
    local M = @__MODULE__
    local FD = ForwardDiff
    local fdf = fused ? (df,) : (f, df)
    quote
        # See: https://github.com/JuliaDiff/ForwardDiff.jl/blob/2ff680824249ad71f55615467bd570c6c29fa673/src/dual.jl#L152
        $(FD).@define_ternary_dual_op(
            $(f),
            $(M).ternary_dual_pushforward($(fdf...), x, y, z, Txyz),
            $(M).ternary_dual_pushforward($(fdf...), x, y, z, Txy),
            $(M).ternary_dual_pushforward($(fdf...), x, y, z, Txz),
            $(M).ternary_dual_pushforward($(fdf...), x, y, z, Tyz),
            $(M).ternary_dual_pushforward($(fdf...), x, y, z, Tx),
            $(M).ternary_dual_pushforward($(fdf...), x, y, z, Ty),
            $(M).ternary_dual_pushforward($(fdf...), x, y, z, Tz),
        )
    end
end

#### Macros

function split_args_kwargs(ex...)
    kws = Pair{Symbol, Any}[]
    i = 0
    while i < length(ex)
        i += 1
        x = ex[i]
        if x isa Expr && x.head === :(=) # kwarg of the form foo=bar
            push!(kws, x.args[1] => x.args[2])
        else
            break
        end
    end
    return esc.(ex[i:end]), kws
end

macro uniform_dual_rule_from_frule(f)
    return uniform_dual_rule_from_frule_inner(f)
end

function uniform_dual_rule_from_frule_inner(f)
    local CRC = ChainRulesCore
    local FD = ForwardDiff
    local _f = esc(f)
    quote
        @inline function $(_f)(x::$(FD).Dual{T}...) where {T}
            vx, px = $(FD).value.(x), $(FD).partials.(x)
            y, dy = $(CRC).frule(($(CRC).NoTangent(), px...), $(_f), vx...)
            return $(FD).Dual{T}(y, dy)
        end
    end
end

macro dual_rule_from_frule(ex)
    return dual_rule_from_frule_inner(ex)
end

function dual_rule_from_frule_inner(ex)
    local CRC = ChainRulesCore
    local FD = ForwardDiff

    local f, args
    if !@capture(ex, f_(args__))
        error("Expected `call` expression; e.g. `f(x, y, !(z::T))`")
    end

    local T = esc(gensym(:T))
    local inputs, primals, tangents = Any[], Any[], Any[]
    for arg in args
        if arg isa Symbol
            x = esc(arg)
            push!(inputs, :($(x)::$(FD).Dual{$(T)}))
            push!(primals, :($(FD).value($(x))))
            push!(tangents, :($(FD).partials($(x))))
        elseif @capture(arg, !(notarg_::Tnotarg_))
            x = esc(notarg)
            Tx = esc(Tnotarg)
            push!(inputs, :($(x)::$(Tx)))
            push!(primals, :($(x)::$(Tx)))
            push!(tangents, :($(CRC).NoTangent()))
        else
            error("Call arguments must be symbols optionally prepended with `!`; e.g. `f(x, y, !z)`")
        end
    end

    local fname = esc(f)
    local fdef = Dict{Symbol, Any}(
        :name        => fname,
        :args        => inputs,
        :kwargs      => Any[],
        :whereparams => (T,),
        :body        => quote
            vx = ($(primals...),)
            px = ($(tangents...),)
            y, dy = $(CRC).frule(($(CRC).NoTangent(), px...), $(fname), vx...)
            return $(FD).Dual{$T}(y, dy)
        end,
    )

    return :(@inline $(combinedef(fdef)))
end
