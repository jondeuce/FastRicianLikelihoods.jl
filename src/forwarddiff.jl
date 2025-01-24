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

@inline untuple_scalar(x::Tuple) = only(x)
@inline untuple_scalar(x::Number) = x

#### Pushforwards

@inline function unary_dual_pushforward(fdf::F, x::Dual{Tag}) where {F, Tag}
    vx, px = unpack_dual(x)
    Ω, dΩ_dx = fdf(vx)
    dΩ = untuple_scalar(dΩ_dx) * px
    return Dual{Tag}(Ω, dΩ)
end

@inline function unary_dual_pushforward(f::F, df::DF, x::Dual{Tag}) where {F, DF, Tag}
    vx, px = unpack_dual(x)
    Ω, dΩ_dx = f(vx), df(vx)
    dΩ = untuple_scalar(dΩ_dx) * px
    return Dual{Tag}(Ω, dΩ)
end

@inline function binary_dual_pushforward(fdf::F, x, y, ::Type{Tag}) where {F, Tag}
    vx, px = unpack_dual(x)
    vy, py = unpack_dual(y)
    Ω, (dΩ_dx, dΩ_dy) = fdf(vx, vy)
    dΩ = dΩ_dx * px + dΩ_dy * py
    return Dual{Tag}(Ω, dΩ)
end

@inline function binary_dual_pushforward(f::F, df::DF, x, y, ::Type{Tag}) where {F, DF, Tag}
    vx, px = unpack_dual(x)
    vy, py = unpack_dual(y)
    Ω, (dΩ_dx, dΩ_dy) = f(vx, vy), df(vx, vy)
    dΩ = dΩ_dx * px + dΩ_dy * py
    return Dual{Tag}(Ω, dΩ)
end

@inline function ternary_dual_pushforward(fdf::F, x, y, z, ::Type{Tag}) where {F, Tag}
    vx, px = unpack_dual(x)
    vy, py = unpack_dual(y)
    vz, pz = unpack_dual(z)
    Ω, (dΩ_dx, dΩ_dy, dΩ_dz) = fdf(vx, vy, vz)
    dΩ = dΩ_dx * px + dΩ_dy * py + dΩ_dz * pz
    return Dual{Tag}(Ω, dΩ)
end

@inline function ternary_dual_pushforward(f::F, df::DF, x, y, z, ::Type{Tag}) where {F, DF, Tag}
    vx, px = unpack_dual(x)
    vy, py = unpack_dual(y)
    vz, pz = unpack_dual(z)
    Ω, (dΩ_dx, dΩ_dy, dΩ_dz) = f(vx, vy, vz), df(vx, vy, vz)
    dΩ = dΩ_dx * px + dΩ_dy * py + dΩ_dz * pz
    return Dual{Tag}(Ω, dΩ)
end

#### Macros which define overloads for `Dual` arguments using macros from `ForwardDiff`

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
    return ex[i:end], kws
end

macro define_unary_dual_scalar_rule(ex...)
    args, kwargs = split_args_kwargs(ex...)
    return define_unary_dual_scalar_rule(args...; kwargs...)
end

function define_unary_dual_scalar_rule(f, df; fused = false)
    local M = @__MODULE__
    local FD = ForwardDiff
    local fdf
    if @capture(f, fobj_::Tfobj_)
        fdf = fused ? (df,) : (fobj, df)
    else
        fdf = fused ? (df,) : (f, df)
    end
    quote
        $(esc(f))(x::$(FD).Dual) = $(M).unary_dual_pushforward($(esc.(fdf)...), x)
    end
end

macro define_binary_dual_scalar_rule(ex...)
    args, kwargs = split_args_kwargs(ex...)
    return define_binary_dual_scalar_rule(args...; kwargs...)
end

function define_binary_dual_scalar_rule(f, df; fused = false)
    local M = @__MODULE__
    local FD = ForwardDiff
    local fdf
    if @capture(f, fobj_::Tfobj_)
        fdf = fused ? (df,) : (fobj, df)
    else
        fdf = fused ? (df,) : (f, df)
    end
    quote
        # See: https://github.com/JuliaDiff/ForwardDiff.jl/blob/2ff680824249ad71f55615467bd570c6c29fa673/src/dual.jl#L136
        $(FD).@define_binary_dual_op(
            $(esc(f)),
            $(M).binary_dual_pushforward($(esc.(fdf)...), x, y, Txy),
            $(M).binary_dual_pushforward($(esc.(fdf)...), x, y, Tx),
            $(M).binary_dual_pushforward($(esc.(fdf)...), x, y, Ty),
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
    local fdf
    if @capture(f, fobj_::Tfobj_)
        fdf = fused ? (df,) : (fobj, df)
    else
        fdf = fused ? (df,) : (f, df)
    end
    quote
        # See: https://github.com/JuliaDiff/ForwardDiff.jl/blob/2ff680824249ad71f55615467bd570c6c29fa673/src/dual.jl#L152
        $(FD).@define_ternary_dual_op(
            $(esc(f)),
            $(M).ternary_dual_pushforward($(esc.(fdf)...), x, y, z, Txyz),
            $(M).ternary_dual_pushforward($(esc.(fdf)...), x, y, z, Txy),
            $(M).ternary_dual_pushforward($(esc.(fdf)...), x, y, z, Txz),
            $(M).ternary_dual_pushforward($(esc.(fdf)...), x, y, z, Tyz),
            $(M).ternary_dual_pushforward($(esc.(fdf)...), x, y, z, Tx),
            $(M).ternary_dual_pushforward($(esc.(fdf)...), x, y, z, Ty),
            $(M).ternary_dual_pushforward($(esc.(fdf)...), x, y, z, Tz),
        )
    end
end

#### Experimental macros which define overloads for `Dual` arguments in terms of `ChainRulesCore.frule`; requires all inputs be promoted to the same `Dual` type

macro uniform_dual_rule_from_frule(f)
    return uniform_dual_rule_from_frule(f)
end

function uniform_dual_rule_from_frule(f)
    local CRC = ChainRulesCore
    local FD = ForwardDiff
    local _f = esc(f)
    quote
        @inline function $(_f)(x::$(FD).Dual{Tag}...) where {Tag}
            vx, px = $(FD).value.(x), $(FD).partials.(x)
            y, dy = $(CRC).frule(($(CRC).NoTangent(), px...), $(_f), vx...)
            return $(FD).Dual{Tag}(y, dy)
        end
    end
end

macro dual_rule_from_frule(ex)
    return dual_rule_from_frule(ex)
end

function dual_rule_from_frule(ex)
    local CRC = ChainRulesCore
    local FD = ForwardDiff

    local f, args
    if !@capture(ex, f_(args__))
        error("Expected `call` expression; e.g. `f(x, y, !(z::T))`")
    end

    local Tag = esc(gensym(:Tag))
    local inputs, primals, tangents = Any[], Any[], Any[]
    for arg in args
        if arg isa Symbol
            x = esc(arg)
            push!(inputs, :($(x)::$(FD).Dual{$(Tag)}))
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
        :whereparams => (Tag,),
        :body        => quote
            vx = ($(primals...),)
            px = ($(tangents...),)
            y, dy = $(CRC).frule(($(CRC).NoTangent(), px...), $(fname), vx...)
            return $(FD).Dual{$(Tag)}(y, dy)
        end,
    )

    return :(@inline $(combinedef(fdef)))
end
