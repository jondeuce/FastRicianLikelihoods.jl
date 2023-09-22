module Utils

using Test

using ArbNumerics
using ForwardDiff: ForwardDiff
using StaticArrays: SVector
using Zygote: Zygote

setworkingprecision(ArbFloat, 500)
setextrabits(128)

@inline common_float_type(args::Tuple) = mapfoldl(typeof, common_float_type, args)

@inline common_float_type(::Type{T1}, ::Type{T2}) where {T1, T2} =
    (T1 <: Real && T2 <: Real) ? (@assert T1 === T2; @assert T1 <: Union{Float32, Float64}; T1) :
    (T1 <: Real) ? (@assert T1 <: Union{Float32, Float64}; T1) :
    (T2 <: Real) ? (@assert T2 <: Union{Float32, Float64}; T2) :
    (T1)

arbify(x) = x
arbify(x::AbstractFloat) = error("Expected typeof(x) = $(typeof(x)) <: Union{Float32, Float64}")
arbify(x::Union{Float32, Float64}) = ArbFloat(x)::ArbFloat
arbify(f::Function) = function f_arbified(args...)
    T = common_float_type(args)
    xs = arbify.(args)
    y = f(xs...)
    return convert.(T, y)
end

∇Zyg(f, args::Real...) = @inferred Zygote.gradient(f, args...)
∇Fwd(f, args::Real...) = @inferred Tuple(ForwardDiff.gradient(Base.splat(f), SVector(args)))

end # module Utils

import .Utils
