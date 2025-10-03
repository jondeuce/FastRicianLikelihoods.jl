module FastRicianLikelihoods

using Base: BroadcastFunction, Fix1, Fix2
using Bessels: Bessels
using ChainRulesCore: ChainRulesCore, @scalar_rule, NoTangent
using Distributions: Distributions, normlogccdf, normlogcdf
using FastGaussQuadrature: FastGaussQuadrature, gausslegendre, gausslaguerre
using ForwardDiff: ForwardDiff, Dual
using GenericLinearAlgebra: GenericLinearAlgebra # need to load this package to extend `LinearAlgebra.eigen` to matrices of `BigFloat`s
using IrrationalConstants: IrrationalConstants, invsqrt2, sqrt2, invsqrt2π, sqrt2π, sqrthalfπ, logtwo, log2π
using LinearAlgebra: LinearAlgebra, dot
using MacroTools: MacroTools, @capture, combinedef, prettify, splitdef
using MuladdMacro: MuladdMacro, @muladd, to_muladd
using Random: Random
using SpecialFunctions: SpecialFunctions, erf, erfc, erfcinv, erfcx
using StaticArrays: StaticArrays, StaticArray, SVector, SMatrix, SHermitianCompact, SOneTo

export Rice, neglogpdf_rician, neglogpdf_qrician

include(to_muladd, "gausshalfhermite.jl")
using .GaussHalfHermite: GaussHalfHermite, gausshalfhermite_gw

include(to_muladd, "gausslegendre.jl")
using .GaussLegendre: GaussLegendre

include(to_muladd, "utils.jl")
include(to_muladd, "forwarddiff.jl")
include(to_muladd, "bessels.jl")
include(to_muladd, "rician.jl")
include(to_muladd, "distributions.jl")

end
