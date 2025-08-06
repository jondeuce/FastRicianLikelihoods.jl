module FastRicianLikelihoods

using Base: BroadcastFunction
using Bessels: Bessels
using ChainRulesCore: ChainRulesCore, @scalar_rule, NoTangent
using Distributions: Distributions, normlogccdf, normlogcdf
using FastGaussQuadrature: FastGaussQuadrature, gausslegendre, gausslaguerre
using ForwardDiff: ForwardDiff, Dual
using GenericLinearAlgebra: GenericLinearAlgebra # need to load this package to extend `LinearAlgebra.eigen` to matrices of `BigFloat`s
using IrrationalConstants: IrrationalConstants, invsqrt2, sqrt2, invsqrt2π, sqrt2π, sqrthalfπ, logtwo, log2π
using LinearAlgebra: LinearAlgebra, dot
using MacroTools: MacroTools, @capture, combinedef, prettify, splitdef
using Random: Random
using SpecialFunctions: SpecialFunctions, erf, erfc, erfcinv, erfcx
using StaticArrays: StaticArrays, StaticArray, SVector, SMatrix, SHermitianCompact, SOneTo

export Rice, neglogpdf_rician, neglogpdf_qrician

include("gausshalfhermite.jl")
using .GaussHalfHermite: GaussHalfHermite, gausshalfhermite_gw

include("utils.jl")
include("forwarddiff.jl")
include("bessels.jl")
include("rician.jl")
include("distributions.jl")

end
