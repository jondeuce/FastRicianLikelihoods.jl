module FastRicianLikelihoods

using Base: BroadcastFunction
using Bessels: Bessels
using ChainRulesCore: ChainRulesCore, @scalar_rule, NoTangent
using Distributions: Distributions, normlogccdf, normlogcdf
using FastGaussQuadrature: gausslegendre, gausslaguerre
using ForwardDiff: ForwardDiff, Dual
using IrrationalConstants: invsqrt2, sqrt2, invsqrt2π, sqrt2π, sqrthalfπ, logtwo, log2π
using LinearAlgebra: dot
using MacroTools: @capture, combinedef, prettify, splitdef
using Random: Random
using SpecialFunctions: erf, erfc, erfcinv, erfcx
using StaticArrays: StaticArrays, SVector, SMatrix

export Rice, neglogpdf_rician, neglogpdf_qrician

include("gausshalfhermite.jl")
using .GaussHalfHermite

include("utils.jl")
include("forwarddiff.jl")
include("bessels.jl")
include("rician.jl")
include("distributions.jl")

end
