module FastRicianLikelihoods

using Base: BroadcastFunction
using Bessels: Bessels
using ChainRulesCore: ChainRulesCore, @scalar_rule, NoTangent
using Distributions: Distributions, normlogccdf, normlogcdf
using FastGaussQuadrature: gausslegendre
using ForwardDiff: ForwardDiff, Dual
using IrrationalConstants: invsqrt2, sqrt2, sqrt2π, sqrthalfπ, logtwo, log2π
using LinearAlgebra: dot
using MacroTools: @capture, combinedef, prettify, splitdef
using Random: Random
using SpecialFunctions: erf, erfc, erfcx
using StaticArrays: StaticArrays, SVector, SMatrix

export Rice, neglogpdf_rician, neglogpdf_qrician

include("forwarddiff.jl")
include("bessels.jl")
include("rician.jl")
include("distributions.jl")

end
