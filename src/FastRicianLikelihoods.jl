module FastRicianLikelihoods

using Base: BroadcastFunction
using Bessels: Bessels
using ChainRulesCore: @scalar_rule
using Distributions: Distributions
using FastGaussQuadrature: gausslegendre
using ForwardDiff: ForwardDiff
using IrrationalConstants: invsqrt2, invsqrt2π, log2π, logtwo, logπ, sqrt2, sqrt2π, sqrthalfπ, sqrtπ
using LinearAlgebra: dot
using Random: Random
using StaticArrays: StaticArrays, SVector, SMatrix

export Rician

include("forwarddiff.jl")
include("bessels.jl")
include("rician.jl")
include("distributions.jl")

end
