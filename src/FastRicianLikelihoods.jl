module FastRicianLikelihoods

using Base: BroadcastFunction
using Bessels: Bessels
using ChainRulesCore: ChainRulesCore, @scalar_rule
using Distributions: Distributions
using FastGaussQuadrature: gausslegendre
using ForwardDiff: ForwardDiff, Dual
using IrrationalConstants: sqrthalfπ
using LinearAlgebra: dot
using Random: Random
using StaticArrays: StaticArrays, SVector, SMatrix

export Rician

include("forwarddiff.jl")
include("bessels.jl")
include("rician.jl")
include("distributions.jl")

end