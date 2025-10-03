using Pkg
Pkg.activate(".")

using Revise
Revise.includet("../../test/utils.jl")

using FastRicianLikelihoods
using .Utils: Utils, arbify

using FastRicianLikelihoods.ForwardDiff
using FastRicianLikelihoods.Random
using FastRicianLikelihoods.SpecialFunctions: SpecialFunctions
using FastRicianLikelihoods.StaticArrays

using ArbNumerics
using Bessels: Bessels
using CUDA
using CairoMakie
using ChainRulesCore: ChainRulesCore
using Chairmarks
using DataStructures
using LaTeXStrings
using PrettyTables
using Printf
using Statistics
using StatsBase

const F = FastRicianLikelihoods
const U = Utils

const OUTDIR = joinpath(@__DIR__, "output")
isdir(OUTDIR) || mkpath(OUTDIR)

ArbNumerics.setworkingprecision(ArbReal; bits = 500)

abserr(x::Number, y::Number) = oftype(x, abs(x - y))
relerr(x::Number, y::Number) = oftype(x, abs(x - y) / abs(y))
minerr(x::Number, y::Number) = oftype(x, Utils.minerr(x, y))
regrelerr(x::Number, y::Number) = oftype(x, abs(x - y) / (1 + abs(y)))

abserr(x::AbstractArray, y::AbstractArray) = maximum(abserr.(x, y))
relerr(x::AbstractArray, y::AbstractArray) = maximum(relerr.(x, y))
minerr(x::AbstractArray, y::AbstractArray) = maximum(minerr.(x, y))
regrelerr(x::AbstractArray, y::AbstractArray) = maximum(regrelerr.(x, y))

# abserr(x::AbstractArray, y::AbstractArray) = convert(eltype(x), norm(x - y))
# relerr(x::AbstractArray, y::AbstractArray) = convert(eltype(x), norm(x - y) / norm(y))
# minerr(x::AbstractArray, y::AbstractArray) = convert(eltype(x), max(relerr(x, y), abserr(x, y)))
# regrelerr(x::AbstractArray, y::AbstractArray) = convert(eltype(x), norm(x - y) / (1 + norm(y)))

function default_theme(; kwargs...)
    theme = merge(
        theme_latexfonts(),
        Theme(;
            font = "CMU Serif",
            fontsize = 16,
            size = (800, 600),
            linestyles = [nothing, :dash, :dash],
            ishollowmarkers = [false, true, false],
            markers = [:circle, :diamond, :rtriangle],
            linecycle = Cycle([:color, :linestyle]; covary = true),
            scattercycle = Cycle([:color => :markercolor, :strokecolor => :color, :marker]; covary = true),
            markerstrokewidth = 1.5,
            colormap = :viridis,
            # colormap = :Hiroshige,
            kwargs...,
        ),
    )
    return theme
end

CairoMakie.activate!(; type = "png")
CairoMakie.set_theme!(default_theme())

function default_pretty_table(io, args...; kwargs...)
    return pretty_table(
        io,
        args...;
        merge_column_label_cells = :auto,
        alignment = :c,
        style = LatexTableStyle(; column_label = ["textbf"]),
        backend = :latex,
        table_format = latex_table_format__booktabs,
        kwargs...,
    )
end

function compile_and_save_table(table_builder, filename)
    filedirname = dirname(filename)
    filebasename = basename(filename)
    @assert isdir(filedirname)
    @assert endswith(filebasename, ".tex")

    # Write table to file
    open(io -> table_builder(io), filename, "w")
    @assert isfile(filename)

    mktempdir() do tmpdir
        current_dir = pwd()
        try
            cd(tmpdir)
            table = table_builder(String)

            write(
                "table.tex",
                """
                \\documentclass[a4paper, 12pt]{article}
                \\pagestyle{empty}
                \\usepackage{amsmath}
                \\usepackage{color}
                \\usepackage{booktabs}
                \\usepackage{xcolor}
                \\usepackage[margin=0.25in,landscape]{geometry}
                \\usepackage{siunitx}
                \\sisetup{round-mode = figures, round-precision = 3}
                \\begin{document}
                $table
                \\end{document}
                """,
            )

            run(`pdflatex table.tex`)
            run(`pdflatex table.tex`)
            run(`convert -density 600 table.pdf -flatten -trim table.png`)
            run(`mv table.pdf $(joinpath(filedirname, replace(filebasename, ".tex" => ".pdf")))`)
            run(`mv table.png $(joinpath(filedirname, replace(filebasename, ".tex" => ".png")))`)
        finally
            cd(current_dir)
        end
    end
end

# Baseline using ArbNumerics (high precision)
function baseline_rician_parts(z::T) where {T <: Union{Float32, Float64}}
    za = arbify(z)
    r = F.besseli1i0(za)
    r′ = 1 - r / za - r^2
    r′′ = -r′ / za + r / (za^2) - 2 * r * r′
    two_r′_plus_z_r′′ = 2r′ + za * r′′
    one_minus_r_minus_z_r′ = 1 - r - za * r′
    return (; r, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′)
end

# Baseline using ArbNumerics for NLL and derivatives
function baseline_rician_nll(x::T, ν::T) where {T <: Union{Float32, Float64}}
    xa, νa = arbify.((x, ν))
    f = F.neglogpdf_rician(xa, νa)
    (fx, fν), (fxx, fxν, fνν), (fxxx, fxxν, fxνν, fννν) = F.∇³neglogpdf_rician_with_gradient_and_hessian(xa, νa)
    return (; f, fx, fν, fxx, fxν, fνν, fxxx, fxxν, fxνν, fννν)
end

# Baseline using ArbNumerics for QRice
function baseline_qrician_nll(x::T, ν::T, δ::T, order::Val = Val(32); method = :analytic) where {T <: Union{Float32, Float64}}
    xa, νa, δa = arbify.((x, ν, δ))
    Ω, (Ωx, Ων, Ωδ), (Ωxx, Ωxν, Ωxδ, Ωνν, Ωνδ, Ωδδ), ∇³Ω = F.∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian(xa, νa, δa, order; method)
    ∇Ω = SVector{3}(Ωx, Ων, Ωδ)
    ∇²Ω = SMatrix{3, 3}(Ωxx, Ωxν, Ωxδ, Ωxν, Ωνν, Ωνδ, Ωxδ, Ωνδ, Ωδδ)
    return (; Ω, ∇Ω, ∇²Ω, ∇³Ω)
end

# Stable piecewise implementation from package internals
function stable_rician_parts(z::T) where {T <: Union{Float32, Float64}}
    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = F._neglogpdf_rician_parts(T(z), Val(2))
    return (; r, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′)
end

# Small-z formulas via a1 Taylor, as in the paper
function stable_rician_parts_smallz(z::T) where {T <: Union{Float32, Float64}}
    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = F._neglogpdf_rician_parts_taylor(T(z), Val(2))
    return (; r, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′)
end

# Large-z asymptotic (simple 3-term for r), recurrences for derivatives
function stable_rician_parts_largez(z::T) where {T <: Union{Float32, Float64}}
    r, r_tail, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′ = F._neglogpdf_rician_parts_asymptotic(T(z), Val(2))
    return (; r, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′)
end

# Stable NLL and derivatives from package
function stable_rician_nll(x::T, ν::T) where {T <: Union{Float32, Float64}}
    f = F.neglogpdf_rician(x, ν)
    (fx, fν), (fxx, fxν, fνν), (fxxx, fxxν, fxνν, fννν) = F.∇³neglogpdf_rician_with_gradient_and_hessian(x, ν)
    return (; f, fx, fν, fxx, fxν, fνν, fxxx, fxxν, fxνν, fννν)
end

# Stable NLL and derivatives from package
function stable_qrician_nll(x::T, ν::T, δ::T, order::Val) where {T <: Union{Float32, Float64}}
    Ω, (Ωx, Ων, Ωδ), (Ωxx, Ωxν, Ωxδ, Ωνν, Ωνδ, Ωδδ), ∇³Ω = F.∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian(x, ν, δ, order)
    ∇Ω = SVector{3}(Ωx, Ων, Ωδ)
    ∇²Ω = SMatrix{3, 3}(Ωxx, Ωxν, Ωxδ, Ωxν, Ωνν, Ωνδ, Ωxδ, Ωνδ, Ωδδ)
    return (; Ω, ∇Ω, ∇²Ω, ∇³Ω)
end

# Naive using ratios + recurrences in the given precision
function naive_rician_parts(z::T) where {T <: Union{Float32, Float64}}
    r = F.besseli1x(z) / F.besseli0x(z) # use scaled to avoid overflow, emulate naive ratio
    r′ = one(z) - r / z - r^2
    r′′ = -r′ / z + r / z^2 - 2 * r * r′
    two_r′_plus_z_r′′ = 2r′ + z * r′′
    one_minus_r_minus_z_r′ = 1 - r - z * r′
    return (; r, r′, r′′, one_minus_r_minus_z_r′, two_r′_plus_z_r′′)
end

# Naive NLL and derivatives (for comparison)
function naive_rician_nll(x::T, ν::T) where {T <: Union{Float32, Float64}}
    z = x * ν
    (; r, r′, r′′) = naive_rician_parts(z)
    z = x * ν

    # Assemble NLL and derivatives from naive parts
    # f = (x^2 + ν^2) / 2 - log(x * F.besseli0(z)) # diverges for large z
    f = (x - ν)^2 / 2 - log(x * F.besseli0x(z))
    fx = x - ν * r - 1 / x
    fν = ν - x * r

    # Second derivatives using naive r′
    fxx = 1 + 1 / x^2 - ν^2 * r′
    fxν = -r - z * r′
    fνν = 1 - x^2 * r′

    # Third derivatives using naive r′′
    fxxx = -2 / x^3 - ν^3 * r′′
    fxxν = -2 * ν * r′ - z * ν * r′′
    fxνν = -2 * x * r′ - z * x * r′′
    fννν = -x^3 * r′′

    return (; f, fx, fν, fxx, fxν, fνν, fxxx, fxxν, fxνν, fννν)
end

# Wrappers to avoid type piracy
@inline bessels_jl_besseli(ν::Int, x::Real) = Bessels.besseli(ν, x)
@inline bessels_jl_besseli0(x::Real) = Bessels.besseli0(x)
@inline bessels_jl_besseli1(x::Real) = Bessels.besseli1(x)
@inline bessels_jl_besselix(ν::Int, x::Real) = Bessels.besselix(ν, x)
@inline bessels_jl_besseli0x(x::Real) = bessels_jl_besselix(0, x)

@inline specialfunctions_jl_besseli0(x::Real) = SpecialFunctions.besseli(0, x)
@inline specialfunctions_jl_besseli0x(x::Real) = SpecialFunctions.besselix(0, x)

# Forward-mode autodiff rules for Bessels.jl methods (taken from SpecialFunctions.jl; see: https://github.com/JuliaMath/SpecialFunctions.jl/blob/1f0527c665994a645a28710f77d183f7edea92c4/ext/SpecialFunctionsChainRulesCoreExt.jl)
ChainRulesCore.@scalar_rule(
    bessels_jl_besseli(ν::Int, x),
    (
        ChainRulesCore.@not_implemented("Derivatives of Bessel functions with respect to the order are not implemented."),
        (bessels_jl_besseli(abs(ν - 1), x) + bessels_jl_besseli(ν + 1, x)) / 2,
    ),
)
F.@dual_rule_from_frule bessels_jl_besseli(!(ν::Int), x)

ChainRulesCore.@scalar_rule bessels_jl_besseli0(x) bessels_jl_besseli1(x)
F.@dual_rule_from_frule bessels_jl_besseli0(x)

ChainRulesCore.@scalar_rule bessels_jl_besseli1(x) (bessels_jl_besseli0(x) + bessels_jl_besseli(2, x)) / 2
F.@dual_rule_from_frule bessels_jl_besseli1(x)

function ChainRulesCore.frule((_, _, Δx), ::typeof(bessels_jl_besselix), ν::Int, x::Number)
    # primal
    Ω = bessels_jl_besselix(ν, x)

    # derivative
    a = (bessels_jl_besselix(abs(ν - 1), x) + bessels_jl_besselix(ν + 1, x)) / 2
    ΔΩ = if Δx isa Real
        muladd(-sign(real(x)), Ω, a) * Δx
    else
        muladd(a, Δx, -sign(real(x)) * real(Δx) * Ω)
    end

    return Ω, ΔΩ
end
F.@dual_rule_from_frule bessels_jl_besselix(!(ν::Int), x)

for (name, _besseli0, _besseli0x) in ((:bessels, bessels_jl_besseli0, bessels_jl_besseli0x), (:specialfunctions, specialfunctions_jl_besseli0, specialfunctions_jl_besseli0x))
    @eval begin
        @inline function $(Symbol(name, :_neglogpdf_rician))(x::D, ν::D) where {D}
            # Negative Rician log-likelihood `-logp(x | ν, σ = 1)`
            z = x * ν
            if z < 1
                return (x^2 + ν^2) / 2 - log(x * $(_besseli0)(z))
            else
                return (x - ν)^2 / 2 - log(x * $(_besseli0x)(z))
            end
        end
        @inline $(Symbol(name, :_neglogpdf_rician))(xν::SVector{2, D}) where {D} = $(Symbol(name, :_neglogpdf_rician))(xν...)

        @inline function $(Symbol(name, :_neglogpdf_rician_with_gradient))(x::D, ν::D) where {D}
            f, (fx, fν) = F.withgradient($(Symbol(name, :_neglogpdf_rician)), SVector{2, D}(x, ν)) # ::Tuple{D, SVector{2, D}}
            return (f, fx, fν)
        end
        @inline $(Symbol(name, :_neglogpdf_rician_with_gradient))(xν::SVector{2, D}) where {D} = SVector{3, D}($(Symbol(name, :_neglogpdf_rician_with_gradient))(xν...))

        @inline function $(Symbol(name, :_neglogpdf_rician_with_gradient_and_hessian))(x::D, ν::D) where {D}
            (f, fx, fν), (fx′, fxx, fνx, fν′, fxν, fνν) = F.withjacobian($(Symbol(name, :_neglogpdf_rician_with_gradient)), SVector{2, D}(x, ν)) # ::Tuple{SVector{3, D}, SMatrix{3, 2, D, 6}} # prime indicates exact duplicate quantity
            return (f, fx, fν, fxx, (fxν + fνx) / 2, fνν)
        end
        @inline $(Symbol(name, :_neglogpdf_rician_with_gradient_and_hessian))(xν::SVector{2, D}) where {D} = SVector{6, D}($(Symbol(name, :_neglogpdf_rician_with_gradient_and_hessian))(xν...))

        @inline function $(Symbol(name, :_neglogpdf_rician_with_gradient_hessian_and_jerk))(x::D, ν::D) where {D}
            (f, fx, fν, fxx, fxν, fνν), (fx′, fxx′, fνx, fxxx, fxνx, fννx, fν′, fxν′, fνν′, fxxν, fxνν, fννν) = F.withjacobian($(Symbol(name, :_neglogpdf_rician_with_gradient_and_hessian)), SVector{2, D}(x, ν)) # ::Tuple{SVector{6, D}, SMatrix{6, 2, D, 12}} # prime indicates exact duplicate quantity
            return (f, fx, fν, fxx, fxν, fνν, fxxx, (fxxν + fxνx) / 2, (fxνν + fννx) / 2, fννν)
        end
        @inline $(Symbol(name, :_neglogpdf_rician_with_gradient_hessian_and_jerk))(xν::SVector{2, D}) where {D} = SVector{10, D}($(Symbol(name, :_neglogpdf_rician_with_gradient_hessian_and_jerk))(xν...))

        function $(Symbol(name, :_rician_nll))(x::T, ν::T) where {T <: Union{Float32, Float64}}
            f, fx, fν, fxx, fxν, fνν, fxxx, fxxν, fxνν, fννν = $(Symbol(name, :_neglogpdf_rician_with_gradient_hessian_and_jerk))(x, ν)
            return (; f, fx, fν, fxx, fxν, fνν, fxxx, fxxν, fxνν, fννν)
        end
    end
end

# Get branch points for annotation
function branch_points(::Type{T}) where {T <: Union{Float32, Float64}}
    return extrema(F.neglogpdf_rician_parts_branches(T))
end

# High-SNR iterator for NLL accuracy evaluation
function high_snr_iterator(ν::T; num_rician_samples) where {T <: Union{Float32, Float64}}
    return Iterators.map(1:num_rician_samples) do _
        # Sample x ~ Rice(ν, 1): x = √((ν + z₁)² + z₂²) where z₁, z₂ ~ N(0,1)
        x = √((ν + randn(T))^2 + randn(T)^2)
        return (x, ν)
    end
end

# Helper function to determine appropriate time units
function converted_time_and_units(time_sec::Real)
    return time_sec >= 1.0 ? (time_sec, "s") :
           time_sec >= 1e-3 ? (time_sec * 1e3, "ms") :
           time_sec >= 1e-6 ? (time_sec * 1e6, "us") :
           (time_sec * 1e9, "ns")
end

# Helper function to format time with appropriate units and optional speedup
function format_time_value(time_sec::Real, baseline_time_sec::Union{Real, Nothing} = nothing; precision::Int = 3, speedup_precision::Int = 2)
    time_in_units, units = converted_time_and_units(time_sec)

    if baseline_time_sec === nothing
        return L"\SI[round-mode = figures, round-precision = %$precision]{%$time_in_units}{\%$units}"
    else
        speedup = time_sec / baseline_time_sec
        return L"\SI[round-mode = figures, round-precision = %$precision]{%$time_in_units}{\%$units} (\num[round-mode = figures, round-precision = %$speedup_precision]{%$speedup}\times)"
    end
end

# Evaluate Bessel ratio accuracy for both precisions
function evaluate_bessel_ratios(::Type{T}; zmin = 1e-6, zmax = 1e6, num = 1001) where {T <: Union{Float32, Float64}}
    zs = exp.(range(log(T(zmin)), log(T(zmax)); length = num))

    rows = NamedTuple[]
    function pushrow!(rows, method, z, q, val, ref)
        push!(rows, (; quantity = q, method, precision = string(T), z = z, value = val, reference = ref, relerr = relerr(val, ref), abserr = abserr(val, ref), minerr = minerr(val, ref), regrelerr = regrelerr(val, ref)))
    end

    for z in zs
        @assert z isa T
        ref = baseline_rician_parts(z)
        for (label, partsfun) in (("naive", naive_rician_parts), ("stable", stable_rician_parts))
            vals = partsfun(T(z))
            pushrow!(rows, label, T(z), "r", vals.r, ref.r)
            pushrow!(rows, label, T(z), "r′", vals.r′, ref.r′)
            pushrow!(rows, label, T(z), "r′′", vals.r′′, ref.r′′)
            pushrow!(rows, label, T(z), "two_r′_plus_z_r′′", vals.two_r′_plus_z_r′′, ref.two_r′_plus_z_r′′)
            pushrow!(rows, label, T(z), "one_minus_r_minus_z_r′", vals.one_minus_r_minus_z_r′, ref.one_minus_r_minus_z_r′)
        end
    end

    return rows
end

# Evaluate NLL accuracy for table
function evaluate_nll_accuracy(
    ::Type{T};
    xmin = 1e-3,
    xmax = 1e3,
    νmin = 1e-3,
    νmax = 1e3,
    num_x = 51,
    num_ν = 51,
    sampling_mode = :grid,
    num_rician_samples = 10,
    methods = ["stable", "naive", "bessels", "specialfunctions"],
) where {T <: Union{Float32, Float64}}

    rows = NamedTuple[]
    function pushrow!(rows, method, x, ν, z, q, val, ref)
        push!(rows, (; quantity = q, method, precision = string(T), x = x, ν = ν, z = z, value = val, reference = ref, relerr = relerr(val, ref), abserr = abserr(val, ref), minerr = minerr(val, ref), regrelerr = regrelerr(val, ref)))
    end

    # Create iterator over (x, ν) pairs based on sampling mode
    if sampling_mode == :grid
        # Current cartesian grid approach
        xs = exp.(range(log(T(xmin)), log(T(xmax)); length = num_x))
        νs = exp.(range(log(T(νmin)), log(T(νmax)); length = num_ν))
        xν_pairs = Iterators.product(xs, νs)
    elseif sampling_mode == :high_snr
        # High-SNR diagonal sampling: x ~ Rice(ν, 1)
        νs = exp.(range(log(T(νmin)), log(T(νmax)); length = num_ν))
        xν_pairs = Iterators.flatten((high_snr_iterator(ν; num_rician_samples) for ν in νs))
    else
        error("sampling_mode must be either :grid or :high_snr")
    end

    for (x, ν) in xν_pairs
        @assert x isa T && ν isa T
        z = x * ν
        ref = baseline_rician_nll(x, ν) # ArbNumerics high-precision

        # Evaluate each specified method
        for method in methods
            if T === Float32 && method == "specialfunctions"
                # SpecialFunctions.jl only has hardcoded Float64 implementations; for Float32 it "cheats" by promoting internally, so we exclude it
                continue
            end
            # Map method names to functions
            vals = method == "stable" ? stable_rician_nll(x, ν) :
                   method == "naive" ? naive_rician_nll(x, ν) :
                   method == "bessels" ? bessels_rician_nll(x, ν) :
                   method == "specialfunctions" ? specialfunctions_rician_nll(x, ν) :
                   error("Unknown method: $method")
            pushrow!(rows, method, x, ν, z, "f", vals.f, ref.f)
            pushrow!(rows, method, x, ν, z, "fx", vals.fx, ref.fx)
            pushrow!(rows, method, x, ν, z, "fν", vals.fν, ref.fν)
            pushrow!(rows, method, x, ν, z, "fxx", vals.fxx, ref.fxx)
            pushrow!(rows, method, x, ν, z, "fxν", vals.fxν, ref.fxν)
            pushrow!(rows, method, x, ν, z, "fνν", vals.fνν, ref.fνν)
            pushrow!(rows, method, x, ν, z, "fxxx", vals.fxxx, ref.fxxx)
            pushrow!(rows, method, x, ν, z, "fxxν", vals.fxxν, ref.fxxν)
            pushrow!(rows, method, x, ν, z, "fxνν", vals.fxνν, ref.fxνν)
            pushrow!(rows, method, x, ν, z, "fννν", vals.fννν, ref.fννν)
        end
    end

    return rows
end

# Evaluate QRice accuracy (high-SNR slab) vs Gauss–Legendre order N
function evaluate_qrice_accuracy(
    ::Type{T};
    νmin = T(1e-3), νmax = T(1e3), num_ν = 13,
    δmin = T(1e-3), δmax = T(1.0), num_δ = 4,
    orders = 2:12,
    # orders = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64],
    num_rician_samples = 10,
) where {T <: Union{Float32, Float64}}

    νs = exp.(range(log(T(νmin)), log(T(νmax)); length = num_ν))
    δs = exp.(range(log(T(δmin)), log(T(δmax)); length = num_δ))

    num_quantities = 10 # Ω, 3 gradient, 6 Hessian
    results_channel = Channel{Vector{NamedTuple}}(num_ν * num_δ)

    function pushrow!(rows, quantity, ν, x, δ, N, z, val, ref)
        push!(rows, (;
            quantity, precision = string(T), ν, x, δ, N, z, val, ref,
            abserr = abserr(val, ref),
            relerr = relerr(val, ref),
            minerr = minerr(val, ref),
            regrelerr = regrelerr(val, ref),
        ))
    end

    # Sample x ~ Rice(ν, 1), then integrate over [x, x + δ]
    Threads.@threads for (ν, δ) in collect(Iterators.product(νs, δs))
        threadlocal_rows = NamedTuple[]
        sizehint!(threadlocal_rows, num_rician_samples * length(orders) * num_quantities)
        for (y, _) in high_snr_iterator(ν; num_rician_samples)
            x = δ * floor(Int, y / δ)
            @assert x isa T && ν isa T && δ isa T
            ref = baseline_qrician_nll(x, ν, δ)
            for N in orders
                val = stable_qrician_nll(x, ν, δ, Val(N))
                z = x * ν
                pushrow!(threadlocal_rows, "Ω", ν, x, δ, N, z, val.Ω, ref.Ω)
                pushrow!(threadlocal_rows, "∇Ω", ν, x, δ, N, z, val.∇Ω, ref.∇Ω)
                pushrow!(threadlocal_rows, "∇²Ω", ν, x, δ, N, z, val.∇²Ω, ref.∇²Ω)
                pushrow!(threadlocal_rows, "∇³Ω", ν, x, δ, N, z, val.∇³Ω, ref.∇³Ω)
            end
        end
        put!(results_channel, threadlocal_rows)
    end

    close(results_channel)
    rows = reduce(vcat, collect(results_channel); init = NamedTuple[])

    return rows
end

# Benchmark Rician NLL and derivatives
function benchmark_rician_nll(::Type{T}; num_samples = 1024, seed = 12345) where {T <: Union{Float32, Float64}}
    # Generate representative (x, ν) samples at high SNR
    Random.seed!(seed)
    xs = exp10.(range(T(-3), T(3); length = num_samples))
    νs = exp10.(range(T(-3), T(3); length = num_samples))
    rows = []

    function benchmark_gpu_barrier(f::F, args::Targs) where {F, Targs}
        gpu_args = map(x -> CUDA.adapt(CuArray, x), args)
        y = f.(gpu_args...)
        return @b CUDA.@sync(y .= f.(gpu_args...))
    end

    # f
    bessels_nll_cpu = @b (rand(xs), rand(νs)) bessels_neglogpdf_rician(_...)
    stable_nll_cpu  = @b (rand(xs), rand(νs)) F.neglogpdf_rician(_...)
    special_nll_cpu = @b (rand(xs), rand(νs)) specialfunctions_neglogpdf_rician(_...)
    bessels_nll_gpu = benchmark_gpu_barrier(bessels_neglogpdf_rician, (xs, νs'))
    stable_nll_gpu  = benchmark_gpu_barrier(F.neglogpdf_rician, (xs, νs'))
    push!(rows, (; quantity = "f", stable_time = stable_nll_cpu.time, stable_gpu_time = stable_nll_gpu.time, bessels_time = bessels_nll_cpu.time, bessels_gpu_time = bessels_nll_gpu.time, specialfunctions_time = special_nll_cpu.time))

    # ∇f
    bessels_grad_cpu = @b (rand(xs), rand(νs)) bessels_neglogpdf_rician_with_gradient(_...)
    stable_grad_cpu  = @b (rand(xs), rand(νs)) F.∇neglogpdf_rician(_...)
    special_grad_cpu = @b (rand(xs), rand(νs)) specialfunctions_neglogpdf_rician_with_gradient(_...)
    bessels_grad_gpu = benchmark_gpu_barrier(bessels_neglogpdf_rician_with_gradient, (xs, νs'))
    stable_grad_gpu  = benchmark_gpu_barrier(F.∇neglogpdf_rician, (xs, νs'))
    push!(rows, (; quantity = "∇f", stable_time = stable_grad_cpu.time, stable_gpu_time = stable_grad_gpu.time, bessels_time = bessels_grad_cpu.time, bessels_gpu_time = bessels_grad_gpu.time, specialfunctions_time = special_grad_cpu.time))

    # ∇²f
    bessels_hess_cpu = @b (rand(xs), rand(νs)) bessels_neglogpdf_rician_with_gradient_and_hessian(_...)
    stable_hess_cpu  = @b (rand(xs), rand(νs)) F.∇²neglogpdf_rician_with_gradient(_...)
    special_hess_cpu = @b (rand(xs), rand(νs)) specialfunctions_neglogpdf_rician_with_gradient_and_hessian(_...)
    bessels_hess_gpu = benchmark_gpu_barrier(bessels_neglogpdf_rician_with_gradient_and_hessian, (xs, νs'))
    stable_hess_gpu  = benchmark_gpu_barrier(F.∇²neglogpdf_rician_with_gradient, (xs, νs'))
    push!(rows, (; quantity = "∇²f", stable_time = stable_hess_cpu.time, stable_gpu_time = stable_hess_gpu.time, bessels_time = bessels_hess_cpu.time, bessels_gpu_time = bessels_hess_gpu.time, specialfunctions_time = special_hess_cpu.time))

    # ∇³f
    bessels_jerk_cpu = @b (rand(xs), rand(νs)) bessels_neglogpdf_rician_with_gradient_hessian_and_jerk(_...)
    stable_jerk_cpu  = @b (rand(xs), rand(νs)) F.∇³neglogpdf_rician_with_gradient_and_hessian(_...)
    special_jerk_cpu = @b (rand(xs), rand(νs)) specialfunctions_neglogpdf_rician_with_gradient_hessian_and_jerk(_...)
    bessels_jerk_gpu = benchmark_gpu_barrier(bessels_neglogpdf_rician_with_gradient_hessian_and_jerk, (xs, νs'))
    stable_jerk_gpu  = benchmark_gpu_barrier(F.∇³neglogpdf_rician_with_gradient_and_hessian, (xs, νs'))
    push!(rows, (; quantity = "∇³f", stable_time = stable_jerk_cpu.time, stable_gpu_time = stable_jerk_gpu.time, bessels_time = bessels_jerk_cpu.time, bessels_gpu_time = bessels_jerk_gpu.time, specialfunctions_time = special_jerk_cpu.time))

    return rows
end

# Benchmark Quantized Rician NLL and derivatives
function benchmark_qrician_nll(::Type{T}; num_samples = 1024, orders = [2, 4, 8]) where {T <: Union{Float32, Float64}}
    # Need to dispatch on the quadrature order to ensure that the order is known at compile time
    function benchmark_cpu_barrier(f::F, args::Targs, ::Val{order}) where {F, Targs, order}
        return @b map(rand, args) f(_..., Val(order))
    end
    function benchmark_gpu_barrier(f::F, args::Targs, ::Val{order}) where {F, Targs, order}
        gpu_args = map(x -> CUDA.adapt(CuArray, x), args)
        y = f.(gpu_args..., Val(order))
        return @b CUDA.@sync(y .= f.(gpu_args..., Val(order)))
    end

    # Generate representative (x, ν, δ) samples at high SNR
    Random.seed!(12345)
    νs = exp10.(range(T(-3), T(3); length = num_samples))
    xs = [√((ν + randn(T))^2 + randn(T)^2) for ν in νs]
    δs = (one(T),)
    Δs = (@SVector(randn(T, 6)),)

    rows = []

    for N in orders
        # Benchmark NLL only
        nll_bench_cpu = benchmark_cpu_barrier(F.neglogpdf_qrician, (xs, νs, δs), Val(N))
        nll_bench_gpu = benchmark_gpu_barrier(F.neglogpdf_qrician, (xs, reshape(νs, 1, :), δs), Val(N))
        push!(rows, (; quantity = "Ω", order = N, time = nll_bench_cpu.time, gpu_time = nll_bench_gpu.time))

        # Benchmark NLL + Gradient
        grad_bench_cpu = benchmark_cpu_barrier(F.∇neglogpdf_qrician_with_primal, (xs, νs, δs), Val(N))
        grad_bench_gpu = benchmark_gpu_barrier(F.∇neglogpdf_qrician_with_primal, (xs, reshape(νs, 1, :), δs), Val(N))
        push!(rows, (; quantity = "∇Ω", order = N, time = grad_bench_cpu.time, gpu_time = grad_bench_gpu.time))

        # Benchmark NLL + Gradient + Hessian
        hess_bench_cpu = benchmark_cpu_barrier(F.∇²neglogpdf_qrician_with_primal_and_gradient, (xs, νs, δs), Val(N))
        hess_bench_gpu = benchmark_gpu_barrier(F.∇²neglogpdf_qrician_with_primal_and_gradient, (xs, reshape(νs, 1, :), δs), Val(N))
        push!(rows, (; quantity = "∇²Ω", order = N, time = hess_bench_cpu.time, gpu_time = hess_bench_gpu.time))

        # Benchmark NLL + Gradient + Hessian + Jerk
        jerk_bench_cpu = benchmark_cpu_barrier(F.∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian, (xs, νs, δs), Val(N))
        jerk_bench_gpu = benchmark_gpu_barrier(F.∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian, (xs, reshape(νs, 1, :), δs), Val(N))
        push!(rows, (; quantity = "∇³Ω", order = N, time = jerk_bench_cpu.time, gpu_time = jerk_bench_gpu.time))

        # # Benchmark NLL + Gradient + Hessian + Jerk
        # vjp_bench_cpu = benchmark_cpu_barrier(F.∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian, (Δs, xs, νs, δs), Val(N))
        # vjp_bench_gpu = benchmark_gpu_barrier(F.∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian, (Δs, xs, reshape(νs, 1, :), δs), Val(N))
        # push!(rows, (; quantity = "VJP", order = N, time = vjp_bench_cpu.time, gpu_time = vjp_bench_gpu.time))
    end

    return rows
end

# Combined table for Bessel ratios (Float64 and Float32)
function save_bessel_table(
    rows64::Vector{<:NamedTuple},
    rows32::Vector{<:NamedTuple},
)
    all_rows = vcat(rows64, rows32)
    metrics = ["minerr"]
    quantities = ["r", "r′", "r′′", "two_r′_plus_z_r′′", "one_minus_r_minus_z_r′"]
    methods = ["naive", "stable"]
    method_labels = Dict("naive" => "Naive", "stable" => "Proposed")
    precisions = ["Float32", "Float64"]

    metric_labels = Dict("relerr" => L"e_{\mathrm{rel}} = |\hat{y} - y| / |y|", "abserr" => L"e_{\mathrm{abs}} = |\hat{y} - y|", "minerr" => L"e_{\min} = \min(|\hat{y} - y|, |\hat{y} - y| / |y|)", "regrelerr" => L"e_{\mathrm{reg}} = |\hat{y} - y| / (1 + |y|)")
    max_errors = Dict(m => OrderedDict((q, m, p) => 0.0 for q in quantities for m in methods for p in precisions) for m in metrics)

    for r in all_rows
        key = (r.quantity, r.method, r.precision)
        for metric in metrics
            max_errors[metric][key] = max(max_errors[metric][key], getfield(r, Symbol(metric)))
        end
    end

    # Create three-level header with requested layout
    headers_top = Any["", [MultiColumn(4, metric_labels[m], :c) for m in metrics]...]
    headers_mid = Any["Quantity", repeat([MultiColumn(2, "Float32", :c), MultiColumn(2, "Float64", :c)], length(metrics))...]
    headers_low = Any[""; repeat([method_labels[m] for m in methods], 2 * length(metrics))...]
    column_labels = [headers_top, headers_mid, headers_low]

    # Total columns = 1 (Quantity) + 4 per metric
    num_cols = 1 + 4 * length(metrics)
    table_data = Matrix{Any}(undef, length(quantities), num_cols)

    # LaTeX quantity labels
    quantity_labels = OrderedDict(
        "r" => L"r(z)",
        "r′" => L"r'(z)",
        "r′′" => L"r''(z)",
        "two_r′_plus_z_r′′" => L"2 r'(z) + z r''(z)",
        "one_minus_r_minus_z_r′" => L"1 - r(z) - z r'(z)",
    )

    for (i, q) in enumerate(quantities)
        table_data[i, 1] = quantity_labels[q]
        col_idx = 2
        for metric in metrics
            err_map = max_errors[metric]
            for p in precisions
                for m in methods
                    val = err_map[(q, m, p)]
                    table_data[i, col_idx] = L"\num[round-mode = figures, round-precision = 2, tight-spacing = true, scientific-notation = true]{%$(val)}"
                    col_idx += 1
                end
            end
        end
    end

    compile_and_save_table(joinpath(OUTDIR, "bessel_ratio_accuracy_table.tex")) do io
        return default_pretty_table(
            io,
            table_data;
            column_labels,
        )
    end
end

# Combined table for NLL and derivatives
function save_nll_accuracy_table(
    rows64::Vector{<:NamedTuple},
    rows32::Vector{<:NamedTuple};
    methods = ["naive", "bessels", "specialfunctions", "stable"],
    metrics = ["minerr"],
)
    all_rows = vcat(rows64, rows32)
    quantities = ["f", "fx", "fν", "fxx", "fxν", "fνν", "fxxx", "fxxν", "fxνν", "fννν"]
    precisions = ["Float32", "Float64"]

    metric_labels = Dict("relerr" => L"e_{\mathrm{rel}} = |\hat{y} - y| / |y|", "abserr" => L"e_{\mathrm{abs}} = |\hat{y} - y|", "minerr" => L"e_{\min} = \min(|\hat{y} - y|, |\hat{y} - y| / |y|)", "regrelerr" => L"e_{\mathrm{reg}} = |\hat{y} - y| / (1 + |y|)")
    method_labels = Dict("naive" => "Naive", "bessels" => "Bessels.jl", "specialfunctions" => "SF.jl", "stable" => "Proposed")
    max_errors = Dict(m => OrderedDict((q, m, p) => 0.0 for q in quantities for m in methods for p in precisions) for m in metrics)

    for r in all_rows
        key = (r.quantity, r.method, r.precision)
        for metric in metrics
            max_errors[metric][key] = max(max_errors[metric][key], getfield(r, Symbol(metric)))
        end
    end

    # Methods available for each precision
    methods_float32 = filter(m -> m != "specialfunctions", methods) # SpecialFunctions.jl doesn't support Float32
    methods_float64 = methods
    num_methods_float32 = length(methods_float32)
    num_methods_float64 = length(methods_float64)

    # Create three-level header with requested layout
    headers_top = Any["", [MultiColumn(num_methods_float32 + num_methods_float64, metric_labels[m], :c) for m in metrics]...]
    headers_mid = Any["Quantity", repeat([MultiColumn(num_methods_float32, "Float32", :c), MultiColumn(num_methods_float64, "Float64", :c)], length(metrics))...]
    headers_low = Any[""; repeat(vcat([method_labels[m] for m in methods_float32], [method_labels[m] for m in methods_float64]), length(metrics))...]
    column_labels = [headers_top, headers_mid, headers_low]

    num_cols = 1 + (num_methods_float32 + num_methods_float64) * length(metrics)
    table_data = Matrix{Any}(undef, length(quantities), num_cols)

    # LaTeX quantity labels
    quantity_labels = OrderedDict(
        "f" => L"f",
        "fx" => L"f_x",
        "fν" => L"f_\nu",
        "fxx" => L"f_{xx}",
        "fxν" => L"f_{x\nu}",
        "fνν" => L"f_{\nu\nu}",
        "fxxx" => L"f_{xxx}",
        "fxxν" => L"f_{xx\nu}",
        "fxνν" => L"f_{x\nu\nu}",
        "fννν" => L"f_{\nu\nu\nu}",
    )

    for (i, q) in enumerate(quantities)
        table_data[i, 1] = quantity_labels[q]
        col_idx = 2
        for metric in metrics
            err_map = max_errors[metric]
            # Float32 columns (without SF.jl)
            for m in methods_float32
                val = err_map[(q, m, "Float32")]
                table_data[i, col_idx] = L"\num[round-mode = figures, round-precision = 2, tight-spacing = true, scientific-notation = true]{%$(val)}"
                col_idx += 1
            end
            # Float64 columns (with all methods)
            for m in methods_float64
                val = err_map[(q, m, "Float64")]
                table_data[i, col_idx] = L"\num[round-mode = figures, round-precision = 2, tight-spacing = true, scientific-notation = true]{%$(val)}"
                col_idx += 1
            end
        end
    end

    compile_and_save_table(joinpath(OUTDIR, "nll_accuracy_table.tex")) do io
        return default_pretty_table(
            io,
            table_data;
            column_labels,
        )
    end
end

# Combined Figure: bessel ratio accuracy across regimes (4 rows: Float32/Float64 × abs/rel, external legend)
function save_bessel_figure(rows64::Vector{<:NamedTuple}, rows32::Vector{<:NamedTuple})
    fig = Figure(; size = (800, 500))
    methods = ["naive", "stable"]
    quantities = ["r", "r′", "r′′", "two_r′_plus_z_r′′", "one_minus_r_minus_z_r′"]
    titles = OrderedDict(
        "r" => L"r",
        "r′" => L"r'",
        "r′′" => L"r''",
        "two_r′_plus_z_r′′" => L"2 r' + z r''",
        "one_minus_r_minus_z_r′" => L"1 - r - z r'",
    )
    colors = OrderedDict(
        "naive" => :tomato,
        "stable" => :deepskyblue3,
    )

    # Define markers for differentiation
    markers = OrderedDict(
        "naive" => :circle,
        "stable" => :utriangle,
    )

    # Create 4x5 grid: (Float32 abs, Float32 rel, Float64 abs, Float64 rel) × quantities
    row_configs = [
        (Float32, rows32, "abserr", L"$e_{\mathrm{abs}}$ (Float32)"),
        (Float32, rows32, "relerr", L"$e_{\mathrm{rel}}$ (Float32)"),
        (Float64, rows64, "abserr", L"$e_{\mathrm{abs}}$ (Float64)"),
        (Float64, rows64, "relerr", L"$e_{\mathrm{rel}}$ (Float64)"),
    ]

    for (row_idx, (T, rows, error_type, row_label)) in enumerate(row_configs)
        # Collect all error values for this row across all quantities and methods
        all_errs = Float64[]
        for q_temp in quantities
            for m_temp in methods
                sub_rows_temp = filter(r -> r.quantity == q_temp && r.method == m_temp, rows)
                errs_temp = [getfield(r, Symbol(error_type)) for r in sub_rows_temp]
                append!(all_errs, errs_temp)
            end
        end

        # Compute y-limits from all error values in this row
        row_ylims = extrema(all_errs)

        for (col_idx, q) in enumerate(quantities)
            # Y-label only shows error type
            ylabel_text = col_idx == 1 ? row_label : ""

            ax = Axis(
                fig[row_idx, col_idx];
                xscale = log10, yscale = log10,
                xlabel = row_idx == 4 ? L"z" : "",  # Only bottom row gets x-label
                ylabel = ylabel_text,
                title = row_idx == 1 ? titles[q] : "",  # Only top row gets titles
                xticklabelsvisible = row_idx == 4,  # Only bottom row shows x tick labels
                yticklabelsvisible = col_idx == 1,  # Only left column shows y tick labels
            )

            # Plot methods
            for m in methods
                sub_rows = filter(r -> r.quantity == q && r.method == m, rows)
                @assert !isempty(sub_rows) "No rows found for $q and $m"
                zs = [r.z for r in sub_rows]

                # Select the appropriate error type
                errs = [getfield(r, Symbol(error_type)) for r in sub_rows]

                # Create max-envelope by sliding window
                n_points = length(errs)
                window = 10
                envelope_zs = Float64[]
                envelope_errs = Float64[]

                # Slide window across the data and keep the maximum in each window
                for i in 1:n_points
                    # Define window bounds centered on point i
                    window_start = max(1, i - window)
                    window_end = min(n_points, i + window)

                    window_errs = errs[window_start:window_end]
                    local_max_idx = argmax(window_errs)
                    global_idx = window_start + local_max_idx - 1

                    push!(envelope_zs, zs[global_idx])
                    push!(envelope_errs, errs[global_idx])
                end

                # Plot the max-envelope as a smooth line
                lines!(
                    ax, envelope_zs, envelope_errs;
                    color = colors[m],
                    label = m,
                    linewidth = 2,
                )

                # Select evenly spaced points from the envelope for scatter plot
                n_scatter_points = 11
                n_envelope = length(envelope_zs)
                scatter_inds = round.(Int, range(1, n_envelope; length = n_scatter_points))

                # Plot the evenly spaced points with markers
                scatter!(
                    ax, envelope_zs[scatter_inds], envelope_errs[scatter_inds];
                    color = colors[m],
                    marker = markers[m],
                    markersize = 8,
                )
            end

            # Add eps(T) line with different dash styles for Float32 vs Float64
            eps_linestyle = T == Float32 ? :dot : :dash
            hlines!(ax, [eps(T)]; color = :black, linestyle = eps_linestyle, linewidth = 1.5)

            # Set y-limits based on computed extrema for this row
            ylims!(ax; low = row_ylims[1], high = row_ylims[2])
        end
    end

    # # Add precision labels on the left side
    # Label(fig[1:2, 0], "Float32"; rotation = π / 2, tellheight = false, fontsize = 16)
    # Label(fig[3:4, 0], "Float64"; rotation = π / 2, tellheight = false, fontsize = 16)

    # Create single unified legend on the right
    legend_elements = []
    legend_labels = []

    # Method legend elements with LaTeX labels
    method_latex_labels = OrderedDict(
        "naive" => LaTeXString("Naive"),
        "stable" => LaTeXString("Proposed"),
    )

    for m in methods
        push!(legend_elements, [LineElement(; color = colors[m], linewidth = 3), MarkerElement(; color = colors[m], marker = markers[m], markersize = 10)])
        push!(legend_labels, method_latex_labels[m])
    end

    # Add eps(T) labels for both precisions with matching colors
    push!(legend_elements, LineElement(; color = :black, linewidth = 3, linestyle = :dot))
    push!(legend_labels, L"$\epsilon_{\mathrm{Float32}}$")

    push!(legend_elements, LineElement(; color = :black, linewidth = 3, linestyle = :dash))
    push!(legend_labels, L"$\epsilon_{\mathrm{Float64}}$")

    Legend(
        fig[1:4, 6], legend_elements, legend_labels;
        tellheight = false, tellwidth = true,
    )

    save(joinpath(OUTDIR, "bessel_ratio_accuracy_figure.png"), fig)
    save(joinpath(OUTDIR, "bessel_ratio_accuracy_figure.pdf"), fig)
    display(fig)

    return nothing
end

# Figure: QRice accuracy vs order N (rows: precisions; cols: metrics; legend outside)
function save_qrice_accuracy(
    rows64::Vector{<:NamedTuple},
    rows32::Vector{<:NamedTuple};
    metric::String = "minerr",
)
    # Columns are now derivative orders
    deriv_quantities = [
        ["Ω"],
        ["∇Ω"],
        ["∇²Ω"],
        ["∇³Ω"],
    ]
    deriv_labels = [L"\Omega", L"\nabla \Omega", L"\nabla^2 \Omega", L"\nabla^3 \Omega"]
    num_derivs = length(deriv_quantities)

    fig = Figure(; size = (800, 400))
    precisions = [(Float32, rows32), (Float64, rows64)]
    metric_names = Dict(
        "abserr" => raw"e_{\mathrm{abs}}",
        "relerr" => raw"e_{\mathrm{rel}}",
        "minerr" => raw"e_{\mathrm{min}}",
        "regrelerr" => raw"e_{\mathrm{reg}}",
    )

    # Canonicalize δ across precisions to ensure one color/legend entry per width
    canonδ(δ) = round(Float64(δ); sigdigits = 6)
    all_rows = vcat(rows64, rows32)
    δ_keys = sort(unique(canonδ(r.δ) for r in all_rows))
    palette = [:tomato, :deepskyblue3, :seagreen, :goldenrod, :purple, :slateblue, :firebrick]
    color_map = Dict(δk => palette[(i-1)%length(palette)+1] for (i, δk) in enumerate(δ_keys))
    markers_list = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :xcross]
    marker_map = Dict(δk => markers_list[(i-1)%length(markers_list)+1] for (i, δk) in enumerate(δ_keys))
    linestyles_list = [:dot, :dashdotdot, :dashdot, :dash, :solid]
    linestyle_map = Dict(δk => linestyles_list[(i-1)%length(linestyles_list)+1] for (i, δk) in enumerate(δ_keys))

    # Grid of axes: rows = precisions, cols = derivative orders
    for (row_idx, (T, rows)) in enumerate(precisions)
        for (col_idx, quantities) in enumerate(deriv_quantities)
            Ns = sort(unique(Int[r.N for r in rows]))

            ax = Axis(
                fig[row_idx, col_idx];
                xscale = identity, yscale = log10,
                xlabel = row_idx == length(precisions) ? L"N" : "",
                ylabel = col_idx == 1 ? L"$%$(metric_names[metric])$ (%$(string(T)))" : "",
                title = row_idx == 1 ? deriv_labels[col_idx] : "",
                xticklabelsvisible = row_idx == length(precisions),
                yticklabelsvisible = col_idx == 1,
                xticks = (Ns[1:2:end], string.(Ns[1:2:end])),
            )

            # Reference line at machine epsilon for the precision
            eps_linestyle = T == Float32 ? :dot : :dash
            hlines!(ax, [eps(T)]; color = :black, linestyle = eps_linestyle)

            # plot max error over high‑SNR slab for each δ (canonical) and order
            for δk in δ_keys
                sub_δ = filter(r -> canonδ(r.δ) == δk, rows)
                sub_deriv = filter(r -> r.quantity in quantities, sub_δ)
                Ns = sort(unique(Int[r.N for r in sub_deriv]))
                isempty(Ns) && (@warn "No Ns found for δ = $δk"; continue)

                ys = Float64[]
                for N in Ns
                    vals = getfield.(filter(r -> r.N == N, sub_deriv), Symbol(metric))
                    isempty(vals) && (@warn "No vals found for N = $N"; continue)
                    push!(ys, max(maximum(vals), eps(Float64)))
                end
                isempty(ys) && (@warn "No ys found for δ = $δk"; continue)
                c = color_map[δk]
                m = marker_map[δk]
                ls = linestyle_map[δk]
                scatterlines!(ax, Ns, ys; color = c, marker = m, linestyle = ls, linewidth = 2, markersize = 8, label = LaTeXString("δ = $(round(δk; sigdigits = 3))"))
            end

            if T == Float32
                ylims!(ax; low = 5e-8, high = 1.0)
            else
                ylims!(ax; low = 5e-17, high = 1.0)
            end
        end
    end

    # External legend on the right across both rows (deduplicated by canonical δ)
    legend_elements = []
    legend_labels = []

    # Add δ entries with color, marker, and linestyle
    for δk in δ_keys
        push!(legend_elements, [LineElement(; color = color_map[δk], linestyle = linestyle_map[δk], linewidth = 3), MarkerElement(; color = color_map[δk], marker = marker_map[δk], markersize = 10)])
        push!(legend_labels, LaTeXString("δ = $(round(δk; sigdigits = 3))"))
    end

    # Add eps(T) labels for both precisions
    push!(legend_elements, LineElement(; color = :black, linewidth = 3, linestyle = :dot))
    push!(legend_labels, L"$\epsilon_{\mathrm{Float32}}$")

    push!(legend_elements, LineElement(; color = :black, linewidth = 3, linestyle = :dash))
    push!(legend_labels, L"$\epsilon_{\mathrm{Float64}}$")

    Legend(fig[1:length(precisions), num_derivs+1], legend_elements, legend_labels; tellheight = false, tellwidth = true)

    save(joinpath(OUTDIR, "qrice_accuracy.png"), fig)
    save(joinpath(OUTDIR, "qrice_accuracy.pdf"), fig)
    display(fig)

    return nothing
end

# Save Rician performance table
function save_rician_performance_table(rows64::Vector, rows32::Vector)
    # Map plain string quantities to LaTeX labels
    quantity_labels = Dict(
        "f"   => L"f",
        "∇f"  => L"\nabla f",
        "∇²f" => L"\nabla^2 f",
        "∇³f" => L"\nabla^3 f",
    )
    quantities = ["f", "∇f", "∇²f", "∇³f"]

    # 8 columns: Quantity | Float32 (CPU: Bessels, Proposed) | Float64 (CPU: SF, Bessels, Proposed) | Float32 (GPU: Bessels, Proposed)
    num_rows = length(quantities)
    table_data = Matrix{Any}(undef, num_rows, 8)

    for (i, q) in enumerate(quantities)
        r32 = rows32[i]
        r64 = rows64[i]
        table_data[i, 1] = quantity_labels[q]

        # Float32 CPU — Bessels.jl (with slowdown vs Proposed CPU)
        table_data[i, 2] = format_time_value(r32.bessels_time, r32.stable_time)

        # Float32 CPU — Proposed
        table_data[i, 3] = format_time_value(r32.stable_time)

        # Float64 CPU — SpecialFunctions.jl (with slowdown vs Proposed CPU)
        table_data[i, 4] = format_time_value(r64.specialfunctions_time, r64.stable_time)

        # Float64 CPU — Bessels.jl (with slowdown vs Proposed CPU)
        table_data[i, 5] = format_time_value(r64.bessels_time, r64.stable_time)

        # Float64 CPU — Proposed
        table_data[i, 6] = format_time_value(r64.stable_time)

        # Float32 GPU — Bessels.jl (with slowdown vs Proposed GPU)
        table_data[i, 7] = format_time_value(r32.bessels_gpu_time, r32.stable_gpu_time)

        # Float32 GPU — Proposed
        table_data[i, 8] = format_time_value(r32.stable_gpu_time)
    end

    # Multi-level headers: top level groups by device (CPU/GPU), middle level by precision
    headers_top = Any[
        "",
        MultiColumn(5, "CPU", :c),
        MultiColumn(2, "GPU", :c),
    ]
    headers_mid = Any[
        "Quantity",
        MultiColumn(2, "Float32", :c),
        MultiColumn(3, "Float64", :c),
        # "", "Float64", "",
        MultiColumn(2, "Float32", :c),
    ]
    headers_low = Any[
        "",
        "Bessels.jl", "Proposed",
        "SF.jl", "Bessels.jl", "Proposed",
        "Bessels.jl", "Proposed",
    ]
    column_labels = [headers_top, headers_mid, headers_low]

    compile_and_save_table(joinpath(OUTDIR, "rician_performance_table.tex")) do io
        return default_pretty_table(
            io,
            table_data;
            column_labels,
        )
    end
end

# Save Quantized Rician performance table
function save_qrician_performance_table(
    rows64::Vector,
    rows32::Vector;
    quantities = ["Ω", "∇Ω", "∇²Ω", "∇³Ω"],
    orders = [2, 8],
)
    # Map plain string quantities to LaTeX labels
    quantity_labels = Dict(
        "Ω" => L"\Omega",
        "∇Ω" => L"\nabla \Omega",
        "∇²Ω" => L"\nabla^2 \Omega",
        "∇³Ω" => L"\nabla^3 \Omega",
        "VJP" => L"\Delta \bullet \nabla^3 \Omega",
    )

    # Quadrature orders to include
    all_orders = unique([r.order for r in rows64])
    filtered_orders = isnothing(orders) ? all_orders : filter(N -> N in orders, all_orders)

    # Rows are quantities, columns are Float32 orders then Float64 orders then Float32 GPU orders then Float64 GPU orders
    num_rows = length(quantities)
    num_cols = 1 + 4 * length(filtered_orders)  # Quantity column + Float32 orders + Float64 orders + Float32 GPU orders + Float64 GPU orders
    table_data = Matrix{Any}(undef, num_rows, num_cols)

    for (i, q) in enumerate(quantities)
        table_data[i, 1] = quantity_labels[q]

        # Fill Float32 CPU columns
        col_idx = 2
        for N in filtered_orders
            r32 = first(filter(r -> r.quantity == q && r.order == N, rows32))
            table_data[i, col_idx] = format_time_value(r32.time)
            col_idx += 1
        end

        # Fill Float64 CPU columns
        for N in filtered_orders
            r64 = first(filter(r -> r.quantity == q && r.order == N, rows64))
            table_data[i, col_idx] = format_time_value(r64.time)
            col_idx += 1
        end

        # Fill Float32 GPU columns
        for N in filtered_orders
            r32 = first(filter(r -> r.quantity == q && r.order == N, rows32))
            table_data[i, col_idx] = format_time_value(r32.gpu_time)
            col_idx += 1
        end

        # Fill Float64 GPU columns
        for N in filtered_orders
            r64 = first(filter(r -> r.quantity == q && r.order == N, rows64))
            table_data[i, col_idx] = format_time_value(r64.gpu_time)
            col_idx += 1
        end
    end

    # Create multi-level headers: top level groups by device (CPU/GPU), middle level by precision, bottom level by order
    headers_top = Any["", MultiColumn(2 * length(filtered_orders), "CPU", :c), MultiColumn(2 * length(filtered_orders), "GPU", :c)]
    headers_mid = Any["Quantity", MultiColumn(length(filtered_orders), "Float32", :c), MultiColumn(length(filtered_orders), "Float64", :c), MultiColumn(length(filtered_orders), "Float32", :c), MultiColumn(length(filtered_orders), "Float64", :c)]
    headers_low = Any[""]

    for N in filtered_orders
        push!(headers_low, L"N=%$N")
    end
    for N in filtered_orders
        push!(headers_low, L"N=%$N")
    end
    for N in filtered_orders
        push!(headers_low, L"N=%$N")
    end
    for N in filtered_orders
        push!(headers_low, L"N=%$N")
    end

    column_labels = [headers_top, headers_mid, headers_low]

    compile_and_save_table(joinpath(OUTDIR, "qrician_performance_table.tex")) do io
        return default_pretty_table(
            io,
            table_data;
            column_labels,
        )
    end
end

@info "Evaluating Bessel ratio accuracy"
rows64_bessel = evaluate_bessel_ratios(Float64)
rows32_bessel = evaluate_bessel_ratios(Float32)

@info "Evaluating NLL accuracy"
rows64_nll = evaluate_nll_accuracy(Float64)
rows32_nll = evaluate_nll_accuracy(Float32)

@info "Evaluating QRice accuracy (high-SNR) vs order"
@time rows64_qrice = evaluate_qrice_accuracy(Float64)
@time rows32_qrice = evaluate_qrice_accuracy(Float32)

@info "Benchmarking Rician NLL performance"
benchmark_rician_nll(Float64)
benchmark_rician_nll(Float32)
rows64_rician_perf = benchmark_rician_nll(Float64)
rows32_rician_perf = benchmark_rician_nll(Float32)

@info "Benchmarking Quantized Rician NLL performance"
benchmark_qrician_nll(Float64)
benchmark_qrician_nll(Float32)
rows64_qrician_perf = benchmark_qrician_nll(Float64)
rows32_qrician_perf = benchmark_qrician_nll(Float32)

function main()
    @info "Generating bessel ratio accuracy figure"
    save_bessel_figure(rows64_bessel, rows32_bessel)

    @info "Generating Bessel ratios table"
    save_bessel_table(rows64_bessel, rows32_bessel)

    @info "Generating NLL table"
    save_nll_accuracy_table(rows64_nll, rows32_nll)

    @info "Generating QRice accuracy figure"
    save_qrice_accuracy(rows64_qrice, rows32_qrice)

    @info "Saving Rician performance table"
    save_rician_performance_table(rows64_rician_perf, rows32_rician_perf)

    @info "Saving Quantized Rician performance table"
    save_qrician_performance_table(rows64_qrician_perf, rows32_qrician_perf)
end

if !isinteractive()
    main()
end
