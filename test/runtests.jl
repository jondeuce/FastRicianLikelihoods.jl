using FastRicianLikelihoods
using Test
using Aqua
using Random

include("utils.jl")

# Test files to run, in order. With no arguments the whole suite runs, otherwise names are matched as substrings of the file name:
#   julia --project=test test/runtests.jl
#   julia --project=test test/runtests.jl bessels
#   julia --project=test test/runtests.jl gauss rician
#   julia -e 'using Pkg; Pkg.test("FastRicianLikelihoods"; test_args = ["bessels"])'

const TEST_FILES = ["gausslegendre", "gausshalfhermite", "forwarddiff", "bessels", "rician", "aqua"]
for arg in ARGS
    any(name -> occursin(arg, name), TEST_FILES) || error("No test files match $arg. Available test files: $(join(TEST_FILES, ", "))")
end
const SELECTED_TEST_FILES = isempty(ARGS) ? TEST_FILES : filter(name -> any(arg -> occursin(arg, name), ARGS), TEST_FILES)

@testset "FastRicianLikelihoods.jl" begin
    for name in SELECTED_TEST_FILES
        Random.seed!(32)
        @testset "$name.jl" verbose = true begin
            include("$name.jl")
        end
    end
end
