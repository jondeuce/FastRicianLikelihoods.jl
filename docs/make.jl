using FastRicianLikelihoods
using Documenter

DocMeta.setdocmeta!(FastRicianLikelihoods, :DocTestSetup, :(using FastRicianLikelihoods); recursive = true)

makedocs(;
    modules = [FastRicianLikelihoods],
    authors = "Jonathan Doucette <jdoucette@physics.ubc.ca> and contributors",
    repo = "https://github.com/jondeuce/FastRicianLikelihoods.jl/blob/{commit}{path}#{line}",
    sitename = "FastRicianLikelihoods.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jondeuce.github.io/FastRicianLikelihoods.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/jondeuce/FastRicianLikelihoods.jl",
    devbranch = "master",
)
