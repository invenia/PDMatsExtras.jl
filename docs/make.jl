using PDMatsExtras
using Documenter

makedocs(;
    modules=[PDMatsExtras],
    authors="Invenia Technical Computing Corporation",
    repo="https://github.com/invenia/PDMatsExtras.jl/blob/{commit}{path}#L{line}",
    sitename="PDMatsExtras.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://invenia.github.io/PDMatsExtras.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/invenia/PDMatsExtras.jl",
)
