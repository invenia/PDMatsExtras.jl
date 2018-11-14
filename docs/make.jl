using Documenter, PSDMats

makedocs(;
    modules=[PSDMats],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://gitlab.invenia.ca/invenia/PSDMats.jl/blob/{commit}{path}#L{line}",
    sitename="PSDMats.jl",
    authors="Eric Davies",
    assets=[
        "assets/invenia.css",
        "assets/logo.png",
    ],
    strict=true,
    html_prettyurls=false,
    checkdocs=:none,
)
