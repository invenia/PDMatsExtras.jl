using PDMatsExtras
using ChainRulesCore
using Distributions
using FiniteDifferences
using LinearAlgebra
using PDMats
using Random
using Test
using Zygote

Random.seed!(1)
@testset "PDMatsExtras.jl" begin
    include("testutils.jl")
    include("test_ad.jl")

    include("psd_mat.jl")
    include("woodbury_pd_mat.jl")
end
