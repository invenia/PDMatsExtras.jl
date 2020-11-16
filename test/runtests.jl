using PDMatsExtras
using Test

using LinearAlgebra
using Random

using Distributions
using PDMats

@testset "PDMatsExtras.jl" begin
    include("testutils.jl")
    include("test_ad.jl")

    include("psd_mat.jl")
    include("woodbury_pd_mat.jl")
end
