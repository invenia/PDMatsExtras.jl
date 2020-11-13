using PDMatsExtras
using Test

using LinearAlgebra
using Random

using Distributions
using PDMats

@testset "PDMatsExtras.jl" begin
    include("testutils.jl")
    include("psd_mat.jl")
end
