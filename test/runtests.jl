using Test
using PDMatsExtras
using ChainRulesCore
using ChainRulesTestUtils
using Distributions
using FiniteDifferences
using LinearAlgebra
using PDMats
using SparseArrays: sparse
using Random
using Test
using Zygote

Random.seed!(1)

# NOTE: We could probably do a more thorough testing job if we able to override the
# `t_tripod` and `t_whiten` test in PDMats.
const TEST_MATRICES = Dict(
    "Positive definite" => [
        0.796911  0.602112  0.766136  0.247788
        0.602112  0.480312  0.605538  0.218218
        0.766136  0.605538  1.28666   0.290052
        0.247788  0.218218  0.290052  0.130588
    ],
    "Positive semi-definite" => [
        10.8145   -9.27226   1.67126   4.02515
        -9.27226   8.08443  -1.48168  -4.27258
        1.67126  -1.48168   1.31866   1.43293
        4.02515  -4.27258   1.43293   6.76801
    ]
)

@testset "PDMatsExtras.jl" begin
    include("testutils.jl")
    include("test_ad.jl")

    include("psd_mat.jl")
    include("chainrules.jl")
    include("woodbury_pd_mat.jl")
    include("utils.jl")
end
