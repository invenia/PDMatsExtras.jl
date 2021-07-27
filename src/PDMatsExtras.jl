module PDMatsExtras
using ChainRulesCore
using LinearAlgebra
using PDMats
using SparseArrays: sparse
using SuiteSparse
import Base: *, \

export PSDMat, WoodburyPDMat
export submat

include("psd_mat.jl")
include("woodbury_pd_mat.jl")
include("utils.jl")
include("chainrules.jl")

end
