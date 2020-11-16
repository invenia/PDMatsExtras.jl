module PDMatsExtras
using ChainRulesCore
using LinearAlgebra
using PDMats
import Base: *, \

export PSDMat, WoodburyPDMat

include("psd_mat.jl")
include("woodbury_pd_mat.jl")

end
