module PDMatsExtras

using LinearAlgebra
using PDMats
import Base: *, \

export PSDMat

include("psd_mat.jl")
include("woodbury_pd_mat.jl")

end
