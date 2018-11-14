module PSDMats

using PDMats

import Base: *, \

export PSDMat

# We're currently reusing this type name to minimize the number of changes from pdmat.jl
CholType{T, S<:AbstractMatrix} = Base.LinAlg.CholeskyPivoted{T, S}

#=
The code below is just a slight modification of the PDMat code in PDMats.jl
https://github.com/JuliaStats/PDMats.jl/blob/master/src/pdmat.jl

TODO: Integrate this with PDMats.jl

References for discussion on supporting degenerate mvnormal distributions:
- https://discourse.julialang.org/t/multivariate-normal-with-positive-semi-definite-covariance-matrix/3029
- https://github.com/JuliaStats/Distributions.jl/issues/366
=#

"""
Positive semi-definite matrix together with a CholeskyPivoted factorization object.
"""
struct PSDMat{T<:Real,S<:AbstractMatrix} <: AbstractPDMat{T}
    dim::Int
    mat::S
    chol::CholType{T, S}

    PSDMat{T,S}(d::Int,m::AbstractMatrix{T},c::CholType{T,S}) where {T,S} = new{T, S}(d, m, c)
end

function PSDMat(mat::AbstractMatrix, chol::CholType)
    d = size(mat, 1)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PSDMat{eltype(mat),typeof(mat)}(d, mat, chol)
end

PSDMat(mat::Matrix) = PSDMat(mat, cholfact(mat, :U, Val{true}))
PSDMat(fac::CholType) = PSDMat(Matrix(fac), fac)

### Conversion
Base.convert(::Type{PSDMat{T}}, a::PSDMat) where {T<:Real} = PSDMat(convert(AbstractArray{T}, a.mat))
Base.convert(::Type{AbstractArray{T}}, a::PSDMat) where {T<:Real} = convert(PSDMat{T}, a)

### Basics

PDMats.dim(a::PSDMat) = a.dim
PDMats.full(a::PSDMat) = copy(a.mat)
Base.Matrix(a::PSDMat) = copy(a.mat)
Base.LinAlg.diag(a::PSDMat) = diag(a.mat)


### Arithmetics

function PDMats.pdadd!(r::Matrix, a::Matrix, b::PSDMat, c)
    PDMats.@check_argdims size(r) == size(a) == size(b)
    PDMats._addscal!(r, a, b.mat, c)
end

*(a::PSDMat{S}, c::T) where {S<:Real, T<:Real} = PSDMat(a.mat * c)
*(a::PSDMat, x::StridedVecOrMat) = a.mat * x
\(a::PSDMat, x::StridedVecOrMat) = a.chol \ x


### Algebra

Base.inv(a::PSDMat) = PSDMat(inv(a.chol))
Base.LinAlg.logdet(a::PSDMat) = logdet(a.chol)
Base.LinAlg.eigmax(a::PSDMat) = eigmax(a.mat)
Base.LinAlg.eigmin(a::PSDMat) = eigmin(a.mat)


### whiten and unwhiten

function PDMats.whiten!(r::StridedVecOrMat, a::PSDMat, x::StridedVecOrMat)
    cf = a.chol[:U]
    istriu(cf) ? Ac_ldiv_B!(cf, PDMats._rcopy!(r, x)) : A_ldiv_B!(cf, PDMats._rcopy!(r, x))
    return r
end

function PDMats.unwhiten!(r::StridedVecOrMat, a::PSDMat, x::StridedVecOrMat)
    cf = a.chol[:U]
    istriu(cf) ? Ac_mul_B!(cf, PDMats._rcopy!(r, x)) : A_mul_B!(cf, PDMats._rcopy!(r, x))
    return r
end


### quadratic forms

PDMats.quad(a::PSDMat, x::StridedVector) = dot(x, a * x)
PDMats.invquad(a::PSDMat, x::StridedVector) = dot(x, a \ x)

"""
    quad!(r::AbstractArray, a::AbstractPDMat, x::StridedMatrix)
Overwrite `r` with the value of the quadratic form defined by `a` applied columnwise to `x`
"""
PDMats.quad!(r::AbstractArray, a::PSDMat, x::StridedMatrix) = PDMats.colwise_dot!(r, x, a.mat * x)

"""
    invquad!(r::AbstractArray, a::AbstractPDMat, x::StridedMatrix)
Overwrite `r` with the value of the quadratic form defined by `inv(a)` applied columnwise to `x`
"""
PDMats.invquad!(r::AbstractArray, a::PSDMat, x::StridedMatrix) = PDMats.colwise_dot!(r, x, a.mat \ x)


### tri products
# NOTE: This code is from 0.8.0 as the newer code is not 0.6/0.7 compliant
# even with Compat.
function PDMats.X_A_Xt(a::PSDMat, x::StridedMatrix)
    z = copy(x)
    cf = a.chol[:U]
    istriu(cf) ? A_mul_Bc!(z, cf) : A_mul_B!(z, cf)
    A_mul_Bt(z, z)
end

function PDMats.Xt_A_X(a::PSDMat, x::StridedMatrix)
    z = copy(x)
    cf = a.chol[:U]
    istriu(cf) ? A_mul_B!(cf, z) : Ac_mul_B!(cf, z)
    At_mul_B(z, z)
end

function PDMats.X_invA_Xt(a::PSDMat, x::StridedMatrix)
    z = copy(x)
    cf = a.chol[:U]
    istriu(cf) ? A_rdiv_B!(z, cf) : A_rdiv_Bc!(z, cf)
    A_mul_Bt(z, z)
end

function PDMats.Xt_invA_X(a::PSDMat, x::StridedMatrix)
    z = copy(x)
    cf = a.chol[:U]
    istriu(cf) ? Ac_ldiv_B!(cf, z) : A_ldiv_B!(cf, z)
    At_mul_B(z, z)
end

end
