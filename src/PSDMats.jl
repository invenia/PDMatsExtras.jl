module PSDMats

using LinearAlgebra
using PDMats

import Base: *, \

export PSDMat

# We're currently reusing this type name to minimize the number of changes from pdmat.jl
CholType{T, S<:AbstractMatrix} = LinearAlgebra.CholeskyPivoted{T, S}

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

PSDMat(mat::Matrix) = PSDMat(mat, cholesky(mat, Val(true); check=false))
PSDMat(mat::Symmetric) = PSDMat(Matrix(mat))
PSDMat(fac::CholType) = PSDMat(Matrix(fac), fac)

### Conversion
Base.convert(::Type{PSDMat{T}}, a::PSDMat) where {T<:Real} = PSDMat(convert(AbstractArray{T}, a.mat))
Base.convert(::Type{AbstractArray{T}}, a::PSDMat) where {T<:Real} = convert(PSDMat{T}, a)

### Basics

PDMats.dim(a::PSDMat) = a.dim
Base.Matrix(a::PSDMat) = copy(a.mat)
Base.getindex(a::PSDMat, i::Int) = getindex(a.mat, i)
Base.getindex(a::PSDMat, I::Vararg{Int, N}) where N = getindex(a.mat, I...)
LinearAlgebra.diag(a::PSDMat) = diag(a.mat)


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
LinearAlgebra.logdet(a::PSDMat) = logdet(a.chol)
LinearAlgebra.eigmax(a::PSDMat) = eigmax(a.mat)
LinearAlgebra.eigmin(a::PSDMat) = eigmin(a.mat)
LinearAlgebra.kron(a::PSDMat, b::PSDMat) = PSDMat(kron(a.mat, b.mat))
LinearAlgebra.kron(a::AbstractPDMat, b::PSDMat) = PSDMat(kron(Matrix(a), b.mat))
LinearAlgebra.kron(a::PSDMat, b::AbstractPDMat) = PSDMat(kron(a.mat, Matrix(b)))


### whiten and unwhiten

function PDMats.whiten!(r::StridedVecOrMat, a::PSDMat, x::StridedVecOrMat)
    cf = a.chol.U
    v = PDMats._rcopy!(r, x)
    istriu(cf) ? ldiv!(transpose(cf), v) : ldiv!(cf, v)
end

function PDMats.unwhiten!(r::StridedVecOrMat, a::PSDMat, x::StridedVecOrMat)
    cf = a.chol.U
    v = PDMats._rcopy!(r, x)
    istriu(cf) ? lmul!(transpose(cf), v) : lmul!(cf, v)
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

function X_A_Xt(a::PSDMat, x::StridedMatrix)
    z = copy(x)
    cf = a.chol.UL
    rmul!(z, istriu(cf) ? transpose(cf) : cf)
    z * transpose(z)
end

function Xt_A_X(a::PSDMat, x::StridedMatrix)
    cf = a.chol.UL
    z = lmul!(istriu(cf) ? cf : transpose(cf), copy(x))
    transpose(z) * z
end

function X_invA_Xt(a::PSDMat, x::StridedMatrix)
    cf = a.chol.UL
    z = rdiv!(copy(x), istriu(cf) ? cf : transpose(cf))
    z * transpose(z)
end

function Xt_invA_X(a::PSDMat, x::StridedMatrix)
    cf = a.chol.UL
    z = ldiv!(istriu(cf) ? transpose(cf) : cf, copy(x))
    transpose(z) * z
end

end
