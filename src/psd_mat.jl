
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
    mat::S
    chol::CholType{T, S}

    PSDMat{T,S}(m::AbstractMatrix{T},c::CholType{T,S}) where {T,S} = new{T, S}(m, c)
end

function PSDMat(mat::AbstractMatrix, chol::CholType{T,S}) where {T,S}
    d = LinearAlgebra.checksquare(mat)
    size(chol, 1) == d ||
        throw(DimensionMismatch("Dimensions of mat and chol are inconsistent."))
    PSDMat{T, S}(d, convert(S, mat), chol)
end

PSDMat(mat::Matrix) = PSDMat(mat, cholesky(mat, VERSION >= v"1.8.0-rc1" ? RowMaximum() : Val(true); check=false))
PSDMat(mat::Symmetric) = PSDMat(Matrix(mat))
PSDMat(fac::CholType) = PSDMat(Matrix(fac), fac)

function Base.getproperty(a::PSDMat, s::Symbol)
    if s === :dim
        return size(getfield(a, :mat), 1)
    end
    return getfield(a, s)
end
Base.propertynames(::PSDMat) = (:mat, :chol, :dim)

### Conversion
Base.convert(::Type{PSDMat{T}}, a::PSDMat{T}) where {T<:Real} = a
function Base.convert(::Type{PSDMat{T}}, a::PSDMat) where {T<:Real}
    chol = convert(CholType{T}, a.chol)
    S = typeof(chol.factors)
    mat = convert(S, a.mat)
    PSDMat{T,S}(mat, chol)
end
Base.convert(::Type{AbstractPDMat{T}}, a::PSDMat) where {T<:Real} = convert(PSDMat{T}, a)

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
LinearAlgebra.cholesky(a::PSDMat) = a.chol
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
    view_v = v isa StridedVector ? view(v, a.chol.p) : view(v, a.chol.p, :)
    istriu(cf) ? ldiv!(transpose(cf), view_v) : ldiv!(cf, view_v)
    return v
end

function PDMats.unwhiten!(r::StridedVecOrMat, a::PSDMat, x::StridedVecOrMat)
    cf = a.chol.U
    v = PDMats._rcopy!(r, x)
    view_v = v isa StridedVector ? view(v, a.chol.p) : view(v, a.chol.p, :)
    istriu(cf) ? lmul!(transpose(cf), view_v) : lmul!(cf, view_v)
    return v
end


### quadratic forms

function PDMats.quad(a::PSDMat, x::AbstractVecOrMat)
    if a.dim != size(x, 1)
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
    # https://github.com/JuliaLang/julia/commit/2425ae760fb5151c5c7dd0554e87c5fc9e24de73
    if VERSION < v"1.4.0-DEV.92"
        z = a.mat * x
        return x isa AbstractVector ? dot(x, z) : map(dot, eachcol(x), eachcol(z))
    else
        return x isa AbstractVector ? dot(x, a.mat, x) : map(Base.Fix1(quad, a), eachcol(x))
    end
end

function PDMats.invquad(a::PSDMat, x::AbstractVecOrMat)
    if a.dim != size(x, 1)
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
    z = a.chol \ x
    return x isa AbstractVector ? dot(x, z) : map(dot, eachcol(x), eachcol(z))
end

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

Base.size(a::PSDMat) = (a.dim, a.dim)

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

