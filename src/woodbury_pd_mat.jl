"""
    WoodburyPDMat(
        A::AbstractMatrix{T}, D::Diagonal{T}, S::Diagonal{T},
    ) where {T<:Real}

Lazily represents matrices of the form
```julia
W = A * D * A' + S
```
`D` and `S` must have only non-negative entries.

Using this matrix type is a good idea if `size(A, 1) > size(A, 2)` as the structure in the
matrix can be exploited to accelerate operations involving `W`'s inverse [1], such as
`invquad`, and it's determinant [2], such as `logdet`.

You probably don't want to use this matrix type if `size(A, 1) < size(A, 2)`.

[1] - https://en.wikipedia.org/wiki/Woodbury_matrix_identity
[2] - https://en.wikipedia.org/wiki/Matrix_determinant_lemma
"""
struct WoodburyPDMat{
    T<:Real, TA<:AbstractMatrix{T}, TD<:Diagonal{T}, TS<:Diagonal{T},
} <: AbstractPDMat{T}
    A::TA
    D::TD
    S::TS
    function WoodburyPDMat(
        A::AbstractMatrix{T}, D::Diagonal{T}, S::Diagonal{T},
    ) where {T<:Real}
        validate_woodbury_arguments(A, D, S)
        return new{T, typeof(A), typeof(D), typeof(S)}(A, D, S)
    end
end

function WoodburyPDMat(
    A::AbstractMatrix{T}, D::AbstractVector{T}, S::AbstractVector{T},
) where {T<:Real}
    return WoodburyPDMat(A, Diagonal(D), Diagonal(S))
end

PDMats.dim(W::WoodburyPDMat) = size(W.A, 1)

# Convesion method. Primarily useful for testing purposes.
Base.Matrix(W::WoodburyPDMat) = W.A * W.D * W.A' + W.S

Base.getindex(W::WoodburyPDMat, inds...) = getindex(Matrix(W), inds...)

function validate_woodbury_arguments(A, D, S)
    if size(A, 1) != size(S, 1)
        throw(ArgumentError("size(A, 1) != size(S, 1)"))
    end
    if size(A, 2) != size(D, 1)
        throw(ArgumentError("size(A, 2) != size(D, 1)"))
    end
    if any(x -> x < 0, diag(D))
        throw(ArgumentError("Detected negative element on diagonal of D: $(D)"))
    end
    if any(x -> x < 0, diag(S))
        throw(ArgumentError("Detected negative element on diagonal of S: $(S)"))
    end
end

@non_differentiable validate_woodbury_arguments(A, D, S)

function LinearAlgebra.logdet(W::WoodburyPDMat)
    C_S = cholesky(W.S)
    B = C_S.U' \ (W.A * cholesky(W.D).U')
    return logdet(C_S) + logdet(cholesky(Symmetric(I + B'B)))
end

# Utilises the matrix inversion lemma to produce an efficient implementation.
function PDMats.invquad(W::WoodburyPDMat{<:Real}, x::AbstractVector{<:Real})
    C_S = cholesky(W.S)
    B = C_S.U' \ (W.A * cholesky(W.D).U')
    α = C_S.U' \ x
    β = B' * α
    return α'α - sum(abs2, cholesky(Symmetric(I + B'B)).U' \ β)
end

# This doesn't get us the computational wins, but it's unclear how to construct a
# root for a Woodbury matrix. Consequently, if performance is very important when sampling,
# it's necessary to implement a method of `rand` or `_rand` that explicitly uses ancestral
# sampling to exploit the approximately low-rank structre in a WoodburyPDMat.
function PDMats.unwhiten!(r::DenseVecOrMat, W::WoodburyPDMat{<:Real}, x::DenseVecOrMat)
    return unwhiten!(r, PDMat(Symmetric(Matrix(W))), x)
end

# NOTE: the parameterisation to scale up the Woodbury matrix is not unique. Here we
# implement one way to scale it.
*(a::WoodburyPDMat, c::Real) = WoodburyPDMat(a.A, a.D * c, a.S * c)
*(c::Real, a::WoodburyPDMat) = a * c
*(a::WoodburyPDMat, c::Diagonal{T}) where {T<:Real} = WoodburyPDMat(sqrt(c) * a.A, a.D, a.S * c)
*(c::Diagonal{T}, a::WoodburyPDMat) where {T<:Real} = a * c
