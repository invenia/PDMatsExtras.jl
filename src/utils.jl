"""
    submat(A::AbstractPDMat, inds)

Select a square submatrix of an `AbstractPDMat` specified by `inds`.
Returns a `AbstractPDMat` of the same type.
"""
submat(A::PDMat, inds) = PDMat(A[inds, inds])
submat(A::PSDMat, inds) = PSDMat(A[inds, inds])
submat(A::PDiagMat, inds) = PDiagMat(A.diag[inds])
submat(A::PDSparseMat, inds) = PDSparseMat(sparse(A[inds, inds]))
submat(A::WoodburyPDMat, inds) = WoodburyPDMat(A.A[inds, :], A.D, Diagonal(A.S.diag[inds]))
function submat(A::ScalMat, inds)
    # inds should not be able to "access" non-existent indices
    checkbounds(Bool, A, inds) || throw(BoundsError(A, inds))
    return ScalMat(length(inds), A.value)
end
