@non_differentiable validate_woodbury_arguments(A, D, S)

# Rule for Woodbury * Real. 
# Ignoring Complex version for now. 
# NOTE: # Can't use Tangent{WoodburyPDMat} here. ChainRules is fine, but Zygote has issues and passes around Tangent{Any}

function ChainRulesCore.rrule(::typeof(*), A::WoodburyPDMat, B::Real)
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    primal = A * B
    times_pullback(ȳ) = _times_pullback(ȳ, primal, A, B, (;A=project_A, B=project_B))
    return primal, times_pullback
end

function ChainRulesCore.rrule(::typeof(*), A::Real, B::WoodburyPDMat, )
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    primal = A * B
    times_pullback(ȳ) = _times_pullback(ȳ, primal, A, B, (;A=project_A, B=project_B))
    return primal, times_pullback
end

_times_pullback(ȳ::AbstractThunk, primal, A, B, proj) = _times_pullback(unthunk(ȳ), primal, A, B, proj)
# If the cotangent is a Matrix we first need to project down, otherwise ignore
_times_pullback(Ȳ::AbstractMatrix, primal, A, B, proj) = _times_pullback(ProjectTo(primal)(Ȳ), A, B, proj)
_times_pullback(ȳ::Tangent, primal, A, B, proj) = _times_pullback(ȳ, A, B, proj)

function _times_pullback(Ȳ::Tangent, A::T, B::Real, proj)  where {T<:WoodburyPDMat}
    Ā = @thunk proj.A(Tangent{WoodburyPDMat}(; A = Ȳ.A, D = Ȳ.D * B', S = Ȳ.S * B'))
    B̄ = @thunk proj.B(dot(Ȳ.D, A.D) + dot(Ȳ.S, A.S))
    return (NoTangent(), Ā, B̄)
end

function _times_pullback(Ȳ::Tangent, A::Real, B::T, proj) where {T<:WoodburyPDMat}
    Ā = @thunk proj.A(dot(Ȳ.D, B.D) + dot(Ȳ.S, B.S))
    B̄ = @thunk proj.B(Tangent{WoodburyPDMat}(; A = Ȳ.A, D = Ȳ.D * A, S = Ȳ.S * A))
    return (NoTangent(), Ā, B̄)
end

# Composite pullbacks
function ChainRulesCore.rrule(
    ::Type{T},
    A::AbstractMatrix,
    D::Diagonal,
    S::Diagonal,
    ) where  {T<:WoodburyPDMat}
    return WoodburyPDMat(A, D, S), X̄ -> WoodburyPDMat_pullback(X̄, A, D, S)
end
WoodburyPDMat_pullback(X̄::Tangent, A, D, S) = (X̄ = (unthunk(X̄); return (NoTangent(), X̄.A, X̄.D, X̄.S)))
WoodburyPDMat_pullback(X̄::AbstractThunk, A, D, S) = WoodburyPDMat_pullback(unthunk(X̄), A, D, S)

function ChainRulesCore.ProjectTo(W::T) where {T<:WoodburyPDMat}
    fields = (A = W.A, D = W.D, S = W.S)
    ChainRulesCore.ProjectTo{T}(; fields...)
end

function (project::ProjectTo{T})(X̄::AbstractMatrix) where {T<:WoodburyPDMat}
    Ā = ProjectTo(project.A)((X̄ + X̄') * (project.A * project.D))
    D̄ = ProjectTo(project.D)(project.A' * (X̄) * project.A)
    S̄ = ProjectTo(project.S)(X̄)
    return Tangent{WoodburyPDMat}(; A = Ā, D = D̄, S = S̄)
end
(project::ProjectTo{T})(W::Tangent) where {T<:WoodburyPDMat} = W
