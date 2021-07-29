@non_differentiable validate_woodbury_arguments(A, D, S)

function _times_pullback(Ȳ::AbstractMatrix, A, B, proj)
    #Ā = @thunk(proj[:A](dot(Ȳ, B)'))
    #B̄ = @thunk(proj[:B](A' * Ȳ))
    Ā = dot(Ȳ, B)'
    B̄ = A' * Ȳ
    return (NoTangent(), Ā, B̄)
end
_times_pullback(ȳ::AbstractThunk, A, B, proj) = _times_pullback(unthunk(ȳ), A, B, proj)
function _times_pullback(Ȳ::Tangent{<:WoodburyPDMat}, A, B, proj)
    W = WoodburyPDMat(Ȳ.A, Ȳ.D, Ȳ.S)
    return _times_pullback(W, A, B, proj)
end

function ChainRulesCore.rrule(::typeof(*), A::Real, B::WoodburyPDMat)
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    times_pullback(ȳ) = _times_pullback(ȳ, A, B, (;A=project_A, B=project_B))
    return A * B, times_pullback
end

function ChainRulesCore.ProjectTo(W::WoodburyPDMat)
    function dW(W̄)
        Ā(W̄) = ProjectTo(W.A)(collect((W.D * W.A' * W̄' + W.D * W.A' * W̄)'))
        D̄(W̄) = ProjectTo(W.D)(W.A' * (W̄) * W.A)
        S̄(W̄) = ProjectTo(W.S)(W̄)
        return Tangent{typeof(W)}(; A = Ā(W̄), D = D̄(W̄), S = S̄(W̄))
    end
    return dW
end
