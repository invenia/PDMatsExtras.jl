@non_differentiable validate_woodbury_arguments(A, D, S)

function ChainRulesCore.rrule(::typeof(*), A::Real, B::WoodburyPDMat)
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    function times_pullback(ȳ::AbstractMatrix)
        Ā = @thunk(project_A(dot(Ȳ, B)'))
        B̄ = @thunk(project_B(A' * Ȳ))
        return (NoTangent(), Ā, B̄)
    end

    function times_pullback(Ȳ::Tangent{<:WoodburyPDMat})
        Ā = dot(Ȳ.A * Ȳ.D * Ȳ.A' + Ȳ.S, B)
        B̄ = Ȳ.A * (A' * Ȳ.D) * Ȳ.A' + A' * Ȳ.S
        return (
           NoTangent(),
           @thunk(project_A(Ā')),
           @thunk(project_B(B̄)),
        )
    end
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
