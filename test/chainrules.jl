@testset "ChainRules" begin

    A = randn(4, 2)
    D = Diagonal(randn(2).^2 .+ 1)
    S = Diagonal(randn(4).^2 .+ 1)

    W = WoodburyPDMat(A, D, S)
    R = 2.0
    Dmat = Diagonal(rand(4,))

    x = randn(size(A, 1))

    @testset "Constructors" begin
        test_rrule(WoodburyPDMat, W.A, W.D, W.S)
        # This is a gradient. Should be able to deal with negative elements
        test_rrule(WoodburyPDMat, W.A, W.D, W.S;
            output_tangent=Tangent{WoodburyPDMat}(;
            A = rand(4,2), D = Diagonal(-1 * rand(2,)), S = Diagonal(-1 * rand(4,)))
        )
    end

    @testset "*(Matrix-Woodbury)" begin
        test_rrule(*, Dmat, W)
        test_rrule(*, W, Dmat)
        test_rrule(*, rand(4,4), W)
    end

    @testset "*(Woodbury-Real)" begin
        test_rrule(*, W, R)
        test_rrule(*, R, W)

        # We can't test test_rrule(*, R, W; output_tangent = rand(size(W)...)) i.e. with a Matrix because
        # FD requires the primal and tangent to be the same size. However, we can just call FD directly and overload
        # the primal computation to return a Matrix:
        res, pb = ChainRulesCore.rrule(*, R, W)
        output_tangent = rand(size(W)...)
        f_jvp = j′vp(ChainRulesTestUtils._fdm, x -> Matrix(*(x...)), output_tangent, (R, W))[1]
        unthunk(pb(output_tangent)[3]).A ≈ f_jvp[2].A
        unthunk(pb(output_tangent)[3]).D ≈ f_jvp[2].D
        unthunk(pb(output_tangent)[3]).S ≈ f_jvp[2].S
        unthunk(pb(output_tangent)[2]) ≈ f_jvp[1]
    end
end
