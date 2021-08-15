@testset "ChainRules" begin

    W = WoodburyPDMat(rand(4,2), Diagonal(rand(2,)), Diagonal(rand(4,)))
    R = 2.0
    D = Diagonal(rand(4,))

    @testset "Constructors" begin
        test_rrule(WoodburyPDMat, W.A, W.D, W.S)
        test_rrule(WoodburyPDMat, W.A, W.D, W.S; output_tangent=Tangent{WoodburyPDMat}(; A = rand(4,2), D = Diagonal(rand(2,)), S = Diagonal(rand(4,))))
    end

    @testset "*(Matrix-Woodbury)" begin
        test_rrule(*, D, W)
        test_rrule(*, W, D)
        test_rrule(*, rand(4,4), W)
    end

    @testset "*(Woodbury-Real)" begin
        test_rrule(*, W, R)
        test_rrule(*, R, W)
    end

end
