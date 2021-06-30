@testset "utils.jl" begin
    @testset "submat" begin

        @testset "PDMat" begin
            M = PDMat(TEST_MATRICES["Positive definite"])
            M_dense = Matrix(M)

            subM = submat(M, [1, 3])
            @test subM isa PDMat
            @test Matrix(subM) == M_dense[[1, 3], [1, 3]]
        end

        @testset "PDDiagMat" begin
            M = PDiagMat(diag(TEST_MATRICES["Positive definite"]))
            M_dense = Matrix(M)

            subM = submat(M, [1, 3])
            @test subM isa PDiagMat
            @test Matrix(subM) == M_dense[[1, 3], [1, 3]]
        end

        @testset "PDSparseMat" begin
            M = PDSparseMat(sparse(TEST_MATRICES["Positive definite"]))
            M_dense = Matrix(M)

            subM = submat(M, [1, 3])
            @test subM isa PDSparseMat
            @test Matrix(subM) == M_dense[[1, 3], [1, 3]]
        end

        @testset "ScalMat" begin
            M = ScalMat(3, 0.1)
            M_dense = Matrix(M)

            subM = submat(M, [1, 3])
            @test subM isa ScalMat
            @test Matrix(subM) == M_dense[[1, 3], [1, 3]]
            @test_throws BoundsError submat(M, [1, 10])
        end

        @testset "PSDMat" begin
            SM = TEST_MATRICES["Positive semi-definite"]
            pivoted = cholesky(SM, Val(true); check=false)
            M = PSDMat(SM, pivoted)
            M_dense = Matrix(M)

            subM = submat(M, [1, 3])
            @test subM isa PSDMat
            @test Matrix(subM) == M_dense[[1, 3], [1, 3]]
        end

        @testset "WoodburyPDMat" begin
            A = randn(4, 2)
            D = Diagonal(randn(2).^2 .+ 1)
            S = Diagonal(randn(4).^2 .+ 1)
            x = randn(size(A, 1))

            W = WoodburyPDMat(A, D, S)
            W_dense = PDMat(Symmetric(Matrix(W)))

            subW = submat(W, [1, 3])
            @test subW isa WoodburyPDMat
            @test subW.A == W.A[[1, 3], :]
            @test subW.D == W.D
            @test subW.S == Diagonal(W.S.diag[[1, 3]])
            @test Matrix(subW) == W_dense[[1, 3], [1, 3]]
        end
    end
end
