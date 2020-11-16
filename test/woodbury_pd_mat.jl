@testset "woodbury_pd_mat" begin
    A = randn(4, 2)
    D = Diagonal(randn(2).^2 .+ 1)
    S = Diagonal(randn(4).^2 .+ 1)
    x = randn(size(A, 1))

    @test_throws ArgumentError WoodburyPDMat(randn(5, 2), D, S)
    @test_throws ArgumentError WoodburyPDMat(randn(4, 3), D, S)
    @test_throws ArgumentError WoodburyPDMat(A, Diagonal(.-randn(2).^2), S)
    @test_throws ArgumentError WoodburyPDMat(A, D, Diagonal(.-randn(4).^2))

    W = WoodburyPDMat(A, D, S)
    W_dense = PDMat(Symmetric(Matrix(W)))

    # Checks getindex works.
    @test all(isapprox.(W, W_dense))

    @test PDMats.unwhiten!(similar(x), W, x) ≈ PDMats.unwhiten!(similar(x), W_dense, x)

    @testset "logdet" begin

        @test logdet(W) ≈ logdet(W_dense)

        test_function = (A, D, S) -> logdet(WoodburyPDMat(A, D, S))

        test_ad(test_function, randn(), A, D, S)
    end

    @testset "invquad" begin
        @test invquad(W, x) ≈ invquad(W_dense, x)

        test_function = (A, D, S, x) -> begin
            W = WoodburyPDMat(A, D, S)
            return HeavyTailedPredictors.invquad(W, x)
        end

        test_ad(test_function, randn(), A, D, S, x)
    end

    @testset "MvNormal logpdf" begin

        m = randn(size(A, 1))
        @test logpdf(MvNormal(m, W), x) ≈ logpdf(MvNormal(m, Symmetric(Matrix(W))), x)

        test_function = (m, A, D, S, x) -> begin
            W = WoodburyPDMat(A, D, S)
            return logpdf(MvNormal(m, W), x)
        end

        test_ad(test_function, randn(), m, A, D, S, x)
    end

    @testset "GenericMvTDist logpdf" begin

        α = 2.1
        m = randn(size(A, 1))
        @test isapprox(
            logpdf(Distributions.GenericMvTDist(α, m, W), x),
            logpdf(Distributions.GenericMvTDist(α, m, W_dense), x),
        )

        test_function = (α, m, A, D, S, x) -> begin
            W = WoodburyPDMat(A, D, S)
            return logpdf(Distributions.GenericMvTDist(α, m, W, false), x)
        end

        test_ad(test_function, randn(), α, m, A, D, S, x)
    end
end
