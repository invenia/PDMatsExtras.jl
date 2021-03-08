@testset "woodbury_pd_mat" begin
    A = randn(4, 2)
    D = Diagonal(randn(2).^2 .+ 1)
    S = Diagonal(randn(4).^2 .+ 1)
    x = randn(size(A, 1))

    W = WoodburyPDMat(A, D, S)
    W_dense = PDMat(Symmetric(Matrix(W)))

    @testset "invalid constructors error" begin
        @test_throws ArgumentError WoodburyPDMat(randn(5, 2), D, S)
        @test_throws ArgumentError WoodburyPDMat(randn(4, 3), D, S)
        @test_throws ArgumentError WoodburyPDMat(A, Diagonal(.-randn(2).^2), S)
        @test_throws ArgumentError WoodburyPDMat(A, D, Diagonal(.-randn(4).^2))
    end

    @testset "Basic functionality" begin
        # Checks getindex works.
        @test all(isapprox.(W, W_dense))
    end

    @testset "unwhiten!" begin
        @test PDMats.unwhiten!(similar(x), W, x) ≈ PDMats.unwhiten!(similar(x), W_dense, x)
    end

    @testset "logdet" begin
        @test logdet(W) ≈ logdet(W_dense)
        test_ad(randn(), A, D, S) do A, D, S
            logdet(WoodburyPDMat(A, D, S))
        end
    end

    @testset "invquad" begin
        @test invquad(W, x) ≈ invquad(W_dense, x)

        test_ad(randn(), A, D, S, x) do A, D, S, x
            W = WoodburyPDMat(A, D, S)
            return invquad(W, x)
        end
    end

    @testset "*" begin
        c = 2.0
        @test c * W == W * c
        @test c * W_dense ≈ c * W atol=1e-6
        @test (c * W) isa WoodburyPDMat

        c = Diagonal(2.0 * ones(4,))
        @test c * W == W * c
        @test c * W_dense ≈ c * W atol=1e-6
        @test (c * W) isa WoodburyPDMat

        c1 = Diagonal(2.0 * ones(4,))
        c2 = Diagonal(3.0 * ones(4,))
        c_neg = Diagonal([1,2,-2,3])

        @test c2 * W * c1 == c1 * W * c2
        @test c1 * W * c2 ≈ c1 * W_dense * c2
        @test (c1 * W * c2) isa WoodburyPDMat

        @test_throws(ArgumentError, c_neg * W)
        @test_throws(ArgumentError, c_neg * W * c2)
        @test_throws(ArgumentError, c1 * W * c_neg)

    end

    @testset "MvNormal logpdf" begin
        m = randn(size(A, 1))
        @test logpdf(MvNormal(m, W), x) ≈ logpdf(MvNormal(m, Symmetric(Matrix(W))), x)

        test_ad(randn(), m, A, D, S, x) do m, A, D, S, x
            W = WoodburyPDMat(A, D, S)
            return logpdf(MvNormal(m, W), x)
        end
    end

    @testset "GenericMvTDist logpdf" begin
        α = 2.1
        m = randn(size(A, 1))
        @test isapprox(
            logpdf(Distributions.GenericMvTDist(α, m, W), x),
            logpdf(Distributions.GenericMvTDist(α, m, W_dense), x),
        )

        test_ad(randn(), α, m, A, D, S, x) do α, m, A, D, S, x
            W = WoodburyPDMat(A, D, S)
            return logpdf(Distributions.GenericMvTDist(α, m, W), x)
        end
    end
end
