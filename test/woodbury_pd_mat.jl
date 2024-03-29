@testset "woodbury_pd_mat" begin
    A = randn(4, 2)
    D = Diagonal(randn(2).^2 .+ 1)
    S = Diagonal(randn(4).^2 .+ 1)
    σ = Diagonal(rand(4,))
    x = randn(size(A, 1))

    W = WoodburyPDMat(A, D, S)
    W_dense = PDMat(Symmetric(Matrix(W)))

    @testset "Valid constructors" begin
        @test W == typeof(W)(W.A, W.D, W.S)
    end

    @testset "invalid constructors error" begin
        @test_throws ArgumentError WoodburyPDMat(randn(5, 2), D, S)
        @test_throws ArgumentError WoodburyPDMat(randn(4, 3), D, S)
        @test_throws ArgumentError WoodburyPDMat(A, Diagonal(.-randn(2).^2), S)
        @test_throws ArgumentError WoodburyPDMat(A, D, Diagonal(.-randn(4).^2))
    end

    @testset "Basic functionality" begin
        # Checks getindex works.
        @test all(isapprox.(W, W_dense))

        for i in 1:length(W)
            @test W[i] ≈ W_dense[i]
            @test W[1:i] ≈ W_dense[1:i]
        end

        for i in size(W, 1)
            @test W[i, :] ≈ W_dense[i, :]
            @test W[1:i, :] ≈ W_dense[1:i, :]
        end
        for i in size(W, 2)
            @test W[:, i] ≈ W_dense[:, i]
            @test W[:, 1:i] ≈ W_dense[:, 1:i]
        end

        # Test particular case of two ranges.
        @test W[2:3, 1:2] ≈ W_dense[2:3, 1:2]

        @test W[:] ≈ W_dense[:]
        @test collect(W) ≈ collect(W_dense)

        @test size(W) == size(W_dense)
        @test size(W, 1) == size(W_dense, 1)
        @test size(W, 2) == size(W_dense, 2)
    end

    @testset "unwhiten!" begin
        @test PDMats.unwhiten!(similar(x), W, x) ≈ PDMats.unwhiten!(similar(x), W_dense, x)
        
    end

    @testset "whiten" begin
        @test PDMats.whiten(W, x) ≈ PDMats.whiten(W_dense, x)
        @test PDMats.whiten!(similar(x), W, x) ≈ PDMats.whiten!(similar(x), W_dense, x)
        @test PDMats.whiten!(W, x) ≈ PDMats.whiten!(W_dense, x)
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
    end

    @testset "$f X::Diagonal" for f in [Xt_A_X, X_A_Xt]
        @test f(W, σ) ≈ σ * W_dense * σ atol=1e-6
        @test f(W, σ) ≈ σ * W * σ atol=1e-6
        @test Matrix(f(W, σ)) ≈ f(W_dense, σ) atol=1e-6
        @test f(W, σ) isa WoodburyPDMat
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
