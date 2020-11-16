function test_ad(test_function, Δoutput, inputs...; atol=1e-7, rtol=1e-7)

    # Verify that the forwards-pass produces the correct answer.
    output, pb = Zygote.pullback(test_function, inputs...)
    @test output ≈ test_function(inputs...)

    # Compute the adjoints using AD and FiniteDifferences.
    dW_ad = pb(Δoutput)
    dW_fd = FiniteDifferences.j′vp(central_fdm(5, 1), test_function, Δoutput, inputs...)

    # Compare AD and FiniteDifferences results.
    @testset "$(typeof(test_function)) argument $n" for n in eachindex(inputs)
        @test dW_ad[n] ≈ dW_fd[n] atol=atol rtol=rtol
    end
end
