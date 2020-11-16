# PDMatsExtras

<!--
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/PDMatsExtras.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/PDMatsExtras.jl/dev)
-->
[![Build Status](https://github.com/invenia/PDMatsExtras.jl/workflows/CI/badge.svg)](https://github.com/invenia/PDMatsExtras.jl/actions)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)


This is a package for extra Positive (Semi-) Definated Matrix types.
It is an extension to [PDMats.jl](https://github.com/JuliaStats/PDMats.jl).

It probably wouldn't exist, except Distributions.jl is currently very tied to the idea that the type of a covariance matrix should subtype `AbstractPDMat`.
There is [an issue open to change that](https://github.com/JuliaStats/Distributions.jl/issues/1219).
When that is resolve the matrix defined here may well move elsewhere, or cease to be required.

## The Matrixes

### PSDMat
A Positive Semi-Definite Matrix.
It still subtypes `AbstractPDMat`.
It's not quite as nice to work with as a truely positive definite matrix, since the math doesn't work out so well.
But this is able to represent all covariences -- which must be positive *semi*-definate.
You might not like the consequences,

```julia
julia> using LinearAlgebra, PDMatsExtras

julia> X = Float64[
               10   -9   2   4
               -9    8  -1  -4
                2   -1   1   1
                4   -4   1   6
           ];

julia> isposdef(X)
false

julia> PSDMat(X)
4×4 PSDMat{Float64, Matrix{Float64}}:
 10.0  -9.0   2.0   4.0
 -9.0   8.0  -1.0  -4.0
  2.0  -1.0   1.0   1.0
  4.0  -4.0   1.0   6.0


julia> # can also construct from a pivoted cholesky, even one of a rank deficient matrix (like this one)

julia> PSDMat(cholesky(X, Val(true); check=false))
4×4 PSDMat{Float64, Matrix{Float64}}:
 10.0  -9.0       2.0   4.0
 -9.0   9.26923  -1.0  -4.0
  2.0  -1.0       1.0   1.0
  4.0  -4.0       1.0   6.0
```

## WoodburyPDMat
It is a positive definite Woodbury matrix.
This is a special case of the Symmetric Woodbury Matrix (see [WoodburyMatrices.jl's](https://github.com/timholy/WoodburyMatrices.jl/) `SymWoodbury` type) which is given by `A*D*A' + S` for `S` and `D` being diagonal,
which has the additional requirement that the diagonal matrices are also non-negative.

```julia
julia> using LinearAlgebra, PDMatsExtras

julia> A = Float64[
         2.0  2.0  -8.0   5.0  -1.0   2.0   6.0
         2.0  7.0  -1.0  -5.0  -4.0   8.0   7.0
        -2.0  9.0  -9.0  -5.0   9.0  -5.0  -3.0
         3.0  4.0  -6.0  -4.0   3.0  -3.0  -3.0
       ];

julia> D = Diagonal(Float64[1, 2, 3, 2, 2, 1, 5]);

julia> S = Diagonal(Float64[4, 2, 3, 6]);

julia> W = WoodburyPDMat(A, D, S)
4×4 WoodburyPDMat{Float64,Array{Float64,2},Diagonal{Float64,Array{Float64,1}},Diagonal{Float64,Array{Float64,1}}}:
 444.0  240.0   80.0   24.0
 240.0  498.0  -18.0  -33.0
  80.0  -18.0  694.0  382.0
  24.0  -33.0  382.0  259.0

julia> A*D*A' + S
4×4 Array{Float64,2}:
 444.0  240.0   80.0   24.0
 240.0  498.0  -18.0  -33.0
  80.0  -18.0  694.0  382.0
  24.0  -33.0  382.0  259.0
```
