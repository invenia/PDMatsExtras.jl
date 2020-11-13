# PDMatsExtras

<!--
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/PDMatsExtras.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/PDMatsExtras.jl/dev)
-->
[![Build Status](https://travis-ci.com/invenia/PDMatsExtras.jl.svg?branch=master)](https://travis-ci.com/invenia/PDMatsExtras.jl)
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
