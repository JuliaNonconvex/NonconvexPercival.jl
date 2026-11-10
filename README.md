<p align="center">
  <img src="assets/logo.png" width="400" alt="NonconvexPercival.jl logo">
</p>

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexPercival.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexPercival.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexPercival.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexPercival.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexPercival.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexPercival.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

# NonconvexPercival.jl

Augmented Lagrangian with Percival.jl

First and second order augmented Lagrangian algorithms for equality- and inequality-constrained problems.

NonconvexPercival.jl is part of the [JuliaNonconvex](https://github.com/JuliaNonconvex) ecosystem built on [NonconvexCore.jl](https://github.com/JuliaNonconvex/NonconvexCore.jl).

## Quick start

```julia
using NonconvexPercival

alg = AugLag()
options = AugLagOptions()
result = optimize(model, alg, x0, options = options)
```

## Documentation

The full documentation is available at [https://julianonconvex.github.io/NonconvexPercival.jl/stable](https://julianonconvex.github.io/NonconvexPercival.jl/stable) (stable) and [https://julianonconvex.github.io/NonconvexPercival.jl/dev](https://julianonconvex.github.io/NonconvexPercival.jl/dev) (development).

## Installation

```julia
using Pkg
Pkg.add("NonconvexPercival")
```
