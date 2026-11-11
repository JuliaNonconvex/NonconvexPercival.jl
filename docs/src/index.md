# Augmented Lagrangian algorithm in pure Julia

```@raw html
<p align="center">
  <img src="https://raw.githubusercontent.com/JuliaNonconvex/NonconvexPercival.jl/master/docs/src/assets/logo.png" alt="JuliaNonconvex logo" width="180"/>
</p>
```



## Description

[Percival.jl](https://github.com/JuliaSmoothOptimizers/Percival.jl) is a pure Julia implementation of the augmented Lagrangian algorithm. Both first and second order versions of the algorithm are available.

## Quick start

Given a model `model` and an initial solution `x0`, the following can be used to optimize the model using Percival.
```julia
using NonconvexPercival

alg = AugLag()
options = AugLagOptions()
result = optimize(model, alg, x0, options = options)
```
Percival is an optional dependency of Nonconvex so you need to import it in order to use it.

## Construct an instance

To construct an instance of the Percival algorithm, use:
```julia
alg = AugLag()
```
`AugLag` and `AugLagOptions` are aliases of `PercivalAlg` and
`PercivalOptions`, respectively, so `PercivalAlg()` and `PercivalOptions(...)`
work as well:
```julia
alg = PercivalAlg()
options = PercivalOptions(first_order = true)
```

## Options

The options keyword argument to the `optimize` function shown above must be an instance of the `AugLagOptions` struct when the algorihm is an `AugLag`. To specify options use keyword arguments in the constructor of `AugLagOptions`, e.g:
```julia
options = AugLagOptions(first_order = false, rtol = 1e-4)
```
The most important option is `first_order` which is `true` by default. When `first_order` is `true`, the first order augmented Lagrangian algorithm will be used. And when it is `false`, the second order augmented Lagrangian algorithm will be used. Other arguments include:
- `atol`: absolute tolerance in the subproblem optimizer
- `rtol`: relative tolerance in the subproblem optimizer
- `ctol`: absolute feasibility tolerance
- `max_iter`: maximum number of iterations
- `max_time`: maximum time in seconds
- `max_eval`: maximum number of function evaluations
- `memory`: the curvature memory of the L-BFGS approximation used by the
  first-order subproblems (default `5`)
- `inity`: a function `(T, nconstraints) -> vector` returning the initial
  Lagrange multipliers (default `ones`)
- `max_cgiter`: maximum number of conjugate-gradient iterations per
  subproblem (default `100`)
- `subsolver_max_eval`: maximum number of evaluations of the subproblem
  solver (default `200`)
- `subsolver_logger`: a Percival logger for the subproblem solver
  (default `Percival.NullLogger()`)

When using the first order augmented Lagrangian and a block constraint (i.e. a constraint function that returns a vector), the use of reverse-mode AD will only require calling the adjoint operator of the block constraint function in order to compute the gradient of the augmented Lagrangian. This is particularly suitable for constraint functions whose Jacobians are expensive but the adjoint operator is relatively inexpensive.

## Result

`optimize` returns a `PercivalResult` with the following fields:

- `minimizer`: the minimizer found by the algorithm
- `minimum`: the minimum objective function value found
- `problem`: the `NLPModels` problem built by the wrapper, which can be
  inspected for evaluation counters (e.g. `neval_hess`)
- `result`: the raw Percival result, e.g. the `GenericExecutionStats` of the
  solve (see the [`Percival` documentation](https://jso.dev/Percival.jl/stable/))
- `fcalls`: the number of objective function evaluations
