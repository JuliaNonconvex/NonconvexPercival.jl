# Quick start

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
