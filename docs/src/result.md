# Result

`optimize` returns a `PercivalResult` with the following fields:

- `minimizer`: the minimizer found by the algorithm
- `minimum`: the minimum objective function value found
- `problem`: the `NLPModels` problem built by the wrapper, which can be
  inspected for evaluation counters (e.g. `neval_hess`)
- `result`: the raw Percival result, e.g. the `GenericExecutionStats` of the
  solve (see the [`Percival` documentation](https://jso.dev/Percival.jl/stable/))
- `fcalls`: the number of objective function evaluations
