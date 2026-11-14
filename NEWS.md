# NEWS

## 0.3.5

- **Zygote is now the default AD backend** (`PercivalOptions(backend=:zygote)`).
  The previous hybrid default used ReverseDiff/ForwardDiff for plain
  models, which crashes on function stacks that convert
  `ReverseDiff.TrackedReal` to `Float64` (e.g. TopOpt's assembly
  functions and `BlockComplianceFun` — the SAND and robust-RBTO models).
  `backend=:auto` restores the old hybrid for the faster compiled tapes.



## 0.2.0

- `optimize` now accepts the generic `callback` keyword. The callback runs
  once per Percival iteration through its `callback(nlp, solver, stats)`
  hook and receives a `Solution` carrying the current iterate (from the
  subproblem solver's stats, or its `x` field on the unconstrained path,
  truncated to the original variables), the objective value, and the
  primal/dual feasibility in the convergence state.

## 0.1.7

- Updated to the modern solver stack: Percival 0.7, NLPModelsModifiers 0.7/0.8
  and ADNLPModels 0.7 (previously pinned to Percival 0.5.2-era releases).
- Requires Julia 1.10+.
