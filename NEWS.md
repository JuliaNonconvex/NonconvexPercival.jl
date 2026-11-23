# NEWS

## 0.3.10

- Docs: the logo now appears only in the sidebar, not in the page body.
## 0.3.9

- Docs: package-name section titles and left-aligned logos (no code changes).
## 0.3.8

- JuMP-converted models get exact derivatives by default: the `jump` backend
  path passes the shared `JuMPEvaluator` through to NonconvexCore's
  ADNLPModels extension, whose backends read the Jacobian and the Lagrangian
  Hessian straight from the MOI evaluator (Float64-safe). No local AD code.

## 0.3.7

- The AD backends now come from NonconvexCore's ADNLPModels compatibility
  extension (requires NonconvexCore 2.2.6); the package no longer defines
  its own AD code.
- The second-order backends use forward-over-reverse, fixing the previous
  nested-Zygote Hessian that could silently return wrong values.
- `inity` accepts an initial-multiplier vector (length-validated).
- The `-Inf`-bounds testitems are enabled (the declared `infinite_bounds`
  capability is now exercised), and the Box-constrained minimizer is
  asserted approximately (the last ULP varies with the AD backend).

## 0.3.6

- Added docstrings for `PercivalAlg`, `PercivalOptions` and the `AugLag`/`AugLagOptions` aliases (REPL help and API reference).


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
