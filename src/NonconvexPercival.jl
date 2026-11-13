module NonconvexPercival

export PercivalAlg, PercivalOptions, AugLag, AugLagOptions

using Percival: Percival
using NLPModelsModifiers: NLPModelsModifiers
using ADNLPModels: ADNLPModels
using NLPModels: NLPModels
using Reexport, Parameters
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractOptimizer
using NonconvexCore:
    AbstractResult, CountingFunction, getnineqconstraints, getneqconstraints
using NonconvexCore: Solution, ConvergenceState, NoCallback
import NonconvexCore: Workspace, reset!, optimize!

include("zygote_backends.jl")

struct PercivalAlg <: AbstractOptimizer end
const AugLag = PercivalAlg

function NonconvexCore.solver_capabilities(::PercivalAlg)
    return ConstraintCapabilities(;
        linear_eq=true,
        nonlinear_eq=true,
        linear_ineq=true,
        nonlinear_ineq=true,
        linear_sd=false,
        nonlinear_sd=false,
        box=true,
        continuous=true,
        binary=false,
        integer=false,
        infinite_bounds=true,
    )
end

struct PercivalOptions{T<:NamedTuple}
    nt::T
    backend::Symbol
end
# Positional constructor on the raw options NamedTuple (JET-1.13 compatibility).
PercivalOptions(nt::NamedTuple) = PercivalOptions(nt, :auto)
function PercivalOptions(; first_order=true, memory=5, backend::Symbol=:auto, kwargs...)
    backend in (:auto, :zygote) ||
        throw(ArgumentError("backend must be :auto (default) or :zygote"))
    return PercivalOptions(
        (; first_order=first_order, memory=memory, inity=ones, kwargs...), backend
    )
end
const AugLagOptions = PercivalOptions

mutable struct PercivalWorkspace{
    TM<:VecModel,
    TP<:Percival.NLPModels.AbstractNLPModel,
    TX<:AbstractVector,
    TO<:PercivalOptions,
    TC<:Base.RefValue{Int},
    TF<:Function,
} <: Workspace
    model::TM
    problem::TP
    x0::TX
    options::TO
    counter::TC
    callback::TF
end
function PercivalWorkspace(
    model::VecModel,
    x0::AbstractVector=NonconvexCore.getinit(model);
    options=PercivalOptions(),
    callback::Function=NoCallback(),
    kwargs...,
)
    problem, counter = get_percival_problem(model, copy(x0), options.backend)
    return PercivalWorkspace(model, problem, copy(x0), options, counter, callback)
end
struct PercivalResult{TM1,TM2,TP,TR,TF} <: AbstractResult
    minimizer::TM1
    minimum::TM2
    problem::TP
    result::TR
    fcalls::TF
end
const AugLagWorkspace = PercivalWorkspace

function optimize!(workspace::PercivalWorkspace)
    @unpack problem, options, x0, counter, callback = workspace
    counter[] = 0
    m = getnineqconstraints(workspace.model) + getneqconstraints(workspace.model)
    problem.meta.x0 .= x0
    inity_values = options.nt.inity(eltype(x0), m)
    inity_values::Union{Nothing,AbstractVector} = inity_values
    result = _percival(
        problem;
        options.nt...,
        inity=inity_values,
        callback=percival_callback(callback, length(x0)),
    )
    result.solution = result.solution[1:length(x0)]
    # Warm start: Percival starts from `nlp.meta.x0` (its native `reset!`
    # seeds the subproblem from the model's `meta.x0`), and `optimize!` fills
    # `problem.meta.x0` from the workspace's `x0` above. Write the minimizer
    # back into the workspace's `x0` so that the next `optimize!` call on this
    # workspace starts from the previous minimizer.
    x0 .= result.solution
    return PercivalResult(
        copy(result.solution), result.objective, problem, result, counter[]
    )
end

"""
    percival_callback(callback)

Adapts the Nonconvex `callback` to Percival's `callback(nlp, solver, stats)`
signature. The Nonconvex callback receives a `Solution` carrying the current
iterate, the objective value, and the primal/dual feasibility in the
convergence state.
"""
percival_callback(::NoCallback) = (args...) -> nothing
function percival_callback(callback::Function, nvars::Int)
    return (nlp, solver, stats) -> begin
        # Percival's own stats never receive the iterate. On the augmented
        # Lagrangian paths the subproblem solver's stats hold the latest
        # subproblem solution (which includes slack variables for
        # inequality-constrained problems); on the unconstrained path the
        # iterate lives in the subproblem solver itself. Keep only the
        # original variables.
        x_all =
            if hasproperty(solver, :sub_stats) && length(solver.sub_stats.solution) >= nvars
                solver.sub_stats.solution
            elseif hasproperty(solver, :x) && length(solver.x) >= nvars
                solver.x
            else
                stats.solution
            end
        length(x_all) >= nvars || return nothing
        x = x_all[1:nvars]
        convstate = ConvergenceState{eltype(x)}()
        convstate.kkt_residual = stats.dual_feas
        convstate.infeas = stats.primal_feas
        solution = Solution(
            copy(x),
            copy(x),
            Float64[],
            stats.objective,
            stats.objective,
            zeros(length(x)),
            Float64[],
            zeros(0, length(x)),
            convstate,
        )
        callback(solution)
    end
end

function _percival(
    nlp::NLPModels.AbstractNLPModel;
    T=eltype(nlp.meta.x0),
    μ::Real=convert(T, 10),
    max_iter::Int=1000,
    max_time::Real=Inf,
    max_eval::Int=100000,
    atol::Real=convert(T, 1e-6),
    rtol::Real=convert(T, 1e-6),
    ctol::Real=convert(T, 1e-6),
    first_order=true,
    memory=5,
    subsolver_logger::Percival.AbstractLogger=Percival.NullLogger(),
    inity::Union{Nothing,AbstractVector}=nothing,
    max_cgiter=100,
    subsolver_max_eval=200,
    callback=(args...) -> nothing,
    kwargs...,
)
    # Percival applies `subproblem_modifier` on every outer augmented Lagrangian
    # iteration, and for inequality-constrained problems it modifies a slack
    # model with more variables than the original problem. Reuse the L-BFGS
    # operator (and its curvature memory) per problem size instead of recreating
    # it on every call.
    op_cache = IdDict{Int,Any}()
    modifier =
        m -> begin
            nvar = m.meta.nvar
            op = get!(op_cache, nvar) do
                return NLPModelsModifiers.LinearOperators.LBFGSOperator(T, nvar; mem=memory)
            end
            return NLPModelsModifiers.LBFGSModel(m.meta, m, op, similar(m.meta.x0))
        end
    _kwargs1 = (
        max_iter=max_iter,
        max_time=max_time,
        max_eval=max_eval,
        atol=atol,
        rtol=rtol,
        subproblem_modifier=first_order ? modifier : identity,
        callback=callback,
    )
    # Percival 0.7 dropped the subsolver logger; only an integer verbosity
    # flag is forwarded to the subsolver.
    _verbose = subsolver_logger isa Percival.NullLogger ? 0 : 1
    if inity !== nothing
        # Percival 0.7 dropped the initial-multiplier vector argument; write
        # the user-provided multipliers into the model's y0 instead.
        nlp.meta.y0 .= inity
    end
    if Percival.unconstrained(nlp) || Percival.bound_constrained(nlp)
        return Percival.percival(
            Val(:tron), nlp; _kwargs1..., max_cgiter=max_cgiter, verbose=_verbose
        )
    elseif Percival.equality_constrained(nlp)
        _kwargs2 = merge(
            _kwargs1,
            (;
                μ=μ,
                ctol=ctol,
                subsolver_max_eval=subsolver_max_eval,
                subsolver_max_cgiter=max_cgiter,
                subsolver_verbose=_verbose,
            ),
        )
        return Percival.percival(Val(:equ), nlp; _kwargs2...)
    else # has inequalities
        _kwargs3 = merge(
            _kwargs1,
            (;
                μ=μ,
                ctol=ctol,
                subsolver_max_eval=subsolver_max_eval,
                subsolver_max_cgiter=max_cgiter,
                subsolver_verbose=_verbose,
            ),
        )
        return Percival.percival(Val(:ineq), nlp; _kwargs3...)
    end
end

function Workspace(model::VecModel, optimizer::PercivalAlg, args...; kwargs...)
    return PercivalWorkspace(model, args...; kwargs...)
end

function reset!(w::AugLagWorkspace, x0=nothing)
    w.counter[] = 0
    if x0 !== nothing
        w.x0 .= x0
    end
    return w
end

function get_percival_problem(model::VecModel, x0::AbstractVector, backend::Symbol=:auto)
    eq = if length(model.eq_constraints.fs) == 0
        nothing
    else
        model.eq_constraints
    end
    ineq = if length(model.ineq_constraints.fs) == 0
        nothing
    else
        model.ineq_constraints
    end
    obj = getobjective(model)
    obj === nothing && throw(ArgumentError("The model has no objective."))
    obj = CountingFunction(obj)
    return get_percival_problem(obj, ineq, eq, x0, getmin(model), getmax(model), backend),
    obj.counter
end
function get_percival_problem(
    obj, ineq_constr, eq_constr, x0, xlb, xub, backend::Symbol=:auto
)
    T = eltype(x0)
    nvars = length(x0)
    if ineq_constr !== nothing
        ineqval = ineq_constr(x0)
        ineq_nconstr = length(ineqval)
    else
        ineqval = T[]
        ineq_nconstr = 0
    end
    if eq_constr !== nothing
        eqval = eq_constr(x0)
        eq_nconstr = length(eqval)
    else
        eqval = T[]
        eq_nconstr = 0
    end
    c = x -> begin
        if ineq_constr !== nothing
            v1 = ineq_constr(x)
        else
            v1 = eltype(x)[]
        end
        if eq_constr !== nothing
            v2 = eq_constr(x)
        else
            v2 = eltype(x)[]
        end
        return [v1; v2]
    end
    lcon = [fill(convert(T, -Inf), ineq_nconstr); zeros(T, eq_nconstr)]
    ucon = zeros(T, ineq_nconstr + eq_nconstr)
    jump = backend === :zygote || _is_jump_converted(obj, ineq_constr, eq_constr)
    nlp = zygote_nlp_model(obj, x0, xlb, xub, c, lcon, ucon; jump)
    nlp::NLPModels.AbstractNLPModel = nlp
    return nlp
end

# A JuMP-converted model exposes its derivatives through the JuMPModel*
# wrappers' ChainRules rrules; only Zygote (which consults ChainRules) can
# pick those up. Other models keep the ReverseDiff/ForwardDiff backends,
# whose compiled tapes are faster for repeated plain-function evaluations.
function _is_jump_converted(obj, ineq_constr, eq_constr)
    raw = obj
    raw = raw isa CountingFunction ? raw.f : raw
    raw = raw isa NonconvexCore.Objective ? NonconvexCore.getfunction(raw) : raw
    raw = raw isa NonconvexCore.FlattenedFunction ? NonconvexCore.getrawfunction(raw) : raw
    raw isa NonconvexCore.JuMPModelObjective || return false
    for c in (ineq_constr === nothing ? () : ineq_constr.fs)
        r = if c.f isa NonconvexCore.FlattenedFunction
            NonconvexCore.getrawfunction(c.f)
        else
            c.f
        end
        r = r isa FunctionWrapper ? NonconvexCore.getfunction(r) : r
        r isa NonconvexCore.JuMPModelIneq || return false
    end
    for c in (eq_constr === nothing ? () : eq_constr.fs)
        r = if c.f isa NonconvexCore.FlattenedFunction
            NonconvexCore.getrawfunction(c.f)
        else
            c.f
        end
        r = r isa FunctionWrapper ? NonconvexCore.getfunction(r) : r
        r isa NonconvexCore.JuMPModelEq || return false
    end
    return true
end

function zygote_nlp_model(
    f, x0::S, lvar::S, uvar::S, c, lcon::S, ucon::S; jump::Bool=false
) where {S}
    T = eltype(S)
    y0 = fill!(similar(lcon), zero(T))
    name::String = "Generic"
    lin::AbstractVector{<:Integer} = Int[]
    minimize::Bool = true
    nvar = length(x0)
    ncon = length(lcon)
    # Vectorized fill: broadcast assignment into the Zygote Buffer is
    # supported (scalar setindex! loops force per-element pullbacks).
    c! = (cx, x) -> (cx.=c(x); cx)
    adbackend = if jump
        # JuMP-converted models: Zygote consults ChainRules and picks up the
        # analytic evaluator derivatives from the JuMPModel* rrules.
        ADNLPModels.ADModelBackend(
            nvar,
            f,
            ncon,
            c!;
            x0=x0,
            gradient_backend=ZygoteGradient,
            hprod_backend=ZygoteHvprod,
            jprod_backend=ZygoteJprod,
            jtprod_backend=ZygoteJtprod,
            jacobian_backend=ZygoteJacobian,
            hessian_backend=ZygoteHessian,
            ghjvprod_backend=ZygoteGHjvprod,
        )
    else
        # Plain models: the compiled ReverseDiff/ForwardDiff tapes are faster
        # for repeated evaluations of user functions.
        ADNLPModels.ADModelBackend(
            nvar,
            f,
            ncon,
            c!;
            x0=x0,
            gradient_backend=ADNLPModels.ReverseDiffADGradient,
            hprod_backend=ADNLPModels.ForwardDiffADHvprod,
            jprod_backend=ADNLPModels.ReverseDiffADJprod,
            jtprod_backend=ADNLPModels.ReverseDiffADJtprod,
            jacobian_backend=ADNLPModels.ForwardDiffADJacobian,
            hessian_backend=ADNLPModels.SparseReverseADHessian,
            ghjvprod_backend=ADNLPModels.ForwardDiffADGHjvprod,
        )
    end
    nnzh = div(nvar * (nvar + 1), 2)
    nnzj = nvar * ncon
    meta = ADNLPModels.NLPModelMeta{T,S}(
        nvar;
        x0=x0,
        lvar=lvar,
        uvar=uvar,
        ncon=ncon,
        y0=y0,
        lcon=lcon,
        ucon=ucon,
        nnzj=nnzj,
        nnzh=nnzh,
        lin=lin,
        minimize=minimize,
        islp=false,
        name=name,
    )
    return ADNLPModels.ADNLPModel(meta, ADNLPModels.Counters(), adbackend, f, c!)
end

end
