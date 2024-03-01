module NonconvexPercival

export PercivalAlg, PercivalOptions, AugLag, AugLagOptions

import Percival, NLPModelsModifiers, ADNLPModels
using Reexport, Parameters
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractOptimizer
using NonconvexCore: AbstractResult, CountingFunction, getnconstraints
import NonconvexCore: Workspace, reset!, optimize!

struct PercivalAlg <: AbstractOptimizer end
const AugLag = PercivalAlg

struct PercivalOptions{T<:NamedTuple}
    nt::T
end
function PercivalOptions(; first_order = true, memory = 5, kwargs...)
    return PercivalOptions((;
        first_order = first_order,
        memory = memory,
        inity = ones,
        kwargs...,
    ),)
end
const AugLagOptions = PercivalOptions

mutable struct PercivalWorkspace{
    TM<:VecModel,
    TP<:Percival.NLPModels.AbstractNLPModel,
    TX<:AbstractVector,
    TO<:PercivalOptions,
    TC<:Base.RefValue{Int},
} <: Workspace
    model::TM
    problem::TP
    x0::TX
    options::TO
    counter::TC
end
function PercivalWorkspace(
    model::VecModel,
    x0::AbstractVector = getinit(model);
    options = PercivalOptions(),
    kwargs...,
)
    problem, counter = get_percival_problem(model, copy(x0))
    return PercivalWorkspace(model, problem, copy(x0), options, counter)
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
    @unpack problem, options, x0, counter = workspace
    counter[] = 0
    m = getnconstraints(workspace.model)
    problem.meta.x0 .= x0
    result = _percival(problem; options.nt..., inity = options.nt.inity(eltype(x0), m))
    result.solution = result.solution[1:length(x0)]
    return PercivalResult(
        copy(result.solution),
        result.objective,
        problem,
        result,
        counter[],
    )
end

function _percival(
    nlp;
    T = eltype(nlp.meta.x0),
    μ::Real = convert(T, 10),
    max_iter::Int = 1000,
    max_time::Real = Inf,
    max_eval::Int = 100000,
    atol::Real = convert(T, 1e-6),
    rtol::Real = convert(T, 1e-6),
    ctol::Real = convert(T, 1e-6),
    first_order = true,
    memory = 5,
    subsolver_logger::Percival.AbstractLogger = Percival.NullLogger(),
    inity = nothing,
    max_cgiter = 100,
    subsolver_max_eval = 200,
    kwargs...,
)
    modifier =
        m -> begin
            op = NLPModelsModifiers.LinearOperators.LBFGSOperator(T, m.meta.nvar; mem = memory)
            return NLPModelsModifiers.LBFGSModel(m.meta, m, op)
        end
    _kwargs1 = (
        max_iter = max_iter,
        max_time = max_time,
        max_eval = max_eval,
        atol = atol,
        rtol = rtol,
        subsolver_logger = subsolver_logger,
        subproblem_modifier = first_order ? modifier : identity,
        subsolver_kwargs = Dict(:max_cgiter => max_cgiter),
    )
    if Percival.unconstrained(nlp) || Percival.bound_constrained(nlp)
        return Percival.percival(Val(:tron), nlp; _kwargs1...)
    elseif Percival.equality_constrained(nlp)
        _kwargs2 = merge(_kwargs1, (; subsolver_max_eval = subsolver_max_eval))
        return Percival.percival(Val(:equ), nlp; inity = inity, _kwargs2...)
    else # has inequalities
        _kwargs3 = merge(_kwargs1, (; subsolver_max_eval = subsolver_max_eval))
        return Percival.percival(Val(:ineq), nlp; inity = inity, _kwargs3...)
    end
end

function Workspace(model::VecModel, optimizer::PercivalAlg, args...; kwargs...)
    return PercivalWorkspace(model, args...; kwargs...)
end

function reset!(w::AugLagWorkspace, x0 = nothing)
    w.counter[] = 0
    if x0 !== nothing
        w.x0 .= x0
    end
    return w
end

function get_percival_problem(model::VecModel, x0::AbstractVector)
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
    obj = CountingFunction(getobjective(model))
    return get_percival_problem(obj, ineq, eq, x0, getmin(model), getmax(model)),
    obj.counter
end
function get_percival_problem(obj, ineq_constr, eq_constr, x0, xlb, xub)
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
    nlp = zygote_nlp_model(obj, x0, xlb, xub, c, lcon, ucon)
    return nlp
end

function zygote_nlp_model(f, x0::S, lvar::S, uvar::S, c, lcon::S, ucon::S) where {S}
    T = eltype(S)
    y0 = fill!(similar(lcon), zero(T))
    name::String = "Generic"
    lin::AbstractVector{<:Integer} = Int[]
    minimize::Bool = true
    nvar = length(x0)
    ncon = length(lcon)
    adbackend = ADNLPModels.ADModelBackend(
        nvar,
        f,
        ncon;
        gradient_backend = ADNLPModels.ZygoteADGradient,
        hprod_backend = ADNLPModels.ForwardDiffADHvprod,
        jprod_backend = ADNLPModels.ZygoteADJprod,
        jtprod_backend = ADNLPModels.ZygoteADJtprod,
        jacobian_backend = ADNLPModels.ZygoteADJacobian,
        hessian_backend = ADNLPModels.ZygoteADHessian,
        ghjvprod_backend = ADNLPModels.ForwardDiffADGHjvprod,
    )
    nnzh = nvar * (nvar + 1) / 2
    nnzj = nvar * ncon
    meta = ADNLPModels.NLPModelMeta{T,S}(
        nvar,
        x0 = x0,
        lvar = lvar,
        uvar = uvar,
        ncon = ncon,
        y0 = y0,
        lcon = lcon,
        ucon = ucon,
        nnzj = nnzj,
        nnzh = nnzh,
        lin = lin,
        minimize = minimize,
        islp = false,
        name = name,
    )
    return ADNLPModels.ADNLPModel(meta, ADNLPModels.Counters(), adbackend, f, c)
end

end
