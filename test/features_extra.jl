# Additional feature coverage for the exported PercivalAlg / PercivalOptions
# API and the AugLag / AugLagOptions aliases. These @testitem blocks exercise
# the PercivalOptions defaults and merging, the alias identities, the
# equality-only (`Val(:equ)`) Percival path, the Workspace API with `reset!`,
# and the fields of PercivalResult.

@testitem "features: PercivalOptions defaults and merging" begin
    using NonconvexPercival, Test

    o = PercivalOptions()
    @test o isa NonconvexPercival.PercivalOptions
    @test o.nt.first_order === true
    @test o.nt.memory == 5
    @test o.nt.inity === ones

    # Extra kwargs are merged, defaults can be overridden.
    o2 = PercivalOptions(first_order=false, memory=10, atol=1e-8, max_iter=100)
    @test o2.nt.first_order === false
    @test o2.nt.memory == 10
    @test o2.nt.atol == 1e-8
    @test o2.nt.max_iter == 100
    @test o2.nt.inity === ones
end

@testitem "features: AugLag / AugLagOptions aliases" begin
    using NonconvexPercival, Test

    @test AugLag === PercivalAlg
    @test AugLagOptions === PercivalOptions
    @test AugLag() isa PercivalAlg
    @test PercivalAlg() isa PercivalAlg
    @test AugLagOptions() isa PercivalOptions
    @test PercivalOptions() isa PercivalOptions
end

@testitem "features: PercivalAlg / PercivalOptions used directly" begin
    using NonconvexPercival, LinearAlgebra, Test, StableRNGs

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    # The same run through the canonical names instead of the aliases.
    m = Model(f)
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    r = optimize(
        m,
        PercivalAlg(),
        [1.234, 2.345];
        options=PercivalOptions(first_order=true, memory=10, atol=1e-8),
    )
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
end

@testitem "features: equality-only problem (Val(:equ) path)" begin
    using NonconvexPercival, LinearAlgebra, Test, StableRNGs

    # A model with equality constraints only routes through Percival's
    # equality-constrained augmented Lagrangian.
    m = Model(x -> sum(abs2, x))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m, x -> x[1] + x[2] - 1.0)
    r = optimize(m, AugLag(), [0.0, 0.0], options=AugLagOptions(first_order=true))
    @test abs(r.minimum - 0.5) < 1e-4
    @test norm(r.minimizer - [0.5, 0.5]) < 1e-3
end

@testitem "features: Workspace API and reset!" begin
    using NonconvexPercival, NonconvexCore, LinearAlgebra, Test, StableRNGs

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model(f)
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    vm, vx0, _ = NonconvexCore.tovecmodel(m, [1.234, 2.345])
    options = AugLagOptions(first_order=true)
    ws = NonconvexPercival.Workspace(vm, AugLag(), vx0; options=options)
    @test ws isa NonconvexPercival.PercivalWorkspace
    @test ws.options === options
    @test ws.x0 == vx0
    ws.counter[] = 12345
    r = NonconvexPercival.optimize!(ws)
    @test r isa NonconvexPercival.PercivalResult
    @test r.fcalls >= 1
    @test r.fcalls <= 12345 # the counter was reset by optimize!
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6

    # reset! clears the counter and optionally overwrites the initial point.
    ws.counter[] = 42
    NonconvexPercival.reset!(ws, [0.1, 0.1])
    @test ws.counter[] == 0
    @test ws.x0 == [0.1, 0.1]
    NonconvexPercival.reset!(ws)
    @test ws.counter[] == 0
    @test ws.x0 == [0.1, 0.1]
end

@testitem "features: PercivalResult fields" begin
    using NonconvexPercival, LinearAlgebra, Test, StableRNGs
    using NLPModels: NLPModels

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model(f)
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    r = optimize(m, AugLag(), [1.234, 2.345], options=AugLagOptions(first_order=true))
    @test r isa NonconvexPercival.PercivalResult
    @test r.minimizer isa AbstractVector
    @test length(r.minimizer) == 2
    @test r.minimum isa Real
    @test r.problem isa NLPModels.AbstractNLPModel
    @test nameof(typeof(r.result)) == :GenericExecutionStats
    @test r.fcalls isa Int
    @test r.fcalls > 0
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
end
