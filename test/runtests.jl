using NonconvexPercival, LinearAlgebra, Test, StableRNGs

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

alg = AugLag()

@testset "Simple constraints" begin
    m = Model(f)
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    for first_order in (true, false)
        options = AugLagOptions(first_order = first_order)
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end
end

@testset "Equality constraints" begin
    m = Model(f)
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)
    for first_order in (true, false)
        options = AugLagOptions(first_order = first_order)
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end
end

@testset "Equality constraints BigFloat" begin
    T = BigFloat
    m = Model(f)
    addvar!(m, T.([1e-4, 1e-4]), T.([10.0, 10.0]))
    add_ineq_constraint!(m, x -> g(x, T(2), T(0)))
    add_ineq_constraint!(m, x -> g(x, T(-1), T(1)))
    add_eq_constraint!(m, x -> sum(x) - T(1 / 3) - T(8 / 27))
    for first_order in (true, false)
        options = AugLagOptions(first_order = first_order)
        r = optimize(m, alg, T.([1.234, 2.345]), options = options)
        @test abs(r.minimum - sqrt(T(8 / 27))) < T(1e-6)
        @test norm(r.minimizer - T.([1 / 3, 8 / 27])) < T(1e-6)
        @test r.minimum isa T
        @test eltype(r.minimizer) <: T
    end
end

@testset "Block constraints" begin
    m = Model(f)
    addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
    add_ineq_constraint!(m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2))

    for first_order in (true, false)
        options = AugLagOptions(first_order = first_order)
        r = optimize(m, alg, [1.234, 2.345], options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end
end

@testset "Infinite bounds" begin
    @testset "Infinite upper bound" begin
        m = Model(f)
        addvar!(m, [1e-4, 1e-4], [Inf, Inf])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        for first_order in (true, false)
            options = AugLagOptions(first_order = first_order)
            r = optimize(m, alg, [1.234, 2.345], options = options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end
    end
    #=
    @testset "Infinite lower bound" begin
        m = Model(f)
        addvar!(m, [-Inf, -Inf], [10, 10])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        for first_order in (true, false)
            options = AugLagOptions(first_order = first_order)
            r = optimize(m, alg, [1.234, 2.345], options = options)
            @test abs(r.minimum - sqrt(8/27)) < 1e-6
            @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
        end
    end
    @testset "Infinite upper and lower bound" begin
        m = Model(f)
        addvar!(m, [-Inf, -Inf], [Inf, Inf])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        for first_order in (true, false)
            options = AugLagOptions(first_order = first_order)
            r = optimize(m, alg, [1.234, 2.345], options = options)
            @test abs(r.minimum - sqrt(8/27)) < 1e-6
            @test norm(r.minimizer - [1/3, 8/27]) < 1e-6
        end
    end
    =#
end

@testset "Box constrained" begin
    component_pdf(x::Real, mu::Real, tau::Real) =
        tau * exp(-tau^2 * (x - mu)^2 / 2) / sqrt(2π)
    data = randn(StableRNG(123), 40)
    loss(params::AbstractVector) =
        -sum(
            log,
            (1 - params[1]) * component_pdf(x, 0, 1) +
            params[1] * component_pdf(x, params[2], params[3]) for x in data
        )
    model = Model(loss)
    addvar!(model, [0, -Inf, 0], [1, Inf, Inf], init = [0.1, 0, 1])
    res = optimize(model, AugLag(), [0.1, -3, 1], options = AugLagOptions())
    @test res.minimizer == [0.0, -2.3140286844577305, 0.06904076799516767]
end

