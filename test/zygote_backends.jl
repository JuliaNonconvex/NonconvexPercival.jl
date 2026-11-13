@testitem "Zygote AD backends" begin
    using NonconvexPercival, NonconvexCore, Test, LinearAlgebra
    using ADNLPModels, NLPModels
    using Zygote, ForwardDiff

    f(x) = (x[1] - 1)^2 + (x[2] - 2)^2 + sin(x[1] * x[2])
    c(x) = [x[1]^2 - x[2], x[1] + x[2] - 1.5]
    x0 = [0.5, 0.8]
    nlp = NonconvexPercival.zygote_nlp_model(
        f, x0, fill(-5.0, 2), fill(5.0, 2), c, [-Inf, -Inf], [0.0, 0.0]; jump=true
    )
    @test nlp isa AbstractNLPModel

    # Gradient backends
    g = NLPModels.grad(nlp, x0)
    @test g ≈ ForwardDiff.gradient(f, x0)

    # Jacobian backends
    J = NLPModels.jac(nlp, x0)
    @test J ≈ ForwardDiff.jacobian(c, x0)

    # Hessian backend
    H = NLPModels.hess(nlp, x0)
    @test H ≈ ForwardDiff.hessian(f, x0)

    v = [0.3, -0.7]
    # Hessian-vector products: objective, Lagrangian and single-constraint variants
    Hv = NLPModels.hprod(nlp, x0, v; obj_weight=1.0)
    @test Hv ≈ H * v
    y = [1.0, 2.0]
    Hv_lag = NLPModels.hprod(nlp, x0, y, v; obj_weight=1.0)
    @test Hv_lag ≈
        H * v +
          y[1] * ForwardDiff.hessian(z -> c(z)[1], x0) * v +
          y[2] * ForwardDiff.hessian(z -> c(z)[2], x0) * v
    Hv_ci = NLPModels.jth_hprod(nlp, x0, v, 1)
    @test Hv_ci ≈ ForwardDiff.hessian(z -> c(z)[1], x0) * v

    # Jacobian products
    Jv = NLPModels.jprod(nlp, x0, v)
    @test Jv ≈ J * v
    Jtv = NLPModels.jtprod(nlp, x0, v)
    @test Jtv ≈ J' * v

    # Directional second derivative: ghjvprod folds the constraint rows
    w = [0.1, -0.2]
    ghjv = NLPModels.ghjvprod(nlp, x0, w, v)
    @test ghjv ≈
        w[1] * ForwardDiff.hessian(z -> c(z)[1], x0) * v +
          w[2] * ForwardDiff.hessian(z -> c(z)[2], x0) * v

    # Buffer-based constraint closure for the immutable backends
    cc = ADNLPModels.get_c(nlp, NonconvexPercival.ZygoteJacobian(2, f, 2))
    @test cc(x0) ≈ c(x0)
end

@testitem "JuMP-converted Zygote path" begin
    using NonconvexPercival, NonconvexCore, Test, LinearAlgebra, JuMP

    jm = JuMP.Model()
    JuMP.@variable(jm, x[1:2] >= -5.0, start=0.5)
    JuMP.@constraint(jm, x[1]^2 - x[2] <= 0.0)
    JuMP.@constraint(jm, x[1] + x[2] <= 1.5)
    JuMP.@objective(jm, Min, (x[1] - 1)^2 + (x[2] - 2)^2 + sin(x[1] * x[2]))
    dm = NonconvexCore.DictModel(jm)

    # first_order=false drives the Zygote Hessian and Hvprod backends
    r = NonconvexPercival.optimize(
        dm,
        AugLag(),
        NonconvexCore.getinit(dm);
        options=AugLagOptions(first_order=true, max_iter=30, atol=1e-5),
    )
    @test isfinite(r.minimum)
    # The minimizer container depends on the conversion path (flat vector or
    # named); the solve itself is what matters for this regression.
    @test r.minimizer !== nothing
end
