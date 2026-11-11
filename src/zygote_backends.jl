# Zygote-based AD backends for ADNLPModels.
#
# ADNLPModels ships equivalent backends (`ZygoteADGradient`, ...) behind a
# `Requires` hook that does not fire on Julia >= 1.12, so the small set is
# defined here instead. Zygote consults ChainRules, so NonconvexCore models
# converted from JuMP models get their analytic derivatives from the
# MathOptInterface Nonlinear evaluator through the `JuMPModel*` rrules.
#
# The second-order backends (Hessian, Hessian-vector and directional second
# derivative) differentiate the gradient through Zygote: exact for the
# closed-form (affine/quadratic) channel of converted models, and only
# exercised by `first_order=false` runs.

using ADNLPModels: ADNLPModels, ADBackend, ImmutableADbackend
using ForwardDiff: ForwardDiff
using Zygote: Zygote

struct ZygoteGradient <: ADBackend end
ZygoteGradient(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteGradient()
function ADNLPModels.gradient(::ZygoteGradient, f, x)
    g = Zygote.gradient(f, x)[1]
    return g === nothing ? zero(x) : g
end
function ADNLPModels.gradient!(::ZygoteGradient, g, f, x)
    _g = Zygote.gradient(f, x)[1]
    g .= _g === nothing ? 0 : _g
    return g
end

struct ZygoteJacobian <: ImmutableADbackend
    nnzj::Int
end
function ZygoteJacobian(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...)
    return ZygoteJacobian(nvar * ncon)
end
ADNLPModels.jacobian(::ZygoteJacobian, c, x) = Zygote.jacobian(c, x)[1]
struct ZygoteJprod <: ImmutableADbackend end
ZygoteJprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteJprod()
function ADNLPModels.Jprod!(::ZygoteJprod, Jv, c, x, v, ::Val)
    Jv .= vec(Zygote.jacobian(t -> c(x + t * v), 0)[1])
    return Jv
end

struct ZygoteJtprod <: ImmutableADbackend end
ZygoteJtprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteJtprod()
function ADNLPModels.Jtprod!(::ZygoteJtprod, Jtv, c, x, v, ::Val)
    g = Zygote.gradient(x -> dot(c(x), v), x)[1]
    g === nothing ? (Jtv .= zero(x)) : (Jtv .= g)
    return Jtv
end

struct ZygoteHessian <: ImmutableADbackend
    nnzh::Int
end
function ZygoteHessian(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...)
    return ZygoteHessian(div(nvar * (nvar + 1), 2))
end
function ADNLPModels.hessian(::ZygoteHessian, f, x)
    # Second order through the ChainRules-first gradient (Zygote has no
    # direct Hessian entry point).
    return Matrix(Zygote.jacobian(t -> Zygote.gradient(f, t)[1], x)[1])
end

struct ZygoteHvprod <: ADBackend end
ZygoteHvprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteHvprod()
# Hessian-vector products through the Lagrangian ADNLPModels constructs:
# its Lagrangian closure mutates a plain array, which Zygote cannot trace, so
# compute the directional derivative of the gradient with forward mode. Dual
# numbers flow through plain user functions and through the closed-form
# (affine/quadratic) channel of JuMP-converted models.
function _Hv!(Hv, ℓ, x, v)
    Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x .+ t .* v), 0)
    return Hv
end
function ADNLPModels.Hvprod!(::ZygoteHvprod, Hv, x, v, ℓ, ::Val{:obj}, obj_weight)
    return _Hv!(Hv, ℓ, x, v)
end
function ADNLPModels.Hvprod!(::ZygoteHvprod, Hv, x, v, ℓ, ::Val{:lag}, y, obj_weight)
    return _Hv!(Hv, ℓ, x, v)
end
ADNLPModels.Hvprod!(::ZygoteHvprod, Hv, x, v, ℓ, ::Val{:ci}) = _Hv!(Hv, ℓ, x, v)

struct ZygoteGHjvprod <: ImmutableADbackend end
ZygoteGHjvprod(nvar, f, ncon=0, c!::Function=(args...) -> []; kwargs...) = ZygoteGHjvprod()
function ADNLPModels.directional_second_derivative(::ZygoteGHjvprod, f, x, v, w)
    # w'∇²f(x) v for the scalar function f.
    return Zygote.gradient(t -> dot(Zygote.gradient(f, x + t * v)[1], w), 0)[1]
end

# The default `get_c` mutates a plain array, which Zygote cannot trace; the
# immutable backends receive a Buffer-based out-of-place constraint closure
# instead (mirroring ADNLPModels' own Zygote integration).
function ADNLPModels.get_c(
    nlp::ADNLPModels.ADModel,
    ::Union{
        ZygoteJacobian,ZygoteJprod,ZygoteJtprod,ZygoteHessian,ZygoteHvprod,ZygoteGHjvprod
    },
)
    function c(x; nnln=nlp.meta.nnln)
        c = Zygote.Buffer(x, nnln)
        nlp.c!(c, x)
        return copy(c)
    end
    return c
end
