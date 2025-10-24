module ForwardDiffTests

using Test

using FastRicianLikelihoods: FastRicianLikelihoods, ChainRulesCore, ForwardDiff, StaticArrays
using FastRicianLikelihoods: @define_unary_dual_scalar_rule, @define_binary_dual_scalar_rule, @define_ternary_dual_scalar_rule
using FastRicianLikelihoods: @uniform_dual_rule_from_frule, @dual_rule_from_frule
using FiniteDifferences: FiniteDifferences
using .StaticArrays: SVector

#### ForwardDiff gradient and Jacobian utilities

@testset "withgradient" begin
    f_grad(x) = sum(abs2, x) + prod(x)
    _x = SVector(rand(3)...)
    y_fd, ∇y_fd = FastRicianLikelihoods.withgradient(f_grad, _x)
    y_ref = f_grad(_x)
    ∇y_ref = ForwardDiff.gradient(f_grad, _x)
    @test y_fd ≈ y_ref
    @test ∇y_fd ≈ ∇y_ref
end

@testset "withjacobian" begin
    f_jac(x) = SVector(x[1]^2 + x[2], x[1] * x[2], sin(x[3]))
    _x = SVector(rand(3)...)
    y_fd, J_fd = FastRicianLikelihoods.withjacobian(f_jac, _x)
    y_ref = f_jac(_x)
    J_ref = ForwardDiff.jacobian(f_jac, _x)
    @test y_fd ≈ y_ref
    @test J_fd ≈ J_ref
end

#### Defining dual rules for scalar functions

f1_unary(x) = exp(x * cos(x))
∇f1_unary(x) = ((cos(x) - x * sin(x)) * exp(x * cos(x)),)
@define_unary_dual_scalar_rule f1_unary ∇f1_unary

f2_unary(x) = f1_unary(x)
f2_∇f2_unary(x) = (f1_unary(x), ∇f1_unary(x))
@define_unary_dual_scalar_rule fused = true f2_unary f2_∇f2_unary

f3_unary(x) = f1_unary(x)
f3_∇f3_unary(x) = (f1_unary(x), only(∇f1_unary(x)))
@define_unary_dual_scalar_rule fused = true f3_unary f3_∇f3_unary

f4_binary(x, y) = exp(x * (y - cos(x * y)))
∇f4_binary(x, y) = (x * y * sin(x * y) - cos(x * y) + y, x^2 * sin(x * y) + x) .* exp(x * (y - cos(x * y)))
@define_binary_dual_scalar_rule f4_binary ∇f4_binary

f5_binary(x, y) = f4_binary(x, y)
f5_∇f5_binary(x, y) = (f4_binary(x, y), ∇f4_binary(x, y))
@define_binary_dual_scalar_rule fused = true f5_binary f5_∇f5_binary

f6_ternary(x, y, z) = exp(x * z + cos(y - z))
∇f6_ternary(x, y, z) = (z, -sin(y - z), (x + sin(y - z))) .* exp(x * z + cos(y - z))
@define_ternary_dual_scalar_rule f6_ternary ∇f6_ternary

f7_ternary(x, y, z) = f6_ternary(x, y, z)
f7_∇f7_ternary(x, y, z) = (f6_ternary(x, y, z), ∇f6_ternary(x, y, z))
@define_ternary_dual_scalar_rule fused = true f7_ternary f7_∇f7_ternary

f8_frule_unary(x) = exp(x * cos(x))
function ChainRulesCore.frule((_, Δx), ::typeof(f8_frule_unary), x)
    y = f8_frule_unary(x)
    Δy = (cos(x) - x * sin(x)) * y * Δx
    return y, Δy
end
@uniform_dual_rule_from_frule f8_frule_unary

f9_frule_binary(x, y) = exp(x * (y - cos(x * y)))
function ChainRulesCore.frule((_, Δx, Δy), ::typeof(f9_frule_binary), x, y)
    z = f9_frule_binary(x, y)
    ∂x = (x * y * sin(x * y) - cos(x * y) + y) * z
    ∂y = (x^2 * sin(x * y) + x) * z
    Δz = ∂x * Δx + ∂y * Δy
    return z, Δz
end
@uniform_dual_rule_from_frule f9_frule_binary

f10_frule_ternary(x, y, z) = exp(x * z + cos(y - z))
function ChainRulesCore.frule((_, Δx, Δy, Δz), ::typeof(f10_frule_ternary), x, y, z)
    w = f10_frule_ternary(x, y, z)
    ∂x = z * w
    ∂y = -sin(y - z) * w
    ∂z = (x + sin(y - z)) * w
    Δw = ∂x * Δx + ∂y * Δy + ∂z * Δz
    return w, Δw
end
@uniform_dual_rule_from_frule f10_frule_ternary

f11_dual_rule_unary(x) = exp(x * cos(x))
function ChainRulesCore.frule((_, Δx), ::typeof(f11_dual_rule_unary), x)
    y = f11_dual_rule_unary(x)
    Δy = (cos(x) - x * sin(x)) * y * Δx
    return y, Δy
end
@dual_rule_from_frule f11_dual_rule_unary(x)

f12_dual_rule_binary(x, y) = exp(x * (y - cos(x * y)))
function ChainRulesCore.frule((_, Δx, Δy), ::typeof(f12_dual_rule_binary), x, y)
    z = f12_dual_rule_binary(x, y)
    ∂x = (x * y * sin(x * y) - cos(x * y) + y) * z
    ∂y = (x^2 * sin(x * y) + x) * z
    Δz = ∂x * Δx + ∂y * Δy
    return z, Δz
end
@dual_rule_from_frule f12_dual_rule_binary(x, y)

f13_dual_rule_ternary(x, y, z) = exp(x * z + cos(y - z))
function ChainRulesCore.frule((_, Δx, Δy, Δz), ::typeof(f13_dual_rule_ternary), x, y, z)
    w = f13_dual_rule_ternary(x, y, z)
    ∂x = z * w
    ∂y = -sin(y - z) * w
    ∂z = (x + sin(y - z)) * w
    Δw = ∂x * Δx + ∂y * Δy + ∂z * Δz
    return w, Δw
end
@dual_rule_from_frule f13_dual_rule_ternary(x, y, z)

f14_dual_rule_binary_mixed(x, y) = exp(x * (y - cos(x * y)))
function ChainRulesCore.frule((_, Δx, _), ::typeof(f14_dual_rule_binary_mixed), x, y)
    z = f14_dual_rule_binary_mixed(x, y)
    ∂x = (x * y * sin(x * y) - cos(x * y) + y) * z
    Δz = ∂x * Δx
    return z, Δz
end
@dual_rule_from_frule f14_dual_rule_binary_mixed(x, !(y::Real))

const fdm = FiniteDifferences.central_fdm(4, 1)

@testset "unary" begin
    unary1(x) = sin(1 + f1_unary(x)^2)
    unary2(x) = sin(1 + f2_unary(x)^2)
    unary3(x) = sin(1 + f3_unary(x)^2)
    map((unary1, unary2, unary3)) do f
        _x = rand()
        @test ForwardDiff.derivative(f, _x) ≈ fdm(f, _x)
    end
end

@testset "binary" begin
    binary1(x, y) = sin(1 + f4_binary(x, y)^2)
    binary2(x, y) = sin(1 + f5_binary(x, y)^2)
    map((binary1, binary2)) do f
        _x, _y = rand(2)
        @test ForwardDiff.derivative(x -> f(x, _y), _x) ≈ fdm(x -> f(x, _y), _x)
        @test ForwardDiff.derivative(y -> f(_x, y), _y) ≈ fdm(y -> f(_x, y), _y)
        @test ForwardDiff.gradient(((x, y),) -> f(x, y), [_x, _y]) ≈ FiniteDifferences.grad(fdm, ((x, y),) -> f(x, y), [_x, _y])[1]
    end
end

@testset "ternary" begin
    ternary1(x, y, z) = sin(1 + f6_ternary(x, y, z)^2)
    ternary2(x, y, z) = sin(1 + f7_ternary(x, y, z)^2)
    map((ternary1, ternary2)) do f
        _x, _y, _z = rand(3)
        @test ForwardDiff.derivative(x -> f(x, _y, _z), _x) ≈ fdm(x -> f(x, _y, _z), _x)
        @test ForwardDiff.derivative(y -> f(_x, y, _z), _y) ≈ fdm(y -> f(_x, y, _z), _y)
        @test ForwardDiff.derivative(z -> f(_x, _y, z), _z) ≈ fdm(z -> f(_x, _y, z), _z)
        @test ForwardDiff.gradient(((x, y),) -> f(x, y, _z), [_x, _y]) ≈ FiniteDifferences.grad(fdm, ((x, y),) -> f(x, y, _z), [_x, _y])[1]
        @test ForwardDiff.gradient(((x, z),) -> f(x, _y, z), [_x, _z]) ≈ FiniteDifferences.grad(fdm, ((x, z),) -> f(x, _y, z), [_x, _z])[1]
        @test ForwardDiff.gradient(((y, z),) -> f(_x, y, z), [_y, _z]) ≈ FiniteDifferences.grad(fdm, ((y, z),) -> f(_x, y, z), [_y, _z])[1]
        @test ForwardDiff.gradient(((x, y, z),) -> f(x, y, z), [_x, _y, _z]) ≈ FiniteDifferences.grad(fdm, ((x, y, z),) -> f(x, y, z), [_x, _y, _z])[1]
    end
end

@testset "uniform_dual_rule_from_frule unary" begin
    frule_unary(x) = sin(1 + f8_frule_unary(x)^2)
    _x = rand()
    @test ForwardDiff.derivative(frule_unary, _x) ≈ fdm(frule_unary, _x)
end

@testset "uniform_dual_rule_from_frule binary" begin
    frule_binary(x, y) = sin(1 + f9_frule_binary(x, y)^2)
    _x, _y = rand(2)
    @test ForwardDiff.derivative(x -> frule_binary(x, _y), _x) ≈ fdm(x -> frule_binary(x, _y), _x)
    @test ForwardDiff.derivative(y -> frule_binary(_x, y), _y) ≈ fdm(y -> frule_binary(_x, y), _y)
    @test ForwardDiff.gradient(((x, y),) -> frule_binary(x, y), [_x, _y]) ≈ FiniteDifferences.grad(fdm, ((x, y),) -> frule_binary(x, y), [_x, _y])[1]
end

@testset "uniform_dual_rule_from_frule ternary" begin
    frule_ternary(x, y, z) = sin(1 + f10_frule_ternary(x, y, z)^2)
    _x, _y, _z = rand(3)
    @test ForwardDiff.derivative(x -> frule_ternary(x, _y, _z), _x) ≈ fdm(x -> frule_ternary(x, _y, _z), _x)
    @test ForwardDiff.derivative(y -> frule_ternary(_x, y, _z), _y) ≈ fdm(y -> frule_ternary(_x, y, _z), _y)
    @test ForwardDiff.derivative(z -> frule_ternary(_x, _y, z), _z) ≈ fdm(z -> frule_ternary(_x, _y, z), _z)
    @test ForwardDiff.gradient(((x, y),) -> frule_ternary(x, y, _z), [_x, _y]) ≈ FiniteDifferences.grad(fdm, ((x, y),) -> frule_ternary(x, y, _z), [_x, _y])[1]
    @test ForwardDiff.gradient(((x, z),) -> frule_ternary(x, _y, z), [_x, _z]) ≈ FiniteDifferences.grad(fdm, ((x, z),) -> frule_ternary(x, _y, z), [_x, _z])[1]
    @test ForwardDiff.gradient(((y, z),) -> frule_ternary(_x, y, z), [_y, _z]) ≈ FiniteDifferences.grad(fdm, ((y, z),) -> frule_ternary(_x, y, z), [_y, _z])[1]
    @test ForwardDiff.gradient(((x, y, z),) -> frule_ternary(x, y, z), [_x, _y, _z]) ≈ FiniteDifferences.grad(fdm, ((x, y, z),) -> frule_ternary(x, y, z), [_x, _y, _z])[1]
end

@testset "dual_rule_from_frule unary" begin
    dual_rule_unary(x) = sin(1 + f11_dual_rule_unary(x)^2)
    _x = rand()
    @test ForwardDiff.derivative(dual_rule_unary, _x) ≈ fdm(dual_rule_unary, _x)
end

@testset "dual_rule_from_frule binary" begin
    dual_rule_binary(x, y) = sin(1 + f12_dual_rule_binary(x, y)^2)
    _x, _y = rand(2)
    @test ForwardDiff.derivative(x -> dual_rule_binary(x, _y), _x) ≈ fdm(x -> dual_rule_binary(x, _y), _x)
    @test ForwardDiff.derivative(y -> dual_rule_binary(_x, y), _y) ≈ fdm(y -> dual_rule_binary(_x, y), _y)
    @test ForwardDiff.gradient(((x, y),) -> dual_rule_binary(x, y), [_x, _y]) ≈ FiniteDifferences.grad(fdm, ((x, y),) -> dual_rule_binary(x, y), [_x, _y])[1]
end

@testset "dual_rule_from_frule ternary" begin
    dual_rule_ternary(x, y, z) = sin(1 + f13_dual_rule_ternary(x, y, z)^2)
    _x, _y, _z = rand(3)
    @test ForwardDiff.derivative(x -> dual_rule_ternary(x, _y, _z), _x) ≈ fdm(x -> dual_rule_ternary(x, _y, _z), _x)
    @test ForwardDiff.derivative(y -> dual_rule_ternary(_x, y, _z), _y) ≈ fdm(y -> dual_rule_ternary(_x, y, _z), _y)
    @test ForwardDiff.derivative(z -> dual_rule_ternary(_x, _y, z), _z) ≈ fdm(z -> dual_rule_ternary(_x, _y, z), _z)
    @test ForwardDiff.gradient(((x, y),) -> dual_rule_ternary(x, y, _z), [_x, _y]) ≈ FiniteDifferences.grad(fdm, ((x, y),) -> dual_rule_ternary(x, y, _z), [_x, _y])[1]
    @test ForwardDiff.gradient(((x, z),) -> dual_rule_ternary(x, _y, z), [_x, _z]) ≈ FiniteDifferences.grad(fdm, ((x, z),) -> dual_rule_ternary(x, _y, z), [_x, _z])[1]
    @test ForwardDiff.gradient(((y, z),) -> dual_rule_ternary(_x, y, z), [_y, _z]) ≈ FiniteDifferences.grad(fdm, ((y, z),) -> dual_rule_ternary(_x, y, z), [_y, _z])[1]
    @test ForwardDiff.gradient(((x, y, z),) -> dual_rule_ternary(x, y, z), [_x, _y, _z]) ≈ FiniteDifferences.grad(fdm, ((x, y, z),) -> dual_rule_ternary(x, y, z), [_x, _y, _z])[1]
end

@testset "dual_rule_from_frule binary mixed" begin
    dual_rule_binary_mixed(x, y) = sin(1 + f14_dual_rule_binary_mixed(x, y)^2)
    _x, _y = rand(2)
    @test ForwardDiff.derivative(x -> dual_rule_binary_mixed(x, _y), _x) ≈ fdm(x -> dual_rule_binary_mixed(x, _y), _x)
end

end # module ForwardDiffTests

using .ForwardDiffTests
