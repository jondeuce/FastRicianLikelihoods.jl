module ForwardDiffTests

using Test

using FastRicianLikelihoods: @define_unary_dual_scalar_rule, @define_binary_dual_scalar_rule, @define_ternary_dual_scalar_rule
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff

f_unary(x) = exp(x * cos(x))
∇f_unary(x) = ((cos(x) - x * sin(x)) * exp(x * cos(x)),)
@define_unary_dual_scalar_rule f_unary ∇f_unary

g_unary(x) = f_unary(x)
g_∇g_unary(x) = (f_unary(x), ∇f_unary(x))
@define_unary_dual_scalar_rule fused = true g_unary g_∇g_unary

h_unary(x) = f_unary(x)
h_∇h_unary(x) = (f_unary(x), only(∇f_unary(x)))
@define_unary_dual_scalar_rule fused = true h_unary h_∇h_unary

f_binary(x, y) = exp(x * (y - cos(x * y)))
∇f_binary(x, y) = (x * y * sin(x * y) - cos(x * y) + y, x^2 * sin(x * y) + x) .* exp(x * (y - cos(x * y)))
@define_binary_dual_scalar_rule f_binary ∇f_binary

g_binary(x, y) = f_binary(x, y)
g_∇g_binary(x, y) = (f_binary(x, y), ∇f_binary(x, y))
@define_binary_dual_scalar_rule fused = true g_binary g_∇g_binary

f_ternary(x, y, z) = exp(x * z + cos(y - z))
∇f_ternary(x, y, z) = (z, -sin(y - z), (x + sin(y - z))) .* exp(x * z + cos(y - z))
@define_ternary_dual_scalar_rule f_ternary ∇f_ternary

g_ternary(x, y, z) = f_ternary(x, y, z)
g_∇g_ternary(x, y, z) = (f_ternary(x, y, z), ∇f_ternary(x, y, z))
@define_ternary_dual_scalar_rule fused = true g_ternary g_∇g_ternary

const fdm = FiniteDifferences.central_fdm(4, 1)

@testset "unary" begin
    unary1(x) = sin(1 + f_unary(x)^2)
    unary2(x) = sin(1 + g_unary(x)^2)
    unary3(x) = sin(1 + h_unary(x)^2)
    map((unary1, unary2, unary3)) do f
        _x = rand()
        @test ForwardDiff.derivative(f, _x) ≈ fdm(f, _x)
    end
end

@testset "binary" begin
    binary1(x, y) = sin(1 + f_binary(x, y)^2)
    binary2(x, y) = sin(1 + g_binary(x, y)^2)
    map((binary1, binary2)) do f
        _x, _y = rand(2)
        @test ForwardDiff.derivative(x -> f(x, _y), _x) ≈ fdm(x -> f(x, _y), _x)
        @test ForwardDiff.derivative(y -> f(_x, y), _y) ≈ fdm(y -> f(_x, y), _y)
        @test ForwardDiff.gradient(((x, y),) -> f(x, y), [_x, _y]) ≈ FiniteDifferences.grad(fdm, ((x, y),) -> f(x, y), [_x, _y])[1]
    end
end

@testset "ternary" begin
    ternary1(x, y, z) = sin(1 + f_ternary(x, y, z)^2)
    ternary2(x, y, z) = sin(1 + g_ternary(x, y, z)^2)
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

end # module ForwardDiffTests

using .ForwardDiffTests
