module UtilsTests

using Test
using FastRicianLikelihoods: estrin, clenshaw

for T in (Float32, Float64)
    @testset "estrin ($T)" begin
        for N in 1:24
            c = ntuple(i -> T((-1)^i * i / 7), N)
            for x in T.((-1, -0.9, -0.1, 0, 0.4, 0.9, 1))
                y = sum(BigFloat(c[i]) * BigFloat(x)^(i - 1) for i in 1:N)
                @test estrin(x, c) ≈ T(y) rtol = 0 atol = 8 * eps(T) * sum(abs, c)
            end
        end
    end

    @testset "clenshaw ($T)" begin
        for N in 0:24
            c = ntuple(i -> T((-1)^i * i / 7), N)
            for x in T.((-1, -0.9, -0.1, 0, 0.4, 0.9, 1))
                θ = acos(BigFloat(x))
                y = sum((BigFloat(c[i]) * cos((i - 1) * θ) for i in 1:N); init = BigFloat(0))
                for coefficients in (c, collect(c))
                    @test clenshaw(x, coefficients) ≈ T(y) rtol = 0 atol = 2N * eps(T) * sum(abs, c; init = zero(T))
                    @test clenshaw(2x + 1, T(-1), T(3), coefficients) ≈ T(y) rtol = 0 atol = 2N * eps(T) * sum(abs, c; init = zero(T))
                end
            end
        end
    end
end

end # module UtilsTests

using .UtilsTests
