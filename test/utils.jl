module Utils

using ArbNumerics

setworkingprecision(ArbFloat, 500)
setextrabits(128)

arbify(f) = function f_arbified(args::T...) where {T <: Union{Float32, Float64}}
    y = f(ArbFloat.(args)...)
    return convert.(T, y)
end

end # module Utils

import .Utils
