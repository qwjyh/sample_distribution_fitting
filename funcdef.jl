using Distributions, LandauDistribution
using Optimization, OptimizationOptimJL

"""
    target(x, p)

TBW
"""
function target(x, p)
    model = MixtureModel(Union{Normal, Landau}[Normal(x[1], x[2]), Landau(x[3], x[4])], [x[5], 1 - x[5]])
    map(p) do p
        - log(pdf(model, p))
    end |> sum
end

cons(res, x, p) = (res .= [x[2], x[4], x[5]])