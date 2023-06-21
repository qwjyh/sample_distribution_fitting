using Distributions
using LandauDistribution
using Optimization
using OptimizationOptimJL
using Plots

# TODO: SVector?

include("funcdef.jl")

model = MixtureModel(Union{Normal, Landau}[Normal(2.0, 1.0), Landau(5.0, 0.5)], [0.55, 0.45])

target([2.0, 1.0, 5.0, 0.5, 0.55], rand(model, 10))

data = rand(model, 1000)
optprob = OptimizationFunction(target, Optimization.AutoForwardDiff(), cons = cons)
prob = OptimizationProblem(optprob, [2.0, 1.0, 5.0, 0.5, 0.55], data, lcons = [0.0, 0.0, 0.0], ucons = [Inf, Inf, 1.0])
sol = solve(prob, IPNewton())
# prob = OptimizationProblem(target, [2.0, 1.0, 5.0, 0.5, 0.55], data, lb = [-Inf, 0.0, -Inf, 0.0, 0.0], ub = [Inf, Inf, Inf, Inf, 1.0])
# sol = solve(prob, ConjugateGradient())

prob = OptimizationProblem(optprob, [5.0, 1.0, 2.0, 0.5, 0.95], data, lcons = [0.0, 0.0, 0.0], ucons = [Inf, Inf, 1.0])
sol = solve(prob, IPNewton())

plot(
    data,
    seriestype = :stephist, bins = range(-5, 1000, 10000), normalize = :pdf,
    xlims = (-5, 20),
    label = "random",
)
result(x) = pdf(
    MixtureModel(Union{Normal, Landau}[Normal(sol.u[1], sol.u[2]), Landau(sol.u[3], sol.u[4])], [sol.u[5], 1 - sol.u[5]]),
    x
)
plot!(result, label = "distribution fitting")
savefig("result.png")
