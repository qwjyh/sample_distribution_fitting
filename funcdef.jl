using Distributions, LandauDistribution
using Optimization, OptimizationOptimJL
using Base.Iterators

"""
    target(x, p)

TBW
"""
function target(x, p)
    model = MixtureModel(Union{Normal, Landau}[Normal(x[1], x[2]), Landau(x[3], x[4])], [x[5], x[6]])
    sum = 0.0
    for p in p
        sum += logpdf(model, p)
    end
    sum
end

cons(res, x, p) = (res .= [x[2], x[4], x[5]])

# Ad hoc
@inline function Distributions.logpdf(d::LandauDistribution.Landau, x::Real)
    log(pdf(d, x))
end

struct InitParam{T<:Real}
    p::T
end

function isinitparam(x::InitParam{T}) where T
    true
end

function isinitparam(x::Any)
    false
end

struct DistributionPrototype{T<:Real}
    dists::AbstractVector{Type}
    givenparams::AbstractVector{Tuple{Vararg{Union{T, InitParam{T}}}}}
    givenpriors::AbstractVector{Union{T, InitParam{T}}}
    nvars::Tuple{<:Integer, <:Integer}

    function DistributionPrototype(dists::AbstractVector, givenparams::AbstractVector{TP}, givenpriors::AbstractVector) where TP<:Tuple
        length(dists) == length(givenparams) == length(givenpriors) || error("All argument vectors must have same length.")
        for dist in dists
            try
                dist <: Distribution || error("dists need to be subtypes of Distribution")
            catch
                error("dists must be a vector of array")
            end
        end
        # TODO: prior sum must be 1
        nvar_params = givenparams |>
            flatten |>
            collect |>
            (x -> filter(isinitparam, x)) |>
            length
        nvar_priors = length(dists)
        nvars = (nvar_params, nvar_priors)
        D = Union{unique(dists)...}
        T = let
            mid = givenparams |>
                flatten .|>
                x -> (typeof(x) <: InitParam ? x.p : x) .|>
                typeof
            mid |>
                unique |>
                # x -> filter(!=(Nothing), x) |>
                x -> promote_type(x...)
        end
        S = let
            mid = givenpriors .|>
                x -> (typeof(x) <: InitParam ? x.p : x) .|>
                typeof
            mid |>
                unique |>
                # x -> filter(!=(Nothing), x) |>
                x -> promote_type(x...)
        end
        T = promote_type(T, S)
        new{T}(dists, givenparams, givenpriors, nvars)
    end
end

struct DistributionPrototypeTargets{T}
    nll::Function # negative log likelihood
    cons::Function # conditions
    lcons::Vector{T}
    ucons::Vector{T}
    inits::Vector
end

function get_distribution_argcheck(_::Type{Normal}, args::Vector)
    [:($(args[2]))], [0.0], [Inf]
end

function get_distribution_argcheck(_::Type{Levy}, args::Vector)
    [:($(args[2]))], [0.0], [Inf]
end

function get_distribution_argcheck(_::Type{Landau}, args::Vector)
    [:($(args[2]))], [0.0], [Inf]
end


"""
# Arguments
- `x`: Variables to optimize. In this case, distribution parameters.
- `p`: Parameters for the problem. In this case, data.
"""
function get_target_function(dp::DistributionPrototype{T})::DistributionPrototypeTargets where T
    # conds exprs
    cond_exprs = Any[]
    lcons = Vector{T}()
    ucons = Vector{T}()
    # model def
    model_def_expr = Any[]
    # component
    model_component_expr = Any[]
    model_dist_type = Union{unique(dp.dists)...}
    model_dists = Any[] # to store Expr for model def
    dist_cond_exprs = Any[] # to store cond Exprs for model def
    inits = Vector{T}()
    arg_id = 1
    for (i, dist) in enumerate(dp.dists)
        # params for the 'dist'
        params#=::Tuple{Vararg{Union{Real, InitParam{Real}}}}=# = dp.givenparams[i]
        params_expr = Any[]
        for param in params
            if isinitparam(param)
                push!(params_expr, :(x[$arg_id]))
                push!(inits, param.p)
                arg_id += 1
            else
                push!(params_expr, param)
            end
        end
        # params_expr = (e -> isinitparam(e) ? :(x[$arg_id]) : e).(params)
        push!(model_dists, Expr(:call, dist, params_expr...))
        con, lcon, ucon = get_distribution_argcheck(dist, params_expr)
        append!(dist_cond_exprs, con)
        append!(lcons, lcon)
        append!(ucons, ucon)
    end
    model_component_expr = :($model_dist_type[$(model_dists...)])
    append!(cond_exprs, dist_cond_exprs)

    # prior
    prior_expr = Any[]
    for prior in dp.givenpriors
        if isinitparam(prior)
            push!(prior_expr, :(x[$arg_id]))
            push!(inits, prior.p)
            arg_id += 1
        else
            push!(prior_expr, prior)
        end
    end
    prior_cond_expr = :($(reduce((x, y) -> (:($x + $y)), prior_expr)))
    push!(cond_exprs, prior_cond_expr)
    push!(lcons, one(T))
    push!(ucons, one(T))

    # combine
    model_def_expr = Expr(
        :call,
        :MixtureModel,
        model_component_expr,
        :([$(prior_expr...)])
    )

    @debug model_def_expr
    @debug cond_exprs
    @debug inits
    # @info quote
    #         (x, p) -> begin
    #             # length(filter(isinitparam, vcat(dp.givenparams, dp.givenpriors))) == length(x) || error("Argument length mismatched")
    #             model = $(model_def_expr)
    #             sum = 0.0
    #             for p in p
    #                 sum += - logpdf(model, p)
    #             end
    #             return sum
    #         end
    # end
    # @info quote
    #         (res, x, p) -> begin
    #             (res .= [$(cond_exprs...)])
    #         end
    # end
    eval(quote
        DistributionPrototypeTargets(
            (x, p) -> begin
                # length(filter(isinitparam, vcat(dp.givenparams, dp.givenpriors))) == length(x) || error("Argument length mismatched")
                model = $(model_def_expr)
                sum = 0.0
                for p in p
                    sum += - logpdf(model, p)
                end
                return sum
            end,
            (res, x, p) -> begin
                (res .= [$(cond_exprs...)])
            end,
            $(lcons),
            $(ucons),
            $(inits)
        )
    end)
end

function get_mixmodel_result(dp::DistributionPrototype{T}, u::AbstractVector{T})::Distributions.MixtureModel where T
    cond_exprs = Any[]
    lcons = Vector{T}()
    ucons = Vector{T}()
    # model def
    model_def_expr = Any[]
    # component
    model_component_expr = Any[]
    model_dist_type = Union{unique(dp.dists)...}
    model_dists = Any[] # to store Expr for model def
    dist_cond_exprs = Any[] # to store cond Exprs for model def
    inits = Vector{T}()
    arg_id = 1
    for (i, dist) in enumerate(dp.dists)
        # params for the 'dist'
        params#=::Tuple{Vararg{Union{Real, InitParam{Real}}}}=# = dp.givenparams[i]
        params_val = Any[]
        for param in params
            if isinitparam(param)
                push!(params_val, u[arg_id])
                push!(inits, param.p)
                arg_id += 1
            else
                push!(params_val, param)
            end
        end
        # params_expr = (e -> isinitparam(e) ? :(x[$arg_id]) : e).(params)
        push!(model_dists, Expr(:call, dist, params_val...))
    end
    model_component_expr = :($model_dist_type[$(model_dists...)])
    append!(cond_exprs, dist_cond_exprs)

    # prior
    prior_val = Any[]
    for prior in dp.givenpriors
        if isinitparam(prior)
            push!(prior_val, u[arg_id])
            push!(inits, prior.p)
            arg_id += 1
        else
            push!(prior_val, prior)
        end
    end
    prior_cond_expr = :($(reduce((x, y) -> (:($x + $y)), prior_val)))
    push!(cond_exprs, prior_cond_expr)
    push!(lcons, zero(T))
    push!(ucons, zero(T))

    # combine
    model_def_expr = Expr(
        :call,
        :MixtureModel,
        model_component_expr,
        :([$(prior_val...)])
    )

    @debug model_def_expr
    @debug cond_exprs
    @debug inits
    eval(model_def_expr)
end

function Distributions.fit(dp::DistributionPrototype{T}, data::AbstractVector) where T
    @eval let
    target::DistributionPrototypeTargets = get_target_function(dp)
    optprob = OptimizationFunction(target.nll, Optimization.AutoForwardDiff(), cons = target.cons)
    prob = OptimizationProblem(optprob, target.inits, data, lcons = target.lcons, ucons = target.ucons)
    sol = solve(prob, IPNewton())
    get_mixmodel_result(dp, sol.u)
    end
end

