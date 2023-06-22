using Distributions, LandauDistribution
using Optimization, OptimizationOptimJL
using Base.Iterators

"""
    target(x, p)

TBW
"""
function target(x, p)
    model = MixtureModel(Union{Normal, Landau}[Normal(x[1], x[2]), Landau(x[3], x[4])], [x[5], 1 - x[5]])
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

struct DistributionPrototype{T<:Real, S<:Real}
    dists::AbstractVector{Type}
    givenparams::AbstractVector{Tuple{Vararg{Union{T, InitParam{T}}}}}
    givenpriors::AbstractVector{Union{S, InitParam{S}}}

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
        new{T, S}(dists, givenparams, givenpriors)
    end
end

function get_target_function(dp::DistributionPrototype)::Function

end

"""
# Arguments
- `x`: Variables to optimize. In this case, distribution parameters.
- `p`: Parameters for the problem. In this case, data.
"""
function (dp::DistributionPrototype)(x, p)
    length(filter(isinitparam, vcat(dp.givenparams, dp.givenpriors))) == length(x) || error("Argument length mismatched")
    # model def
    model_def_expr = Any[]
    # component
    model_component_expr = Any[]
    model_dist_type = Union{unique(dp.dists)...}
    model_dists = Any[] # to store Expr for model def
    model_priors = Vector{Real}[]
    arg_id = 1
    for (i, dist) in enumerate(dp.dists)
        params#=::Tuple{Vararg{Union{Real, InitParam{Real}}}}=# = dp.givenparams[i]
        params_expr = Any[]
        for param in params
            if isinitparam(param)
                push!(params_expr, :(x[$arg_id]))
                arg_id += 1
            else
                push!(params_expr, param)
            end
        end
        # params_expr = (e -> isinitparam(e) ? :(x[$arg_id]) : e).(params)
        push!(model_dists, Expr(:call, dist, params_expr...))
    end
    model_component_expr = :($model_dist_type[$(model_dists...)])

    # prior
    prior_expr = Any[]
    for prior in dp.givenpriors
        if isinitparam(prior)
            push!(prior_expr, :(x[$arg_id]))
            arg_id += 1
        else
            push!(prior_expr, prior)
        end
    end

    # combine
    push!(model_def_expr, Expr(
        :call,
        :MixtureModel,
        model_component_expr,
        :([$(prior_expr...)])
    ))
    # eval(quote
    #     (x, p) -> begin
    #         model = $(model_dist_type)[]
    #     end
    # end)
    return model_def_expr
end

function mixmodel_result(distributions::Vector{T})::Distributions.MixtureModel where {T<:Distribution}
    
end
