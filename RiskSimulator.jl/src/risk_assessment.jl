##############################################################################
# Risk Assessment
##############################################################################
using Distributions
using Plots
using PGFPlotsX
using LaTeXStrings
using StatsBase
using Statistics
using LinearAlgebra
using Parameters
using Markdown

@with_kw struct RiskMetrics
    Z # cost data
    α # probability threshold

    𝒫 # emperical CDF
    𝒞 # conditional distribution

    mean # expected value
    var # Value at Risk
    cvar # Conditional Value at Risk
    worst # worst case
end


function RiskMetrics(Z,α)
    𝒫 = ecdf(Z)
    𝒞 = conditional_distr(𝒫, Z, α)
    𝔼 = mean(Z)
    var = VaR(𝒞)
    cvar = CVaR(𝒞)
    return RiskMetrics(Z=Z, α=α, 𝒫=𝒫, 𝒞=𝒞, mean=𝔼, var=VaR(𝒞), cvar=CVaR(𝒞), worst=worst_case(Z))
end

function RiskMetricsModeled(Z, α, ℱ; length=1000)
    𝒫 = z->cdf(ℱ, z)
    𝒞 = conditional_distr_model(𝒫, α, ℱ; length=length)
    𝔼 = mean(ℱ)
    var = VaR(𝒞)
    cvar = CVaR(𝒞)
    return RiskMetrics(Z=Z, α=α, 𝒫=𝒫, 𝒞=𝒞, mean=𝔼, var=VaR(𝒞), cvar=CVaR(𝒞), worst=worst_case(Z))
end

conditional_distr(𝒫,Z,α) = filter(z->1-𝒫(z) ≤ α, Z)
conditional_distr_model(𝒫,α,ℱ;length=1000) = filter(z->1-𝒫(z) ≤ α, rand(ℱ, length))

VaR(𝒫,Z,α) = minimum(conditional_distr(𝒫,Z,α))
VaR(𝒞) = minimum(𝒞)

worst_case(Z) = maximum(Z)

CVaR(𝒫,Z,α) = mean(conditional_distr(𝒫,Z,α))
CVaR(𝒞) = mean(𝒞)


# TODO: Rename to `risk_metrics`?
risk_assessment(planner, α=0.2) = risk_assessment(planner.mdp.dataset, α)
function risk_assessment(𝒟::Vector, α=0.2)
    metrics = RiskMetrics(cost_data(𝒟), α)
    return metrics
end


"""
Combine datasets from different runs then collect risk metrics.
"""
function collect_metrics(planners, α)
    dataset = vcat(map(planner->planner.mdp.dataset, planners)...)
    metrics = risk_assessment(dataset, α)
    return metrics
end


"""
Return the cost data (Z) of the failures or `nonfailures` (i.e., rate/severity).
"""
function cost_data(𝒟; nonfailures=false, terminal_only=true)
    if typeof(𝒟[1][1]) <: Vector{Vector{Any}}
        # [end] in the 𝐱 data, and [2:end] to remove the first rate value (0 - first distance)
        costs = [d[1][end][2:end] for d in filter(d->nonfailures ? !d[2] : d[2], 𝒟)]
        # when we collect data for FULL trajectory (not just at the terminal state)
        if terminal_only
            return convert(Vector{Real}, vcat(last.(costs)...))
        else
            return convert(Vector{Real}, vcat(costs...))
        end
    else
        return [d[1][end] for d in filter(d->nonfailures ? !d[2] : d[2], 𝒟)]
    end
end


"""
Return the distance data (𝐝) of the failures or `nonfailures`.
"""
function distance_data(𝒟; nonfailures=false, terminal_only=true)
    if typeof(𝒟[1][1]) <: Vector{Vector{Any}}
        # [end] in the 𝐱 data, and [2:end] to match the removal of the first rate value (0 - first distance)
        distances = [d[1][end-1][2:end] for d in filter(d->nonfailures ? !d[2] : d[2], 𝒟)]
        # when we collect data for FULL trajectory (not just at the terminal state)
        if terminal_only
            return convert(Vector{Real}, vcat(last.(distances)...))
        else
            return convert(Vector{Real}, vcat(distances...))
        end
    else
        return [d[1][end-1] for d in filter(d->nonfailures ? !d[2] : d[2], 𝒟)]
    end
end


"""
Display risk metrics in a LaTeX enviroment.
Useful in Pluto.jl notebooks.
"""
function latex_metrics(metrics::RiskMetrics)
    # Note indenting is important here to render correctly.
    return Markdown.parse(string("
\$\$\\begin{align}",
"\\alpha &=", metrics.α, "\\\\",
"\\mathbb{E}[Z] &=", round(metrics.mean, digits=3), "\\\\",
"\\operatorname{VaR}_{", metrics.α, "}(Z) &=", round(metrics.var, digits=3), "\\\\",
"\\operatorname{CVaR}_{", metrics.α, "}(Z) &=", round(metrics.cvar, digits=3), "\\\\",
"\\text{worst case} &=", round(metrics.worst, digits=3), "\\\\",
"\\end{align}\$\$"))
end
