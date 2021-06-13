### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 9ab2bb62-b97c-11eb-2bd2-01a3bbe4935c
begin
	using Revise
	using RiskSimulator
	using AutomotiveSimulator
	using AutomotiveVisualization
	using AdversarialDriving
	using PlutoUI
	using Random
	using Distributions
end

# ╔═╡ e738e8ec-b31c-4dc7-9f77-71a728743d86
AutomotiveVisualization.colortheme["background"] = colorant"white";

# ╔═╡ a1f73e35-287a-4d1f-9909-868c3e023903
AutomotiveVisualization.set_render_mode(:fancy);

# ╔═╡ bf57e93d-10bc-42d5-8c81-497044966372
md"""
# AST
"""

# ╔═╡ 6dfd1372-6ef4-4927-b4ba-bd40e2360d8b
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the mod. that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
		Expr(:toplevel,
			 :(eval(x) = $(Expr(:core, :eval))($name, x)),
			 :(include(x) = $(Expr(:top, :include))($name, x)),
			 :(include(mapexpr::Function, x) =
				$(Expr(:top, :include))(mapexpr, $name, x)),
			 :(include($path))))
	return m
end

# ╔═╡ b2cf6275-b491-4d12-915c-08ea7c48109b
begin
	SEED = 1000
	Random.seed!(SEED);
end;

# ╔═╡ 6d5dacb1-c68f-4a36-9d65-87f0d92ed234
SEEDS = 1000:1001

# ╔═╡ 41bed420-834b-4bf3-bac6-9fa349142d10
md"""
## `Revise.retry`
"""

# ╔═╡ 23e94f2c-0c69-4831-b355-be8905ca98a3
Revise.retry()

# ╔═╡ acc8b2df-c45e-44e5-a253-29c9d9741891
md"""
## Scenario
"""

# ╔═╡ c08fb802-3a51-4813-a905-970b9cda35e7
SCENARIO

# ╔═╡ 582542ee-09d3-48e8-b8de-787293618302
begin
	SC = CROSSWALK
	scenario = get_scenario(SC)
	scenario_string = get_scenario_string(SC)
end

# ╔═╡ 90860f18-72a4-442b-8404-4bd3d717ec77
function change_noise_disturbance!(sim)
    σ0 = 1e-300

	# Scenario specific noise
	if SC == CROSSING 
		σ = 7
		σᵥ = 1
	elseif SC == T_HEAD_ON
		σ = 10
		σᵥ = 4
	elseif SC == T_LEFT
		σ = 10
		σᵥ = 1
	elseif SC == STOPPING
		σ = 2
    	σᵥ = σ/100
	elseif SC == MERGING
		σ = 2
    	σᵥ = 1
	elseif SC == CROSSWALK
		σ = 1
		σᵥ = 1e-8
	end
	
    sim.xposition_noise_veh = Normal(0, σ)
    sim.yposition_noise_veh = Normal(0, σ)
    sim.velocity_noise_veh = Normal(0, σᵥ)

    sim.xposition_noise_sut = Normal(0, σ)
    sim.yposition_noise_sut = Normal(0, σ)
    sim.velocity_noise_sut = Normal(0, σᵥ)
end

# ╔═╡ 595cf7d7-9559-4427-9d99-0ba25f9c3212
md"""
## State-Proxy
"""

# ╔═╡ 7bc840a4-981b-4656-b496-2da65989cab1
state_proxy = :distance # :distance, :rate, :actual, :none

# ╔═╡ ab6b9bed-4485-43eb-81c4-2fce07d4f2d2
md"""
## Solver
"""

# ╔═╡ cb0cf557-0c2f-4b4e-acf3-4c5803c550dd
which_solver = :mcts

# ╔═╡ c741297a-bd52-45db-b0da-4b1441af8470
use_nn_obs_model = false

# ╔═╡ e7945ae2-a67c-4ca5-9b62-3dc3f6e5e95f
adjust_noise = true

# ╔═╡ 635259b1-555e-4233-8afd-9fb13dd55bc4
md"""
## Run IDM Search
"""

# ╔═╡ 3904c3e0-5a6f-4652-a8d5-3458a14b5aaf
md"""
### Learn failure policy phase
"""

# ╔═╡ 1e2f3595-6b02-4c92-bffa-22f64b740bbc
learned_solver = :ppo

# ╔═╡ 2dbec591-d445-4ff1-a8bd-638314ac149e
begin
	system = IntelligentDriverModel(v_des=12.0)

	learned_planner = setup_ast(sut=system, scenario=scenario, seed=SEED,
		nnobs=use_nn_obs_model, state_proxy=state_proxy, which_solver=learned_solver,
		noise_adjustment=adjust_noise ? change_noise_disturbance! : nothing)

	# Run AST.
	search!(learned_planner)
	learned_fail_metrics = failure_metrics(learned_planner)
	RiskSimulator.POMDPStressTesting.latex_metrics(learned_fail_metrics)
end

# ╔═╡ bfc678f4-744b-4f3a-bbf1-bcfdbb2d718a
learned_rollout = (mdp, s, d) -> ppo_rollout(mdp, s, d, learned_planner)

# ╔═╡ fc0e2a65-5257-4a5b-b3c9-2e9573f3fb1d
md"""
### Efficient MCTS with learned rollouts
"""

# ╔═╡ 74276517-e275-4e3b-9be0-968961d413cc
use_learned_rollout = true

# ╔═╡ a84a95cc-3e99-405a-aa40-133b26ea5f58
begin
	failure_metrics_vector::Vector{FailureMetrics} = []
	planner = nothing
	for seed in SEEDS
		planner = setup_ast(sut=system, scenario=scenario, seed=seed,
			nnobs=use_nn_obs_model, state_proxy=state_proxy,
			which_solver=which_solver,
			noise_adjustment=adjust_noise ? change_noise_disturbance! : nothing,
			rollout=use_learned_rollout ? learned_rollout : RiskSimulator.AST.rollout
		)

		# Run AST.
		search!(planner)
		fail_metrics = failure_metrics(planner)
		push!(failure_metrics_vector, fail_metrics)
	end
	RiskSimulator.POMDPStressTesting.latex_metrics(
		mean(failure_metrics_vector), std(failure_metrics_vector))
end

# ╔═╡ bd85ebcb-ab90-4549-bbde-f99c822491c2
# TODO: Move to cem.jl
function RiskSimulator.get_action(policy::Dict{Symbol, Vector{Sampleable}}, state)
	# CEM policy is state-less (it's a distribution we sample from)
	return rand(policy)
end

# ╔═╡ be4e1476-b05b-4fd1-b8ae-457c41799813
md"""
## Run Princeton Search
"""

# ╔═╡ d35ce4ee-9a38-4c6a-8ab0-78bd2ccdc249
begin
	system2 = PrincetonDriver(v_des=12.0)
	failure_metrics_vector2::Vector{FailureMetrics} = []
	planner2 = nothing
	for seed in SEEDS
		planner2 = setup_ast(sut=system2, scenario=scenario, seed=seed,
			nnobs=use_nn_obs_model, state_proxy=state_proxy,
			which_solver=which_solver,
			noise_adjustment=adjust_noise ? change_noise_disturbance! : nothing,
			rollout=use_learned_rollout ? learned_rollout : RiskSimulator.AST.rollout
		)

		# Run AST.
		search!(planner2)
		fail_metrics2 = failure_metrics(planner2)
		push!(failure_metrics_vector2, fail_metrics2)
	end
	RiskSimulator.POMDPStressTesting.latex_metrics(
		mean(failure_metrics_vector2), std(failure_metrics_vector2))
end

# ╔═╡ abbb60b3-fff0-4ee0-89d6-6ba3aadc1b31
planner2.mdp.sim.state;

# ╔═╡ 8e606b42-b881-44ff-a3ac-3760bc699e2e
md"""
## Plotting
"""

# ╔═╡ 2f016a52-87b0-42dd-ab0b-af1f31b9eb79
begin
	α = 0.2 # Risk tolerance.
	𝒟 = planner.mdp.dataset

	# Plot cost distribution.
	metrics = risk_assessment(𝒟, α)
	p_risk = plot_risk(metrics; mean_y=3.33, var_y=3.25, cvar_y=2.1, α_y=2.8)
end

# ╔═╡ bfdec09b-9c4a-4883-8c27-f8633d9b40f9
begin
	𝒟2 = planner2.mdp.dataset

	# Plot cost distribution.
	metrics2 = risk_assessment(𝒟2, α)
	p_risk2 = plot_risk(metrics2; mean_y=3.33, var_y=3.25, cvar_y=2.1, α_y=2.8)
end

# ╔═╡ de7c77a0-cf88-4484-867d-2fd3680e34ee
begin
	𝐰 = ones(7)
	𝐰[end] = 1/max(maximum(map(fm->exp(fm.highest_loglikelihood), failure_metrics_vector)), maximum(map(fm->exp(fm.highest_loglikelihood), failure_metrics_vector2))) # 1/max(p)

	areas = overall_area([planner, planner2], weights=𝐰, α=α)
	area_idm = round(areas[1], digits=5)
	area_princeton = round(areas[2], digits=5)
	p_metrics = plot_overall_metrics([planner, planner2],
		["IDM ($area_idm)", "Princeton ($area_princeton)"]; 
		weights=𝐰, α=α, title="Overall Risk:\n$scenario_string")
end

# ╔═╡ c1bf67e9-cfee-486b-8db9-9f7f0e40125b
md"""
## Visualize Playback
"""

# ╔═╡ e44b5f2b-faa9-4e7a-956e-702547f54788
# TODO. `roadway`
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, veh::Entity{BlinkerState, VehicleDef, Int64})
    reg_veh = Entity(veh.state.veh_state, veh.def, veh.id)
    add_renderable!(rendermodel, FancyCar(car=reg_veh))

	noisy_veh = Entity(noisy_entity(veh, scenario.roadway).state.veh_state, veh.def, veh.id)
    ghost_color = weighted_color_mean(0.3, colorant"blue", colorant"white")
    add_renderable!(rendermodel, FancyCar(car=noisy_veh, color=ghost_color))

    li = laneid(veh)
    bo = BlinkerOverlay(on = blinker(veh), veh = reg_veh, right=Tint_signal_right[li])
    add_renderable!(rendermodel, bo)
    return rendermodel
end; md"**TODO**: `AutomotiveVisualization.add_renderable!`"

# ╔═╡ dcc3d3e3-3c44-4aaf-9d73-02818160afba
md"Visualize? $(@bind viz CheckBox())"

# ╔═╡ 8d1b9eb5-b2c9-46c5-a7f1-05413d0a4034
viz && fail_metrics.num_failures > 0 && visualize_most_likely_failure(planner)

# ╔═╡ a2df95b7-f768-407f-9481-e88cb79a74d6
PlutoUI.TableOfContents()

# ╔═╡ Cell order:
# ╠═9ab2bb62-b97c-11eb-2bd2-01a3bbe4935c
# ╠═e738e8ec-b31c-4dc7-9f77-71a728743d86
# ╠═a1f73e35-287a-4d1f-9909-868c3e023903
# ╟─bf57e93d-10bc-42d5-8c81-497044966372
# ╟─6dfd1372-6ef4-4927-b4ba-bd40e2360d8b
# ╠═b2cf6275-b491-4d12-915c-08ea7c48109b
# ╠═6d5dacb1-c68f-4a36-9d65-87f0d92ed234
# ╟─41bed420-834b-4bf3-bac6-9fa349142d10
# ╠═23e94f2c-0c69-4831-b355-be8905ca98a3
# ╟─acc8b2df-c45e-44e5-a253-29c9d9741891
# ╠═c08fb802-3a51-4813-a905-970b9cda35e7
# ╠═582542ee-09d3-48e8-b8de-787293618302
# ╠═90860f18-72a4-442b-8404-4bd3d717ec77
# ╟─595cf7d7-9559-4427-9d99-0ba25f9c3212
# ╠═7bc840a4-981b-4656-b496-2da65989cab1
# ╠═abbb60b3-fff0-4ee0-89d6-6ba3aadc1b31
# ╟─ab6b9bed-4485-43eb-81c4-2fce07d4f2d2
# ╠═cb0cf557-0c2f-4b4e-acf3-4c5803c550dd
# ╠═c741297a-bd52-45db-b0da-4b1441af8470
# ╠═e7945ae2-a67c-4ca5-9b62-3dc3f6e5e95f
# ╟─635259b1-555e-4233-8afd-9fb13dd55bc4
# ╟─3904c3e0-5a6f-4652-a8d5-3458a14b5aaf
# ╠═1e2f3595-6b02-4c92-bffa-22f64b740bbc
# ╠═2dbec591-d445-4ff1-a8bd-638314ac149e
# ╠═bfc678f4-744b-4f3a-bbf1-bcfdbb2d718a
# ╟─fc0e2a65-5257-4a5b-b3c9-2e9573f3fb1d
# ╠═74276517-e275-4e3b-9be0-968961d413cc
# ╠═a84a95cc-3e99-405a-aa40-133b26ea5f58
# ╠═bd85ebcb-ab90-4549-bbde-f99c822491c2
# ╟─be4e1476-b05b-4fd1-b8ae-457c41799813
# ╠═d35ce4ee-9a38-4c6a-8ab0-78bd2ccdc249
# ╟─8e606b42-b881-44ff-a3ac-3760bc699e2e
# ╠═2f016a52-87b0-42dd-ab0b-af1f31b9eb79
# ╠═bfdec09b-9c4a-4883-8c27-f8633d9b40f9
# ╠═de7c77a0-cf88-4484-867d-2fd3680e34ee
# ╟─c1bf67e9-cfee-486b-8db9-9f7f0e40125b
# ╟─e44b5f2b-faa9-4e7a-956e-702547f54788
# ╟─dcc3d3e3-3c44-4aaf-9d73-02818160afba
# ╠═8d1b9eb5-b2c9-46c5-a7f1-05413d0a4034
# ╠═a2df95b7-f768-407f-9481-e88cb79a74d6
