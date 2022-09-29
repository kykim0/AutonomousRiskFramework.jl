module AVExperiments

# Set up experiments with configurations
# Save results
    # costs
    # tree
    # planner
    # info!
    # (naming schemes)
    # sub-directories
# Generate figures
    # histogram of costs
    # re-weighted histogram
    # anything from info?
    # polar plots?
# [] TODO: monitor carla.exe, kick off `carla-start` if missing.
# [] TODO: periodically save planner/costs a la "Weights and Biases" monitoring

using Reexport
# @reexport using JLD
@reexport using BSON
@reexport using ImportanceWeightedRiskMetrics
@reexport using TreeImportanceSampling
@reexport using ParallelTreeSampling
@reexport using Random
@reexport using MCTS
@reexport using D3Trees
using POMDPPolicies
using ProgressMeter
using POMDPSimulators

include("mdp_carla.jl")
include("monitor.jl")

export
    NEAT,
    WorldOnRails,
    GNSS,
    AVExperimentConfig,
    ExperimentResults,
    show_tree,
    CARLAScenarioMDP,
    GenericDiscreteNonParametric,
    Weather,
    Agent,
    run_carla_experiment,
    pyreload,
    generate_dirname!,
    load_data,
    get_costs,
    ScenarioState,
    run_scenario


function disturbance(m::CARLAScenarioMDP, s::ScenarioState)
    xs = POMDPs.actions(m, s)
    return xs
end


show_tree(planner::ISDPWPlanner) = show_tree(planner.tree.dpw_tree)
function show_tree(tree)
    t = D3Tree(tree, init_expand=1);
    for i in 1:length(t.text)
        if t.text[i][1:4] == "JLD2"
            t.text[i] = split(t.text[i], "\n")[end-1]
        end
    end
    inchrome(t)
end


# save_data(data, filename) = save(filename, "data", data)
# load_data(filename) = load(filename)[:data]

save_data(data, filename) = BSON.@save(filename, data)
load_data(filename) = BSON.load(filename, @__MODULE__)[:data]


@with_kw mutable struct AVExperimentConfig
    seed           = 0xC0FFEE  # RNG seed for determinism
    agent          = NEAT      # AV policy/agent to run. Options: [NEAT, WorldOnRails, GNSS]
    N              = 100       # Number of scenario selection iterations
    dir            = "results" # Directory to save results
    use_tree_is    = true      # Use tree importance sampling (IS) for scenario selection (SS) [`false` will use Monte Carlo SS]
    leaf_noise     = true      # Apply adversarial noise disturbances at the leaf nodes
    resume         = true      # Resume previous run?
    additional     = false     # Resume experiment by running an additional N iterations (as opposed to "finishing" the remaining `N-length(results)`)
    rethrow        = false     # Choose to rethrow the errors or simply provide warning.
    retry          = true      # Restart the run if an error was encountered.
    monitor        = @task start_carla_monitor() # Task to monitor that CARLA is still running.
    render_carla   = false      # Show CARLA rendered display.
    save_frequency = 1         # After X iterations, save results.
    s0             = nothing   # initial state
    iterations_per_process = 3 # Number of runs to make in separate Julia process (due to CARLA memory leak).
end


@with_kw mutable struct ExperimentResults
    planner
    costs
    info
end


function generate_dirname!(config::AVExperimentConfig)
    dir = isempty(config.dir) ? "results" : config.dir
    if config.agent == WorldOnRails
        dir = "$(dir)_wor"
    elseif config.agent == NEAT
        dir = "$(dir)_neat"
    elseif config.agent == GNSS
        dir = "$(dir)_gnss"
    end

    if config.use_tree_is
        dir = "$(dir)_SS-IS"
    else
        dir = "$(dir)_SS-MC"
    end

    if config.leaf_noise
        dir = "$(dir)_leaf-MC"
    else
        dir = "$(dir)_leaf-none"
    end

    dir = "$(dir)_seed-$(uppercase(string(config.seed, base=16)))"

    config.dir = dir
end


get_costs(results::Vector) = map(res->res.hist[end].r, results)
get_costs(planner::ISDPWPlanner) = planner.mdp.costs


function run_carla_experiment(config::AVExperimentConfig)
    # Monitor that CARLA executable is still alive.
    if !istaskstarted(config.monitor) && !haskey(ENV, "CARLA_MONITOR_STARTED")
        schedule(config.monitor) # Done asynchronously.
        ENV["CARLA_MONITOR_STARTED"] = true
    end

    rmdp = CARLAScenarioMDP(seed=config.seed,
                            agent=config.agent,
                            leaf_noise=config.leaf_noise,
                            render_carla=config.render_carla,
                            iterations_per_process=config.iterations_per_process)
    Random.seed!(rmdp.seed) # Determinism

    !isdir(config.dir) && mkdir(config.dir)
    planner_filename = joinpath(config.dir, "planner.bson")

    N = config.N # number of samples drawn from the tree

    if isnothing(config.s0)
        s0 = rand(initialstate(rmdp))
    else
        s0 = config.s0
    end

    function new_planner()
        tree_mdp = TreeMDP(rmdp, 1.0, [], [], disturbance, "sum")
        c = 0.0 # exploration bonus (NOTE: keep at 0)
        α = rmdp.α # VaR/CVaR risk parameter
        tree_is_params = TreeISParams(c, α, 0.0, 0.0, 1.0, 1e-6)
        return TreeImportanceSampling.mcts_isdpw(tree_mdp, tree_is_params; N)
    end

    function new_planner_p()
        c = 0.0
        # nominal_distrib_fn = (mdp, s) -> actions(mdp, s)
        experiment_config = ParallelTreeSampling.ExperimentConfig(nominal_steps=0)
        solver = PISSolver(; depth=100,
                           exploration_constant=c,
                           n_iterations=N,
                           enable_action_pw=false,  # Needed for discrete cases.
                           k_state=Inf,             # Needed for discrete cases (to always transition).
                           virtual_loss=0.0,
                           keep_tree=true,
                           rollout_strategy=:uniform,
                           action_selection=:var_sigmoid,
                           experiment_config=experiment_config,
                           show_progress=false)
        planner = solve(solver, rmdp)
        return planner
    end

    use_pis = true
    if config.use_tree_is
        if use_pis
            s0_tree = s0
        else
            s0_tree = TreeImportanceSampling.TreeState(s0)
        end
        if config.resume && !isfile(planner_filename)
            @info "Trying to resume a file that doesn't exist, starting from scratch: $planner_filename"
            planner = use_pis ? new_planner_p() : new_planner()
        elseif config.resume
            @info "Resuming: $planner_filename"
            planner = load_data(planner_filename)
            if config.additional
                planner.solver.n_iterations = N
            else
                costs = use_pis ? planner.tree.costs : planner.mdp.costs
                planner.solver.n_iterations = N - length(costs)
            end
            @info "Resuming for N = $(planner.solver.n_iterations) runs."
        else
            planner = use_pis ? new_planner_p() : new_planner()
        end

        tree_in_info = false
        β = 0.01 # for picking equal to Monte Carlo strategy (if β=1 then exactly MC)
        γ = 0.01 # for better estimate of VaR (γ=1 would give minimum variance estimate of VaR)

        try
            save_callback = (planner) -> save_data(planner, planner_filename)
            if use_pis
                α = 0.1; min_s = 5.0; mix_w_fn = linear_decay_schedule(1.0, 0.90, 1_000)
                a, info = action_info(planner, s0_tree; tree_in_info=tree_in_info,
                                      save_freq=config.save_frequency, save_callback=save_callback,
                                      α=α, mix_w_fn=mix_w_fn, min_s=min_s)
            else
                planner.solver.tree_in_info = tree_in_info
                a, info = action_info(planner, s0_tree; tree_in_info=tree_in_info,
                                      save_frequency=config.save_frequency, save_callback=save_callback, β, γ)
            end
            # show_tree(planner)
        catch err
            if config.rethrow
                rethrow(err)
            else
                @warn err
                if config.retry
                    @info "Retrying AV experiment!"
                    config.additional = false
                    config.resume = true
                    exp_results = run_carla_experiment(config) # Retry if there was an error.
                    planner = exp_results.planner # Make sure to copy over the recursed planner results.
                end
            end
        end
        save_data(planner, planner_filename)
        costs = use_pis ? planner.tree.costs : planner.mdp.costs
        return ExperimentResults(planner, costs, [])
    else
        # Use Monte Carlo scenario selection instead of tree-IS.
        policy = RandomPolicy(rmdp)
        results_filename = joinpath(config.dir, "random_scenario_results.bson")
        new_results = ()->ExperimentResults(rmdp, [], [])
        if config.resume && !isfile(results_filename)
            @info "Trying to resume a file that doesn't exist, starting from scratch: $results_filename"
            results = new_results()
        elseif config.resume
            @info "Resuming: $results_filename"
            results = load_data(results_filename)
            if !config.additional
                N = N - length(results.info)
            end
            @info "Resuming for N = $N runs."
        else
            results = new_results()
        end

        if length(results.info) != 0
            rmdp.seed = rmdp.seed + length(results.info) # Change seed to where we left off.
        end

        try
            @showprogress for i in 1:N
                res = POMDPSimulators.simulate(HistoryRecorder(), rmdp, policy, s0)
                push!(results.info, res)

                if i % config.save_frequency == 0
                    costs = get_costs(results.info)
                    exp_results = ExperimentResults(rmdp, costs, results.info)
                    save_data(exp_results, results_filename)
                end
            end
        catch err
            if config.rethrow
                rethrow(err)
            else
                @warn err
                if config.retry
                    @info "Retrying AV experiment!"
                    config.additional = false
                    config.resume = true
                    exp_results = run_carla_experiment(config) # Retry if there was an error.
                    rmdp = merge(rmdp, exp_results.planner)
                    results.info = exp_results.info
                end
            end
        end

        costs = get_costs(results.info)
        exp_results = ExperimentResults(rmdp, costs, results.info)
        save_data(exp_results, results_filename)
        return exp_results
    end
end


function run_scenario(planner::ISDPWPlanner, s::ScenarioState, seed::Int)
    rmdp = deepcopy(planner.mdp.rmdp)
    rmdp.seed = seed
    rmdp.counter = 0
    return POMDPs.reward(rmdp, s)
end


end # module
