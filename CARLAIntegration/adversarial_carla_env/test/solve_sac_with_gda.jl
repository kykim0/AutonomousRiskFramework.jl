using POMDPs
using Crux
using Flux
using Distributions
using POMDPGym
using PyCall
using GaussianDiscriminantAnalysis
import POMDPPolicies:FunctionPolicy
pyimport("adv_carla")

sensors = [
    Dict(
        "id" => "GPS",
        "lat" => Dict("mean" => 0, "std" => 0.0000001, "upper" => 0.0000001, "lower" => -0.0000001),
        "lon" => Dict("mean" => 0, "std" => 0.0000001, "upper" => 0.0000001, "lower" => -0.0000001),
        "alt" => Dict("mean" => 0, "std" => 0.00000001, "upper" => 0.0000001, "lower" => 0),
    ),
]
seed = 1
scenario_type = "Scenario2"
gym_args = (sensors=sensors, seed=seed, scenario_type=scenario_type, no_rendering=false)
mdp = GymPOMDP(Symbol("adv-carla"); gym_args...)


function sac_solver(mdp)
    global sensors

    S = state_space(mdp)
    amin = [Float32(sensors[1][k]["lower"]) for k in ["lat", "lon", "alt"]]
    amax = [Float32(sensors[1][k]["upper"]) for k in ["lat", "lon", "alt"]]
    rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

    state_dim = length(rand(initialstate(mdp)))
    action_dim = length(amin)
    QSA() = ContinuousNetwork(Chain(Dense(state_dim+action_dim, 64, relu), Dense(64, 64, relu), Dense(64, action_dim)))

    function SAC_A()
        base = Chain(Dense(state_dim, 64, relu), Dense(64, 64, relu))
        mu = ContinuousNetwork(Chain(base..., Dense(64, action_dim)))
        logΣ = ContinuousNetwork(Chain(base..., Dense(64, action_dim)))
        SquashedGaussianPolicy(mu, logΣ)
    end

    off_policy = (S=S,
                  ΔN=50,
                  N=2, # NOTE: was 30_000 (then 100)
                  buffer_size=Int(5e5),
                  buffer_init=1000,
                  c_opt=(batch_size=100, optimizer=ADAM(1e-3)),
                  a_opt=(batch_size=100, optimizer=ADAM(1e-3)),
                  π_explore=FirstExplorePolicy(5, rand_policy, GaussianNoiseExplorationPolicy(0.0000001f0, a_min=amin, a_max=amax))) # NOTE: was 1000
    return SAC(; π=ActorCritic(SAC_A(), DoubleNetwork(QSA(), QSA())), off_policy...)
end


function reward_mod(sarsp::NamedTuple)
    s, a, r, sp, info = sarsp.s, sarsp.a, sarsp.r, sarsp.sp, sarsp.info
    return r
end


function extract_info(info)
    if info["done"] == true
        d = info["distance"]
        r = info["rate"]
        y = info["collision"]
        data_point = ([r,d], y)
    else
        data_point = missing
    end
    return data_point
end


function collect_and_train_gda(mdp)
    # 1. run AST with data collection ON (records distance and rate at terminal state)
    @info "Running AST to collect dataset..."
    mdp_info = InfoCollector(mdp, extract_info)
    @time π_train = solve(sac_solver(mdp_info), mdp_info)

    # 2. compile data set of ([d,r], y)
    dataset = mdp_info.dataset
    failure_rate = sum(map(last, dataset)) / length(dataset)
    @info "Failure rate: $failure_rate"

    # 3. train GDA predict function
    @info "Training GDA prediction function..."
    predict, mv₀, mv₁ = qda(dataset)

    # 4. pass back reward mod function.
    reward_mod = sarsp -> begin
        info = sarsp.info
        rate = info["rate"]
        dist = info["distance"]
        ŷ = predict([rate, dist])
        reward = sarsp.r
        return reward + ŷ
    end

    # 5 (external). re-run solver using reward augmented MDP
    @info "Re-running AST with reward augmentation..."
    mdp_aug = RewardMod(mdp, reward_mod)
    @time π_aug = solve(sac_solver(mdp_aug), mdp_aug)

    return π_aug, dataset
end


function run(mdp, π, show_render=false)
    s = rand(initialstate(mdp))
    total_reward = 0
    while !isterminal(mdp, s)
        a = action(π, Vector(s))
        s, o, r = gen(mdp, s, a)
        total_reward += r
        show_render && render(mdp.pomdp.env)
    end
    return total_reward
end

# policy, dataset = collect_and_train_gda(mdp)