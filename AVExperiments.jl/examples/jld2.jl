using BSON
using FileIO

using AVExperiments
using POMDPSimulators


function process(input_filename, output_filename)
    # Read the input file. Can be either the MC or IS data in BSON.
    input_bson = BSON.load(input_filename)[:data]
    if !hasproperty(input_bson, :tree)
        save(output_filename, Dict("costs" => input_bson.costs))
    else
        tree = input_bson.tree
        save(output_filename, Dict("costs" => tree.costs, "weights" => tree.weights))
    end
end

basedir = "/home/kykim/dev/sisl/AutonomousRiskFramework.jl/AVExperiments.jl/examples"
in_file = joinpath(basedir, "golden", "exp_10-12_gnss_SS-IS_c10000_s100_leaf-none_seed-FF1CE.bson")
out_file = joinpath(basedir, "golden_jld2", "gnss_SS-IS_c10000_s100_leaf-none_seed-FF1CE.jld2")
process(in_file, out_file)
