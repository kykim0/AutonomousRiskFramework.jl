using Distributed

function is_carla_running()
    if Sys.iswindows()
        tasks = read(`tasklist`, String)
        return occursin("CarlaUE4.exe", tasks)
    else
        tasks = read(`ps -aux`, String)
        return occursin("CarlaUE4.sh", tasks)
    end
end

function start_carla_monitor(time=10)
    # Check if CARLA executable is running
    while true
        # CARLA already open
        if is_carla_running()
            sleep(time)
        else
            tasks = read(`ps -aux`, String)
            # @show tasks
            @show occursin("CarlaUE4.sh", tasks)
            if Sys.iswindows()
                # CARLA not open, so open it.
                carla_start = joinpath(@__DIR__, "..", "..", "CARLAIntegration", "adversarial_carla_env", "carla-start.bat")
                @info "Re-opening CARLA executable."
                run(`cmd /c $carla_start`)
            else
                carla_start = joinpath(@__DIR__, "..", "..", "CARLAIntegration", "adversarial_carla_env", "carla-start.sh")
                @info "Re-opening CARLA executable."
                # @async run(`$carla_start`)
                run(`$carla_start`, wait=false)
            end
            sleep(15)
        end
    end
end
