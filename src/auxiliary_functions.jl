module auxiliary_functions

import HomotopyContinuation: @var, evaluate, Expression
import LinearAlgebra: inv, det
using ProgressMeter

#=
Here, the correct θ's are calculated for all possible circuits. 
=#
function createθcircuits(points, coefficients, lineconfigurations, triangconfigurations)
    m = (2,1)
    λ = []
    θ = []

    for config in triangconfigurations
        global barycenter = Matrix{Float64}(undef,3,3); barycenter[1,:] = [points[entry][1] for entry in config[1]]; barycenter[2,:] = [points[entry][2] for entry in config[1]]; barycenter[3,:] = [1 for entry in config[1]];
        global msolve = [2,1,1];
        global λ1 = inv(barycenter)*msolve;
        global λ1 = collect(λ1 / sum(λ1));

        barycenter[1,:] = [points[entry][1] for entry in config[2]]; barycenter[2,:] = [points[entry][2] for entry in config[2]];
        global λ2 = inv(barycenter)*msolve;
        global λ2 = collect(λ2 / sum(λ2));

        global barycenterline = Matrix{Float64}(undef,2,2); barycenterline[1,:] = [points[entry][1] for entry in config[3]]; barycenterline[2,:] = [points[entry][2] for entry in config[3]]; 
        if det(barycenterline) == 0
            global λ3 = [0.5,0.5]
        else
            global λ3 = inv(barycenterline)*[2,1];
            global λ3 = collect(λ3 / sum(λ3))
        end

        barycenterline[1,:] = [points[entry][1] for entry in config[4]]; barycenterline[2,:] = [points[entry][2] for entry in config[4]]; 
        if det(barycenterline) == 0
            global λ4 = [0.5,0.5]
        else
            global λ4 = inv(barycenterline)*[2,1];
            global λ4 = collect(λ4 / sum(λ4))
        end
        
        θ1 = prod([(coefficients[config[1][i]]/λ1[i])^(λ1[i]) for i in 1:length(config[1])])
        θ2 = prod([(coefficients[config[2][i]]/λ2[i])^(λ2[i]) for i in 1:length(config[2])])
        θ3 = prod([(coefficients[config[3][i]]/λ3[i])^(λ3[i]) for i in 1:length(config[3])])
        θ4 = prod([(coefficients[config[4][i]]/λ4[i])^(λ4[i]) for i in 1:length(config[4])])
        push!(θ,θ1+θ2+θ3+θ4)
    end

    for config in lineconfigurations
        global barycenterline = Matrix{Float64}(undef,2,2); barycenterline[1,:] = [points[entry][1] for entry in config[1]]; barycenterline[2,:] = [points[entry][2] for entry in config[1]]; 
        if det(barycenterline) == 0
            global λ1=[0.5,0.5]
        else
            global λ1 = inv(barycenterline)*[2,1];
            global λ1 = collect(λ1 / sum(λ1))
        end

        barycenterline[1,:] = [points[entry][1] for entry in config[2]]; barycenterline[2,:] = [points[entry][2] for entry in config[2]]; 
        if det(barycenterline) == 0
            global λ2=[0.5,0.5]
        else
            global λ2 = inv(barycenterline)*[2,1];
            global λ2 = collect(λ2 / sum(λ2))
        end

        barycenterline[1,:] = [points[entry][1] for entry in config[3]]; barycenterline[2,:] = [points[entry][2] for entry in config[3]]; 
        if det(barycenterline) == 0
            global λ3=[0.5,0.5]
        else
            global λ3 = inv(barycenterline)*[2,1];
            global λ3 = collect(λ3 / sum(λ3))
        end

        barycenterline[1,:] = [points[entry][1] for entry in config[4]]; barycenterline[2,:] = [points[entry][2] for entry in config[4]]; 
        if det(barycenterline) == 0
            global λ4=[0.5,0.5]
        else
            global λ4 = inv(barycenterline)*[2,1];
            global λ4 = collect(λ4 / sum(λ4))
        end

        global λ5 = [0.5,0.5]
        barycenterline[1,:] = [points[entry][1] for entry in config[5]]; barycenterline[2,:] = [points[entry][2] for entry in config[5]]; 
        if det(barycenterline) == 0
            global λ5=[0.5,0.5]
        else
            global λ5 = inv(barycenterline)*[2,1];
            global λ5 = collect(λ5 / sum(λ5))
        end

        θ1 = prod([(coefficients[config[1][i]]/λ1[i])^(λ1[i]) for i in 1:length(config[1])])
        θ2 = prod([(coefficients[config[2][i]]/λ2[i])^(λ2[i]) for i in 1:length(config[2])])
        θ3 = prod([(coefficients[config[3][i]]/λ3[i])^(λ3[i]) for i in 1:length(config[3])])
        θ4 = prod([(coefficients[config[4][i]]/λ4[i])^(λ4[i]) for i in 1:length(config[4])])
        θ5 = prod([(coefficients[config[5][i]]/λ5[i])^(λ5[i]) for i in 1:length(config[5])])
        push!(θ,θ1+θ2+θ3+θ4+θ5)
    end

    return θ
end

#=
Sample randomly from the region [0,boxsize]^n. Whenever aη>0 and bη<0, the sample is accepted. 
We compare all samples to the baseline given by `θbaseline`. Whenever the new model recognizes
nonnegativity, while the baseline model does not, we add 1 to `ourmodel` and if the opposite is
true, we add 1 to `prevmodel`. If neither model recognizes nonnegativity, we add 1 to `nomodel`.
`pointnumber` is a counter of the samples drawn. 

The obtained data is saved in the file `./data/$(prefix)$(suffix)triangstoredsolutions$(boxsize).txt`
=#
function runSamplingComparison(θ, κs, aη, bη, mcoef, θbaseline; boxsize=100, numberOfSamplingRuns=250, prefix="NEW", suffix="")
    #If the file exists, we add to the previously run tests. Else, we set everything to 0.
    global relDict = Dict()
    try
        f = open("../data/multistationarity_points.txt", "r")
        global multistationary_points = []
        while ! eof(f)  
            k_point = [parse(Float64,entry) for entry in split(readline(f)[2:end-1], ",")]
            push!(multistationary_points, k_point)
         end
    catch e
        display(e)
        global multistationary_points = []
    end

    try
        f = open("../data/$(prefix)$(suffix)triangstoredsolutions$(boxsize).txt", "r")

        global pointnumber = parse(Int,readline(f))
        global allmodels = parse(Int,readline(f))
        global onlyonemodel = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]
        global newmodel = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]
        global ourmodel = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]
        global prevmodel = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]
        global nomodel = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]

        while ! eof(f)  
            sstring = split(readline(f), ": ")
            keystring = split(sstring[1], ", ")
            key = (parse(Int,keystring[1]), (parse(Int,keystring[2])))
            relDict[key] = parse(Int,sstring[2])
         end

        close(f)
    catch
        global pointnumber = 0
        global allmodels = 0
        global onlyonemodel = [0 for _ in 1:length(θ)]
        global newmodel = [0 for _ in 1:length(θ)]
        global ourmodel = [0 for _ in 1:length(θ)]
        global prevmodel = [0 for _ in 1:length(θ)]
        global nomodel = [0 for _ in 1:length(θ)]

        for i in 1:length(θ), j in 1:length(θ)
            if i!=j
                relDict[(i,j)] = 0
            end
        end
    end

    for sampleindex in 1:numberOfSamplingRuns
        display("Run: $(sampleindex)")
        global sampling = filter(sampler -> !any(t->isapprox(t,0), sampler) && evaluate(aη, κs=>sampler)>0 && evaluate(bη, κs=>sampler)<0, [boxsize * abs.(rand(Float64, length(κs))) for _ in 1:1000000])
        global pointnumber = pointnumber+length(sampling)
        @showprogress for ind in 1:length(sampling)
            sampler = sampling[ind]
            mval = evaluate(mcoef, κs=>sampler)
            prevval = evaluate(θbaseline, κs=>sampler)
            θ_eval = []
            for j in 1:length(θ)
                ourval = evaluate(θ[j], κs=>sampler)
                push!(θ_eval, ourval)
                if ourval >= -mval
                    global newmodel[j] += 1
                end
                if ourval >= -mval && prevval < -mval
                    global ourmodel[j] += 1
                elseif ourval < -mval && prevval >= -mval
                    global prevmodel[j] += 1
                elseif ourval < -mval && prevval < -mval
                    global nomodel[j] += 1
                end
            end
            
            indicator, winnerarray = false, []
            for i in 1:length(θ), j in i+1:length(θ)
                val1 = θ_eval[i]
                val2 = θ_eval[j]
                if val1 >= -mval && val2 < -mval
                    push!(winnerarray, i)
                    relDict[(i,j)] += 1
                elseif val2 >= -mval && val1 < -mval
                    push!(winnerarray, j)
                    relDict[(j,i)] += 1
                end

                if val2 >= -mval || val1 >= -mval
                    indicator = true
                end
            end

            if length(collect(Set(winnerarray))) == 1
                onlyonemodel[winnerarray[1]]+=1
            end
            if !indicator
                push!(multistationary_points, sampler)
            end
            global allmodels += indicator ? 1 : 0 
        end

        #SAVE the data to the file `NEWtriangstoredsolutions.txt`
        open("../data/$(prefix)$(suffix)triangstoredsolutions$(boxsize).txt", "w") do file
            write(file, "$(pointnumber)\n")
            write(file, "$(allmodels)\n")
            write(file, "$(onlyonemodel)\n")
            write(file, "$(newmodel)\n")
            write(file, "$(ourmodel)\n")
            write(file, "$(prevmodel)\n")
            write(file, "$(nomodel)\n")
            for key in keys(relDict)
                write(file, "$(key[1]), $(key[2]): $(relDict[key])\n")
            end
        end

        open("../data/multistationarity_points.txt", "w") do file
            for pt in multistationary_points
                write(file, "[")
                for p in pt[1:end-1]
                    write(file, "$(p),")
                end
                write(file, "$(pt[end])")
                write(file, "]\n")
            end
        end
    end
    return multistationary_points
end

function runTest_noDependencies( ; boxsize=1000, numberOfSamplingRuns=62, prefix="", suffix="noDependencies")               
    @var κ[1:11]

    hexPoints = [(0,0),(1,0),(2,0),(4,1),(4,2),(3,2),(2,2),(0,1),(3,1),(1,1)]
    triangconfigurations = [[[2,7,9],[3,6,10],[1,5],[4,8]], [[3,6,10],[2,4,7],[1,5],[8,9]], [[1,3,6],[2,5,7],[9,10],[4,8]],
    [[1,3,6],[2,5,7],[8,9],[4,10]], [[3,6,8],[2,7,9],[4,10],[1,5]], [[2,4,7],[3,6,8],[1,5],[9,10]],
    [[1,4,6],[2,5,8],[3,7],[9,10]], [[1,7,9],[3,5,10],[2,6],[4,8]], [[3,5,8],[1,4,7],[9,10],[2,6]],
    [[1,7,9],[3,5,8],[2,6],[4,10]], [[1,6,9],[2,5,8],[3,7],[4,10]], [[1,4,7],[3,5,10],[8,9],[2,6]],
    [[2,5,10],[1,4,6],[3,7],[8,9]], [[1,6,9],[2,5,10],[3,7],[4,8]]]
    #Check if all covers are legit
    for config in triangconfigurations
        all(t->t in union(config[1],config[2],config[3],config[4]),1:10) || display(config)&&throw(error("The triangles don't cover the entire region"))
        isempty(intersect(config[1],config[2])) && isempty(intersect(config[1],config[3])) && isempty(intersect(config[1],config[4])) && isempty(intersect(config[2],config[3])) && isempty(intersect(config[2],config[4])) && isempty(intersect(config[3],config[4])) || display(config)&&throw(error("Each vertex should only be used once"))
    end

    lineconfigurations = [[[5,1],[7,3],[8,9],[6,2],[4,10]], [[5,1],[7,3],[9,10],[6,2],[8,4]]]

    θ = createθcircuits(hexPoints, Vector{Expression}(κ[1:10]), lineconfigurations, triangconfigurations)
    runSamplingComparison(θ, κ[1:11], Vector{Float64}([]), Expression(1), Expression(-1), Expression(-κ[11]), θ[9]; boxsize=boxsize, numberOfSamplingRuns=numberOfSamplingRuns, prefix=prefix, suffix=suffix)
end


function compareTwoCovers(cover_suggested::Int, cover_baseline::Int; numberOfSamplingRuns=100, boxsizes=[5 for _ in 1:8])
    @var K[1:4] κ[1:12]

    #We choose colors with maximum distinguishability
    hexPoints = [(0,0),(1,0),(2,0),(4,1),(4,2),(3,2),(2,2),(0,1),(3,1),(1,1)]
    K = [(κ[2]+κ[3])/κ[1], (κ[5]+κ[6])/κ[4], (κ[8]+κ[9])/κ[7], (κ[11]+κ[12])/κ[10]]
    aη = κ[3]*κ[12] - κ[6]*κ[9]
    bη = (K[2] + K[3])*κ[3]*κ[12] - (K[1]+K[4])*κ[6]*κ[9]
    coefficients = [K[1]^3*K[3]^2*κ[6]^3*κ[12]^2, K[1]^2*K[2]*K[3]^2*κ[3]*κ[6]^2*κ[12]^2, K[1]^2*K[2]*K[3]*K[4]*κ[3]*κ[6]^2*κ[9]*κ[12], K[1]*K[2]^2*K[4]*κ[3]^2*κ[6]*κ[9]^2,
                    K[2]^2*K[4]*κ[3]^2*κ[9]*aη, K[2]^2*K[3]*κ[3]^2*κ[12]*aη, K[1]*K[2]*K[3]*κ[3]*κ[6]*κ[12]*aη, K[1]^2*K[3]^2*κ[6]^3*κ[12]^2, 
                    2*K[1]*K[2]*K[3]*K[4]*κ[3]^2*κ[6]*κ[9]*κ[12], 2*K[1]^2*K[2]*K[3]*κ[3]*κ[6]^2*κ[12]^2]
    mcoef = K[1]*K[2]*K[3]*κ[3]*κ[6]*κ[12]*bη

    triangconfigurations = [[[2,7,9],[3,6,10],[1,5],[4,8]], [[3,6,10],[2,4,7],[1,5],[8,9]], [[1,3,6],[2,5,7],[9,10],[4,8]],
    [[1,3,6],[2,5,7],[8,9],[4,10]], [[3,6,8],[2,7,9],[4,10],[1,5]], [[2,4,7],[3,6,8],[1,5],[9,10]],
    [[1,4,6],[2,5,8],[3,7],[9,10]], [[1,7,9],[3,5,10],[2,6],[4,8]], [[3,5,8],[1,4,7],[9,10],[2,6]],
    [[1,7,9],[3,5,8],[2,6],[4,10]], [[1,6,9],[2,5,8],[3,7],[4,10]], [[1,4,7],[3,5,10],[8,9],[2,6]],
    [[2,5,10],[1,4,6],[3,7],[8,9]], [[1,6,9],[2,5,10],[3,7],[4,8]]]
    #Check if all covers are legit
    for config in triangconfigurations
        all(t->t in union(config[1],config[2],config[3],config[4]),1:10) || display(config)&&throw(error("The triangles don't cover the entire region"))
        isempty(intersect(config[1],config[2])) && isempty(intersect(config[1],config[3])) && isempty(intersect(config[1],config[4])) && isempty(intersect(config[2],config[3])) && isempty(intersect(config[2],config[4])) && isempty(intersect(config[3],config[4])) || display(config)&&throw(error("Each vertex should only be used once"))
    end

    lineconfigurations = [[[5,1],[7,3],[8,9],[6,2],[4,10]], [[5,1],[7,3],[9,10],[6,2],[8,4]]]

    θ = createθcircuits(hexPoints, K, κ, coefficients, lineconfigurations, triangconfigurations)
    empiricalComparisonOfTwoCovers(θ[cover_suggested], θ[cover_baseline], K, κ, aη, bη, mcoef; numberOfSamplingRuns=250, boxsizes=[5 for _ in 1:8], cover_1=cover_suggested, cover_2 = cover_baseline)
end


function createθcircuit_montecarlo(points, coefficients, configurations, cur_weights)
    output = []
    _configurations = copy(configurations)
    for (index, configuration) in enumerate(configurations)
        _configuration = copy(configuration)
        while !isempty(_configuration)
            global weight = cur_weights[index]
            config = pop!(_configuration)
            global foundindex = nothing
            for (j, o) in enumerate(output)
                if Set(o[3]) == Set(config)
                    global weight = weight + o[2]
                    global foundindex = j
                    break
                end
            end
            if foundindex != nothing
                deleteat!(output, foundindex)
            end    
    
            if length(config)==2
                global barycenter = Matrix{Float64}(undef,2,2); barycenter[1,:] = [points[entry][1] for entry in config]; barycenter[2,:] = [points[entry][2] for entry in config]; 
                global λ = (det(barycenter) == 0) ? [0.5,0.5] : collect(inv(barycenter)*[2,1] / sum(inv(barycenter)*[2,1]))
                push!(output, [prod([(weight*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]), weight, config])
            elseif length(config)==3
                global barycenter, msolve = Matrix{Float64}(undef,3,3), [2,1,1]; barycenter[1,:] = [points[entry][1] for entry in config]; barycenter[2,:] = [points[entry][2] for entry in config]; barycenter[3,:] = [1 for entry in config];
                global λ = collect(inv(barycenter)*msolve / sum(inv(barycenter)*msolve));
                push!(output, [prod([(weight*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]), weight, config])
            end
        end
    end
    display(sum([o[2] for o in output]))
    return sum([o[1] for o in output])
end

function createθcircuits_weighted(points, coefficients, configurations; discretization=25)
    θdict = Dict()
    length(configurations)>=3 || throw(error("Since we are providing a 2D heatmap of the covers, at least 3 simplicial configurations need to be provided!"))
    for i in 1:length(configurations), j in i+1:length(configurations), k in j+1:length(configurations)
        ijkDict = Dict()
        for ω1 in 0:1/discretization:1
            ijkDict[ω1] = []
        end

        for ω1 in 0:round(1/discretization, sigdigits=5):1, ω2 in 0:round(1/discretization, sigdigits=5):round(1-ω1, sigdigits=5)
            configuration1, configuration2, configuration3, helper = Base.copy(configurations[i]), Base.copy(configurations[j]), Base.copy(configurations[k]), []
            while !isempty(configuration1)
                config = pop!(configuration1)
                if length(config)==2
                    global barycenter = Matrix{Float64}(undef,2,2); barycenter[1,:] = [points[entry][1] for entry in config]; barycenter[2,:] = [points[entry][2] for entry in config]; 
                    global λ = (det(barycenter) == 0) ? [0.5,0.5] : collect(inv(barycenter)*[2,1] / sum(inv(barycenter)*[2,1]))

                    if config in configuration2 && config in configuration3
                        deleteat!(configuration2, findfirst(entry -> config==entry, configuration2))
                        deleteat!(configuration3, findfirst(entry -> config==entry, configuration3))
                        push!(helper, prod([(coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    elseif config in configuration2 || config in configuration3
                        (config in configuration2) ? deleteat!(configuration2, findfirst(entry -> config==entry, configuration2)) : deleteat!(configuration3, findfirst(entry -> config==entry, configuration3))
                        push!(helper, prod([(((config in configuration2) ? ω1+ω2 : 1-ω2)*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    else
                        push!(helper, prod([(ω1*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    end
                elseif length(config)==3
                    global barycenter, msolve = Matrix{Float64}(undef,3,3), [2,1,1]; barycenter[1,:] = [points[entry][1] for entry in config]; barycenter[2,:] = [points[entry][2] for entry in config]; barycenter[3,:] = [1 for entry in config];
                    global λ = collect(inv(barycenter)*msolve / sum(inv(barycenter)*msolve));
                    if config in configuration2 && config in configuration3
                        deleteat!(configuration2, findfirst(entry -> config==entry, configuration2))
                        push!(helper, prod([(coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    elseif config in configuration2 || config in configuration3
                        (config in configuration2) ? deleteat!(configuration2, findfirst(entry -> config==entry, configuration2)) : deleteat!(configuration3, findfirst(entry -> config==entry, configuration3))
                        push!(helper, prod([(((config in configuration2) ? ω1+ω2 : 1-ω2)*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    else
                        push!(helper, prod([(ω1*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    end
                end
            end

            while !isempty(configuration2)
                config = pop!(configuration2)
                if length(config)==2
                    global barycenter = Matrix{Float64}(undef,2,2); barycenter[1,:] = [points[entry][1] for entry in config]; barycenter[2,:] = [points[entry][2] for entry in config]; 
                    global λ = (det(barycenter) == 0) ? [0.5,0.5] : collect(inv(barycenter)*[2,1] / sum(inv(barycenter)*[2,1]))
                    if config in configuration3
                        deleteat!(configuration3, findfirst(entry -> config==entry, configuration3))
                        push!(helper, prod([((1-ω1)*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    else
                        push!(helper, prod([(ω2*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    end
                elseif length(config)==3
                    global barycenter, msolve = Matrix{Float64}(undef,3,3), [2,1,1]; barycenter[1,:] = [points[entry][1] for entry in config]; barycenter[2,:] = [points[entry][2] for entry in config]; barycenter[3,:] = [1 for entry in config];
                    global λ = collect(inv(barycenter)*msolve / sum(inv(barycenter)*msolve));
                    if config in configuration3
                        deleteat!(configuration3, findfirst(entry -> config==entry, configuration3))
                        push!(helper, prod([((1-ω1)*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    else
                        push!(helper, prod([(ω2*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                    end
                end
            end

            while !isempty(configuration3)
                config = pop!(configuration3)
                if length(config)==2
                    global barycenter = Matrix{Float64}(undef,2,2); barycenter[1,:] = [points[entry][1] for entry in config]; barycenter[2,:] = [points[entry][2] for entry in config]; 
                    global λ = (det(barycenter) == 0) ? [0.5,0.5] : collect(inv(barycenter)*[2,1] / sum(inv(barycenter)*[2,1]))
                    push!(helper, prod([((1-ω1-ω2)*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                elseif length(config)==3
                    global barycenter, msolve = Matrix{Float64}(undef,3,3), [2,1,1]; barycenter[1,:] = [points[entry][1] for entry in config]; barycenter[2,:] = [points[entry][2] for entry in config]; barycenter[3,:] = [1 for entry in config];
                    global λ = collect(inv(barycenter)*msolve / sum(inv(barycenter)*msolve));
                    push!(helper, prod([((1-ω1-ω2)*coefficients[config[i]]/λ[i])^(λ[i]) for i in 1:length(config)]))
                end
            end

            push!(ijkDict[ω1], sum(helper))
        end
        θdict[(i,j,k)] = ijkDict
    end
    return θdict
end



function runSamplingComparison_weighted(θ, θ_weighted, κs, aη, bη, mcoef; discretization, boxsize=100, numberOfSamplingRuns=250, prefix="linearweight", suffix="")
    #If the file exists, we add to the previously run tests. Else, we set everything to 0.
    θkeys = keys(θ_weighted)
    ourmodel = Dict()
    no_other = Dict()
    try
        f = open("../data/$(prefix)$(suffix)storedsolutions$(boxsize).txt", "r")
        global pointnumber = parse(Int,readline(f))
        while ! eof(f)  
            sstring = split(readline(f), ": ")
            keystring = split(sstring[1], "; ")
            key = (parse(Int,keystring[1][2]), (parse(Int,keystring[1][5])))
            weight = parse(Float64,keystring[2])
            ourmodel[(key, weight)] = [parse(Int,entry) for entry in split(sstring[2][2:end-1], ", ")]
         end
        close(f)
    catch
        global pointnumber = 0
        for key in θkeys
            ijkkeys = keys(θ_weighted[key])
            for weight in ijkkeys
                ourmodel[(key, weight)] = [0 for _ in 1:length(θ_weighted[key][weight])]
            end
        end
    end

    for sampleindex in 1:numberOfSamplingRuns
        display("Run: $(sampleindex)")
        global sampling = filter(sampler -> !any(t->isapprox(t,0), sampler) && evaluate(aη,κs=>sampler)>0 && evaluate(bη,κs=>sampler)<0, [boxsize * abs.(rand(Float64,length(κs))) for _ in 1:1000000])
        global pointnumber = pointnumber+length(sampling)
        for ind in 1:length(sampling)
            sampler = sampling[ind]
            mval = real(evaluate(mcoef,κs=>sampler))
            for key in θkeys
                for weight in keys(θ_weighted[key])
                    for j in 1:length(θ_weighted[key][weight])
                        ourval = evaluate(θ_weighted[key][weight][j], κs=>sampler)
                        if real(ourval) >= -mval
                            global ourmodel[(key,weight)][j] += 1
                        end
                    end
                end
            end
            
        end

        #SAVE the data to the file `NEWtriangstoredsolutions.txt`
        open("../data/$(prefix)$(suffix)storedsolutions$(boxsize).txt", "w") do file
            write(file, "$(pointnumber)\n")
            for key in keys(ourmodel)
                write(file, "$(key[1]); $(key[2]): $(ourmodel[key])\n")
            end
        end
    end
end

function runSamplingComparison_MC(sampling, θ_MC, κs, aη, bη, mcoef; discretization, boxsize=100, numberOfSamplingRuns=250, prefix="linearweight", suffix="")
    #If the file exists, we add to the previously run tests. Else, we set everything to 0.
    global ourmodel = 0
    global pointnumber = 0
    
    #TODO Create a single list of samples that is re-used every time. 

    @showprogress for ind in 1:length(sampling)
        sampler = sampling[ind]
        mval = real(evaluate(mcoef,κs=>sampler))
        ourval = evaluate(θ_MC, κs=>sampler)
        if real(ourval) >= -mval
            global ourmodel += 1
        end            
    end
    return ourmodel
end

end