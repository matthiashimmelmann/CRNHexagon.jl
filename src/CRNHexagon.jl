module CRNHexagon

import HomotopyContinuation: @var, evaluate, Expression

include("auxiliary_functions.jl")
include("plotting_functionality.jl")

export runTest, computeCoverInvariants, printValues, runTest_weighted, runTest_montecarlo


# Initialization
global hexPoints = [(0,0),(1,0),(2,0),(4,1),(4,2),(3,2),(2,2),(0,1),(3,1),(1,1)]
@var κ[1:12]
global K = [(κ[2]+κ[3])/κ[1], (κ[5]+κ[6])/κ[4], (κ[8]+κ[9])/κ[7], (κ[11]+κ[12])/κ[10]]
global aη = κ[3]*κ[12] - κ[6]*κ[9]
global bη = (K[2] + K[3])*κ[3]*κ[12] - (K[1]+K[4])*κ[6]*κ[9]
coefficients = [K[1]^3*K[3]^2*κ[6]^3*κ[12]^2, K[1]^2*K[2]*K[3]^2*κ[3]*κ[6]^2*κ[12]^2, K[1]^2*K[2]*K[3]*K[4]*κ[3]*κ[6]^2*κ[9]*κ[12], 
                K[1]*K[2]^2*K[4]*κ[3]^2*κ[6]*κ[9]^2, K[2]^2*K[4]*κ[3]^2*κ[9]*aη, K[2]^2*K[3]*κ[3]^2*κ[12]*aη, 
                K[1]*K[2]*K[3]*κ[3]*κ[6]*κ[12]*aη, K[1]^2*K[3]^2*κ[6]^3*κ[12]^2, 2*K[1]*K[2]*K[3]*K[4]*κ[3]^2*κ[6]*κ[9]*κ[12], 
                2*K[1]^2*K[2]*K[3]*κ[3]*κ[6]^2*κ[12]^2]
global mcoef = K[1]*K[2]*K[3]*κ[3]*κ[6]*κ[12]*bη

global triangConfigurations = [[[2,7,9],[3,6,10],[1,5],[4,8]], [[3,6,10],[2,4,7],[1,5],[8,9]], [[1,3,6],[2,5,7],[9,10],[4,8]],
    [[1,3,6],[2,5,7],[8,9],[4,10]], [[3,6,8],[2,7,9],[4,10],[1,5]], [[2,4,7],[3,6,8],[1,5],[9,10]],
    [[1,4,6],[2,5,8],[3,7],[9,10]], [[1,7,9],[3,5,10],[2,6],[4,8]], [[3,5,8],[1,4,7],[9,10],[2,6]],
    [[1,7,9],[3,5,8],[2,6],[4,10]], [[1,6,9],[2,5,8],[3,7],[4,10]], [[1,4,7],[3,5,10],[8,9],[2,6]],
    [[2,5,10],[1,4,6],[3,7],[8,9]], [[1,6,9],[2,5,10],[3,7],[4,8]]]

# Check if all covers are legitimate
for config in triangConfigurations
    all(t->t in union(config[1],config[2],config[3],config[4]),1:10) || display(config)&&throw(error("The triangles don't cover the entire region"))
    isempty(intersect(config[1],config[2])) && isempty(intersect(config[1],config[3])) && isempty(intersect(config[1],config[4])) && isempty(intersect(config[2],config[3])) && isempty(intersect(config[2],config[4])) && isempty(intersect(config[3],config[4])) || display(config)&&throw(error("Each vertex should only be used once"))
end
global lineConfigurations = [[[1,5],[7,3],[8,9],[6,2],[4,10]], [[1,5],[7,3],[9,10],[6,2],[8,4]]]



global montecarloconfigurations = [[[1,4,6],[2,5,8],[3,7],[9,10]], [[1,7,9],[3,5,10],[2,6],[4,8]], [[3,5,8],[1,4,7],[9,10],[2,6]],
[[1,7,9],[3,5,8],[2,6],[4,10]], [[1,6,9],[2,5,8],[3,7],[4,10]], [[1,4,7],[3,5,10],[8,9],[2,6]],
[[2,5,10],[1,4,6],[3,7],[8,9]], [[1,6,9],[2,5,10],[3,7],[4,8]], [[1,5],[7,3],[8,9],[6,2],[4,10]], [[1,5],[7,3],[9,10],[6,2],[8,4]]]
# Methods

#=
This is the main method. Use it to run all tests. By generating the circuit numbers of all 16 circuit covers
associated to the hexagonal face H of the Newton polytope of the dual phosphorylation network, we are able to 
compare their quality. All covers are compared to the cover CC(9) introduced by Feliu et al.
=#
function runTest( ; boxsize=1, numberOfSamplingRuns=150, prefix="michaelismentontest", suffix="NEW")
    θ = auxiliary_functions.createθcircuits(hexPoints, coefficients, lineConfigurations, triangConfigurations)
    auxiliary_functions.runSamplingComparison(θ, κ, aη, bη, mcoef, θ[9]; boxsize=boxsize, numberOfSamplingRuns=numberOfSamplingRuns, prefix=prefix, suffix=suffix)
end

#=
This is the test functionality for weighted covers. Currently, the test is implemented for the combinations of 3 covers,
creating a simplicial homotopy.
=#
function runTest_montecarlo(; boxsize=1, numberOfSamplingRuns=20, discretization=25)    
    # Check if the input `coversToCompare` has the correct format
    global curval = [1/10 for _ in 1:10]
    global cur_coverage = [0, curval]
    global none_found = 0
    global sampling = []
    global pointnumber = 0
    for sampleindex in 1:numberOfSamplingRuns
        global sampling = vcat(sampling, filter(sampler -> !any(t->isapprox(t,0), sampler) && evaluate(aη,κ=>sampler)>0 && evaluate(bη,κ=>sampler)<0, [boxsize * abs.(rand(Float64,length(κ))) for _ in 1:1000000]))
    end
    global pointnumber = length(sampling)
    while true
        global prev_val = curval
        θ_montecarlo = auxiliary_functions.createθcircuit_montecarlo(hexPoints, coefficients, montecarloconfigurations, curval)
        our_model = auxiliary_functions.runSamplingComparison_MC(sampling, θ_montecarlo, κ, aη, bη, mcoef; boxsize=boxsize, numberOfSamplingRuns=numberOfSamplingRuns, discretization=discretization)
        cur_error = our_model/pointnumber
        display(cur_error)
        if cur_coverage[1] < cur_error
            global cur_coverage = [cur_error, curval]
            global none_found = 0
        else
            none_found = none_found+1
            if none_found > 10
                return cur_coverage
            end
            global curval = prev_val
        end
        global curval = map(t->t<0 ? 0 : t, curval + 1/discretization .* rand([-1,0,1],10))
        global curval = curval/sum(curval)    
    end
    return cur_coverage
end


#=
This is the test functionality for weighted covers. Currently, the test is implemented for the combinations of 3 covers,
creating a simplicial homotopy.
=#
function runTest_weighted(coversToCompare; boxsize=1, numberOfSamplingRuns=100, prefix="linearweight", suffix="4,9", discretization=10)
    length(coversToCompare)==3 || throw(error("Exactly 3 distinct covers need to be provided!"))
    
    # Check if the input `coversToCompare` has the correct format
    foreach(conf->sort(conf) in [sort(en) for en in vcat(triangConfigurations, lineConfigurations)] || throw(error("The configurations do not have the correct format. Feel free to copy the list from the follwoing:\n
        [[2,7,9],[3,6,10],[1,5],[4,8]], [[3,6,10],[2,4,7],[1,5],[8,9]], [[1,3,6],[2,5,7],[9,10],[4,8]],
        [[1,3,6],[2,5,7],[8,9],[4,10]], [[3,6,8],[2,7,9],[4,10],[1,5]], [[2,4,7],[3,6,8],[1,5],[9,10]],
        [[1,4,6],[2,5,8],[3,7],[9,10]], [[1,7,9],[3,5,10],[2,6],[4,8]], [[3,5,8],[1,4,7],[9,10],[2,6]],
        [[1,7,9],[3,5,8],[2,6],[4,10]], [[1,6,9],[2,5,8],[3,7],[4,10]], [[1,4,7],[3,5,10],[8,9],[2,6]],
        [[2,5,10],[1,4,6],[3,7],[8,9]], [[1,6,9],[2,5,10],[3,7],[4,8]], [[1,5],[7,3],[8,9],[6,2],[4,10]], 
        [[1,5],[7,3],[9,10],[6,2],[8,4]]."))
    , coversToCompare)

    θ_weighted = auxiliary_functions.createθcircuits_weighted(hexPoints, coefficients, coversToCompare; discretization=discretization)
    auxiliary_functions.runSamplingComparison_weighted([], θ_weighted, κ, aη, bη, mcoef; boxsize=boxsize, numberOfSamplingRuns=numberOfSamplingRuns, prefix=prefix, suffix=suffix, discretization=discretization)    
end

#=
This function produces the Latex output for the sampled data from the method `runTest`. This data leads to the
compilation of Tables 1 and 2 in the article connected to this repository.
=#
function computeCoverInvariants( ; boxsizes=["0.1","1","10","100"], prefix="michaelismentontest", suffix="NEW")
    for boxsize in boxsizes
        try
            f = open("../data/$(prefix)$(suffix)triangstoredsolutions$(boxsize).txt", "r")
            global pointnumber = parse(Int,readline(f))
            global allmodels = parse(Int,readline(f))
            global onlyone = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]
            global allpoints = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]
            global ourmodel = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]
            global prevmodel = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]
            global nomodel = [parse(Int,entry) for entry in split(readline(f)[2:end-1], ", ")]
            close(f)
        catch
            continue
        end
        println("$(boxsize): $(pointnumber) points contained.")
        global allmodeldots, ourmodeldots, prevmodeldots, nomodeldots, puremodel = [[] for _ in 1:length(onlyone)], [[] for _ in 1:length(onlyone)], [[] for _ in 1:length(onlyone)], [[] for _ in 1:length(onlyone)], [[] for _ in 1:length(onlyone)]

        permille_allmodels, permille_ourmodel, permille_prevmodel, permille_nomodel, percent_ourmodel_pure = round.(round.(10000*allpoints ./ pointnumber, digits=2)/10000, digits=5), round.(100*ourmodel ./ pointnumber, digits=2), round.(100*prevmodel ./ pointnumber, digits=2), round.(100*nomodel ./ pointnumber, digits=2), round.(100 * (pointnumber .- (nomodel .+ prevmodel)) ./ pointnumber, digits=3)
        foreach(i->push!(allmodeldots[i], permille_allmodels[i]), 1:length(permille_allmodels))
        foreach(i->push!(ourmodeldots[i], permille_ourmodel[i]), 1:length(permille_ourmodel))
        foreach(i->push!(prevmodeldots[i], permille_prevmodel[i]), 1:length(permille_ourmodel))
        foreach(i->push!(nomodeldots[i], permille_nomodel[i]), 1:length(permille_ourmodel))
        foreach(i->push!(puremodel[i], percent_ourmodel_pure[i]), 1:length(percent_ourmodel_pure))
    end

    
    print("~&+&-&0&+&-&0&+&-&0\\\\ \\thickhline \n\n")
    for θ in 1:Int(length(ourmodeldots))
        print("$(θ)")
        for i in 1:length(ourmodeldots[θ])
            print("&$(ourmodeldots[θ][i])&$(prevmodeldots[θ][i])&$(nomodeldots[θ][i])")
        end
        print("\\\\ \\hline \n\n")
    end


    for θ in 1:Int(length(allmodeldots))
        print("$(θ)&")
        for i in 1:length(allmodeldots[θ])-1
            print("$(allmodeldots[θ][i])&")
        end
        print("$(allmodeldots[θ][end])")
        print("\\\\ \\hline \n")
    end
end


#=
The method `printValues` displays for the data previously obtained from `runTest`. This data is subsequently
used to compose Figure 6 in the article associated with this repository.
=#
function printValues( ; boxsizes = ["0.1","1","10","100"], prefix="michaelismentontest", suffix="NEW")
    for boxsize in boxsizes
        print("$(boxsize): \n")
        f = open("../data/$(prefix)$(suffix)triangstoredsolutions$(boxsize).txt", "r")
        global relDict = Dict()
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
        containmentArray, almostContainmentArray = [[] for _ in 1:16], [[] for _ in 1:16]
        for key in keys(relDict)
            if relDict[key]==0
                push!(containmentArray[key[1]], key[2])
            elseif relDict[key]<10
                push!(almostContainmentArray[key[1]], key[2])
            end
        end
        println("Containment")
        foreach(t->println("$(t): $(containmentArray[t])"), 1:16)
        println("AlmostContainment")
        foreach(t->println("$(t): $(almostContainmentArray[t])"), 1:16)
    end
end

end 
