f = open("../data/multistationarity_points.txt", "r")
global multistationary_points = []
while ! eof(f)  
    k_point = [parse(Float64,entry) for entry in split(readline(f)[2:end-1], ",")]
    push!(multistationary_points, k_point)
end
using HomotopyContinuation
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

using JuMP
using DynamicPolynomials
using SumOfSquares
import CSDP
total_points = []
for (i,point) in enumerate(multistationary_points)
    display(i)
    current_coefficients = evaluate.(coefficients, κ=>point)
    current_mcoef = evaluate(mcoef, κ=>point)
    @polyvar x y
    model = SOSModel(CSDP.Optimizer)
    S = @set x >= 0 && y >= 0
    @variable(model, L)
    set_silent(model)
    @constraint(model, sum(current_coefficients[i]*x^hexPoints[i][1]*y^hexPoints[i][2] for i in 1:length(hexPoints)) + current_mcoef * x^2*y >= L, domain = S)
    @objective(model, Max, L)
    optimize!(model)
    if objective_value(model) > 0 && is_solved_and_feasible(model)
        display(point)
        push!(total_points,point)
    end
end