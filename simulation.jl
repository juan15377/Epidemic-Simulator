using NearestNeighbors
using Distances 
using Random
using SparseArrays
using JLD2
using Distributions


const HEALTHY = :healthy 
const INFECTED = :infected
const INMUNIZED = :inmunized
const DEAD = :dead 

const STATES = [HEALTHY, INFECTED, INMUNIZED, DEAD]


begin 

const infected_states_probs::Vector{Float32} = 
#  healthy  infected  inmunized  dead
   [0.001,      1 - 2* 0.001 - 0.1,     0.001,     0.1]  # infected

const DISTR_AGENT_STATE = Categorical(infected_states_probs)

@inline const GET_AGENT_STATE() = STATES[rand(DISTR_AGENT_STATE)]
# solo hay una forma de pasar de sano a infectado


const interval_beetween_infected::Int16 = 10 # 60 is a 1 one seg 

const prob_infect::Float32 = 0.9

const max_radius_infected = 50

const max_infected::Int16 = 4

const size_agent::Int8 = 20

const num_agents::Int16 = 10

const Height::Int16 = 1200

const Width::Int16 = 1200

const initial_infected::Int16 = 2

const speed_infected::Int8, speed_healthy::Int8 = 20, 100

const time_rate::Float32 = 60

end 

mutable struct VelocityVector
    x::Float32
    y::Float32
    speed::Int8

    function VelocityVector(speed::Int8)
        x = rand() * 2 - 1
        y = rand() * 2 - 1

        x = x / sqrt(x^2 + y^2)
        y = y / sqrt(x^2 + y^2)

        x = speed * x 
        y = speed * y

        return new(x, y, speed)
    end

    function VelocityVector(x::Float32, y::Float32, speed::Int8)

        return new(x, y, speed)
    end

end


function normalize(v::VelocityVector)
    v.x = v.x / √(v.x^2 + v.y^2)
    v.y = v.y / √(v.x^2 + v.y^2)
    return v
end

import Base.*
import Base.+

*(m::Number, v::VelocityVector) = VelocityVector(m * v.x, m * v.y, v.speed)
+(v1::VelocityVector, v2::VelocityVector) = v1.speed * normalize(VelocityVector(v1.x - v2.x, v1.y -  v2.y, v1.speed))


mutable struct Agent
    x::Float32
    y::Float32 
    state::Symbol 
    velocity_vector::VelocityVector
    time_infected::Int32 # measured in frames
end 

function check_wall_collission(agents::Vector{Agent})
    for agent in agents
        if agent.state == DEAD
            continue
        end
        if agent.x < 0 || agent.x > Width
            agent.velocity_vector.x = -agent.velocity_vector.x
        end
        if agent.y < 0 || agent.y > Height
            agent.velocity_vector.y = -agent.velocity_vector.y
        end
    end 
end

function update_velocities_on_collision(agent1::Agent, agent2::Agent)

    agent1.velocity_vector = agent1.velocity_vector + agent2.velocity_vector
    agent2.velocity_vector = agent2.velocity_vector + agent1.velocity_vector

end


function get_pair_on_collision(matrix_distances::Matrix{Float32})

    index_agents_collision = []

    n, n = size(matrix_distances)

    for i in 1:n
        for j in i+1:n
            if matrix_distances[i, j] < size_agent
                push!(index_agents_collision, (i, j))
            end
        end
    end
    return index_agents_collision

end

# include wall colissions
function update_velocities_on_collision(agents::Vector{Agent}, matrix_distances::Matrix{Float32})
    index_agents_collision = get_pair_on_collision(matrix_distances)

    for (i, j) in index_agents_collision
        update_velocities_on_collision(agents[i], agents[j])
    end

    check_wall_collission(agents)
end

function update_positions(agents::Array{Agent})
    for agent in agents
        if agent.state == DEAD
            continue
        end
        agent.x += agent.velocity_vector.x * (1 / time_rate) 
        agent.y += agent.velocity_vector.y * (1 / time_rate)
    end
end

function distance(a1::Agent, a2::Agent)
    return sqrt((a1.x - a2.x)^2 + (a1.y - a2.y)^2)
end


function update_states(agents::Array{Agent}, matrix_adjacency::Matrix{Bool}, matrix_distances::Matrix{Float32}, time::Int64)


    function infect_agents(agent)
        # infecta a todos los que se encuentran en la matriz de adyacencia
        @inbounds for i in 1:length(agents)
            if matrix_adjacency[agent, i] && agents[i].state == HEALTHY && matrix_distances[agent, i] < max_radius_infected
                agents[i].state = INFECTED
            end
        end
    end 
    # update healthy to infected  
    for i in 1:length(agents)
        if agents[i].state == INFECTED
            agents[i].time_infected += 1
            if time % time_rate == 0
                state = GET_AGENT_STATE()
                agents[i].state = state
            end
            if agents[i].state == INFECTED
                if rand() < prob_infect && agents[i].time_infected % interval_beetween_infected == 0
                    infect_agents(i)
                end
            end 
        end
    end
end


function generate_agents(n::Int16, x_max::Int16, y_max::Int16, num_infected::Int16, speed_infected::Int8, speed_healthy::Int8)
    agents = []

    for i in 1:n-num_infected
        x::Float32 = rand() * x_max
        y::Float32 = rand() * y_max
        state = HEALTHY
        velocity_vector = VelocityVector(speed_healthy)
        push!(agents, Agent(x, y, state, velocity_vector, 0))
    end

    for i in 1:num_infected
        x = rand() * x_max
        y = rand() * y_max
        state = INFECTED
        velocity_vector = VelocityVector(speed_infected)
        push!(agents, Agent(x, y, state, velocity_vector, 0))
    end

    return agents
end


mutable struct MatrixDistances
    agents::Array{Agent}
    matrix::Matrix{Float32}

    function MatrixDistances(agents)
        x_points = [agent.x for agent in agents]
        y_points = [agent.y for agent in agents]

        xy_points = vcat(x_points', y_points')

        matrix_distances = pairwise(Euclidean(), xy_points, xy_points)

        return new(agents, matrix_distances)

    end
end


mutable struct AdjacencyMatrixGraph
    angents::Array{Agent}
    matrix::Matrix{Bool}

    function AdjacencyMatrixGraph(agents::Vector{Agent}, max_infected::Int16)
        x_points = [agent.x for agent in agents]
        y_points = [agent.y for agent in agents]

        xy_points = vcat(x_points', y_points')

        matrix_graph = sparse(zeros(Bool, length(agents), length(agents)))

        kd_tree = KDTree(xy_points)

        indices, distances = knn(kd_tree, xy_points, Int64(max_infected), true)

        @inbounds for ind in indices
            row = ind[1]
            for index in ind[2:end] 
                matrix_graph[row, index] = true
            end
        end

        return new(agents, matrix_graph)

    end
end



mutable struct FrameSimulation
    matrix_points::Matrix{Float32}
    states::Vector{Symbol}
    graph_matrix::AbstractMatrix{Bool}
    time::Int64

    function FrameSimulation(agents::Vector{Agent}, graph_matrix::Matrix{Bool}, time::Int64)

        x_points = [agent.x for agent in agents]
        y_points = [agent.y for agent in agents]

        matrix_points = vcat(x_points', y_points')

        states = [agent.state for agent in agents]

        return new(matrix_points, states, graph_matrix, time)
    
    end 

end


function generate_simulation(path_simulation::AbstractString, time_seg::Int64)
    agents::Vector{Agent} = generate_agents(num_agents, Width, Height, initial_infected, speed_infected, speed_healthy)
    matrix_distances = MatrixDistances(agents)
    matrix_adjacency = AdjacencyMatrixGraph(agents, max_infected)
    frame_simulation = FrameSimulation(agents, matrix_adjacency.matrix, 0)

    JLD2.jldopen(path_simulation, "w") do file
        file["frame_simulation_0"] = frame_simulation
        file["WIDTH"] = Width
        file["HEIGHT"] = Height
        file["SIZE_AGENT"] = size_agent
    end

    for i in 1:time_seg * 60
        update_positions(agents)
        update_velocities_on_collision(agents, matrix_distances.matrix)
        update_states(agents, matrix_adjacency.matrix, matrix_distances.matrix, i)
        matrix_adjacency = AdjacencyMatrixGraph(agents, max_infected)
        matrix_distances = MatrixDistances(agents)

        frame_simulation = FrameSimulation(agents, matrix_adjacency.matrix, i)

        JLD2.jldopen(path_simulation, "a") do file
            file["frame_simulation_$i"] = frame_simulation
        end
    end

end



@time generate_simulation("simulations/sim_3.jld2", 10000)

simulacion = load("simulations/sim_3.jld2")

using GameZero
rungame("run.jl")
