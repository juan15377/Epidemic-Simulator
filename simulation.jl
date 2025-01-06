using NearestNeighbors
using Distances 
using Random
using SparseArrays
using JLD2
using Distributions
using Colors

const HEALTHY = :healthy 
const INFECTED = :infected
const INMUNIZED = :inmunized
const DEAD = :dead 

const STATES = [HEALTHY, INFECTED, INMUNIZED, DEAD]

mutable struct ParametersSimulation
    infected_states_probs::AbstractVector{Float32}  # Permitir diferentes tipos de vectores
    distribution_states_agent::Categorical
    get_agent_state::Function
    interval_between_infected::Int16
    prob_infect::Float32
    max_radius_infected::Int16
    max_infected::Int16
    size_agent::Int8
    num_agents::Int16
    height::Int16
    width::Int16
    initial_infected::Int16
    speed_infected::Int8
    speed_healthy::Int8
    time_rate::Float32

    function ParametersSimulation(
        infected_states_probs::AbstractVector{Float32},
        interval_between_infected::Int16,
        prob_infect::Float32,
        max_radius_infected::Int16,
        max_infected::Int16,
        size_agent::Int8,
        num_agents::Int16,
        height::Int16,
        width::Int16,
        initial_infected::Int16,
        speed_infected::Int8,
        speed_healthy::Int8,
        time_rate::Float32;
    )
        # Validaciones
        @assert 0 <= prob_infect <= 1 "La probabilidad de infección debe estar entre 0 y 1"
        @assert initial_infected <= num_agents "Los infectados iniciales no pueden exceder el número total de agentes"

        # Función para obtener el estado del agente
        distribution_states_agent = Categorical(infected_states_probs)
        get_agent_state = () -> STATES[rand(distribution_states_agent)]

        return new(
            infected_states_probs,
            distribution_states_agent,
            get_agent_state,
            interval_between_infected,
            prob_infect,
            max_radius_infected,
            max_infected,
            size_agent,
            num_agents,
            height,
            width,
            initial_infected,
            speed_infected,
            speed_healthy,
            time_rate
        )
    end
end

# Resto del código sin cambios
infected_states_probs ::Vector{Float32} = 
   [0.001, 1 - 2*0.001 - 0.1, 0.001, 0.1]

interval_between_infected::Int16 = 10
prob_infect::Float32 = 0.9
max_radius_infected::Int16 = 50
max_infected::Int16 = 4
size_agent::Int8 = 30
num_agents::Int16 = 20
height::Int16 = 800
width::Int16 = 1300
initial_infected::Int16 = 1
speed_infected::Int8, speed_healthy::Int8 = 10, 100
time_rate::Float32 = 60

params = ParametersSimulation(
    infected_states_probs,
    interval_between_infected,
    prob_infect,
    max_radius_infected,
    max_infected,
    size_agent,
    num_agents,
    height,
    width,
    initial_infected,
    speed_infected,
    speed_healthy,
    time_rate
)


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

        x = x / sqrt(x^2 + y^2)
        y = y / sqrt(x^2 + y^2)

        x = speed * x 
        y = speed * y

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

function check_wall_collission(agents::Vector{Agent}, params::ParametersSimulation)
    for agent in agents
        if agent.state == DEAD
            continue
        end
        if agent.x - params.size_agent < 0 || agent.x > params.width - params.size_agent
            agent.velocity_vector.x = -agent.velocity_vector.x
        end
        if agent.y - params.size_agent < 0 || agent.y > params.height - params.size_agent
            agent.velocity_vector.y = -agent.velocity_vector.y
        end
    end 
end

function update_velocities_on_collision(agent1::Agent, agent2::Agent)

    agent1.velocity_vector = agent1.velocity_vector + agent2.velocity_vector
    agent2.velocity_vector = agent2.velocity_vector + agent1.velocity_vector

end


function get_pair_on_collision(matrix_distances::Matrix{Float32}, params::ParametersSimulation)

    index_agents_collision = []

    n, n = size(matrix_distances)

    for i in 1:n
        for j in i+1:n
            if matrix_distances[i, j] < 2*params.size_agent
                push!(index_agents_collision, (i, j))
            end
        end
    end
    return index_agents_collision

end


# include wall colissions
function update_velocities_on_collision(agents::Vector{Agent}, matrix_distances::Matrix{Float32}, params::ParametersSimulation)
    index_agents_collision = get_pair_on_collision(matrix_distances, params)

    for (i, j) in index_agents_collision
        update_velocities_on_collision(agents[i], agents[j])
    end

    check_wall_collission(agents, params)
end

function update_positions(agents::Array{Agent}, params::ParametersSimulation)
    for agent in agents
        if agent.state == DEAD
            continue
        end
        agent.x += agent.velocity_vector.x * (1 / params.time_rate) 
        agent.y += agent.velocity_vector.y * (1 / params.time_rate)
    end
end

function distance(a1::Agent, a2::Agent)
    return sqrt((a1.x - a2.x)^2 + (a1.y - a2.y)^2)
end


function update_states(agents::Array{Agent}, matrix_adjacency::Matrix{Bool}, matrix_distances::Matrix{Float32}, time::Int64, params::ParametersSimulation)


    function infect_agents(agent)
        # infecta a todos los que se encuentran en la matriz de adyacencia
        @inbounds for i in 1:length(agents)
            if matrix_adjacency[agent, i] && agents[i].state == HEALTHY && matrix_distances[agent, i] < params.max_radius_infected
                agents[i].state = INFECTED
            end
        end
    end 
    # update healthy to infected  
    for i in 1:length(agents)
        if agents[i].state == INFECTED
            agents[i].time_infected += 1
            if time % params.time_rate == 0
                state = params.get_agent_state()
                agents[i].state = state
            end
            if agents[i].state == INFECTED
                if rand() < params.prob_infect && agents[i].time_infected % params.interval_between_infected == 0
                    infect_agents(i)
                end
            end 
        end
    end
end


function generate_agents(params::ParametersSimulation)
    agents = []

    for i in 1:params.num_agents-params.initial_infected
        x::Float32 = rand() * (params.width - 2*params.size_agent)
        y::Float32 = rand() * (params.height - 2*params.size_agent)
        state = HEALTHY
        velocity_vector = VelocityVector(params.speed_healthy)
        push!(agents, Agent(x, y, state, velocity_vector, 0))
    end

    for i in 1:params.initial_infected
        x = rand() * (params.width - 2*params.size_agent)
        y = rand() *  (params.height - 2*params.size_agent)
        state = INFECTED
        velocity_vector = VelocityVector(params.speed_infected)
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

    function AdjacencyMatrixGraph(agents::Vector{Agent}, params::ParametersSimulation)
        x_points = [agent.x for agent in agents]
        y_points = [agent.y for agent in agents]

        xy_points = vcat(x_points', y_points')

        matrix_graph = sparse(zeros(Bool, length(agents), length(agents)))

        kd_tree = KDTree(xy_points)

        indices, distances = knn(kd_tree, xy_points, Int64(params.max_infected), true)

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


function generate_simulation(path_simulation::AbstractString, time_seg::Int64, params::ParametersSimulation)
    agents::Vector{Agent} = generate_agents(params)
    matrix_distances = MatrixDistances(agents)
    matrix_adjacency = AdjacencyMatrixGraph(agents, params)
    frame_simulation = FrameSimulation(agents, matrix_adjacency.matrix, 0)

    JLD2.jldopen(path_simulation, "w") do file
        file["frame_simulation_0"] = frame_simulation
        file["WIDTH"] = params.width
        file["HEIGHT"] = params.height
        file["SIZE_AGENT"] = params.size_agent
    end

    for i in 1:time_seg * 60
        update_positions(agents, params)
        matrix_distances = MatrixDistances(agents)
        update_velocities_on_collision(agents, matrix_distances.matrix, params)
        update_states(agents, matrix_adjacency.matrix, matrix_distances.matrix, i, params)
        matrix_adjacency = AdjacencyMatrixGraph(agents, params)

        frame_simulation = FrameSimulation(agents, matrix_adjacency.matrix, i)

        JLD2.jldopen(path_simulation, "a") do file
            file["frame_simulation_$i"] = frame_simulation
        end
    end

end



@time generate_simulation("simulations/sim_2.jld2", 5, params)

using GameZero
rungame("run.jl")