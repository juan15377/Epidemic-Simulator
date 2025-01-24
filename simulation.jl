using NearestNeighbors
using Distances 
using Random
using JLD2
using Distributions
using Colors
using CodecZlib
using SparseArrays

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
infected_states_probs::Vector{Float32} = 
   [0.01, 1 - 2*0.01 - .1, 0.01, 0.1]
 # [HEALTHY, INFECTED,   INMUNIZED, DEAD]

interval_between_infected::Int16 = 30
prob_infect::Float32 = 0.3
max_radius_infected::Int16 = 50
max_infected::Int16 = 3
size_agent::Int8 = 7
num_agents::Int16 = 500
height::Int16 = 800
width::Int16 = 1800
initial_infected::Int16 = 2
speed_infected::Int8, speed_healthy::Int8 = 2, 50
time_rate::Float32 = 30

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

const G = VelocityVector(Float32(0.0), Float32(0.1), Int8(1)) # effect of gravity


function normalize(v::VelocityVector)
    v.x = v.x / √(v.x^2 + v.y^2)
    v.y = v.y / √(v.x^2 + v.y^2)
    return v
end

import Base.*
import Base.+

*(m::Number, v::VelocityVector) = VelocityVector(m * v.x, m * v.y, v.speed)
+(v1::VelocityVector, v2::VelocityVector) = VelocityVector(v1.x + v2.x, v1.y + v2.y, v1.speed)


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



function defined_angle(angulo_auxiliar, x1, y1, x2=0, y2=0)
    # Calcula el ángulo según los cuadrantes
    if x2 == 0 && y2 == 0
        x2, y2 = x1, y1  # Si se usan componentes vectoriales
    end

    if y2 >= y1 && x2 >= x1
        return angulo_auxiliar
    elseif y2 >= y1 && x2 < x1
        return π - angulo_auxiliar
    elseif y2 < y1 && x2 < x1
        return π + angulo_auxiliar
    else
        return 2π - angulo_auxiliar
    end
end

function get_angle(v::VelocityVector)::Float32
    m = sqrt(v.x^2 + v.y^2)
    angle_aux=asin(abs(v.y/m))
    angle = defined_angle(angle_aux, v.x , v.y)
    return angle
end 

function rotate(v::VelocityVector, angle::Float32)
    v.x::Float32 = v.speed * cos(angle)
    v.y::Float32 = v.speed * sin(angle)
end 

@inline function ⊕(v1::VelocityVector, v2::VelocityVector)
    v1 = VelocityVector(v1.x + v2.x, v1.y + v2.y, v1.speed)
    angle_rotate = get_angle(v1)
    rotate(v1, angle_rotate)
    return v1
end 


function get_pair_on_collision(matrix_distances::Matrix{Float32}, params::ParametersSimulation)

    index_agents_collision = []

    n, n = size(matrix_distances)

    for i in 1:n
        for j in i+1:n
            if matrix_distances[i, j] < params.size_agent * 2 + 5
                push!(index_agents_collision, (i, j))
            end
        end
    end
    return index_agents_collision

end

# Calculate the absolute or relative angle of a vector based on its components or coordinates
function compute_vector_angle(aux_angle::Float32, vx::Float32, vy::Float32; is_relative::Bool=false, coord1::Union{Vector{Float32}, Nothing}=nothing, coord2::Union{Vector{Float32}, Nothing}=nothing)::Float32
    """
    Compute the absolute or relative angle of a vector.
    
    Parameters:
    - aux_angle: Precomputed angle (e.g., using `asin` or `atan`).
    - vx, vy: Components of the vector (for absolute angle) or coordinates (for relative angle).
    - is_relative: If true, computes the relative angle between two coordinates.
    - coord1, coord2: Coordinates of two points (required if `is_relative=true`).
    
    Returns:
    - The adjusted angle in the range [0, 2π].
    """
    if is_relative
        # Compute relative angle between two coordinates
        if coord2[2] >= coord1[2] && coord2[1] >= coord1[1]
            return aux_angle
        elseif coord2[2] >= coord1[2] && coord1[1] >= coord2[1]
            return Float32(π) - aux_angle
        elseif coord1[2] >= coord2[2] && coord1[1] >= coord2[1]
            return Float32(π) + aux_angle
        else
            return Float32(2π) - aux_angle
        end
    else
        # Compute absolute angle of a vector
        if vx >= 0 && vy >= 0
            return aux_angle
        elseif vx <= 0 && vy >= 0
            return Float32(π) - aux_angle
        elseif vx <= 0 && vy <= 0
            return Float32(π) + aux_angle
        else
            return Float32(2π) - aux_angle
        end
    end
end

# Decompose a vector into directional and perpendicular components
function decompose_velocity(magnitude::Float32, angle::Float32, reference_angle::Float32)::Tuple{Vector{Float32}, Vector{Float32}}
    """
    Decompose a velocity vector into directional and perpendicular components.
    
    Parameters:
    - magnitude: Magnitude of the velocity vector.
    - angle: Angle of the velocity vector.
    - reference_angle: Reference angle for decomposition (e.g., collision angle).
    
    Returns:
    - A tuple containing the directional and perpendicular components.
    """
    if sin(angle - reference_angle) <= 0
        return ([0.0f0, 0.0f0], [magnitude * cos(angle), magnitude * sin(angle)])
    end

    dir_magnitude = abs(magnitude * sin(angle - reference_angle))
    perp_magnitude = abs(magnitude * cos(angle - reference_angle))

    directional = [dir_magnitude * cos(Float32(π / 2) + reference_angle), dir_magnitude * sin(Float32(π / 2) + reference_angle)]
    perpendicular = [perp_magnitude * cos(reference_angle), perp_magnitude * sin(reference_angle)]

    return (directional, perpendicular)
end

# Combine vectors to simulate velocity change during collision
function compute_resultant_velocity(original_velocity::Vector{Float32}, component1::Vector{Float32}, component2::Vector{Float32})::Vector{Float32}
    """
    Combine two velocity components to simulate the resultant velocity after a collision.
    
    Parameters:
    - original_velocity: Original velocity vector (before collision).
    - component1, component2: Velocity components to combine.
    
    Returns:
    - The resultant velocity vector with the same magnitude as `original_velocity`.
    """
    resultant = [component1[1] + component2[1], component1[2] + component2[2]]
    resultant_magnitude = Float32(sqrt(resultant[1]^2 + resultant[2]^2))
    resultant_angle_aux = Float32(asin(abs(resultant[2] / resultant_magnitude)))
    resultant_angle = compute_vector_angle(resultant_angle_aux, resultant[1], resultant[2])

    original_magnitude = Float32(sqrt(original_velocity[1]^2 + original_velocity[2]^2))
    vx = original_magnitude * cos(resultant_angle)
    vy = original_magnitude * sin(resultant_angle)

    return [vx, vy]
end

# Simulate an elastic collision between two agents
function elastic_collision!(agent1::Agent, agent2::Agent)
    """
    Simulate an elastic collision between two agents.
    
    Parameters:
    - agent1, agent2: Agents with `x`, `y`, and `velocity_vector` fields.
    
    Updates:
    - The velocities of `agent1` and `agent2` after the collision.
    """
    # Skip if either agent is at the origin
    if (agent1.x == 0 && agent1.y == 0) || (agent2.x == 0 && agent2.y == 0)
        return nothing
    end

    # Extract positions and velocities
    position1 = [agent1.x, agent1.y]
    position2 = [agent2.x, agent2.y]
    velocity1 = [agent1.velocity_vector.x, agent1.velocity_vector.y]
    velocity2 = [agent2.velocity_vector.x, agent2.velocity_vector.y]

    # Compute angles for both agents' velocities
    velocity1_magnitude = Float32(sqrt(velocity1[1]^2 + velocity1[2]^2))
    theta1_aux = Float32(asin(abs(velocity1[2] / velocity1_magnitude)))
    theta1 = compute_vector_angle(theta1_aux, velocity1[1], velocity1[2])

    velocity2_magnitude = Float32(sqrt(velocity2[1]^2 + velocity2[2]^2))
    theta2_aux = Float32(asin(abs(velocity2[2] / velocity2_magnitude)))
    theta2 = compute_vector_angle(theta2_aux, velocity2[1], velocity2[2])

    # Compute relative angles between agents
    distance = Float32(sqrt((position1[1] - position2[1])^2 + (position1[2] - position2[2])^2))
    alpha_aux = Float32(asin(abs((position1[2] - position2[2]) / distance)))
    alpha1 = compute_vector_angle(alpha_aux, position1[1] - position2[1], position1[2] - position2[2]; is_relative=true, coord1=position1, coord2=position2)
    omega1 = alpha1 - Float32(π / 2)

    alpha2 = compute_vector_angle(alpha_aux, position2[1] - position1[1], position2[2] - position1[2]; is_relative=true, coord1=position2, coord2=position1)
    omega2 = alpha2 - Float32(π / 2)

    # Decompose velocity vectors
    dir1, perp1 = decompose_velocity(velocity1_magnitude, theta1, omega1)
    dir2, perp2 = decompose_velocity(velocity2_magnitude, theta2, omega2)

    # Compute new velocities after collision
    new_velocity1 = compute_resultant_velocity(velocity1, perp1 .- dir1, dir2)
    new_velocity2 = compute_resultant_velocity(velocity2, perp2 .- dir2, dir1)

    # Update agent velocities
    agent1.velocity_vector.x = new_velocity1[1]
    agent1.velocity_vector.y = new_velocity1[2]
    agent2.velocity_vector.x = new_velocity2[1]
    agent2.velocity_vector.y = new_velocity2[2]
end

"""
update the velocity vector that effect on gravity when an agent is dead

"""
function gravity_on_agent_dead!(agent::Agent, params::ParametersSimulation)
    agent.velocity_vector = agent.velocity_vector + G
end



# include wall colissions
function update_velocities_on_collision(agents::Vector{Agent}, matrix_distances::Matrix{Float32}, params::ParametersSimulation)
    index_agents_collision = get_pair_on_collision(matrix_distances, params)

    for (i, j) in index_agents_collision
        elastic_collision!(agents[i], agents[j])
    end

    check_wall_collission(agents, params)
end



function update_positions(agents::Array{Agent}, params::ParametersSimulation)
    for agent in agents
        if agent.state == DEAD 
            if agent.y >= params.width - params.size_agent
                continue 
            end 
            gravity_on_agent_dead!(agent, params)
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

    num_agents_non_infected = params.num_agents-params.initial_infected

    for _ in 1:num_agents_non_infected
        x::Float32 = rand() * (params.width - 2*params.size_agent)
        y::Float32 = rand() * (params.height - 2*params.size_agent)
        state = HEALTHY
        velocity_vector = VelocityVector(params.speed_healthy)
        push!(agents, Agent(x, y, state, velocity_vector, 0))
    end

    for _ in 1:params.initial_infected
        x = rand() * (params.width - 2*params.size_agent)
        y = rand() *  (params.height - 2*params.size_agent)
        state = INFECTED
        velocity_vector = VelocityVector(params.speed_infected)
        push!(agents, Agent(x, y, state, velocity_vector, 0))
    end

    return agents
end

mutable struct MatrixDistances
    agents::Vector{Agent}
    matrix::Matrix{Float32}

    function MatrixDistances(agents::Vector{Agent})
        # Verificar que agents no esté vacío
        if isempty(agents)
            throw(ArgumentError("La lista de agentes no puede estar vacía."))
        end

        # Extraer puntos (x, y) de los agentes
        xy_points = hcat([agent.x for agent in agents], [agent.y for agent in agents])'

        # Calcular la matriz de distancias
        matrix_distances = pairwise(Euclidean(), xy_points, dims=2)

        # Convertir a Float32 si es necesario
        matrix_distances = Float32.(matrix_distances)

        return new(agents, matrix_distances)
    end
end

mutable struct AdjacencyMatrixGraph
    angents::Array{Agent}
    matrix::Matrix{Bool}

    function AdjacencyMatrixGraph(agents::Vector{Agent}, params::ParametersSimulation, matrix_distances::Matrix{Float32})
        x_points = [agent.x for agent in agents]
        y_points = [agent.y for agent in agents]

        xy_points = vcat(x_points', y_points')

        matrix_graph = sparse(zeros(Bool, length(agents), length(agents)))

        kd_tree = KDTree(xy_points)

        indices, distances = knn(kd_tree, xy_points, Int64(params.max_infected), true)

        @inbounds for ind in indices
            row = ind[1]
            for index in ind[2:end]
                if matrix_distances[row, index] < params.max_radius_infected
                    matrix_graph[row, index] = true
                end
            end
        end

        return new(agents, matrix_graph)

    end
end


mutable struct FrameSimulation
    matrix_points::Matrix{Float16}
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
    matrix_adjacency = AdjacencyMatrixGraph(agents, params, matrix_distances.matrix)
    update_velocities_on_collision(agents, matrix_distances.matrix, params)
    update_states(agents, matrix_adjacency.matrix, matrix_distances.matrix, 0, params)

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
        matrix_adjacency = AdjacencyMatrixGraph(agents, params, matrix_distances.matrix)
        
        update_velocities_on_collision(agents, matrix_distances.matrix, params)
        update_states(agents, matrix_adjacency.matrix, matrix_distances.matrix, i, params)

        frame_simulation = FrameSimulation(agents, matrix_adjacency.matrix, i)

        jldopen(path_simulation, "a"; compress=true) do file
            file["frame_simulation_$i"] = frame_simulation
        end
    end
end

@time generate_simulation("simulations/sim_2.jld2", 30, params)

using GameZero
rungame("run.jl")
