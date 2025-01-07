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

interval_between_infected::Int16 = 3
prob_infect::Float32 = 0.3
max_radius_infected::Int16 = 50
max_infected::Int16 = 3
size_agent::Int8 = 10
num_agents::Int16 = 300
height::Int16 = 800
width::Int16 = 1700
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


function definir_angulo(angulo_auxiliar,vx,vy)
    angulo = 0 
    if vx>=0 && vy>=0
        angulo = angulo_auxiliar 
    end
    if vx<=0 && vy>=0
        angulo= pi - angulo_auxiliar 
    end
    if vx<=0 && vy<=0
        angulo= pi + angulo_auxiliar 
    end
    if vx>=0 && vy<=0
        angulo= 2*pi - angulo_auxiliar 
    end

    return angulo 
end


function definir_angulo_2(angulo_aux,cor1,cor2)
    angulo=0
    if cor2[2]>=cor1[2] && cor2[1]>=cor1[1]
        angulo=angulo_aux
    end
    if cor2[2]>=cor1[2] && cor1[1]>=cor2[1]
        angulo=pi -angulo_aux
    end
    if cor1[2]>=cor2[2] && cor1[1]>=cor2[1]
        angulo= pi + angulo_aux
    end
    if cor1[2]>=cor2[2] && cor2[1]>=cor1[1]
        angulo=2*pi - angulo_aux
    end
    return angulo
end



function cambio_angular(m0,m1,m2)
    mr=[m1[1]+ m2[1],m1[2]+m2[2]]
    __mr__=sqrt((mr[1]^2 + mr[2]^2))
    angulo_mr_a=asin(abs(mr[2]/__mr__))
    angulo_mr = definir_angulo(angulo_mr_a,mr[1],mr[2])
    __m0__ = sqrt((m0[1])^2+(m0[2])^2)
    vx =__m0__*cos(angulo_mr)
    vy = __m0__*sin(angulo_mr)
    return [vx,vy]

end


function elastic_collision!(a1::Agent, a2::Agent)
    ############################# haremos una consideracion, si uno de ellos tiene los vectores anlados el mantiene la misma direccion y el que los
    #tenia anuladossiguen anulados 
    if a1.x == 0 && a1.y == 0
        return nothing
    end

    if a2.x == 0 && a2.y == 0
        return nothing
    end

    cor1 = [a1.x, a1.y]
    cor2 = [a2.x, a2.y]
    m1 = [a1.velocity_vector.x, a1.velocity_vector.y]
    m2 = [a2.velocity_vector.x, a2.velocity_vector.y]
    
    
    ######### primero definiremos a los angulos ##############
    longitud_vector_1=sqrt(m1[1]^2 + m1[2]^2)
    theta1_auxiliar=asin(abs(m1[2]/longitud_vector_1))
    theta1=definir_angulo(theta1_auxiliar,m1[1],m1[2])



    distancia_entre_particulas=sqrt((cor1[1]-cor2[1])^2 + (cor1[2]-cor2[2])^2)
    alpha1_auxiliar= asin(abs((cor1[2]-cor2[2])/distancia_entre_particulas))
    alpha1=definir_angulo_2(alpha1_auxiliar,cor1,cor2)
    # es importante recalcar que para la particula 1, en los argumentos las coordenadas1 deben ir primero
    # por que se estan haciendo relativo a ella 
    omega1= alpha1 - pi/2 
    ## particula de los modulos m1 
    longitud_vector_2=sqrt(m2[1]^2 + m2[2]^2)
    theta2_auxiliar=asin(abs(m2[2]/longitud_vector_2))
    theta2=definir_angulo(theta2_auxiliar,m2[1],m2[2])
    

    ## una consideracion a tener en cuenta es que alpha1_auxiliar es igual a alpha2_auxiliar 
    alpha2_auxiliar= alpha1_auxiliar
    alpha2 = definir_angulo_2(alpha2_auxiliar,cor2,cor1)
    #### si nos fijamos aqui se invirtieron papeles con los argumentos 
    omega2= alpha2 - pi/2 
    ## particula de los modulos m2
    
    # hay que hacer un programa que al momento de rotar un angulo omega, nos regrese el vector que apunte 
    # al centro de la otra particula y su perpendicular 
    # para un mejor contexto definamos de otra manera las variables como
    __m1__ = longitud_vector_1
    __m2__ = longitud_vector_2

    # por lo que las nuevas coordenadas en x y y de los modulos 1 y 2 respecto a ellas serian, __m1__*cos(theta1-omega1) y __m2__*sen(theta2-omega2)
    # el modulo 1 por el sen con el angulo invertido me da el modulo del vector direccional y el modulo1 por el cos del angulo invertido me da el modulo perpendicular
    # y asi para la particula 2, por lo que me queda solo obtener el valor en x y en y de dos modulos cuyos valores ya se y que el direccional se encuentra a un angulo de 90 + omega y el
    # perpedicular se encuentra a un angulo de omega o 180 + omega 

    #     
    
    # primero, obtengamos al vector direccional y su perpendicular de la particula 1
    # primero, si el vector direccional no apunta hacia el centro no se hace nada y el vector direccional se hace cero y el 
    # perpendicular lo tomas como el mismo, una manera de saber esto es si el sen(theta-omega)<=0
    if sin(theta1-omega1)<=0
        md1=[0,0]
        mp1=m1 
    else
        # para modulo direccional
        __md1__ = abs(__m1__*sin(theta1-omega1))
        md1=[__md1__*cos(pi/2 + omega1),__md1__*sin(pi/2 + omega1)]

        # para el vector perpendicular 
        __mp1__ = abs(__m1__*cos(theta1-omega1))
        
        if cos(theta1-omega1)<=0
            mp1=[__mp1__*cos(omega1 + pi), __mp1__*sin(omega1 + pi )]
        else
            mp1 = [__mp1__*cos(omega1),__mp1__*sin(omega1)]
        end
    end
    
    ## para la particula 2

    if sin(theta2-omega2)<=0
        md2=[0,0]
        mp2=m2 
    else
        # para modulo direccional
        __md2__ = abs(__m2__*sin(theta2-omega2))
        md2=[__md2__*cos(pi/2 + omega2),__md2__*sin(pi/2 + omega2)]

        # para el vector perpendicular 
        __mp2__ = abs(__m2__*cos(theta2-omega2))
        
        if cos(theta2-omega2)<=0
            mp2=[__mp2__*cos(omega2 + pi), __mp2__*sin(omega2 + pi )]
        else
            mp2 = [__mp2__*cos(omega2),__mp2__*sin(omega2)]
        end
    end


    # por lo que ya obtuvimos como informacion a los vectores md1 y mp1 e md2 y mp2 

    # ya por ultimo lo que debemos hacer es hacer elcmabio angular de m1 y m2 usando como suma ordinaria de vectores,
    # para la particula 1

    modulos_1=cambio_angular(m1,[(mp1-md1)[1],(mp1-md1)[2]],[md2[1],md2[2]])
    modulos_2=cambio_angular(m2,[(mp2-md2)[1],(mp2-md2)[2]],[md1[1],md1[2]])
 
    a1.velocity_vector.x = Float32(modulos_1[1])
    a2.velocity_vector.x = Float32(modulos_2[1])

    a1.velocity_vector.y = Float32(modulos_1[2])
    a2.velocity_vector.y = Float32(modulos_2[2])
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

        JLD2.jldopen(path_simulation, "a") do file
            file["frame_simulation_$i"] = frame_simulation
        end
    end

end

@time generate_simulation("simulations/sim_2.jld2", 50, params)

using GameZero
rungame("run.jl")

