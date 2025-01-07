using JLD2
using Colors

BACKGROUND = colorant"black"

global path_simulation = "simulations/sim_2.jld2"
global num_frame = 0 
global simulation = load(path_simulation)
WIDTH = simulation["WIDTH"]
HEIGHT = simulation["HEIGHT"]
SIZE_AGENT = simulation["SIZE_AGENT"]

global frame = simulation["frame_simulation_$(num_frame)"]

function update()
    global num_frame += 1
    global frame = simulation["frame_simulation_$(num_frame)"]
end

const LINE_COLOR = RGB(.1, .1, .8)

const AGENT_COLORS = Dict([
    :infected => RGB(.8, .1, .1),
    :healthy => RGB(.1, .9, .1),
    :inmunized => RGB(.4, .6, .7),
    :dead => RGB(0.9, 0.9, 0.9)
])

function draw_frame(frame, size_agent)
    # pintar todas las lineas de los grafos
    num_agents = length(frame.matrix_points[1,:])

    for i in 1:num_agents
        x::Int = round(frame.matrix_points[1, i])
        y::Int = round(frame.matrix_points[2, i])

        for j in 1:num_agents
            if frame.graph_matrix[i, j]
                x2::Int = round(frame.matrix_points[1, j])
                y2::Int = round(frame.matrix_points[2, j])
                line = Line(x, y, x2, y2)
                draw(line, LINE_COLOR)
            end
        end
    end

    # pintar todos los agentes
    for i in 1:num_agents
        x::Int64 = round(frame.matrix_points[1, i])
        y::Int64 = round(frame.matrix_points[2, i])
        agent = Circle(x, y, SIZE_AGENT)
        state = frame.states[i]
        agent_color = AGENT_COLORS[state]

        draw(agent, agent_color, fill = true)
    end
end

function draw()
    draw_frame(frame, 5)
end