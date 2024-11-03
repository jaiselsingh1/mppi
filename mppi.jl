
using MuJoCo
using LinearAlgebra
using Random 
using Base.Threads

model = load_model("cartpole.xml")
data = init_data(model)

# system parameters 
const K = 30 #number of samples 
const H = 100 #horizon 
const lambda = 1.0 
const sigma = 1.0 

const nx = 2 * model.nv 
const nu = model.nu 

const control_limits = (-10,10)
const U = zeros(nu, H)


function state_cost(data)
    wx_pos = 1 
    wtheta = 20
    wx_dot = 0.1 
    wtheta_dot = 0.1 
    wctrl = 0.05

    x_pos = data.qpos[1]
    theta = data.qpos[2]
    x_dot = data.qvel[1]
    theta_dot = data.qvel[2]
    ctrl = data.ctrl[1]

    state_cost = wx_pos * x_pos^2
    state_cost += wtheta*(cos(theta) - 1)^2
    state_cost += wx_dot*(x_dot^2)
    state_cost += wtheta_dot*(theta_dot^2)
    state_cost += wctrl*(ctrl[1]^2)
    
    return state_cost
end 


function rollout(model,data, u, noise)
    costs = zeros(K)

    @threads for k in 1:K
        cost_k = 0.0 
        d_copy = init_data(model)
        d_copy.qpos .= data.qpos
        d_copy.qvel .= data.qvel

        for t in 1:H
            u_t = U[:,t] + noise[:, t,k]
            u_t = clamp.(u_t, control_limits[1], control_limits[2])
            d_copy.ctrl .= u_t
            mj_step(model, d_copy)
            cost_k += state_cost(d_copy)
        end 
        costs[k] = cost_k + 10*state_cost(d_copy)       
    end 
    return costs 

end 

function mppi_step!(model, data)
    noise = randn(nu, H, K) * sigma
    costs = rollout(model, data, U, noise)

    beta = minimum(costs)
    weights = exp.(-1 / lambda * (costs .- beta))
    weights ./= sum(weights)

    for t in 1:H
        U[:, t] .= U[:, t] .+ sum(weights[k] * noise[:, t, k] for k in 1:K)
    end
end

function mppi_controller!(model, data)
    mppi_step!(model, data)
    data.ctrl .= U[:, 1]
    U[:, 1:end-1] .= U[:, 2:end]
    U[:, end] .= 0.1 * U[:, end-1]
end

# set initial state
data.qpos[1] = 0
data.qpos[2] = 180

init_visualiser()
visualise!(model, data, controller=mppi_controller!)


