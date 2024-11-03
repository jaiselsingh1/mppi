using MuJoCo
using LinearAlgebra 
using MatrixEquations 

model = load_model("cartpole.xml")
data = init_data(model)

nx = 2*model.nv
nu = model.nu 

#finite difference parameters
c = 1e-6 
centred = true 

A = zeros(nx, nx)
B = zeros(nx, nu)
mjd_transitionFD(model, data, c, centred, A, B, nothing, nothing)

Q = diagm([1, 10, 1, 5])
R = diagm([1])

S = zeros(nx, nu)
_,_,K,_ = ared(A,B,R,Q,S)

function lqr_controller!(model, data)
    state = vcat(data.qpos, data.qvel)
    data.ctrl .= -K*state
    nothing
end 

init_visualiser()
visualise!(model, data, controller=lqr_controller!)
