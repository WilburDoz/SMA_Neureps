# Load up a load of shit
import numpy as np
from jax import vmap, value_and_grad, grad, jit, random
try:
    from jax.experimental import optimizers
except:
    from jax.example_libraries import optimizers
import os
from datetime import datetime

# And functions I've written
from NRT_functions import helper_functions
from NRT_functions import losses


###### Set a load of parameters #####

parameters = {
    'T': 10000,
    'num_angs': 16,
    'print_iter': 5000,
    'random_seed': 30,
    'N': 20,
    'step_size': 1e-3,
    'sigma_w': 0.25,
    'Runs': 100,

    'mu_fit': 50,
    'mu_act': 1,
    'mu_L2': 1,
    'mu_weight': 1,
    'mu_path': 30,
    'mu_pos': 200
}

######

# Setup save file locations
today = datetime.strftime(datetime.now(), '%y%m%d')
now = datetime.strftime(datetime.now(), '%H%M%S')
filepath = f"./data/{today}/{now}/"
# Make sure folder is there
if not os.path.isdir(f"./data/"):
    os.mkdir(f"./data/")
if not os.path.isdir(f"data/{today}/"):
    os.mkdir(f"data/{today}/")
# Now make a folder in there for this run
savepath = f"data/{today}/{now}/"
if not os.path.isdir(f"data/{today}/{now}"):
    os.mkdir(f"data/{today}/{now}")

helper_functions.save_obj(parameters, "parameters", savepath)
print("\nOPTIMISATION BEGINNING\n")

# Define Targets
targets = np.zeros([5, 8])
targets_indices = [1,2,3,4,3,2,1,0]
for i in range(8):
    targets[targets_indices[i], i] = 1

jl_path = jit(losses.loss_path)
jl_fit = jit(losses.loss_fit)
jl_act = jit(losses.loss_act)
jl_pos = jit(losses.loss_pos)
jl_weight = jit(losses.loss_weight)
opt_init, opt_update, get_params = optimizers.adam(parameters['step_size'])

def loss_func(params, targets, parameters):
    return parameters['mu_fit']*losses.loss_fit(params, targets, parameters, g) + parameters['mu_act']*losses.loss_act(params, g) + parameters['mu_weight']*losses.loss_weight(params) + parameters['mu_path']*losses.loss_path(params, g) + parameters['mu_pos']*losses.loss_pos(params, g)


for run in range(parameters['Runs']):
    print(f"RUN: {run}")
    key = random.PRNGKey(int(now))
    g0 = random.normal(key, (parameters['N'], ))
    W = random.normal(key, (parameters['N'], parameters['N']))*parameters['sigma_w']
    params = [g0, W]
    opt_state = opt_init([g0, W])
    g = np.zeros([parameters['N'], parameters['num_angs']])

    @jit
    def update(params, targets, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = value_and_grad(loss_func)(params, targets, parameters)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, value


    min_L = np.infty
    params_init = params
    Losses = np.zeros([int(parameters['T']/parameters['print_iter'])+1, 7])
    counter = 0

    for t in range(parameters['T']):
        params, opt_state, loss = update(params, targets, opt_state)

        if t % parameters['print_iter'] == 0:
            L_f = jl_fit(params, targets, parameters, g)
            L_a = jl_act(params, g)
            L_w = jl_weight(params)
            L_p = jl_path(params, g)
            L_n = jl_pos(params, g)

            Losses[counter, :] = [t, loss, L_f, L_a, L_w, L_p, L_n]
            counter += 1

            print(f"Step {t}, Loss: {loss:.5f}, Fit: {L_f:.5f}, Act: {L_a:.5f}, Wei: {L_w:.5f}, W_path: {L_p:.5f}, Neg Prop: {L_n:.5f}")

        if loss < min_L:
            params_best = params
            min_L = loss

    # Now save the weights and the losses
    helper_functions.save_obj(params_best, f"params_best_{run}", savepath)
    helper_functions.save_obj(params_init, f"params_init_{run}", savepath)
    helper_functions.save_obj(params, f"params_{run}", savepath)
    helper_functions.save_obj(Losses, f"L_{run}", savepath)
    helper_functions.save_obj(min_L, f"min_L_{run}", savepath)