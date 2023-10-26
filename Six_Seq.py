# Load up a load of shit
from jax import vmap, value_and_grad, grad, jit, random
try:
    from jax.experimental import optimizers
except:
    from jax.example_libraries import optimizers
import os
from datetime import datetime
import numpy as np

# And functions I've written
from NRT_functions import helper_functions
from NRT_functions import Six_Seq_helpers

###### Set a load of parameters #####

parameters = {
    'T': 1000,
    'print_iter': 5000,
    'N': 20,
    'step_size': 1e-3,
    'sigma_w': 0.25,
    'Runs': 10,

    'mu_fit': 100,
    'mu_act': 5,
    'mu_weight': 5,
    'mu_pos': 500
}

targets = np.array([[1, 0, 1, 0, 0, 0]])

######

deltas = [2*np.pi/6*n for n in range(6)]
Ts = np.stack([Six_Seq_helpers.irrep_transform(delta) for delta in deltas])

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
helper_functions.save_obj(targets, "targets", savepath)
print("\nOPTIMISATION BEGINNING\n")

jl_fit = jit(Six_Seq_helpers.loss_fit)
jl_act = jit(Six_Seq_helpers.loss_act)
jl_pos = jit(Six_Seq_helpers.loss_pos)
jl_weight = jit(Six_Seq_helpers.loss_weight)
opt_init, opt_update, get_params = optimizers.adam(parameters['step_size'])
def loss_func(A, targets, Ts):
    return parameters['mu_fit']*Six_Seq_helpers.loss_fit(A, targets, Ts) + parameters['mu_act']*Six_Seq_helpers.loss_act(A, Ts) + parameters['mu_weight']*Six_Seq_helpers.loss_weight(A, Ts) + parameters['mu_pos']*Six_Seq_helpers.loss_pos(A, Ts)


for run in range(parameters['Runs']):
    print(f"RUN: {run}")
    now = datetime.strftime(datetime.now(), '%H%M%S')
    key = random.PRNGKey(int(now))
    A = random.normal(key, (parameters["N"], 6))
    opt_state = opt_init(A)

    @jit
    def update(A, targets, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = value_and_grad(loss_func)(A, targets, Ts)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, value

    min_L = np.infty
    A_init = A
    Losses = np.zeros([int(parameters['T']/parameters['print_iter'])+1, 6])
    counter = 0

    for t in range(parameters['T']):
        A, opt_state, loss = update(A, targets, opt_state)

        if t % parameters['print_iter'] == 0:
            L_f = jl_fit(A, targets, Ts)
            L_a = jl_act(A, Ts)
            L_w = jl_weight(A, Ts)
            L_n = jl_pos(A, Ts)

            Losses[counter, :] = [t, loss, L_f, L_a, L_w, L_n]
            counter += 1

            print(f"Step {t}, Loss: {loss:.5f}, Fit: {L_f:.5f}, Act: {L_a:.5f}, Wei: {L_w:.5f}, Neg Prop: {L_n:.5f}")

        if loss < min_L:
            A_best = A
            min_L = loss

    # Now save the weights and the losses
    helper_functions.save_obj(A_best, f"params_best_{run}", savepath)
    helper_functions.save_obj(A_init, f"params_init_{run}", savepath)
    helper_functions.save_obj(A, f"params_{run}", savepath)
    helper_functions.save_obj(Losses, f"L_{run}", savepath)
    helper_functions.save_obj(min_L, f"min_L_{run}", savepath)