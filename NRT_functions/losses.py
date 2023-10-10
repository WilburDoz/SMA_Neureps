# File containing all the various loss functions I have written
import jax.numpy as jnp
from NRT_functions import helper_functions

def loss_fit(params, targets, parameters, g):
    g = helper_functions.generate_rep(params, g)
    g = g[:, :8]
    N = g.shape[0]
    g = g - jnp.multiply(g, g < 0)
    Q = jnp.matmul(g, g.T)
    R = jnp.matmul(g, targets.T)
    P = jnp.matmul(jnp.linalg.inv(Q + 0.01 * jnp.eye(N)), R)
    L_fit = jnp.linalg.norm(targets - jnp.matmul(P.T, g))
    return L_fit + jnp.linalg.norm(P) * parameters['mu_L2']


def loss_pos(params, g):
    g = helper_functions.generate_rep(params, g)
    g_neg = (g - jnp.abs(g)) / 2
    L_pos = -jnp.mean(g_neg)
    return L_pos


def loss_act(params, g):
    g = helper_functions.generate_rep(params, g)
    return jnp.mean(jnp.power(g, 2))


def loss_weight(params):
    return jnp.linalg.norm(params[1])


def loss_path(params, g):
    g = helper_functions.generate_rep(params, g)
    return jnp.linalg.norm(g[:, :8] - g[:, 8:])