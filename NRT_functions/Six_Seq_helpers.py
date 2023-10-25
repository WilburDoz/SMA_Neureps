import numpy as np
import jax.numpy as jnp

def irrep_transform(delta):
    transform_mat = np.zeros([6,6])
    transform_mat[0,0] = 1
    transform_mat[1,1] = np.cos(delta)
    transform_mat[2,2] = np.cos(delta)
    transform_mat[1,2] = -np.sin(delta)
    transform_mat[2,1] = np.sin(delta)
    transform_mat[3,3] = np.cos(delta*2)
    transform_mat[3,4] = -np.sin(delta*2)
    transform_mat[4,3] = np.sin(delta*2)
    transform_mat[4,4] = np.cos(delta*2)
    transform_mat[5,5] = np.cos(delta*3)
    return transform_mat
def generate_rep(a, Ts):
    g0 = a[:,0] + a[:,1] + a[:,3] + a[:,5]
    W = jnp.einsum('ij,kjl->kil', a, jnp.einsum('kij,jl->kil',Ts,jnp.linalg.pinv(a)))
    return jnp.einsum('ijk,k->ji', W, g0)
def loss_fit(A, targets, Ts):
    g = generate_rep(A, Ts)
    N = g.shape[0]
    g = g - jnp.multiply(g, g<0)
    Q = jnp.matmul(g, g.T)
    R = jnp.matmul(g, targets.T)
    P = jnp.matmul(jnp.linalg.inv(Q + 0.0001*jnp.eye(N)), R)
    P_norm = P/jnp.linalg.norm(P, axis = 0)[None,:]
    L_fit = jnp.linalg.norm(targets - jnp.matmul(P_norm.T, g))
    return L_fit
def loss_pos(A, Ts):
    g = generate_rep(A, Ts)
    g_neg = (g - jnp.abs(g))/2
    L_pos = -jnp.mean(g_neg)
    return L_pos
def loss_act(A, Ts):
    g = generate_rep(A, Ts)
    return jnp.mean(jnp.power(g, 2))

all_weights = 1
if all_weights:
    def loss_weight(A, Ts):
        W = jnp.einsum('ij,kjl->kil', A, jnp.einsum('kij,jl->kil',Ts,jnp.linalg.pinv(A)))
        return jnp.linalg.norm(W)
else:
    def loss_weight(A, Ts):
        W = jnp.einsum('ij,kjl->kil', A, jnp.einsum('kij,jl->kil',Ts[[1,3],:,:],jnp.linalg.pinv(A)))
        return jnp.linalg.norm(W)

