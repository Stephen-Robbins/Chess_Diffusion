import jax.random as jr
import jax.numpy as jnp
from jax.scipy.special import i0  # Modified Bessel function of the first kind
import jax
import numpy as np
from scipy.special import iv

jax.config.update("jax_enable_x64", True)


    
def _random_chi2(key, df, shape=(), dtype=jnp.float_):
    return 2.0 * jr.gamma(key, 0.5 * df, shape=shape, dtype=dtype)

def sample_from_ncx2(key, df, nc, sample_shape=()):
    
    shape = sample_shape + jnp.shape(df) + jnp.shape(nc)

    key1, key2, key3 = jr.split(key, 3)

    i = jr.poisson(key1, 0.5 * nc, shape=shape)
    n = jr.normal(key2, shape=shape) + jnp.sqrt(nc)
    cond = jnp.greater(df, 1.0)
    chi2 = _random_chi2(key3, jnp.where(cond, df - 1.0, df + 2.0 * i), shape=shape)
    return jnp.where(cond, chi2 + n * n, chi2)

def sample_CIR(key, theta_0, a, b, t):
    exp_bt = jnp.exp(-b * t)
    d = 2 * a  # degrees of freedom
    mu = 2 * theta_0 * exp_bt / (1 - exp_bt)  # non-centrality parameter

    # Sample from the non-central chi-squared distribution
    theta_t_sample = sample_from_ncx2(key, df=d, nc=mu)

    theta_t = (1 - exp_bt) / 2 * theta_t_sample
    return theta_t



def sample_CIR_multi(key, thetas, a, b, T):
   
    theta_shape = thetas.shape
    cir_processes = jnp.zeros(theta_shape)
    num_processes = int(jnp.prod(jnp.array(theta_shape)))

    keys = jr.split(key, num_processes)

    idx = 0  
    for theta_idx in np.ndindex(theta_shape):
        theta = thetas[theta_idx]  # Initial value for this CIR process
        theta = sample_CIR(keys[idx], theta, a, b, T)
        cir_processes = cir_processes.at[theta_idx].set(theta)
        idx += 1

    return cir_processes

def score_cir(theta_t, theta_0, a, b, t):

    """
    
    Compute the score (gradient of the log transition density) for the CIR process.
    :param theta_t: The value at time t.
    :param theta_0: The initial value of the process.
    :param a: The long-term mean level parameter.
    :param b: The speed of reversion parameter.
    :param t: The time increment.
    :return: The score value.
    """

    c = 1 / (1 - jnp.exp(-b * t))
    exp_bt = jnp.exp(-b * t)
    z = 2 * c * jnp.sqrt(exp_bt * theta_t * theta_0)

    # Compute the parts of the score function based on the Wolfram Alpha result
    term1 = -c
    term2 = (a - 1) / (2 * theta_t)
    term3_numerator = c * exp_bt * theta_0 * (iv(a - 2, z) + iv(a, z))

    term3_denominator = 2 * jnp.sqrt(exp_bt * theta_t * theta_0) * iv(a - 1, z)
    term3 = term3_numerator / term3_denominator
    # Combine all terms to get the score
    score = term1 + term2 + term3
    return score


def transition_density_cir(theta_t, theta_0,  b, t):
    """
    Compute the transition density for the CIR process using the provided formula.
    ONLY FOR a=1
    :param theta_t: The value at time t.
    :param theta_0: The initial value of the process.
    :param b: The speed of reversion parameter.
    :return: The transition density value.
    """
    c=1/(1-jnp.exp(-b*t))
    bessel_term = i0(2 * c * jnp.sqrt(theta_0 * theta_t * jnp.exp(-b * t)))
    density = c * jnp.exp(-c * (theta_0 * jnp.exp(-b *t) + theta_t))  * bessel_term

    return density

def log_transition_density_cir(theta_t, theta_0, b, t):
    return jnp.log(transition_density_cir(theta_t, theta_0, b, t))

# Gradient function for log_transition_density_cir
grad_log_transition = jax.grad(log_transition_density_cir, argnums=0)

# Vectorize the gradient computation
elementwise_grad = jax.vmap(grad_log_transition, in_axes=(0, 0, None, None))

def score_function_numerical(theta_t, theta_0, a, b, t):
    # Flatten the theta_t and theta_0 arrays to 1D
    theta_t_flat = theta_t.reshape(-1)
    theta_0_flat = theta_0.reshape(-1)

    # Define a function that computes the gradient for a single element
    def single_element_grad(theta_t_elem, theta_0_elem):
        return jax.grad(log_transition_density_cir, argnums=0)(theta_t_elem, theta_0_elem, b, t)

    # Vectorize the single-element gradient computation
    vectorized_grad = jax.vmap(single_element_grad, in_axes=(0, 0))

    # Compute the gradient for each element
    gradients_flat = vectorized_grad(theta_t_flat, theta_0_flat)

    # Reshape the gradients back to the original shape
    gradients = gradients_flat.reshape(13, 8, 8)

    return gradients
