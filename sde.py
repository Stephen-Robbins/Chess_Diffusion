from abc import ABC, abstractmethod
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx  # https://github.com/patrick-kidger/diffrax
import functools as ft
from CIR_Helper import sample_CIR_multi, score_cir, score_function_numerical



class SDE(ABC):
  """Abstract base class for a diffusion process."""
  
  @abstractmethod
  def score_loss(self, model, data, data_y, t, y=None):
    """
    Calculates the loss of a neural network approximation of a score function

    """
    raise NotImplementedError
  
  
  @abstractmethod
  def Drift(self, x, t, y=None):
    """
    Samples from the backward sde

    """
    raise NotImplementedError
  
  @abstractmethod
  def Diffusion(self, x, t):
    """
    Samples from the backward sde

    """
    raise NotImplementedError

  def Backward_Drift(self, score, x, t, y=None):
    """
    Samples from the backward sde

    """
    if y is not None:
        x_and_y = jnp.concatenate([x, y], axis=0)
        score=score(x_and_y, t)
    else:
        score=score(t, x)

    return self.Drift(x, t, y)-(self.Diffusion(x, t)**2*score)-2
  
  def forward_sample(self, x, key,  t0=0, t1=5, dt=0.001, y=None):
    eps=0.
    num_steps = int((t1 - t0) / dt)
    times = jnp.linspace(t0+eps, t1-eps, num_steps)
    # Create an array of times
    key, subkey = jr.split(key)
    current_x = x # Initialize state at t1

    for time in times:
      
        drift = self.Drift(current_x, time, y)
        diffusion = self.Diffusion(current_x, time)
        noise = jr.normal(key, current_x.shape) * jnp.sqrt(dt)
        
        
        # Euler-Maruyama method for SDE integration, with negative time step for backward process
        current_x += drift * dt + diffusion * noise
        key, _ = jr.split(key)
    return current_x
  
  #@eqx.filter_jit
  def backward_sample(self, score, data_shape, t1, key, t0=0,  dt=0.001, y=None):
    num_steps = int((t1 - t0) / dt)
    times = jnp.linspace(t1, t0, num_steps)  # Create an array of times
    key, subkey = jr.split(key)

    current_x = self.Prior(subkey, data_shape)
    
    for time in times:
        backward_drift = self.Backward_Drift(score, current_x, time, y)
        diffusion = self.Diffusion( current_x, time)
        noise = jr.normal(key, current_x.shape) * jnp.sqrt(dt)

        # Euler-Maruyama method for SDE integration, with negative time step for backward process
        current_x += backward_drift * (-dt) + diffusion * noise
        key, _ = jr.split(key)
        
    return current_x

  
class CIR(SDE):
    def __init__(self,  a=1, b=1,  T=5.0):
        self.T=T
        self.a=a
        self.b=b
       
    def score_loss(self, model, theta_0, data_y, t, key):
        weight=1
        theta_t=sample_CIR_multi(key, theta_0, self.a, self.b, t)
        true_score=score_function_numerical(theta_t, theta_0, self.a, self.b, t)
        pred_score = model(t, theta_t)
        return weight * jnp.mean((true_score - pred_score) ** 2)


    def Drift(self, theta_t,  t, y=None):
        """
        SDE drift

        """
        return self.b * (self.a - theta_t)

    def Diffusion(self, theta_t, t, y=None):
        """
        SDE Diffusion

        """
        condition = 2 * self.b * theta_t < 0

        # Set negative values to 0, keep positive values as they are
        adjusted_theta_t = jnp.where(condition, 0, theta_t)

        return jnp.sqrt(2 * self.b * adjusted_theta_t)

    def Backward_Drift(self, score, x, t, y=None):
        """
        Samples from the backward sde

        """
      
        score=score( t, x)

        return self.Drift(x, t, y)-(self.Diffusion(x, t)**2*score)-2*self.b
    
    def Prior(self, key, data_shape):
        thetas = jnp.ones(data_shape)*self.a
        return sample_CIR_multi(key, thetas, self.a, self.b, self.T)
       