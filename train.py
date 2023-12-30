
import functools as ft
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax  # https://github.com/deepmind/optax
from plotting_functions import print_chessboard_from_bitboard

def one_hot_encode_max_value(batch):
    # Find the indices of the maximum values along the channel axis
    max_indices = jnp.argmax(batch, axis=1)

    # Create a one-hot encoded array where only the maximum values are 1
    one_hot_encoded = jax.nn.one_hot(max_indices, num_classes=batch.shape[1])

    return one_hot_encoded


def batch_loss_fn(model,  sde, data, data_y, t1, key):
    batch_size = data.shape[0]
    
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # Low-discrepancy sampling over t to reduce variance
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
   
    loss_fn = ft.partial(sde.score_loss, model)
    loss_fn = jax.vmap(loss_fn)
    
    return jnp.mean(loss_fn(data, data_y, t, losskey))


@eqx.filter_jit
def make_step(model, sde, data, data_y, t1, key, opt_state, opt_update):
    
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, sde, data, data_y, t1, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state

def dataloader(data, batch_size, *, key):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size

def main(
    model,
    data,
    sde,
    data_y=None,
    t1=5.0,
    # Optimisation hyperparameters
    num_steps=1_000_000,
    lr=1e-4,
    batch_size=32,
    print_every=100,
    # Sampling hyperparameters
    dt0=0.1,
    sample_size=4,
    # Seed
    seed=12,
):
    key = jr.PRNGKey(seed)
    train_key, loader_key,loader_key2, sample_key = jr.split(key, 4)
    data_shape = data.shape[1:]
    opt = optax.adabelief(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    total_value = 0
    total_size = 0
    
    for step, data in zip(
        range(num_steps), dataloader(data, batch_size, key=loader_key)
    ):
        data_Y=None
        value, model, train_key, opt_state = make_step(
            model, sde, data, data_y, t1, train_key, opt_state, opt.update
        )
        
        total_value += value.item()
        total_size += 1
    
        print((f"Step={step} Loss={value.item()}"))
        if ((step % print_every) == 0) or step == num_steps - 1:
            print(f"Step={step} Loss={total_value / total_size}")
            total_value = 0
            total_size = 0
            
            vmap_key = jr.split(sample_key, sample_size)
            
            sample_fn = ft.partial(sde.backward_sample, model, data_shape, t1)
            sample = jax.vmap(sample_fn)(vmap_key, y=data_y)
            sample = one_hot_encode_max_value(sample)

            sample_key = jr.split(sample_key, 1)[0]
            print_chessboard_from_bitboard(sample)