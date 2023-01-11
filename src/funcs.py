import hydra
from hydra.utils import get_original_cwd

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from jax import lax
import einops

import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
from flax.metrics import tensorboard
from src.model import NCNet

import optax

import os
from tqdm import tqdm
from typing import Callable


def l1_loss(x, y):
    return jnp.mean(jnp.abs(x - y))


def metrics(x, y):
    x_tf = tf.math.round(tf.convert_to_tensor(x))
    y_tf = tf.math.round(tf.convert_to_tensor(y))
    return {
        'l1_loss': float(l1_loss(x, y)),
        'psnr': float(tf.reduce_mean(tf.image.psnr(x_tf, y_tf, max_val=255.))),
        'ssim': float(tf.reduce_mean(tf.image.ssim(x_tf, y_tf, max_val=255.))),
    }


def create_train_state(config: dict):
    rng = jax.random.PRNGKey(config['seed'])
    model = NCNet(config['train']['n_filters'], config['train']['scale'])

    params = model.init(rng, jnp.ones((1, 64, 64, 3)))
    print(model.tabulate(rng, jnp.ones((1, 64, 64, 3))))

    boundaries = list(
        range(0, config['train']['steps'] + config['fine_tuning']['steps'], config['train']['lr']['decay_steps'])
    )[1:]
    # lrs = [
    #     config['train']['lr']['init'] * (config['train']['decay'] ** (-1 * i)) for i in range(len(boundaries) + 1)
    # ]
    decay_dict = {
        k: config['train']['lr']['decay'] for k in boundaries
    }
    schedular = optax.piecewise_constant_schedule(
        config['train']['lr']['init'], decay_dict
    )
    optimizer = optax.adam(schedular, b1=0.9, b2=0.999, eps=1e-7)  # keras default

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )


@jax.jit
def train_step(state, batch):
    x, y = batch

    def loss_fn(params):
        y_hat = state.apply_fn(params, x)
        loss = l1_loss(y, y_hat)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def inference(x, state):
    y_hat = state.apply_fn(state.params, x)
    return y_hat
