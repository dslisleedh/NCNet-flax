from omegaconf import OmegaConf

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import einops

import flax
import flax.linen as nn
from flax.training import checkpoints, train_state
from flax.metrics import tensorboard

import optax

import os
from functools import partial
from tqdm import tqdm
from src.funcs import create_train_state, train_step, inference, metrics
from src.datasets import load_dataset
from src.utils import check_trained_model_exists
from jax.experimental import jax2tf
from src.model import NCNet

from tensorflow.lite.python import interpreter as interpreter_wrapper
import matplotlib.pyplot as plt


def representative_dataset_gen_model_none():
    config = dict(OmegaConf.load('./models/trained/.hydra/config.yaml'))
    _, valid_ds = load_dataset(scale=config['train']['scale'])
    for batch in valid_ds:
        x, y = batch
        yield [tf.cast(tf.convert_to_tensor(x), tf.float32)]


def main():
    os.makedirs('./models/quantization', exist_ok=True)
    check_trained_model_exists()

    config = dict(OmegaConf.load('./models/trained/.hydra/config.yaml'))
    model = NCNet(config['train']['n_filters'], config['train']['scale'])
    params = checkpoints.restore_checkpoint('./models/trained/ckpts', None)['params']

    predict = lambda x: model.apply(params, x)

    tf_predict = tf.function(
        jax2tf.convert(predict, enable_xla=False, polymorphic_shapes=["1, h, w, 3"]),
        input_signature=[tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32)],
        autograph=False
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_predict.get_concrete_function()], tf_predict)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS
    ]

    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen_model_none
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open('./models/quantization/model_none.tflite', 'wb') as f:
        f.write(tflite_model)

    interpreter = interpreter_wrapper.Interpreter(model_path='./models/quantization/model_none.tflite')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, valid_ds = load_dataset(scale=config['train']['scale'])

    for batch in valid_ds:
        x, y = batch
        interpreter.resize_tensor_input(input_details[0]['index'], x.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], tf.cast(tf.convert_to_tensor(x), tf.uint8))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_float = predict(jnp.array(x))
        break

    fig, ax = plt.subplots(1, 3, figsize=(25, 15))
    ax[0].imshow(np.array(x[0]).astype(np.uint8))
    ax[0].set_title('LR(Input)')
    ax[1].imshow(np.array(output_data[0]).astype(np.uint8))
    ax[1].set_title('SR/jax/float32')
    ax[2].imshow(np.array(output_float[0]).astype(np.uint8))
    ax[2].set_title('SR/tflite/uint8')
    fig.tight_layout()
    plt.savefig('./models/quantization/model_none.png')

    mae = jnp.mean(jnp.abs(output_float - jnp.array(output_data, dtype=jnp.float32)))
    print('MAE: ', mae)
    print('Done')


if __name__ == '__main__':
    main()
