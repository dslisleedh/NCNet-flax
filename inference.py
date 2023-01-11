"""
Run Super resolution on given image(under /test_images) and save the result in /results
"""


from PIL import Image
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np

import flax
import flax.linen as nn
from flax.training import checkpoints, train_state
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from src.model import NCNet
from src.funcs import create_train_state
from src.utils import check_trained_model_exists

from tqdm import tqdm
import os
from glob import glob
from natsort import natsorted
import shutil
from omegaconf import OmegaConf
import argparse


def run_super_resolution():
    # Load the model. Load from ./models. If not found, load checkpoint from ./logs
    check_trained_model_exists()
    config = dict(OmegaConf.load('./models/trained/.hydra/config.yaml'))
    state = create_train_state(config)

    results = OmegaConf.load('./models/trained/result.yaml')
    if results.train_psnr >= results.fine_tune_psnr:
        state = checkpoints.restore_checkpoint('./models/trained/ckpts', state)
    else:
        state = checkpoints.restore_checkpoint('./models/trained/ckpts_fine_tune', state)

    # Iteratively load image and run super resolution. Save the result in /results
    os.makedirs('./results', exist_ok=True)
    files = glob('./test_images/*')
    files = natsorted(files)

    for file in tqdm(files, desc='Iterating images'):
        img = Image.open(file).convert("RGB")
        img = np.array(img, dtype=np.float32)[jnp.newaxis, :, :, :]
        img = jnp.array(img)

        img = state.apply_fn(state.params, img)
        img = img[0]
        img = np.array(img).astype(np.uint8)
        plt.imsave('./results/{}'.format(file.split('/')[-1]), img)

    print('Done !!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['trained', 'quantization'], default='trained')
    args = parser.parse_args()

    if args.model == 'trained':
        run_super_resolution()
    elif args.model == 'quantization':
        # check_quantization_model_exists()
        pass
    else:
        raise ValueError('Unknown model type')
