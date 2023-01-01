import hydra
from hydra.utils import get_original_cwd
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
from flax.training import checkpoints
from flax.metrics import tensorboard

import optax

import os
from tqdm import tqdm
from src.funcs import create_train_state, train_step, inference, metrics
from src.datasets import load_dataset


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(config):
    seed = config.seed
    tf.random.set_seed(seed)
    np.random.seed(seed)

    result = dict()

    state = create_train_state(config)

    # Train with small patch size
    train_ds, valid_ds = load_dataset(
        batch_size=config['train']['batch_size'], scale=config['train']['scale'],
        train_lr_image_size=config['train']['lr_image_size'], steps=config['train']['steps'],
    )
    summary_writer = tensorboard.SummaryWriter('./train_logs')
    watch_val = -jnp.inf
    valid_loss = jnp.inf
    valid_ssim = -jnp.inf
    patience = 0

    with tqdm(total=config['train']['steps'], colour='CYAN', position=0) as pbar:
        pbar.set_description('Train steps')
        loss_history = []

        for i in range(1, config['train']['steps'] + 1):
            pbar.update(1)
            batch = train_ds.next()
            state, loss = train_step(state, batch)
            summary_writer.scalar('train_loss', loss, step=i)
            loss_history.append(loss)
            loss_history = loss_history[-100:]
            pbar.set_postfix(
                train_loss=sum(loss_history) / len(loss_history), valid_psnr=watch_val, patience=patience,
                valid_loss=valid_loss, valid_ssim=valid_ssim
            )

            if (i % config['train']['check_every'] == 0) and i > 0:
                valid_loss_list = []
                valid_psnr_list = []
                valid_ssim_list = []

                with tqdm(range(100), colour='YELLOW', desc='Valid steps', position=1, leave=False) as tbar:
                    valid_np_ds = valid_ds.as_numpy_iterator()
                    for n in tbar:
                        valid_batch = next(valid_np_ds)
                        y, y_hat = inference(state, valid_batch)
                        val_metrics = metrics(y, y_hat)
                        valid_loss_list.append(val_metrics['l1_loss'])
                        valid_psnr_list.append(val_metrics['psnr'])
                        valid_ssim_list.append(val_metrics['ssim'])

                    valid_loss = sum(valid_loss_list) / len(valid_loss_list)
                    valid_psnr = sum(valid_psnr_list) / len(valid_psnr_list)
                    valid_ssim = sum(valid_ssim_list) / len(valid_ssim_list)
                    cur_step = int(i // config['train']['check_every'])
                    summary_writer.scalar('valid_loss', valid_loss, step=cur_step)
                    summary_writer.scalar('valid_psnr', valid_psnr, step=cur_step)
                    summary_writer.scalar('valid_ssim', valid_ssim, step=cur_step)

                # Early stopping
                if valid_psnr > watch_val:
                    watch_val = valid_psnr
                    patience = 0
                    checkpoints.save_checkpoint(ckpt_dir='ckpts', target=state, step=state.step)

                else:
                    patience += 1
                    if patience > config['train']['patience']:
                        break

            if i == config['train']['steps']:
                break

    result['train_psnr'] = watch_val

    # Fine-tune with large patch size
    state = checkpoints.restore_checkpoint(ckpt_dir='ckpts', target=state)
    summary_writer_fine_tune = tensorboard.SummaryWriter('./fine_tune_logs')

    train_ds, valid_ds = load_dataset(
        batch_size=config['train']['batch_size'], scale=config['train']['scale'],
        train_lr_image_size=config['fine_tuning']['lr_image_size']
    )
    watch_val = -jnp.inf
    valid_loss = jnp.inf
    valid_ssim = -jnp.inf
    patience = 0

    with tqdm(total=config['fine_tuning']['steps'], colour='CYAN', position=0) as pbar:
        pbar.set_description('Fine-tuning steps')
        loss_history = []

        for i in range(1, config['fine_tuning']['steps'] + 1):
            pbar.update(1)
            state, loss = train_step(state, batch)
            summary_writer_fine_tune.scalar('train_loss', loss, step=i)
            loss_history.append(loss)
            loss_history = loss_history[-100:]
            pbar.set_postfix(
                train_loss=sum(loss_history) / len(loss_history), valid_psnr=watch_val, patience=patience,
                valid_loss=valid_loss, valid_ssim=valid_ssim
            )

            if (i % config['train']['check_every'] == 0) and i > 0:
                valid_loss_list = []
                valid_psnr_list = []
                valid_ssim_list = []

                with tqdm(range(100), colour='YELLOW', desc='Valid steps', position=1, leave=False) as tbar:
                    valid_np_ds = valid_ds.as_numpy_iterator()
                    for n in tbar:
                        valid_batch = next(valid_np_ds)
                        y, y_hat = inference(state, valid_batch)
                        val_metrics = metrics(y, y_hat)
                        valid_loss_list.append(val_metrics['l1_loss'])
                        valid_psnr_list.append(val_metrics['psnr'])
                        valid_ssim_list.append(val_metrics['ssim'])

                    valid_loss = sum(valid_loss_list) / len(valid_loss_list)
                    valid_psnr = sum(valid_psnr_list) / len(valid_psnr_list)
                    valid_ssim = sum(valid_ssim_list) / len(valid_ssim_list)
                    cur_step = int(i // config['train']['check_every'])
                    summary_writer.scalar('valid_loss', valid_loss, step=cur_step)
                    summary_writer.scalar('valid_psnr', valid_psnr, step=cur_step)
                    summary_writer.scalar('valid_ssim', valid_ssim, step=cur_step)

                # Early stopping
                if valid_psnr > watch_val:
                    watch_val = valid_psnr
                    patience = 0
                    checkpoints.save_checkpoint('ckpts_fine_tune', state, state.step)

                else:
                    patience += 1
                    if patience > config['train']['patience']:
                        break

            if i == config["fine_tuning"]["steps"]:
                break

    result['fine_tune_psnr'] = watch_val

    OmegaConf.save(OmegaConf.create(result), 'result.yaml')

if __name__ == '__main__':
    main()
