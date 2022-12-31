import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import tensorflow_datasets as tfds


def load_dataset(
        batch_size: int = 64, scale: int = 3, degradation: str = 'bicubic', train_lr_image_size: int = 64,
        return_type: tf.DType = tf.float32, steps: int = 500000
):
    train_ds, valid_ds = tfds.load(
        f'div2k/{degradation}_x{scale}', split=['train', 'validation'], as_supervised=True
    )
    n_samples = train_ds.cardinality()
    n_repeat = max(1, (steps * batch_size) // n_samples)
    train_ds = train_ds.repeat(n_repeat).shuffle(n_samples).map(
            lambda lr, hr: (tf.cast(lr, return_type), tf.cast(hr, return_type)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .map(
            lambda lr, hr: random_crop(lr, hr, scale=scale, hr_crop_size=train_lr_image_size * scale),
            num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

    valid_ds = valid_ds.map(
            lambda lr, hr: (tf.cast(lr, return_type), tf.cast(hr, return_type)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(1, drop_remainder=False) \
        .prefetch(tf.data.AUTOTUNE)

    train_ds = train_ds.as_numpy_iterator()
    # valid_ds = valid_ds.as_numpy_iterator()
    return train_ds, valid_ds


'''
Belows are preprocessing funcs from https://github.com/Algolzw/NCNet/blob/main/div2k.py
'''


def random_crop(lr_img, hr_img, hr_crop_size=192, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)
