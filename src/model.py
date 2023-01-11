import jax
import jax.numpy as jnp
from jax import lax
import einops

import flax
from flax import linen as nn

from typing import Optional, Callable


def nearest_conv_init(
        input_c: int, out_c: int, scale: int
) -> jnp.ndarray:
    kernel = jnp.transpose(
        jnp.reshape(
            jnp.repeat(
                jnp.eye(input_c), (out_c * (scale ** 2)) // input_c, axis=0
            ), (1, 1, -1, input_c)
        ), (0, 1, 3, 2)
    )
    return lax.stop_gradient(kernel)


class NCNet(nn.Module):
    n_filters: int
    scale_factor: int
    out_c: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        out_c = self.out_c or x.shape[-1]
        nearest_conv_kernel = self.variable(
            'nearest_conv', 'kernel', lambda: nearest_conv_init(x.shape[-1], out_c, self.scale_factor)
        )
        nc_kernel = lax.stop_gradient(nearest_conv_kernel.value)
        skip = lax.conv_general_dilated(
            x, nc_kernel, (1, 1), 'VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        for i in range(7):
            x = nn.Conv(
                self.n_filters if i < 5 else out_c * (self.scale_factor ** 2), (3, 3),
                padding='SAME', kernel_init=nn.initializers.glorot_normal()
            )(x)
            if i != 6:
                x = nn.relu(x)
        x += skip
        x = einops.rearrange(
            x, 'b h w (s1 s2 c) -> b (h s1) (w s2) c', s1=self.scale_factor, s2=self.scale_factor
        )
        x = jnp.clip(x, 0., 255.)
        return x


if __name__ == '__main__':
    # Test and Visualize
    x = jnp.ones((1, 256, 256, 3))
    model = NCNet(32, 3)
    rng = jax.random.PRNGKey(0)
    print(model.tabulate(rng, x))
