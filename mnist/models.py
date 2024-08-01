from typing import Tuple, Callable, Union, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array
from einops import repeat, rearrange


class ResidualNetwork(eqx.Module):
    _in: eqx.nn.Linear
    layers: Tuple[eqx.nn.Linear]
    dropouts: Tuple[eqx.nn.Dropout]
    _out: eqx.nn.Linear
    activation: Callable
    dropout_rate: float
    y_dim: Optional[int] = None
    time_embedder: Optional[Callable] = None

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        width_size: int, 
        depth: int, 
        y_dim: int, 
        activation: Callable,
        dropout_rate: float = 0.,
        time_embedder: Optional[Callable] = None,
        *, 
        key: Key
    ):
        in_key, *net_keys, out_key = jr.split(key, 2 + depth)
        self._in = eqx.nn.Linear(
            in_size + y_dim if y_dim is not None else in_size, width_size, 
            key=in_key
        )
        layers = [
            eqx.nn.Linear(
                width_size + y_dim if y_dim is not None else width_size, width_size, 
                key=_key
            )
            for _key in net_keys 
        ]
        self._out = eqx.nn.Linear(width_size, out_size, key=out_key) # For mu, sigma
        dropouts = [
            eqx.nn.Dropout(p=dropout_rate) for _ in layers
        ]
        self.layers = tuple(layers)
        self.dropouts = tuple(dropouts)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.y_dim = y_dim
        self.time_embedder = time_embedder
    
    def __call__(
        self, 
        x: Array, 
        t: Union[float, Array], 
        y: Optional[Array] = None,
        *, 
        key: Key = None
    ) -> Array:
        t = jnp.atleast_1d(t)
        if self.time_embedder is not None:
            t = self.time_embedder(t)
        xyt = jnp.concatenate([x, y, t] if y is not None else [x, t])
        h0 = self.activation(self._in(xyt))
        h = h0
        for l, d in zip(self.layers, self.dropouts):
            # Condition on time at each layer
            hyt = jnp.concatenate([h, y, t] if y is not None else [h, t])
            h = l(hyt)
            h = d(h, key=key)
            h = self.activation(h)
            h = h0 + h
        o = self._out(h)
        return o


class MixerBlock(eqx.Module):
    patch_mixer: eqx.nn.MLP
    hidden_mixer: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key
    ):
        tkey, ckey = jr.split(key, 2)
        self.patch_mixer = eqx.nn.MLP(
            num_patches, num_patches, mix_patch_size, depth=1, key=tkey
        )
        self.hidden_mixer = eqx.nn.MLP(
            hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey
        )
        self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))

    def __call__(self, y):
        y = y + jax.vmap(self.patch_mixer)(self.norm1(y))
        y = rearrange(y, "c p -> p c")
        y = y + jax.vmap(self.hidden_mixer)(self.norm2(y))
        y = rearrange(y, "p c -> c p")
        return y


class Mixer2d(eqx.Module):
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.ConvTranspose2d
    blocks: list
    norm: eqx.nn.LayerNorm
    t1: float

    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        *,
        key,
    ):
        input_size, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        self.conv_in = eqx.nn.Conv2d(
            input_size + 1, hidden_size, patch_size, stride=patch_size, key=inkey
        )
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, input_size, patch_size, stride=patch_size, key=outkey
        )
        self.blocks = [
            MixerBlock(
                num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=bkey
            )
            for bkey in bkeys
        ]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.t1 = t1

    def __call__(self, y, t):
        t = jnp.array(t / self.t1)
        _, height, width = y.shape
        t = repeat(t, "-> 1 h w", h=height, w=width)
        y = jnp.concatenate([y, t])
        y = self.conv_in(y)
        _, patch_height, patch_width = y.shape
        y = rearrange(y, "c h w -> c (h w)")
        for block in self.blocks:
            y = block(y)
        y = self.norm(y)
        y = rearrange(y, "c (h w) -> c h w", h=patch_height, w=patch_width)
        return self.conv_out(y)