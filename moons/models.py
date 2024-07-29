from typing import Tuple, Callable, Union, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array


class ResidualNetwork(eqx.Module):
    _in: eqx.nn.Linear
    layers: Tuple[eqx.nn.Linear]
    dropouts: Tuple[eqx.nn.Dropout]
    _out: eqx.nn.Linear
    activation: Callable
    dropout_rate: float
    y_dim: Optional[int] = None

    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        width_size: int, 
        depth: int, 
        y_dim: int, 
        activation: Callable,
        dropout_rate: float = 0.,
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
    
    def __call__(
        self, 
        x: Array, 
        t: Union[float, Array], 
        y: Optional[Array] = None,
        *, 
        key: Key = None
    ) -> Array:
        t = jnp.atleast_1d(t)
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