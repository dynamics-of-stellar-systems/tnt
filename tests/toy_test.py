from tnt import main
import jax.numpy as jnp

def test_square():
    arr = jnp.array([1,2,3])
    assert main.add(arr) == 6