import jax.numpy as jnp
import unxt as u

from tnt import quantity_conversions


def test_angular_to_physical():
    angle = u.Quantity(jnp.array([0.5, 2.0]), "rad")
    distance = u.Quantity(30.5, "Mpc")

    physical = quantity_conversions.angular_to_physical(angle, distance)

    assert physical.unit == u.unit("Mpc")
    assert jnp.allclose(physical.ustrip("Mpc"), jnp.array([15.25, 61.0]))
