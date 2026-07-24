import jax.numpy as jnp
import unxt as u

from tnt import units


def test_angular_to_physical():
    angle = u.Quantity(jnp.array([0.5, 2.0]), "rad")
    distance = u.Quantity(30.5, "Mpc")

    physical = units.angular_to_physical(angle, distance)

    assert physical.unit == u.unit("Mpc")
    assert jnp.allclose(physical.ustrip("Mpc"), jnp.array([15.25, 61.0]))


def test_physical_to_angular():
    length = u.Quantity(jnp.array([15.25, 61.0]), "Mpc")
    distance = u.Quantity(30.5, "Mpc")

    angular = units.physical_to_angular(length, distance)

    assert angular.unit == u.unit("rad")
    assert jnp.allclose(angular.ustrip("rad"), jnp.array([0.5, 2.0]))


def test_round_trip():
    angle = u.Quantity(jnp.array([0.5, 2.0]), "rad")
    distance = u.Quantity(30.5, "Mpc")

    round_tripped = units.physical_to_angular(
        units.angular_to_physical(angle, distance), distance
    )

    assert jnp.allclose(round_tripped.ustrip("rad"), angle.ustrip("rad"))
