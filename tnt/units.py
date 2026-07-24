"""General angle-to-physical-length conversions, used across the project."""

from unxt import Quantity


def angular_to_physical(angle: Quantity, distance: Quantity) -> Quantity:
    """Convert a small angle to a projected physical length at a distance.

    Uses the small-angle approximation: ``physical_length = distance *
    angle_in_radians``.

    Args:
        angle: An angular size (e.g. in rad or arcsec).
        distance: The distance to the object.

    Returns:
        The projected physical length, in `distance`'s unit.
    """
    return distance * angle.ustrip("rad")


def physical_to_angular(length: Quantity, distance: Quantity) -> Quantity:
    """Convert a projected physical length to a small angle at a distance.

    Inverse of `angular_to_physical`.

    Args:
        length: A projected physical length.
        distance: The distance to the object.

    Returns:
        The angular size, in radians.
    """
    return Quantity((length / distance).ustrip(""), "rad")
