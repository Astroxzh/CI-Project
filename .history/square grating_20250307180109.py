import jax.numpy as jnp
from chromatix.element import PointSource

def square_lattice(heigt: float, width: float, interval:float):
    '''
    Generates a square lattice of round holes with given height, width, and interval.

    Args:
        height (float): The height of the square lattice in micrometers.
        width (float): The width of the square lattice in micrometers.
        interval (float): The interval between round holes in micrometers.
    
    Returns:
        A array of square lattice
    '''