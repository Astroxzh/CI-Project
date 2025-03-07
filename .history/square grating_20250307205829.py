import jax.numpy as jnp
from chromatix import PointSource

def square_lattice(heigt: float, width: float, interval:float, r_holes:float):
    '''
    Generates a square lattice of round holes with given height, width, and interval.

    Args:
        height (float): The height of the square lattice in micrometers.
        width (float): The width of the square lattice in micrometers.
        interval (float): The interval between round holes in micrometers.
        r_holes (float): The radius of the round holes in micrometers.
    
    Returns:
        A 2D array representing the transmission function of the square lattice,
        where 1 indicates points inside the holes and 0 indicates points outside.'
    '''
    # Step size for the grid resolution
