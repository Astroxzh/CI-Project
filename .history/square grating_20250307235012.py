import jax.numpy as jnp
from chromatix.elements import PointSource
import matplotlib.pyplot as plt
from chromatix.field import Field

def square_lattice(field: Field, interval:float, r_holes:float):
    '''
    Generates a square lattice of round holes that machtes the grid of chromatix.field

    Args:
        
        interval (float): The interval between round holes in micrometers.
        r_holes (float): The radius of the round holes in micrometers.
    
    Returns:
        A 2D array representing the transmission function of the square lattice,
        where 1 indicates points inside the holes and 0 indicates points outside.'
    '''
    # Step size for the grid resolution
    res = field.dx
    height = field.u.shape[-4] * res
    width = field.u.shape[-3] * res

    # Generate the grid coordinates
    x = jnp.arange(0, width, res)
    y = jnp.arange(0, height, res)
    X, Y = jnp.meshgrid(x, y, indexing = "ij")

    # Initialize the transmission array
    trans = jnp.zeros_like(X, dtype = "bool")

    # Centers of holes
    centers_x = jnp.arange(0, width, interval)
    centers_y = jnp.arange(0, height, interval)

    # For each hole center, set the circle around the center with aa radius True.
    for cx in centers_x:
        for cy in centers_y:
            r = jnp.sqrt((X - cx)**2 + (Y - cy)**2)
            trans = trans | (r < r_holes)
    
    return trans

