import jax.numpy as jnp
from chromatix.elements import PointSource
import matplotlib.pyplot as plt

def square_lattice(height: float, width: float, interval:float, r_holes:float):
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
    res = interval / 10 

    # Generate the grid coordinates
    x = jnp.arange(0, width, res)
    y = jnp.arange(0, height, res)
    X, Y = jnp.meshgrid(x, y, indexing = "ij")

    # Initialize the transmission array
    trans = jnp.zeros(width, height, dtype = "boolean")

    # Centers of holes
    centers_x = jnp.arange(0, width, interval)
    centers_y = jnp.arange(0, height, interval)

    # For each hole center, set the circle around the center with aa radius True.
    for cx, cy in zip(centers_x, centers_y):
        r = jnp.sqrt((X - cx)**2 + (Y - cy)**2)
        trans = trans | (r < r_holes)
    
    return trans

trans = square_lattice(10, 10, 1, 0.2)

# Plot the transmission function
plt.figure(figsize=(5, 5))

plt.imshow(trans, origin='lower', extent=(0, 10, 0, 10), cmap='gray')

