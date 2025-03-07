import jax.numpy as jnp
from chromatix.elements import PointSource
import matplotlib.pyplot as plt
from chromatix.field import Field
from chromatix.utils.shapes import _broadcast_2d_to_spatial

def square_lattice(field: Field, interval:float, r_holes:float):
    '''
    Generates a square lattice of round holes that machtes the grid of chromatix.field
    and perturb the input field

    Args:
        field: the complex field to be perturbed
        interval (float): The interval between round holes in micrometers.
        r_holes (float): The radius of the round holes in micrometers.
    
    Returns:
        The field after being pertuebed
    '''
    # Step size for the grid resolution
    res = field.dx
    height = field.shape[0] * res
    width = field.shape[1] * res

    # Generate the grid coordinates
    x = jnp.arange(0, width, res)
    y = jnp.arange(0, height, res)
    X, Y = jnp.meshgrid(x, y, indexing = "ij")

    # Initialize the transmission array
    trans = jnp.zeros_like(X, dtype = "bool")

    # Centers of holes
    centers_x = jnp.arange(0, width, interval)
    offset = width/2 - (centers_x[0] + centers_x[-1])/2
    centers_x += offset
    
    centers_y = jnp.arange(0, height, interval)
    offset = height/2 - (centers_y[0] + centers_y[-1])/2
    centers_y += offset

    # For each hole center, set the circle around the center with a radius as True.
    for cx in centers_x:
        for cy in centers_y:
            r = jnp.sqrt((X - cx)**2 + (Y - cy)**2)
            trans = trans | (r < r_holes)
    
    trans = _broadcast_2d_to_spatial(trans, field.ndim)
    return field * trans


from chromatix.elements import PlaneWave
# Create a field
field = Field(
    u = jnp.ones((512, 512), dtype = jnp.complex64)
    _spectrum = 0.5,
    _dx = 0.1,
    _spectral_density = 1
)

# Perturb the field with a square lattice of round holes
field = square_lattice(field, 10, 2)

# Plot the field
plt.imshow(jnp.abs(field.u.squeeze()))
plt.show()