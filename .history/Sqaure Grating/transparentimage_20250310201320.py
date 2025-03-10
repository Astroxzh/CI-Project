from flax import linen as nn
import jax.numpy as jnp
from chex import Array, assert_rank
from chromatix.field import Field
from chromatix.functional import circular_pupil
from chromatix.utils import center_crop
from chromatix.utils import _broadcast_2d_to_spatial

def transparent_image(field: Field, image: Array, transparency: float) -> Field:
    """
    Simulate a transparent image assuming that the centers 
    of the image and the field on the optical axis. 

    Args:
        field (Field): The input field
        image (Array): The transparent image

    Returns:
        Field: The resulting field.
    """
    image = _broadcast_2d_to_spatial(image, field.ndim)
    image_H, image_W = image.shape[1:3]

    field_H, field_W = field.u.shape[1:3]

    if image_H >= field_H and image_W >= field_W:
        image = center_crop(image, [0, (image_H-field_H)//2, (image_W-field_W)//2, 0, 0])
    else:
         raise ValueError(
        f"Field.u dimensions ({field_H}, {field_W}) can't be larger than "
        f"image dimensions ({image_H}, {image_W}).")
    
    return field.replace(u=field.u * jnp.sqrt(image * transparency))


class TransparentImage(nn.Module):
    """
    Simulate a transparent image using transparent_image
    """

    image: Array
    transparency: float

    @nn.compact
    def __call__(self, field: jnp.ndarray) -> jnp.ndarray:
        return transparent_image(field, self.image, self.transparency)