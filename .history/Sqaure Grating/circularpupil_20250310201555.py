

class CircularPupil(nn.Module):
    """
    Simulate a circular pupil function using chromatix.functional.circular_pupil
    """

    w: float

    @nn.compact
    def __call__(self, field: jnp.ndarray) -> jnp.ndarray:
        return circular_pupil(field, self.w)