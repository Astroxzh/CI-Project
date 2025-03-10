from chromatix.systems import OpticalSystem
from chromatix.elements import ObjectivePointSource
from square_grating import SquareGrating
from circularpupil import CircularPupil
from transparentimage import TransparentImage
from chromatix.elements import BasicSensor

ptychography_system = OpticalSystem(
    elements = [
        ObjectivePointSource(),
        SquareGrating(interval = 1, r_holes = 0.2),
        CircularPupil(w = 0.5),
        TransparentImage(image = jnp.ones((500, 500), dtype = jnp.complex64), transparency = 0.5)，
        BasicSensor()
    ]
)