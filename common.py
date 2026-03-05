import colour
import numpy as np
from colour import SpectralDistribution, SpectralShape


def better_reshape(
    sd: SpectralDistribution, shape: SpectralShape
) -> SpectralDistribution:
    return sd.extrapolate(
        shape, extrapolator_kwargs={"method": "Constant", "left": 0, "right": 0}
    ).interpolate(shape, colour.CubicSplineInterpolator)


def curve_to_sd(x: np.ndarray, reshape: bool = True) -> SpectralDistribution:
    sd = SpectralDistribution(x[1], x[0])
    if reshape:
        shape = SpectralShape(np.floor(sd.shape.start), np.ceil(sd.shape.end), 1)
        sd = better_reshape(sd, shape)

    return sd
