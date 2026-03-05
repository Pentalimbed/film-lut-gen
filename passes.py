from common import better_reshape
from film import RgbFilm, DEFAULT_FILM_LOADER

from abc import ABC, abstractmethod
from typing import Dict
import colour
from colour import SpectralDistribution
import numpy as np


class PassBase(ABC):
    @abstractmethod
    def forward(self, v):
        pass

    @staticmethod
    @abstractmethod
    def from_dict(pdict: Dict):
        pass


class GainPass(PassBase):
    def __init__(self, gain: float):
        self.gain = gain

    def forward(self, v: np.ndarray | SpectralDistribution) -> np.ndarray | SpectralDistribution:
        if isinstance(v, np.ndarray):
            return v * self.gain
        elif isinstance(v, SpectralDistribution):
            sd = v.copy()
            sd.values *= self.gain
            return sd
        else:
            assert False, f"Invalid input type for {self.__name__}"

    @staticmethod
    def from_dict(pdict: Dict):
        return GainPass(pdict["gain"])


class RgbUpsamplePass(PassBase):
    def __init__(self, colour_space: str, apply_cctf_decoding: bool):
        self.colour_space = colour_space
        self.apply_cctf_decoding = apply_cctf_decoding

    def forward(self, v: np.ndarray) -> SpectralDistribution:
        # TODO better upsampling method
        xyz = colour.RGB_to_XYZ(v, self.colour_space, apply_cctf_decoding=self.apply_cctf_decoding)
        sd = better_reshape(colour.XYZ_to_sd(xyz, method="Mallett 2019"), colour.SPECTRAL_SHAPE_DEFAULT)
        sd.values = np.maximum(0, sd.values)
        return sd

    @staticmethod
    def from_dict(pdict: Dict):
        return RgbUpsamplePass(pdict["colour_space"], pdict["apply_cctf_decoding"])


class RgbDownsamplePass(PassBase):
    def __init__(self, colour_space: str, apply_cctf_encoding: bool):
        self.colour_space = colour_space
        self.apply_cctf_encoding = apply_cctf_encoding

    def forward(self, v: SpectralDistribution) -> np.ndarray:
        sd = colour.sd_blackbody(5000) / colour.sd_to_XYZ(colour.sd_blackbody(5000))[1]
        xyz = colour.sd_to_XYZ(v, illuminant=sd) / 100
        return colour.XYZ_to_RGB(xyz, self.colour_space, apply_cctf_encoding=self.apply_cctf_encoding)

    @staticmethod
    def from_dict(pdict: Dict):
        return RgbDownsamplePass(pdict["colour_space"], pdict["apply_cctf_encoding"])


class FilmExposePass(PassBase):
    def __init__(self, film: RgbFilm):
        self.film = film

    def forward(self, v: SpectralDistribution) -> np.ndarray:
        dye_densities = []
        for film in self.film.films:
            exposure = np.trapezoid(v.values * film.sen.values, v.wavelengths) / 300
            exposure = np.maximum(exposure, 10 ** film.char[0].min())
            dye_dens = np.interp(np.log10(exposure), film.char[0], film.char[1])
            dye_densities.append(dye_dens)
        return np.array(dye_densities)

    @staticmethod
    def from_dict(pdict: Dict):
        return FilmExposePass(DEFAULT_FILM_LOADER.load(pdict["film_name"]))


class FilmProjectPass(PassBase):
    def __init__(self, film: RgbFilm):
        self.film = film

    def forward(self, v: np.ndarray) -> SpectralDistribution:
        dye_specs = np.stack([film.dye_spec.values for film in self.film.films])
        den_sd = np.einsum("kx,k->x", dye_specs, v)
        tr_sd = 10 ** (-den_sd)
        return SpectralDistribution(tr_sd, colour.SPECTRAL_SHAPE_DEFAULT)

    @staticmethod
    def from_dict(pdict: Dict):
        return FilmProjectPass(DEFAULT_FILM_LOADER.load(pdict["film_name"]))


PASS_NAME_MAP = {
    tp.__name__: tp
    for tp in [
        GainPass,
        RgbUpsamplePass,
        RgbDownsamplePass,
        FilmExposePass,
        FilmProjectPass,
    ]
}


def dict_to_pass(pdict: Dict):
    return PASS_NAME_MAP[pdict["type"]].from_dict(pdict)
