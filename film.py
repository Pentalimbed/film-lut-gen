from common import better_reshape

import os
from typing import Collection

import colour
from colour import SpectralDistribution
import numpy as np


def curve_to_sd(x: np.ndarray) -> SpectralDistribution:
    return better_reshape(SpectralDistribution(x[1], x[0]), colour.SPECTRAL_SHAPE_DEFAULT)


class Film:
    def __init__(self, char: np.ndarray, sen: np.ndarray, dye_spec: np.ndarray):
        self.char = char

        self.sen = sen.copy()
        self.sen[1] = 10 ** self.sen[1]  # log sensitivity
        self.sen = curve_to_sd(self.sen)

        self.dye_spec = curve_to_sd(dye_spec)


class RgbFilm:
    def __init__(self, films: Collection[Film]):
        self.films = list(films)

    @staticmethod
    def from_dir(path: str):
        films = [
            Film(
                np.load(os.path.join(path, f"char-{label}.npy")),
                np.load(os.path.join(path, f"sen-{label}.npy")),
                np.load(os.path.join(path, f"dye-spec-{label}.npy")),
            )
            for label in ("rc", "gm", "by")
        ]
        return RgbFilm(films)


class FilmLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.cache = dict()

    def load(self, film_name: str):
        if film_name in self.cache:
            return self.cache[film_name]

        film = RgbFilm.from_dir(os.path.join(self.base_path, film_name))
        self.cache[film_name] = film
        return film


DEFAULT_FILM_LOADER = FilmLoader("./data")