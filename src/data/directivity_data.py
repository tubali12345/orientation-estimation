from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
from scipy.io import loadmat


@dataclass
class PyData:
    fs: int
    grid: pra.doa.GridSphere
    impulse_responses: np.ndarray


@dataclass
class MatData:
    fs: int
    azimuth: np.ndarray
    colatitude: np.ndarray
    irs: np.ndarray

    @classmethod
    def from_file(cls, file_path: str | Path) -> "MatData":
        data = loadmat(file_path)
        return cls(
            fs=data["fs"].item(),
            azimuth=data["azimuth"],
            colatitude=data["colatitude"],
            irs=data["irs"],
        )

    def __str__(self):
        return (
            f"fs: {self.fs}, azimuth: {self.azimuth.shape}, colatitude: {self.colatitude.shape}, irs: {self.irs.shape}"
        )

    def to_pydata(self) -> PyData:
        return PyData(
            fs=self.fs,
            grid=self.get_grid(),
            impulse_responses=self.irs.transpose(),
        )

    def get_grid(self) -> pra.doa.GridSphere:
        """Turn the azimuth and colatitude into a grid spehere with cartesian coordinates"""
        x = np.sin(self.colatitude) * np.cos(self.azimuth)
        y = np.sin(self.colatitude) * np.sin(self.azimuth)
        z = np.cos(self.colatitude)
        grid = np.vstack((x, y, z))
        return pra.doa.GridSphere(n_points=grid.shape[1], cartesian_points=grid)
