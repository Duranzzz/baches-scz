"""Proyeccion de pixeles al plano de la calzada para una camara frontal fija (modelo pinhole).

Ejes del mundo: X a la derecha, Y hacia abajo, Z hacia adelante (horizontal); origen en la camara.
El suelo es el plano Y = altura. La camara solo tiene cabeceo (pitch) hacia abajo; sin alabeo ni guinada.
No corrige la distorsion del lente: con una dashcam gran angular hay que desdistorsionar antes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int  # px
    height: int  # px
    hfov_deg: float  # campo de vision horizontal
    height_m: float  # altura del lente sobre el suelo
    pitch_deg: float  # cabeceo hacia abajo (positivo = mira al suelo)

    @property
    def f(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def horizon_v(self) -> float:
        """Fila (px) del horizonte; por encima de ella ningun pixel toca el suelo."""
        return self.height / 2 - self.f * math.tan(math.radians(self.pitch_deg))

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float] | None:
        """(u, v) en px -> (X, Z) en metros sobre el suelo, o None si el rayo no toca el suelo."""
        th = math.radians(self.pitch_deg)
        dx = (u - self.width / 2) / self.f
        dy = (v - self.height / 2) / self.f
        wy = dy * math.cos(th) + math.sin(th)
        if wy <= 1e-9:
            return None
        t = self.height_m / wy
        return t * dx, t * (math.cos(th) - dy * math.sin(th))

    def ground_to_pixel(self, x: float, z: float) -> tuple[float, float] | None:
        """(X, Z) sobre el suelo -> (u, v) en px, o None si queda detras de la camara."""
        th = math.radians(self.pitch_deg)
        y = self.height_m
        yc = y * math.cos(th) - z * math.sin(th)
        zc = y * math.sin(th) + z * math.cos(th)
        if zc <= 1e-9:
            return None
        return self.f * x / zc + self.width / 2, self.f * yc / zc + self.height / 2


@dataclass(frozen=True)
class Footprint:
    distance_m: float  # Z del borde inferior (el mas cercano)
    width_m: float  # ancho del borde inferior
    length_m: float  # extension a lo largo de la calle
    area_m2: float  # area del trapecio proyectado (cota superior del dano)


def polygon_area(points: list[tuple[float, float]]) -> float:
    a = np.asarray(points, dtype=np.float64)
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def bbox_footprint(cam: Camera, xyxy: tuple[float, float, float, float], max_distance_m: float = 15.0) -> Footprint | None:
    """Proyecta una caja al suelo. None si alguna esquina no toca el suelo o queda mas lejos que max_distance_m.

    Lejos de la camara un pixel cubre mucha calle y el error crece con Z al cuadrado,
    por eso conviene medir solo en la franja cercana y quedarse con la mejor medicion de cada seguimiento.
    """
    x1, y1, x2, y2 = xyxy
    corners = [cam.pixel_to_ground(u, v) for u, v in ((x1, y2), (x2, y2), (x2, y1), (x1, y1))]
    if any(c is None for c in corners):
        return None
    bl, br, tr, tl = corners
    if max(tl[1], tr[1]) > max_distance_m:
        return None
    return Footprint(
        distance_m=(bl[1] + br[1]) / 2,
        width_m=br[0] - bl[0],
        length_m=(tl[1] + tr[1]) / 2 - (bl[1] + br[1]) / 2,
        area_m2=polygon_area(corners),
    )
