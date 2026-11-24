"""Calibración de distancia a lo largo de la calle para estimar velocidades.

Con perspectiva (y más con teleobjetivo), un solo `px_por_metro` no sirve: un
metro cerca de la cámara ocupa muchos más píxeles que un metro a lo lejos. Para
una calle plana y casi recta, la distancia s (en metros) a lo largo de la calle
y la fila y de la imagen se relacionan por una transformación proyectiva 1D:

    s(y) = (a·y + b) / (c·y + 1)

Bastan 3 puntos con distancia conocida para fijar a, b y c; con más se ajusta por
mínimos cuadrados. Los puntos típicos son el inicio de líneas discontinuas
consecutivas del carril, que se pintan a una distancia fija (el «periodo»).

La velocidad sale de cuánto avanza s por segundo. Se ignora el movimiento lateral
(los cambios de carril), que es pequeño frente al avance. Solo se mide entre la
primera y la última marca usadas (con un pequeño margen): fuera de ese tramo el
modelo extrapola, y si la calle se curva (como al fondo del video por defecto)
el avance deja de verse en y.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Calibracion:
    ancho: int
    alto: int
    puntos: list            # [[y_px, s_m], ...]
    limite_kmh: float | None = None
    nota: str = ""

    def __post_init__(self):
        p = np.asarray(self.puntos, float)
        if len(p) < 3:
            raise ValueError("La calibración necesita al menos 3 puntos.")
        y, s = p[:, 0], p[:, 1]
        # a·y + b − c·y·s = s   (lineal en a, b, c)
        A = np.column_stack([y, np.ones_like(y), -y * s])
        self.a, self.b, self.c = np.linalg.lstsq(A, s, rcond=None)[0]
        self.error_m = float(np.sqrt(np.mean((self._s(y) - s) ** 2)))
        margen = 0.25 * float(np.min(np.abs(np.diff(np.sort(y))))) if len(y) > 1 else 0.0
        self.y_min, self.y_max = float(y.min()) - margen, float(y.max()) + margen

    def _s(self, y):
        y = np.asarray(y, float)
        return (self.a * y + self.b) / (self.c * y + 1.0)

    def metros(self, x: float, y: float) -> float | None:
        """Distancia a lo largo de la calle del punto (x, y), o None fuera del tramo calibrado."""
        if not (self.y_min <= y <= self.y_max):
            return None
        return float(self._s(y))

    def como_dict(self) -> dict:
        return {"ancho": self.ancho, "alto": self.alto, "puntos": self.puntos,
                "limite_kmh": self.limite_kmh, "nota": self.nota, "error_m": round(self.error_m, 3)}

    def guardar(self, ruta: Path):
        Path(ruta).write_text(json.dumps(self.como_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: Path) -> "Calibracion":
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(d["ancho"], d["alto"], d["puntos"], d.get("limite_kmh"), d.get("nota", ""))
