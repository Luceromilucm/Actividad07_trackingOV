"""Registro del historial para el programa de tracking (Nivel 1).

Uso típico dentro del bucle del Nivel 1:

    from logger import RegistroHistorial

    with RegistroHistorial(fuente="video.mp4", fps=fps, ancho=w, alto=h,
                           limite_kmh=40, analizar_al_cerrar=True) as reg:
        for frame_idx, frame in enumerate(video):
            reg.guardar_frame(frame)              # para el frame de referencia
            for det in detecciones:               # salida del tracker
                reg.registrar(det.track_id, frame_idx, det.clase, *det.caja,
                              velocidad_kmh=det.v, excede_limite=det.v > 40)
    # al salir del `with` (también si hay una excepción o Ctrl+C) se cierra el CSV,
    # se escribe meta.json, se calcula frame_referencia.jpg y se genera el análisis.

Cada corrida crea su propia carpeta: data/sesiones/AAAA-MM-DD_HH-MM-SS/.
El CSV se vacía a disco cada `flush_cada` filas, así que si el programa muere
de golpe queda lo registrado hasta ese momento. En Windows, cerrar la consola
con la X no ejecuta el `finally`; en ese caso el análisis se lanza a mano:
    python src/main_analitica.py data/sesiones/<carpeta>
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

COLUMNAS = ["track_id", "frame", "timestamp_s", "clase", "cx", "cy",
            "x1", "y1", "x2", "y2", "velocidad_kmh", "excede_limite"]

RAIZ = Path(__file__).resolve().parents[1]


def crear_carpeta_sesion(base: str | Path | None = None, inicio: datetime | None = None) -> Path:
    base = Path(base) if base else RAIZ / "data" / "sesiones"
    inicio = inicio or datetime.now()
    carpeta = base / f"{inicio:%Y-%m-%d_%H-%M-%S}"
    n = 2
    while carpeta.exists():
        carpeta = base / f"{inicio:%Y-%m-%d_%H-%M-%S}_{n}"
        n += 1
    carpeta.mkdir(parents=True)
    return carpeta


class RegistroHistorial:
    def __init__(self, carpeta: str | Path | None = None, fuente: str = "", fps: float = 30.0,
                 ancho: int | None = None, alto: int | None = None, limite_kmh: float | None = None,
                 px_por_metro: float | None = None, linea_conteo=None,
                 inicio: datetime | None = None, en_vivo: bool = False,
                 flush_cada: int = 200, frames_referencia: int = 30,
                 analizar_al_cerrar: bool = False):
        """
        en_vivo=True (webcam): timestamp_s se toma del reloj.
        en_vivo=False (video): timestamp_s = frame / fps, nunca la hora del reloj.
        `inicio` es la hora real del frame 0 (por defecto, ahora).
        """
        self.inicio = inicio or datetime.now()
        self.carpeta = Path(carpeta) if carpeta else crear_carpeta_sesion(inicio=self.inicio)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.en_vivo = en_vivo
        self.fps = float(fps)
        self.flush_cada = flush_cada
        self.analizar_al_cerrar = analizar_al_cerrar
        self.meta = {
            "fuente": fuente, "fps": self.fps, "ancho": ancho, "alto": alto,
            "inicio_datetime": self.inicio.isoformat(timespec="seconds"),
            "limite_kmh": limite_kmh, "px_por_metro": px_por_metro,
            "linea_conteo": linea_conteo,
        }
        self._archivo = open(self.carpeta / "historial.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._archivo)
        self._csv.writerow(COLUMNAS)
        self._pendientes = 0
        self._ultimo_frame = 0
        self._ultimo_t = 0.0
        self._n_ref = frames_referencia
        self._muestras: list[np.ndarray] = []
        self._vistos = 0
        self._paso = max(1, int(round(self.fps)))   # al principio, una muestra por segundo
        self._cerrado = False
        self._escribir_meta()

    # --- registro ---
    def registrar(self, track_id: int, frame: int, clase: str, x1: float, y1: float, x2: float, y2: float,
                  velocidad_kmh: float | None = None, excede_limite: bool | int | None = None,
                  timestamp_s: float | None = None):
        if timestamp_s is None:
            timestamp_s = ((datetime.now() - self.inicio).total_seconds() if self.en_vivo
                           else frame / self.fps)
        cx, cy = (x1 + x2) / 2, y2   # punto inferior central: donde el vehículo toca el suelo
        self._csv.writerow([int(track_id), int(frame), round(float(timestamp_s), 4), clase,
                            round(cx, 1), round(cy, 1), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                            "" if velocidad_kmh is None else round(float(velocidad_kmh), 1),
                            "" if excede_limite is None else int(bool(excede_limite))])
        self._ultimo_frame = max(self._ultimo_frame, int(frame))
        self._ultimo_t = max(self._ultimo_t, float(timestamp_s))
        self._pendientes += 1
        if self._pendientes >= self.flush_cada:
            self._archivo.flush()
            self._pendientes = 0

    def guardar_frame(self, frame_bgr: np.ndarray):
        """Guarda una muestra cada cierto número de frames para el fondo sin autos.

        Se conservan hasta `frames_referencia` muestras repartidas en el tiempo;
        al cerrar se calcula la mediana píxel a píxel, que borra los vehículos.
        """
        if self.meta["ancho"] is None:
            self.meta["alto"], self.meta["ancho"] = frame_bgr.shape[:2]
        self._vistos += 1
        if self._vistos % self._paso:
            return
        self._muestras.append(frame_bgr.copy())
        if len(self._muestras) > self._n_ref:
            # se descarta una de cada dos y se duplica el paso: las muestras siguen
            # repartidas en toda la sesión y la memoria queda acotada
            self._muestras = self._muestras[::2]
            self._paso *= 2

    # --- cierre ---
    def _escribir_meta(self):
        (self.carpeta / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False),
                                                encoding="utf-8")

    def cerrar(self):
        if self._cerrado:
            return
        self._cerrado = True
        self._archivo.flush()
        self._archivo.close()
        self.meta["duracion_s"] = round(self._ultimo_t if self.en_vivo else (self._ultimo_frame + 1) / self.fps, 3)
        self._escribir_meta()
        if self._muestras:
            import cv2
            fondo = np.median(np.stack(self._muestras), axis=0).astype(np.uint8)
            cv2.imwrite(str(self.carpeta / "frame_referencia.jpg"), fondo)
        print(f"Historial guardado en {self.carpeta}")
        if self.analizar_al_cerrar:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from main_analitica import analizar
            try:
                analizar(self.carpeta)
            except Exception as e:  # el análisis nunca debe impedir que el tracking cierre
                print(f"No se pudo generar el análisis: {e}\n"
                      f"Puedes lanzarlo a mano: python src/main_analitica.py \"{self.carpeta}\"")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()
        return False

