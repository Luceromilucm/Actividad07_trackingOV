"""Carga de una sesión (historial.csv + meta.json + frame_referencia.jpg) y limpieza.

El Nivel 2 depende solo de estos archivos: no importa YOLO ni el tracker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

COLUMNAS_OBLIGATORIAS = ["track_id", "frame", "timestamp_s", "clase", "x1", "y1", "x2", "y2"]

NOMBRE_CLASE = {
    "car": "Auto", "truck": "Camión", "bus": "Bus", "motorcycle": "Moto",
    "bicycle": "Bicicleta", "van": "Furgoneta",
}


class ErrorSesion(Exception):
    """La carpeta de sesión no tiene lo necesario para analizar."""


@dataclass
class Sesion:
    carpeta: Path
    df: pd.DataFrame                       # detecciones (ya con px, py)
    meta: dict
    fondo: np.ndarray | None               # frame_referencia en RGB, o None
    inicio: datetime | None                # hora real del frame 0, si se conoce
    duracion_s: float
    limpieza: dict = field(default_factory=dict)

    @property
    def ancho(self) -> int:
        return int(self.meta["ancho"])

    @property
    def alto(self) -> int:
        return int(self.meta["alto"])


def cargar_sesion(carpeta: str | Path) -> Sesion:
    carpeta = Path(carpeta)
    csv = carpeta / "historial.csv"
    if not csv.exists():
        raise ErrorSesion(f"No se encontró {csv}. La carpeta de sesión debe contener historial.csv.")

    df = pd.read_csv(csv)
    faltan = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise ErrorSesion(f"A historial.csv le faltan columnas obligatorias: {', '.join(faltan)}")
    if df.empty:
        raise ErrorSesion("historial.csv no tiene detecciones.")

    meta = {}
    if (carpeta / "meta.json").exists():
        meta = json.loads((carpeta / "meta.json").read_text(encoding="utf-8"))

    fondo = None
    ruta_fondo = carpeta / "frame_referencia.jpg"
    if ruta_fondo.exists():
        bgr = cv2.imread(str(ruta_fondo))
        if bgr is not None:
            fondo = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resolución: meta.json > frame de referencia > lo que alcanzan las cajas
    if "ancho" not in meta or "alto" not in meta:
        if fondo is not None:
            meta["alto"], meta["ancho"] = fondo.shape[:2]
        else:
            meta["ancho"] = int(np.ceil(df["x2"].max()))
            meta["alto"] = int(np.ceil(df["y2"].max()))
    meta.setdefault("fps", 30)

    # Cajas cortadas por el borde de la imagen: cuando un vehículo entra o sale del
    # cuadro, su caja se recorta y el punto inferior central se desliza por el borde
    # (un auto que sale por abajo parece moverse en horizontal). Esas detecciones no
    # sirven para ubicar al vehículo y se quitan antes de todo lo demás.
    margen_x = max(3.0, 0.01 * meta["ancho"])
    margen_y = max(3.0, 0.01 * meta["alto"])
    en_borde = ((df["y2"] >= meta["alto"] - margen_y) | (df["x1"] <= margen_x)
                | (df["x2"] >= meta["ancho"] - margen_x))
    df = df[~en_borde].copy()
    if df.empty:
        raise ErrorSesion("Todas las detecciones tocan el borde de la imagen; revisa ancho y alto en meta.json.")
    meta["_detecciones_en_borde"] = int(en_borde.sum())

    # Punto de referencia: el punto inferior central de la caja (donde el auto toca el suelo)
    df["px"] = (df["x1"] + df["x2"]) / 2.0
    df["py"] = df["y2"].astype(float)
    df["clase"] = df["clase"].astype(str).str.strip().str.lower()
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    inicio = None
    if meta.get("inicio_datetime"):
        try:
            inicio = datetime.fromisoformat(str(meta["inicio_datetime"]))
        except ValueError:
            inicio = None

    duracion = float(meta.get("duracion_s") or df["timestamp_s"].max())
    return Sesion(carpeta, df, meta, fondo, inicio, duracion)


def limpiar(sesion: Sesion, min_frames: int = 8, min_desplazamiento: float = 40.0) -> Sesion:
    """Descarta tracks cortos (IDs fragmentados, fantasmas) y tracks que no se mueven.

    Los tracks quietos se guardan aparte en `sesion.limpieza['quietos']`, por si se
    quieren analizar vehículos detenidos.
    """
    df = sesion.df
    g = df.groupby("track_id")
    n = g.size()
    primero = g[["px", "py"]].first()
    ultimo = g[["px", "py"]].last()
    # desplazamiento: diagonal de la región que recorre el track (robusto a idas y vueltas)
    extension = np.hypot(g["px"].max() - g["px"].min(), g["py"].max() - g["py"].min())
    neto = np.hypot(*(ultimo - primero).to_numpy().T)
    desplaz = pd.Series(np.maximum(extension.to_numpy(), neto), index=n.index)

    cortos = n.index[n < min_frames]
    quietos = n.index[(n >= min_frames) & (desplaz < min_desplazamiento)]
    validos = n.index[(n >= min_frames) & (desplaz >= min_desplazamiento)]

    sesion.limpieza = {
        "tracks_totales": int(len(n)),
        "descartados_cortos": int(len(cortos)),
        "descartados_quietos": int(len(quietos)),
        "tracks_validos": int(len(validos)),
        "detecciones_en_borde": int(sesion.meta.pop("_detecciones_en_borde", 0)),
        "min_frames": min_frames,
        "min_desplazamiento": min_desplazamiento,
        "quietos": df[df["track_id"].isin(quietos)].copy(),
    }
    sesion.df = df[df["track_id"].isin(validos)].reset_index(drop=True)
    return sesion


def resumen_por_vehiculo(df: pd.DataFrame, limite_kmh: float | None = None) -> pd.DataFrame:
    """Una fila por track_id: clase, tiempos, posición y velocidad.

    La clase es la más frecuente del track (YOLO a veces alterna auto/camión).
    La velocidad del vehículo es la mediana de sus mediciones; se considera que
    excedió el límite si la mayoría de sus detecciones lo marcan (o, si no existe
    `excede_limite`, si su mediana supera el límite).
    """
    g = df.groupby("track_id")
    res = pd.DataFrame({
        "clase": g["clase"].agg(lambda s: s.value_counts().idxmax()),
        "t_inicio": g["timestamp_s"].min(),
        "t_fin": g["timestamp_s"].max(),
        "n_detecciones": g.size(),
        "px_ini": g["px"].first(), "py_ini": g["py"].first(),
        "px_fin": g["px"].last(), "py_fin": g["py"].last(),
    })
    if "velocidad_kmh" in df.columns and df["velocidad_kmh"].notna().any():
        res["velocidad_kmh"] = g["velocidad_kmh"].median()
        if "excede_limite" in df.columns and df["excede_limite"].notna().any():
            res["excedio"] = g["excede_limite"].mean() >= 0.5
        elif limite_kmh:
            res["excedio"] = res["velocidad_kmh"] > limite_kmh
    return res


def nombre_clase(clase: str) -> str:
    return NOMBRE_CLASE.get(clase, clase.capitalize())
