"""Matriz origen-destino por zonas (norte, sur, este, oeste).

En la imagen: norte = arriba, sur = abajo, este = derecha, oeste = izquierda.

La «escena» es el rectángulo por donde realmente circulan los vehículos
(percentiles 1 y 99 de todas las posiciones). Desde el punto donde aparece cada
vehículo se lanza un rayo en sentido contrario a su rumbo de entrada: el borde de
la escena por el que sale ese rayo es el ORIGEN. Igual con el punto donde
desaparece y su rumbo de salida: ese borde es el DESTINO.

Ni la posición sola ni el rumbo solo bastan: con perspectiva, la calle es un
trapecio y un auto del carril izquierdo aparece pegado al borde oeste aunque
viene del sur; y en una calle en diagonal el rumbo queda a medio camino entre
dos zonas. El rayo combina las dos cosas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONAS = ["Norte", "Sur", "Este", "Oeste"]
DESCRIPCION = {"Norte": "arriba", "Sur": "abajo", "Este": "derecha", "Oeste": "izquierda"}
OPUESTA = {"Norte": "Sur", "Sur": "Norte", "Este": "Oeste", "Oeste": "Este"}


def rumbo(d: np.ndarray) -> np.ndarray:
    """Zona hacia la que apunta cada vector (dx, dy) en coordenadas de imagen (y hacia abajo)."""
    d = np.atleast_2d(np.asarray(d, float))
    vertical = np.abs(d[:, 1]) >= np.abs(d[:, 0])
    return np.where(vertical, np.where(d[:, 1] < 0, "Norte", "Sur"),
                    np.where(d[:, 0] > 0, "Este", "Oeste"))


def limites_escena(df: pd.DataFrame) -> tuple[float, float, float, float]:
    x0, x1 = np.percentile(df["px"], [1, 99])
    y0, y1 = np.percentile(df["py"], [1, 99])
    return float(x0), float(x1), float(y0), float(y1)


def borde_del_rayo(punto, direccion, limites) -> str:
    """Borde de la escena por el que sale el rayo punto + t·direccion (t > 0)."""
    x0, x1, y0, y1 = limites
    u = np.clip((punto[0] - x0) / max(x1 - x0, 1e-6), 0, 1)
    v = np.clip((punto[1] - y0) / max(y1 - y0, 1e-6), 0, 1)
    du = direccion[0] / max(x1 - x0, 1e-6)
    dv = direccion[1] / max(y1 - y0, 1e-6)
    if abs(du) < 1e-12 and abs(dv) < 1e-12:
        return "Norte"
    tiempos = {
        "Norte": v / -dv if dv < 0 else np.inf,
        "Sur": (1 - v) / dv if dv > 0 else np.inf,
        "Este": (1 - u) / du if du > 0 else np.inf,
        "Oeste": u / -du if du < 0 else np.inf,
    }
    return min(tiempos, key=tiempos.get)


def _tramo(p: np.ndarray, desde_inicio: bool, fraccion: float = 0.3) -> np.ndarray:
    """Desplazamiento del primer (o último) `fraccion` del recorrido, medido por longitud de arco."""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.zeros(2)
    if desde_inicio:
        j = int(np.searchsorted(s, fraccion * s[-1]))
        return p[min(max(j, 1), len(p) - 1)] - p[0]
    j = int(np.searchsorted(s, (1 - fraccion) * s[-1]))
    return p[-1] - p[min(j, len(p) - 2)]


def matriz(df: pd.DataFrame) -> dict:
    """Origen y destino de cada vehículo, y la matriz de conteos."""
    lim = limites_escena(df)
    filas = {}
    for tid, g in df.groupby("track_id"):
        p = g[["px", "py"]].to_numpy(float)
        entrada, salida = _tramo(p, True), _tramo(p, False)
        filas[tid] = {"origen": borde_del_rayo(p[0], -entrada, lim),
                      "destino": borde_del_rayo(p[-1], salida, lim),
                      "x_ini": p[0, 0], "y_ini": p[0, 1], "x_fin": p[-1, 0], "y_fin": p[-1, 1]}
    por_vehiculo = pd.DataFrame.from_dict(filas, orient="index")
    tabla = pd.crosstab(por_vehiculo["origen"], por_vehiculo["destino"]).reindex(
        index=ZONAS, columns=ZONAS, fill_value=0)
    # se quitan las zonas que no aparecen ni como origen ni como destino
    usadas = [z for z in ZONAS if tabla.loc[z].sum() or tabla[z].sum()]
    tabla = tabla.loc[usadas, usadas]
    pares = (por_vehiculo[["origen", "destino"]].value_counts().rename("vehiculos").reset_index()
             .sort_values("vehiculos", ascending=False, ignore_index=True))
    total = int(pares["vehiculos"].sum())
    pares["porcentaje"] = (100 * pares["vehiculos"] / max(total, 1)).round(1)
    principal = pares.iloc[0].to_dict() if len(pares) else None
    # dónde entran y salen, en promedio, los vehículos de cada par (para dibujar flechas)
    flechas = []
    for r in pares.itertuples():
        sel = por_vehiculo[(por_vehiculo["origen"] == r.origen) & (por_vehiculo["destino"] == r.destino)]
        flechas.append({"origen": r.origen, "destino": r.destino, "vehiculos": int(r.vehiculos),
                        "desde": [float(sel["x_ini"].median()), float(sel["y_ini"].median())],
                        "hasta": [float(sel["x_fin"].median()), float(sel["y_fin"].median())]})
    return {"por_vehiculo": por_vehiculo[["origen", "destino"]], "tabla": tabla, "pares": pares,
            "principal": principal, "flechas": flechas}


def frase(od: dict) -> str | None:
    p = od["principal"]
    if not p:
        return None
    return (f"El movimiento más frecuente fue de {p['origen'].lower()} ({DESCRIPCION[p['origen']]}) "
            f"a {p['destino'].lower()} ({DESCRIPCION[p['destino']]}): {int(p['vehiculos'])} vehículos, "
            f"el {p['porcentaje']:.0f}%")
