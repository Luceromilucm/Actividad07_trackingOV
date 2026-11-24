"""Cálculos de la analítica: vehículos por intervalo, por carril, por clase y por línea virtual.

Todo se cuenta por vehículo (track_id único), nunca por detección.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from carriles import FUERA

INTERVALOS = {"hora": 3600, "minuto": 60, "10s": 10}
NOMBRE_INTERVALO = {"hora": "hora", "minuto": "minuto", "10s": "intervalo de 10 s"}


def elegir_intervalo(duracion_s: float, pedido: str = "auto") -> str:
    """'auto' elige según la duración: una sesión corta agrupada por hora daría una sola barra."""
    if pedido != "auto":
        return pedido
    if duracion_s >= 2 * 3600:
        return "hora"
    if duracion_s >= 3 * 60:
        return "minuto"
    return "10s"


def _etiqueta(inicio_bin: float, intervalo: str, inicio: datetime | None) -> str:
    paso = INTERVALOS[intervalo]
    if inicio is not None:
        a = inicio + timedelta(seconds=inicio_bin)
        b = a + timedelta(seconds=paso)
        if intervalo == "hora":
            return f"{a:%H:%M}–{b:%H:%M}"
        if intervalo == "minuto":
            return f"{a:%H:%M}"
        return f"{a:%H:%M:%S}"
    # sin hora real: tiempo relativo desde el inicio del video
    m, s = divmod(int(inicio_bin), 60)
    h, m = divmod(m, 60)
    if intervalo == "hora":
        return f"h {h}"
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def vehiculos_por_intervalo(vehiculos: pd.DataFrame, duracion_s: float, intervalo: str,
                            inicio: datetime | None = None) -> pd.DataFrame:
    """Vehículos únicos por intervalo, según el momento en que aparecen por primera vez.

    Si hay hora real, los intervalos se alinean al reloj (15:00, 16:00...). Los
    intervalos que la sesión no cubre por completo se marcan como parciales.
    """
    paso = INTERVALOS[intervalo]
    # desfase para alinear los intervalos al reloj
    desfase = 0.0
    if inicio is not None:
        seg_del_dia = inicio.hour * 3600 + inicio.minute * 60 + inicio.second + inicio.microsecond / 1e6
        desfase = seg_del_dia % paso
    t_abs = vehiculos["t_inicio"].to_numpy() + desfase
    fin_abs = duracion_s + desfase
    n_bins = max(1, math.ceil(fin_abs / paso - 1e-9))
    idx = np.clip((t_abs // paso).astype(int), 0, n_bins - 1)
    conteo = np.bincount(idx, minlength=n_bins)

    tolerancia = 1.0  # segundos
    filas = []
    for b in range(n_bins):
        a, z = b * paso, (b + 1) * paso
        parcial = (desfase > tolerancia and b == 0) or (fin_abs < z - tolerancia)
        filas.append({
            "intervalo": _etiqueta(a - desfase, intervalo, inicio),
            "inicio_s": a - desfase,
            "vehiculos": int(conteo[b]),
            "parcial": bool(parcial),
        })
    return pd.DataFrame(filas)


def fin_intervalo(tabla_intervalos: pd.DataFrame, indice: int, intervalo: str,
                  inicio: datetime | None = None) -> str:
    """Etiqueta del final del intervalo `indice` (para decir «de 15:37 a 15:38»)."""
    return _etiqueta(float(tabla_intervalos.loc[indice, "inicio_s"]) + INTERVALOS[intervalo], intervalo, inicio)


def clases_por_intervalo(vehiculos: pd.DataFrame, tabla_intervalos: pd.DataFrame, intervalo: str) -> pd.DataFrame:
    """Matriz intervalo × clase con el número de vehículos."""
    paso = INTERVALOS[intervalo]
    inicios = tabla_intervalos["inicio_s"].to_numpy()
    idx = np.clip(np.searchsorted(inicios, vehiculos["t_inicio"].to_numpy(), side="right") - 1, 0, len(inicios) - 1)
    tabla = pd.crosstab(tabla_intervalos["intervalo"].to_numpy()[idx], vehiculos["clase"].to_numpy())
    tabla = tabla.reindex(tabla_intervalos["intervalo"], fill_value=0)
    orden = tabla.sum().sort_values(ascending=False).index
    return tabla[orden]


def pico(tabla_intervalos: pd.DataFrame) -> dict:
    i = int(tabla_intervalos["vehiculos"].idxmax())
    return {"intervalo": tabla_intervalos.loc[i, "intervalo"],
            "vehiculos": int(tabla_intervalos.loc[i, "vehiculos"]), "indice": i}


def conteo_por_carril(asignacion: pd.Series, carriles: list[dict]) -> pd.DataFrame:
    nombres = [c["nombre"] for c in carriles]
    conteo = asignacion.value_counts()
    if FUERA in conteo.index:
        nombres = nombres + [FUERA]
    conteo = conteo.reindex(nombres, fill_value=0)
    total = int(conteo.sum())
    return pd.DataFrame({
        "carril": conteo.index,
        "vehiculos": conteo.to_numpy().astype(int),
        "porcentaje": (100 * conteo.to_numpy() / max(total, 1)).round(1),
    })


def carril_mas_usado(tabla_carriles: pd.DataFrame) -> dict:
    t = tabla_carriles[tabla_carriles["carril"] != FUERA]
    fila = t.loc[t["vehiculos"].idxmax()]
    return {"carril": fila["carril"], "vehiculos": int(fila["vehiculos"]), "porcentaje": float(fila["porcentaje"])}


def frase_carril(info: dict) -> str:
    return (f"El carril más utilizado fue {info['carril']} "
            f"({info['vehiculos']} vehículos, {info['porcentaje']:.0f}%)")


# --- Línea virtual ------------------------------------------------------------

def nombres_sentidos(linea) -> tuple[str, str]:
    """Nombre legible de cada sentido de cruce según la orientación de la línea."""
    (x1, y1), (x2, y2) = linea
    if abs(x2 - x1) >= abs(y2 - y1):          # línea más bien horizontal
        return "Hacia arriba (se alejan)", "Hacia abajo (se acercan)"
    return "Hacia la derecha", "Hacia la izquierda"


def cruces_linea(df: pd.DataFrame, linea, duracion_s: float) -> dict:
    """Cuenta cruces de la línea virtual por sentido.

    Para cada track se mira el lado de la línea entre frames consecutivos; un cambio
    de lado dentro del segmento es un cruce. Se cuenta como máximo un cruce por
    vehículo (el primero), para que el temblor de la caja sobre la línea no sume de más.
    """
    (x1, y1), (x2, y2) = np.asarray(linea, float)
    d = np.array([x2 - x1, y2 - y1])
    # normal que apunta hacia arriba de la imagen (o hacia la derecha si la línea es vertical)
    n = np.array([-d[1], d[0]])
    if abs(d[0]) >= abs(d[1]):
        n = n if n[1] < 0 else -n
    else:
        n = n if n[0] > 0 else -n
    nombre_pos, nombre_neg = nombres_sentidos(linea)

    eventos = []
    for tid, g in df.groupby("track_id"):
        p = g[["px", "py"]].to_numpy()
        lado = (p - [x1, y1]) @ n
        cambia = np.nonzero(np.sign(lado[:-1]) * np.sign(lado[1:]) < 0)[0]
        for i in cambia:
            # punto de cruce: ¿cae dentro del segmento?
            a, b = p[i], p[i + 1]
            t = lado[i] / (lado[i] - lado[i + 1])
            q = a + t * (b - a)
            s = ((q - [x1, y1]) @ d) / (d @ d)
            if 0 <= s <= 1:
                sentido = nombre_pos if lado[i + 1] > 0 else nombre_neg
                eventos.append({"track_id": tid, "t": float(g["timestamp_s"].iloc[i + 1]),
                                "sentido": sentido, "x": float(q[0]), "y": float(q[1])})
                break
    ev = pd.DataFrame(eventos, columns=["track_id", "t", "sentido", "x", "y"])
    minutos = max(duracion_s / 60, 1e-9)
    n_min = max(1, math.ceil(duracion_s / 60 - 1e-9))
    por_minuto = pd.DataFrame({
        s: np.bincount(np.clip((ev.loc[ev["sentido"] == s, "t"] // 60).astype(int), 0, n_min - 1),
                       minlength=n_min)
        for s in (nombre_pos, nombre_neg)
    })
    totales = {nombre_pos: int((ev["sentido"] == nombre_pos).sum()),
               nombre_neg: int((ev["sentido"] == nombre_neg).sum())}
    return {
        "linea": [[float(x1), float(y1)], [float(x2), float(y2)]],
        "totales": totales,
        "flujo_por_minuto": {k: round(v / minutos, 2) for k, v in totales.items()},
        "por_minuto": por_minuto,
        "eventos": ev,
    }


def resumen_velocidades(vehiculos: pd.DataFrame) -> dict | None:
    """Velocidad promedio por vehículo (media de las medianas de cada track)."""
    if "velocidad_kmh" not in vehiculos.columns:
        return None
    v = vehiculos["velocidad_kmh"].dropna()
    if v.empty:
        return None
    res = {"promedio_kmh": round(float(v.mean()), 1), "maxima_kmh": round(float(v.max()), 1)}
    if "excedio" in vehiculos.columns:
        res["excesos"] = int(vehiculos["excedio"].sum())
    return res
