"""Detección de anomalías: vehículos detenidos, en sentido contrario y trayectorias atípicas.

Todo se calcula por vehículo (track_id) a partir del CSV:

- **Detenido:** el punto de referencia se mueve menos de `umbral_px_s` píxeles por
  segundo durante al menos `min_detenido_s` segundos seguidos. También entran los
  tracks que la limpieza descartó por no moverse nunca (estacionados), pero solo si
  están en la zona por donde circula el tráfico: un objeto inmóvil fuera de la
  calzada es casi siempre un cartel confundido con un vehículo.
- **Sentido contrario:** el desplazamiento neto del vehículo forma más de 120° con el
  sentido de su carril (`sentido` de carriles.json o, si los carriles son automáticos,
  el sentido mayoritario de los vehículos de ese carril).
- **Trayectoria atípica:** cada trayectoria se remuestrea en puntos equiespaciados
  y se agrupa con DBSCAN; las que no caen en ningún grupo (ruido) son atípicas
  (cambios de carril bruscos, giros raros, errores del tracker).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from carriles import FUERA

TIPOS = {
    "detenido": "Detenido",
    "estacionado": "Estacionado",
    "contramano": "Sentido contrario",
    "atipica": "Trayectoria atípica",
}


def _velocidad_px_s(t: np.ndarray, p: np.ndarray, ventana_s: float = 1.0) -> np.ndarray:
    """Velocidad del punto de referencia (px/s) en una ventana centrada de ~1 s.

    Con ventana (y no frame a frame) el temblor de la caja no parece movimiento.
    """
    i0 = np.searchsorted(t, t - ventana_s / 2, side="left")
    i1 = np.clip(np.searchsorted(t, t + ventana_s / 2, side="right") - 1, 0, len(t) - 1)
    dt = t[i1] - t[i0]
    dist = np.linalg.norm(p[i1] - p[i0], axis=1)
    return np.where(dt > 1e-6, dist / np.maximum(dt, 1e-6), np.inf)


def fuera_de_la_via(df: pd.DataFrame, x: float, y: float, margen_px: float) -> bool:
    """¿El punto queda fuera de la zona de tráfico (envolvente convexa de las trayectorias)?"""
    pts = df[["px", "py"]].to_numpy(np.float32)
    if len(pts) < 3:
        return False
    envolvente = cv2.convexHull(pts)
    return cv2.pointPolygonTest(envolvente, (float(x), float(y)), True) < -margen_px


def detenidos(df: pd.DataFrame, quietos: pd.DataFrame | None, alto: int,
              min_detenido_s: float = 10.0, umbral_px_s: float | None = None,
              ancho: int | None = None) -> tuple[pd.DataFrame, int]:
    """Vehículos que estuvieron quietos al menos `min_detenido_s` segundos seguidos.

    Devuelve la tabla y cuántos objetos fijos se descartaron por estar fuera de la vía.
    """
    umbral = umbral_px_s if umbral_px_s is not None else 0.012 * alto   # ~1 % de la altura por segundo
    filas = []
    for tid, g in df.groupby("track_id"):
        t = g["timestamp_s"].to_numpy(float)
        p = g[["px", "py"]].to_numpy(float)
        if t[-1] - t[0] < min_detenido_s:
            continue
        quieto = _velocidad_px_s(t, p) < umbral
        # tramo quieto más largo
        mejor, ini = (0.0, 0, 0), None
        for i, q in enumerate(np.append(quieto, False)):
            if q and ini is None:
                ini = i
            elif not q and ini is not None:
                dur = t[i - 1] - t[ini]
                if dur > mejor[0]:
                    mejor = (dur, ini, i - 1)
                ini = None
        dur, a, b = mejor
        if dur >= min_detenido_s:
            filas.append({"track_id": int(tid), "tipo": "detenido", "t": float(t[a]), "t_fin": float(t[b]),
                          "x": float(np.median(p[a:b + 1, 0])), "y": float(np.median(p[a:b + 1, 1])),
                          "detalle": f"detenido {dur:.0f} s"})
    fuera = 0
    margen = 0.02 * max(alto, ancho or alto)
    if quietos is not None and len(quietos):
        for tid, g in quietos.groupby("track_id"):
            dur = float(g["timestamp_s"].max() - g["timestamp_s"].min())
            if dur < min_detenido_s:
                continue
            x, y = float(g["px"].median()), float(g["py"].median())
            if fuera_de_la_via(df, x, y, margen):
                fuera += 1
                continue
            filas.append({"track_id": int(tid), "tipo": "estacionado", "t": float(g["timestamp_s"].min()),
                          "t_fin": float(g["timestamp_s"].max()), "x": x, "y": y,
                          "detalle": f"no se movió en {dur:.0f} s"})
    return pd.DataFrame(filas, columns=["track_id", "tipo", "t", "t_fin", "x", "y", "detalle"]), fuera


def contramano(df: pd.DataFrame, asignacion: pd.Series, carriles_def: list[dict],
               min_recorrido_px: float = 60.0, angulo_min: float = 120.0) -> pd.DataFrame:
    """Vehículos que avanzan en contra del sentido de su carril."""
    sentido = {c["nombre"]: np.asarray(c["sentido"], float) for c in carriles_def if c.get("sentido")}
    cos_max = np.cos(np.radians(angulo_min))
    filas = []
    for tid, g in df.groupby("track_id"):
        carril = asignacion.get(tid)
        if carril in (None, FUERA) or carril not in sentido:
            continue
        p = g[["px", "py"]].to_numpy(float)
        d = p[-1] - p[0]
        norma = np.linalg.norm(d)
        s = sentido[carril]
        if norma < min_recorrido_px or np.linalg.norm(s) < 1e-9:
            continue
        cos = float(d @ s / (norma * np.linalg.norm(s)))
        if cos <= cos_max:
            filas.append({"track_id": int(tid), "tipo": "contramano", "t": float(g["timestamp_s"].iloc[0]),
                          "t_fin": float(g["timestamp_s"].iloc[-1]),
                          "x": float(np.median(p[:, 0])), "y": float(np.median(p[:, 1])),
                          "detalle": f"en {carril}, {np.degrees(np.arccos(np.clip(cos, -1, 1))):.0f}° del sentido"})
    return pd.DataFrame(filas, columns=["track_id", "tipo", "t", "t_fin", "x", "y", "detalle"])


def remuestrear(p: np.ndarray, n: int = 10) -> np.ndarray:
    """n puntos equiespaciados a lo largo de la trayectoria (por longitud de arco)."""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(p[:1], n, axis=0)
    u = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])])


def atipicas(df: pd.DataFrame, ancho: int, alto: int, n_puntos: int = 10, min_muestras: int = 5,
             factor_eps: float = 3.5, excluir: set | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Trayectorias que no se parecen a ninguna otra (ruido de DBSCAN).

    `eps` se elige solo: `factor_eps` veces la mediana de la distancia de cada
    trayectoria a su `min_muestras`-ésimo vecino, así el umbral se adapta a la
    resolución del video y a cuánto varían las trayectorias. Devuelve las
    atípicas y el grupo (patrón de movimiento) de cada track (-1 = atípica).
    """
    excluir = excluir or set()
    escala = float(max(ancho, alto))
    ids, X = [], []
    for tid, g in df.groupby("track_id"):
        ids.append(tid)
        X.append((remuestrear(g[["px", "py"]].to_numpy(float), n_puntos) / escala).ravel())
    vacio = pd.DataFrame(columns=["track_id", "tipo", "t", "t_fin", "x", "y", "detalle"])
    if len(X) <= min_muestras + 1:
        return vacio, pd.Series(0, index=ids, name="grupo")
    X = np.asarray(X) / np.sqrt(n_puntos)          # distancia = desvío medio por punto (en fracción de la imagen)
    d_k = NearestNeighbors(n_neighbors=min_muestras + 1).fit(X).kneighbors(X)[0][:, -1]
    tipico = max(float(np.median(d_k)), 1e-6)
    eps = max(factor_eps * tipico, 0.01)
    etiquetas = DBSCAN(eps=eps, min_samples=min_muestras).fit_predict(X)
    grupo = pd.Series(etiquetas, index=ids, name="grupo")

    filas = []
    for i in np.nonzero(etiquetas == -1)[0]:
        tid = ids[i]
        if tid in excluir:
            continue
        g = df[df["track_id"] == tid]
        filas.append({"track_id": int(tid), "tipo": "atipica", "t": float(g["timestamp_s"].iloc[0]),
                      "t_fin": float(g["timestamp_s"].iloc[-1]),
                      "x": float(g["px"].median()), "y": float(g["py"].median()),
                      "detalle": f"se aparta {d_k[i] / tipico:.0f}× más que una trayectoria típica"})
    return (pd.DataFrame(filas, columns=vacio.columns) if filas else vacio), grupo


def detectar(sesion, asignacion: pd.Series, carriles_def: list[dict],
             min_detenido_s: float = 10.0) -> dict:
    """Corre los tres detectores y devuelve la tabla unida y un resumen por tipo."""
    quietos = sesion.limpieza.get("quietos") if sesion.limpieza else None
    det, fijos_fuera = detenidos(sesion.df, quietos, sesion.alto, min_detenido_s, ancho=sesion.ancho)
    con = contramano(sesion.df, asignacion, carriles_def)
    # las ya explicadas (detenidos, contramano) no se repiten como atípicas
    ya = set(det["track_id"]) | set(con["track_id"])
    ati, grupos = atipicas(sesion.df, sesion.ancho, sesion.alto, excluir=ya)
    partes = [t for t in (det, con, ati) if len(t)]
    tabla = (pd.concat(partes, ignore_index=True) if partes
             else pd.DataFrame(columns=["track_id", "tipo", "t", "t_fin", "x", "y", "detalle"]))
    tabla = tabla.sort_values("t").reset_index(drop=True)
    conteo = {nombre: int((tabla["tipo"] == clave).sum()) for clave, nombre in TIPOS.items()}
    return {"tabla": tabla, "conteo": conteo, "grupos": grupos, "min_detenido_s": min_detenido_s,
            "fijos_fuera_de_la_via": fijos_fuera,
            "n_patrones": int(grupos[grupos >= 0].nunique())}
