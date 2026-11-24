"""Imágenes para el dashboard: mapa de calor, carriles, trayectorias y línea virtual
dibujados sobre el frame de referencia con OpenCV (rápido, sin matplotlib).

Devuelven arrays RGB uint8 listos para `st.image`.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

import graficos as gr
from carriles import FUERA


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


COLORES_CARRIL = [_hex(c) for c in gr.COLORES_CARRIL]
TEXTO = _hex(gr.TEXTO)
FONDO = _hex(gr.FONDO)
CORAL = _hex(gr.CORAL)
TURQUESA = _hex(gr.TURQUESA)
FUCSIA = _hex(gr.FUCSIA)


def _base(sesion, oscurecer: float) -> np.ndarray:
    return (np.clip(gr._fondo(sesion, oscurecer), 0, 1) * 255).astype(np.uint8)


def _rotulo(img, texto: str, org, escala: float = 0.7, color=TEXTO, grosor: int = 2, centrado: bool = True):
    (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    x, y = int(org[0] - (w / 2 if centrado else 0)), int(org[1])
    x = int(np.clip(x, 4, img.shape[1] - w - 4))
    y = int(np.clip(y, h + 4, img.shape[0] - 4))
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, FONDO, grosor + 4, cv2.LINE_AA)
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def _escala_texto(img) -> float:
    return max(0.5, img.shape[1] / 1600)


def mapa_calor(sesion, df: pd.DataFrame, carriles_def: list[dict], tabla_carr: pd.DataFrame,
               mas_usado: str | None, por_deteccion: bool) -> np.ndarray:
    calor = gr.calcular_mapa_calor(df, sesion.ancho, sesion.alto, por_deteccion)
    img = (np.clip(gr._superponer(gr._fondo(sesion, 0.8), calor), 0, 1) * 255).astype(np.uint8)
    pct = dict(zip(tabla_carr["carril"], tabla_carr["porcentaje"]))
    s = _escala_texto(img)
    for c in carriles_def:
        p = np.asarray(c["poligono"])
        base = p[np.argsort(p[:, 1])[-2:]].mean(axis=0)
        destacado = c["nombre"] == mas_usado
        _rotulo(img, f"{c['nombre']}: {pct.get(c['nombre'], 0):.0f}%", (base[0], base[1] - 0.04 * img.shape[0]),
                s * (1.15 if destacado else 0.9), TURQUESA if destacado else TEXTO)
    return img


def carriles(sesion, carriles_def: list[dict], tabla_carr: pd.DataFrame, mas_usado: str | None) -> np.ndarray:
    img = _base(sesion, 0.55)
    capa = img.copy()
    cuenta = dict(zip(tabla_carr["carril"], tabla_carr["vehiculos"]))
    pct = dict(zip(tabla_carr["carril"], tabla_carr["porcentaje"]))
    for i, c in enumerate(carriles_def):
        p = np.round(np.asarray(c["poligono"])).astype(np.int32)
        cv2.fillPoly(capa, [p], COLORES_CARRIL[i % len(COLORES_CARRIL)])
    img = cv2.addWeighted(capa, 0.38, img, 0.62, 0)
    s = _escala_texto(img)
    for i, c in enumerate(carriles_def):
        p = np.round(np.asarray(c["poligono"])).astype(np.int32)
        destacado = c["nombre"] == mas_usado
        cv2.polylines(img, [p], True, COLORES_CARRIL[i % len(COLORES_CARRIL)], 4 if destacado else 2, cv2.LINE_AA)
        centro = p.mean(axis=0)
        _rotulo(img, c["nombre"], (centro[0], centro[1] - 12 * s), s * 0.95)
        _rotulo(img, f"{cuenta.get(c['nombre'], 0)} ({pct.get(c['nombre'], 0):.0f}%)",
                (centro[0], centro[1] + 26 * s), s * 0.85, TURQUESA if destacado else TEXTO)
    return img


def color_velocidad(v: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Verde (lento) → ámbar → coral (rápido), igual que el gráfico estático."""
    t = np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    return (np.asarray(gr.CMAP_VELOCIDAD(t))[..., :3] * 255).astype(np.uint8)


def trayectorias(sesion, df: pd.DataFrame, vehiculos: pd.DataFrame, limite: float | None) -> np.ndarray:
    img = _base(sesion, 0.45)
    hay_vel = "velocidad_kmh" in vehiculos.columns and vehiculos["velocidad_kmh"].notna().any()
    if hay_vel:
        v = vehiculos["velocidad_kmh"]
        vmin, vmax = float(np.nanpercentile(v, 2)), float(np.nanpercentile(v, 98))
        colores = dict(zip(vehiculos.index, color_velocidad(v.to_numpy(), vmin, vmax)))
    capa = img.copy()
    for tid, g in df.groupby("track_id"):
        pts = np.round(g[["px", "py"]].to_numpy()).astype(np.int32)
        if hay_vel and tid in colores:
            col = tuple(int(x) for x in colores[tid])
        else:
            col = COLORES_CARRIL[int(tid) % len(COLORES_CARRIL)]
        exceso = bool(vehiculos["excedio"].get(tid, False)) if "excedio" in vehiculos.columns else False
        cv2.polylines(capa, [pts], False, col, 3 if exceso else 2, cv2.LINE_AA)
        if exceso:
            cv2.circle(capa, tuple(pts[-1]), 5, CORAL, -1, cv2.LINE_AA)
    return cv2.addWeighted(capa, 0.85, img, 0.15, 0)


def linea_virtual(sesion, df: pd.DataFrame, cruces: dict) -> np.ndarray:
    img = _base(sesion, 0.55)
    for _, g in df.groupby("track_id"):
        pts = np.round(g[["px", "py"]].to_numpy()).astype(np.int32)
        cv2.polylines(img, [pts], False, (80, 100, 125), 1, cv2.LINE_AA)
    (x1, y1), (x2, y2) = np.round(cruces["linea"]).astype(int)
    cv2.line(img, (x1, y1), (x2, y2), TEXTO, 4, cv2.LINE_AA)
    sentidos = list(cruces["totales"])
    colores = {sentidos[0]: TURQUESA, sentidos[1]: FUCSIA}
    for e in cruces["eventos"].itertuples():
        cv2.circle(img, (int(e.x), int(e.y)), 6, colores[e.sentido], -1, cv2.LINE_AA)
    s = _escala_texto(img)
    y = min(y1, y2) - 18 * s
    for i, (k, n) in enumerate(cruces["totales"].items()):
        _rotulo(img, f"{k}: {n}", ((x1 + x2) / 2, y - i * 34 * s), s * 0.9, colores[k])
    return img


COLOR_ANOMALIA = {k: _hex(c) for k, c in gr.COLOR_ANOMALIA.items()}


def anomalias(sesion, df: pd.DataFrame, tabla: pd.DataFrame) -> np.ndarray:
    """Trayectorias normales en gris tenue y las anómalas resaltadas con su número."""
    img = _base(sesion, 0.45)
    quietos = sesion.limpieza.get("quietos") if sesion.limpieza else None
    marcados = set(tabla["track_id"])
    for tid, g in df.groupby("track_id"):
        if tid not in marcados:
            pts = np.round(g[["px", "py"]].to_numpy()).astype(np.int32)
            cv2.polylines(img, [pts], False, (70, 88, 110), 1, cv2.LINE_AA)
    s = _escala_texto(img)
    for r in tabla.itertuples():
        fuente = quietos if r.tipo == "estacionado" and quietos is not None else df
        pts = np.round(fuente.loc[fuente["track_id"] == r.track_id, ["px", "py"]].to_numpy()).astype(np.int32)
        col = COLOR_ANOMALIA[r.tipo]
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, col, 4, cv2.LINE_AA)
            if r.tipo in ("contramano", "atipica"):
                cv2.arrowedLine(img, tuple(pts[max(len(pts) - 6, 0)]), tuple(pts[-1]), col, 4, cv2.LINE_AA,
                                tipLength=0.6)
        centro = (int(r.x), int(r.y))
        cv2.circle(img, centro, int(16 * s), col, 3, cv2.LINE_AA)
        _rotulo(img, f"#{r.track_id}", (r.x, r.y - 24 * s), s * 0.85, col)
    return img


def vehiculos_filtrados(df: pd.DataFrame, vehiculos: pd.DataFrame, clases: list[str], carriles_sel: list[str],
                        t_rango: tuple[float, float]) -> pd.Index:
    """track_id que pasan los filtros del dashboard."""
    v = vehiculos
    m = v["clase"].isin(clases) & v["t_inicio"].between(*t_rango)
    if carriles_sel is not None:
        m &= v["carril"].isin(carriles_sel)
    return v.index[m]


__all__ = ["mapa_calor", "carriles", "trayectorias", "linea_virtual", "anomalias", "vehiculos_filtrados", "FUERA"]
