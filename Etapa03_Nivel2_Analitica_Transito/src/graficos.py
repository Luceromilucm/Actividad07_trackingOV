"""Figuras de la analítica: barras por intervalo, mapa de calor, carriles, clases,
trayectorias y línea virtual. Todo el texto en español, con un estilo oscuro común.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Polygon
from scipy.ndimage import gaussian_filter

from carriles import FUERA
from datos import nombre_clase

# --- Paleta -----------------------------------------------------------------
FONDO = "#0E1621"
PANEL = "#131F2D"
BORDE = "#2C4560"
TEXTO = "#F2F5F8"
GRIS = "#A3B3C4"
TURQUESA = "#2DE2CF"
FUCSIA = "#E879F9"
CORAL = "#FF5D73"
AMBAR = "#FFB627"
AZUL = "#5AA9FF"
VERDE = "#3DDC84"

COLORES_CARRIL = [TURQUESA, FUCSIA, AMBAR, AZUL, VERDE, "#C3A6FF"]
COLORES_CLASE = {"car": TURQUESA, "motorcycle": FUCSIA, "truck": AMBAR, "bus": AZUL}
CMAP_VELOCIDAD = LinearSegmentedColormap.from_list("velocidad", [VERDE, AMBAR, CORAL])

plt.rcParams.update({
    "font.family": ["Segoe UI", "Calibri", "DejaVu Sans"],
    "figure.facecolor": FONDO, "axes.facecolor": PANEL, "savefig.facecolor": FONDO,
    "axes.edgecolor": BORDE, "axes.labelcolor": GRIS, "axes.titlecolor": TEXTO,
    "xtick.color": GRIS, "ytick.color": GRIS, "text.color": TEXTO,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": BORDE, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.labelcolor": TEXTO,
})

CONTORNO = [patheffects.withStroke(linewidth=3, foreground=FONDO)]


def _titulo(fig, titulo: str, subtitulo: str | None = None):
    fig.text(0.015, 0.975, titulo, fontsize=17, fontweight="bold", color=TEXTO, va="top")
    if subtitulo:
        fig.text(0.015, 0.915, subtitulo, fontsize=11.5, color=GRIS, va="top", style="italic")


def _guardar(fig, ruta: Path) -> Path:
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta


def _fondo(sesion, oscurecer: float = 1.0) -> np.ndarray:
    if sesion.fondo is not None:
        img = sesion.fondo.astype(np.float32) / 255.0
    else:
        img = np.full((sesion.alto, sesion.ancho, 3), 0.12, np.float32)
    return img * oscurecer


def _imagen(ax, img):
    ax.imshow(img, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --- 1. Barras por intervalo ----------------------------------------------------

def barras_intervalo(tabla: pd.DataFrame, intervalo: str, pico: dict, total: int, ruta: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.17)
    x = np.arange(len(tabla))
    colores = [CORAL if i == pico["indice"] else TURQUESA for i in range(len(tabla))]
    barras = ax.bar(x, tabla["vehiculos"], color=colores, width=0.72, zorder=3)
    for b, parcial in zip(barras, tabla["parcial"]):
        if parcial:
            b.set_alpha(0.45)
            b.set_hatch("///")
            b.set_edgecolor(FONDO)

    ymax = tabla["vehiculos"].max()
    ax.set_ylim(0, ymax * 1.18 + 1)
    muchas = len(tabla) > 24
    for i, (b, v) in enumerate(zip(barras, tabla["vehiculos"])):
        es_pico = i == pico["indice"]
        if muchas and not es_pico:
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, f"{v}",
                ha="center", va="bottom", fontsize=12 if es_pico else 10,
                fontweight="bold" if es_pico else "normal", color=CORAL if es_pico else TEXTO)

    paso = max(1, int(np.ceil(len(tabla) / 24)))
    ax.set_xticks(x[::paso])
    ax.set_xticklabels(tabla["intervalo"].iloc[::paso], rotation=0 if len(tabla) <= 12 else 45,
                       ha="center" if len(tabla) <= 12 else "right")
    unidad = {"hora": "Hora del día", "minuto": "Minuto", "10s": "Intervalo de 10 s"}[intervalo]
    ax.set_xlabel(unidad)
    ax.set_ylabel("Vehículos (únicos)")
    ax.grid(axis="y", zorder=0)
    if len(tabla) < 6:   # pocas barras: que no ocupen todo el ancho
        extra = (6 - len(tabla)) / 2
        ax.set_xlim(-0.5 - extra, len(tabla) - 0.5 + extra)
    if intervalo == "hora" and tabla["parcial"].all():
        fig.text(0.5, 0.012, "Nota: la grabación no cubre ninguna hora completa. Con una sesión más larga "
                             "(por ejemplo, la webcam encendida varias horas) aparece una barra por hora.",
                 ha="center", va="bottom", fontsize=10.5, color=AMBAR, style="italic")

    leyenda = [Patch(color=TURQUESA, label="Vehículos"), Patch(color=CORAL, label="Intervalo pico")]
    if tabla["parcial"].any():
        leyenda.append(Patch(facecolor=TURQUESA, alpha=0.45, hatch="///", edgecolor=FONDO,
                             label="Intervalo parcial (la sesión no lo cubre entero)"))
    ax.legend(handles=leyenda, loc="upper right", fontsize=10, ncols=len(leyenda))

    por = {"hora": "hora", "minuto": "minuto", "10s": "intervalo de 10 segundos"}[intervalo]
    _titulo(fig, "¿A qué hora pasaron más vehículos?" if intervalo == "hora" else "¿Cuándo pasaron más vehículos?",
            f"Vehículos por {por} · pico: {pico['intervalo']} con {pico['vehiculos']} vehículos · total: {total}")
    return _guardar(fig, ruta)


# --- 2. Mapa de calor ------------------------------------------------------------

def calcular_mapa_calor(df: pd.DataFrame, ancho: int, alto: int, por_deteccion: bool = False,
                        reduccion: int = 2, sigma: float = 5.0) -> np.ndarray:
    """Grilla de presencia normalizada a [0, 1], del tamaño de la imagen.

    Por defecto cada vehículo suma una sola vez por celda: se dibuja su trayectoria
    en una máscara binaria y se suma la máscara. Así un auto detenido no domina el mapa.
    Con `por_deteccion` se suma cada detección (equivale a tiempo de permanencia).
    """
    w, h = int(np.ceil(ancho / reduccion)), int(np.ceil(alto / reduccion))
    acum = np.zeros((h, w), np.float32)
    if por_deteccion:
        xs = np.clip((df["px"] / reduccion).astype(int), 0, w - 1)
        ys = np.clip((df["py"] / reduccion).astype(int), 0, h - 1)
        np.add.at(acum, (ys, xs), 1.0)
    else:
        mascara = np.zeros((h, w), np.uint8)
        for _, g in df.groupby("track_id"):
            mascara[:] = 0
            pts = np.round(g[["px", "py"]].to_numpy() / reduccion).astype(np.int32)
            cv2.polylines(mascara, [pts], False, 1, thickness=1)
            acum += mascara
    acum = gaussian_filter(acum, sigma)
    # Se normaliza con el percentil 99,5 de las celdas con tráfico (no con el máximo):
    # por la perspectiva, las trayectorias se juntan a lo lejos y ese punto saturaría
    # la escala. La gamma 0,6 deja ver también los carriles con poco tráfico.
    if acum.max() > 0:
        tope = np.percentile(acum[acum > acum.max() * 0.01], 99.5)
        acum = np.clip(acum / tope, 0, 1) ** 0.6
    return cv2.resize(acum, (ancho, alto), interpolation=cv2.INTER_LINEAR)


def _superponer(fondo: np.ndarray, calor: np.ndarray, umbral: float = 0.05, opacidad: float = 0.62):
    rgba = plt.get_cmap("inferno")(np.clip(calor, 0, 1))
    alpha = np.clip((calor - umbral) / 0.2, 0, 1) * opacidad
    return fondo * (1 - alpha[..., None]) + rgba[..., :3] * alpha[..., None]


def mapa_calor(sesion, calor: np.ndarray, carriles: list[dict], tabla_carriles: pd.DataFrame,
               mas_usado: dict, por_deteccion: bool, ruta: Path) -> Path:
    alto, ancho = calor.shape
    fig = plt.figure(figsize=(12.5, 12.5 * alto / ancho * 0.84 + 1.3))
    ax = fig.add_axes([0.015, 0.02, 0.88, 0.84])
    cax = fig.add_axes([0.915, 0.12, 0.018, 0.62])
    _imagen(ax, np.clip(_superponer(_fondo(sesion, 0.8), calor), 0, 1))

    # rótulo de cada carril en su borde inferior
    pct = dict(zip(tabla_carriles["carril"], tabla_carriles["porcentaje"]))
    for c in carriles:
        p = np.asarray(c["poligono"])
        base = p[np.argsort(p[:, 1])[-2:]].mean(axis=0)
        destacado = c["nombre"] == mas_usado["carril"]
        rotulo = f"{c['nombre']}\n{pct.get(c['nombre'], 0):.0f}%" + ("  · más usado" if destacado else "")
        ax.text(base[0], min(base[1], alto - 4) - 0.03 * alto, rotulo,
                ha="center", va="bottom", fontsize=13 if destacado else 10.5, linespacing=1.1,
                fontweight="bold", color=TEXTO, path_effects=CONTORNO)

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["baja", "media", "máxima"])
    cb.outline.set_edgecolor(BORDE)
    cb.set_label("Presencia de vehículos", color=GRIS)
    modo = "cada detección suma (tiempo de permanencia)" if por_deteccion else "cada vehículo cuenta una vez por zona"
    _titulo(fig, "Mapa de calor: ¿por dónde pasan los vehículos?",
            f"{len(sesion.df['track_id'].unique())} vehículos · {modo} · "
            f"{mas_usado['carril']} concentra el {mas_usado['porcentaje']:.0f}% del tráfico")
    return _guardar(fig, ruta)


# --- 3. Carriles -----------------------------------------------------------------

def carriles(sesion, carriles_def: list[dict], tabla: pd.DataFrame, mas_usado: dict,
             origen: str, frase: str, ruta: Path) -> Path:
    fig = plt.figure(figsize=(14, 6.4))
    ax_img = fig.add_axes([0.015, 0.04, 0.52, 0.78])
    ax_bar = fig.add_axes([0.62, 0.13, 0.35, 0.66])
    _imagen(ax_img, _fondo(sesion, 0.7))
    ax_img.set_xlim(0, sesion.ancho)
    ax_img.set_ylim(sesion.alto, 0)

    color = {c["nombre"]: COLORES_CARRIL[i % len(COLORES_CARRIL)] for i, c in enumerate(carriles_def)}
    color[FUERA] = GRIS
    for c in carriles_def:
        p = np.asarray(c["poligono"])
        destacado = c["nombre"] == mas_usado["carril"]
        ax_img.add_patch(Polygon(p, closed=True, facecolor=color[c["nombre"]], alpha=0.38 if destacado else 0.22,
                                 edgecolor=color[c["nombre"]], linewidth=2.5 if destacado else 1.4))
        base = p[np.argsort(p[:, 1])[-2:]].mean(axis=0)
        ax_img.text(base[0], min(base[1], sesion.alto) - 0.05 * sesion.alto, c["nombre"], ha="center",
                    fontsize=11, fontweight="bold", color=TEXTO, path_effects=CONTORNO)
        if c.get("sentido"):
            # flecha del sentido de circulación en el centro del polígono
            centro = p.mean(axis=0)
            v = np.asarray(c["sentido"], float)
            v = v / (np.linalg.norm(v) + 1e-9) * 0.09 * sesion.alto
            ax_img.annotate("", xy=centro + v, xytext=centro - v,
                            arrowprops=dict(arrowstyle="-|>", color=TEXTO, lw=2.2, mutation_scale=18))
    nota = "dibujados a mano (carriles.json)" if origen == "dibujados" else "detectados automáticamente"
    ax_img.set_title(f"Carriles {nota}", fontsize=11.5, color=GRIS, loc="left", fontweight="normal")

    t = tabla.iloc[::-1]
    y = np.arange(len(t))
    ax_bar.barh(y, t["vehiculos"], color=[color.get(n, GRIS) for n in t["carril"]], height=0.62, zorder=3)
    for yi, (n, v, pc) in enumerate(zip(t["carril"], t["vehiculos"], t["porcentaje"])):
        destacado = n == mas_usado["carril"]
        ax_bar.text(v + t["vehiculos"].max() * 0.02, yi, f"{v}  ({pc:.0f}%)", va="center",
                    fontsize=12 if destacado else 10.5, fontweight="bold" if destacado else "normal",
                    color=TEXTO)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(t["carril"], fontsize=11)
    for lab in ax_bar.get_yticklabels():
        if lab.get_text() == mas_usado["carril"]:
            lab.set_fontweight("bold")
            lab.set_color(TEXTO)
    ax_bar.set_xlim(0, t["vehiculos"].max() * 1.35)
    ax_bar.set_xlabel("Vehículos")
    ax_bar.grid(axis="x", zorder=0)
    ax_bar.set_title("Vehículos por carril", fontsize=12.5, loc="left")

    _titulo(fig, frase, f"{int(tabla['vehiculos'].sum())} vehículos asignados al carril donde pasaron "
                        f"la mayor parte de su recorrido")
    return _guardar(fig, ruta)


# --- 4. Clases por intervalo -------------------------------------------------------

def clases_intervalo(tabla: pd.DataFrame, intervalo: str, ruta: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.17)
    x = np.arange(len(tabla))
    abajo = np.zeros(len(tabla))
    extra = iter(["#C3A6FF", VERDE, "#FF9F6B"])
    for clase in tabla.columns:
        v = tabla[clase].to_numpy()
        ax.bar(x, v, bottom=abajo, width=0.72, zorder=3,
               color=COLORES_CLASE.get(clase) or next(extra, GRIS),
               label=f"{nombre_clase(clase)} ({int(v.sum())})", edgecolor=FONDO, linewidth=0.6)
        abajo += v
    paso = max(1, int(np.ceil(len(tabla) / 24)))
    ax.set_xticks(x[::paso])
    ax.set_xticklabels(tabla.index[::paso], rotation=0 if len(tabla) <= 12 else 45,
                       ha="center" if len(tabla) <= 12 else "right")
    ax.set_ylim(0, abajo.max() * 1.15 + 1)
    ax.set_xlabel({"hora": "Hora del día", "minuto": "Minuto", "10s": "Intervalo de 10 s"}[intervalo])
    ax.set_ylabel("Vehículos (únicos)")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right", ncols=len(tabla.columns), fontsize=10)
    total = tabla.to_numpy().sum()
    principal = tabla.sum().idxmax()
    _titulo(fig, "¿Qué tipo de vehículos pasaron?",
            f"Desglose por clase · {nombre_clase(principal).lower()}s: "
            f"{100 * tabla[principal].sum() / max(total, 1):.0f}% del total")
    return _guardar(fig, ruta)


# --- 5. Trayectorias ------------------------------------------------------------------

def trayectorias(sesion, vehiculos: pd.DataFrame, asignacion: pd.Series, carriles_def: list[dict],
                 limite: float | None, ruta: Path) -> Path:
    df = sesion.df
    fig = plt.figure(figsize=(12.5, 12.5 * sesion.alto / sesion.ancho * 0.84 + 1.3))
    ax = fig.add_axes([0.015, 0.03, 0.88, 0.80])
    cax = fig.add_axes([0.915, 0.12, 0.018, 0.6])
    _imagen(ax, _fondo(sesion, 0.45))

    grupos = {t: g[["px", "py"]].to_numpy() for t, g in df.groupby("track_id")}
    ids = list(grupos)
    hay_vel = "velocidad_kmh" in vehiculos.columns and vehiculos["velocidad_kmh"].notna().any()
    if hay_vel:
        # los más rápidos se dibujan al final, encima del resto
        ids = list(vehiculos.loc[ids, "velocidad_kmh"].fillna(0).sort_values().index)
    segmentos = [grupos[t] for t in ids]
    if hay_vel:
        v = vehiculos.loc[ids, "velocidad_kmh"].to_numpy()
        vmin, vmax = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        if limite:
            vmax = max(vmax, limite * 1.1)
        norm = Normalize(vmin, vmax)
        lc = LineCollection(segmentos, cmap=CMAP_VELOCIDAD, norm=norm, linewidths=1.3, alpha=0.75)
        lc.set_array(v)
        ax.add_collection(lc)
        cb = fig.colorbar(lc, cax=cax)
        cb.set_label("Velocidad del vehículo (km/h)", color=GRIS)
        cb.outline.set_edgecolor(BORDE)
        cb.solids.set_alpha(1)
        if limite and vmin < limite < vmax:
            cax.axhline(limite, color=TEXTO, lw=2)
            cax.text(-0.6, limite, f"límite {limite:g}", transform=cax.get_yaxis_transform(),
                     ha="right", va="center", fontsize=10, color=TEXTO, fontweight="bold")
        rapidos = vehiculos.loc[ids, "excedio"].sum() if "excedio" in vehiculos.columns else None
        sub = (f"{len(ids)} vehículos · velocidad media {np.nanmean(v):.0f} km/h"
               + (f" · {int(rapidos)} superaron el límite" if rapidos is not None else ""))
        titulo = "Trayectorias coloreadas por velocidad"
    else:
        color = {c["nombre"]: COLORES_CARRIL[i % len(COLORES_CARRIL)] for i, c in enumerate(carriles_def)}
        cols = [color.get(asignacion.get(t), GRIS) for t in ids]
        ax.add_collection(LineCollection(segmentos, colors=cols, linewidths=1.2, alpha=0.7))
        cax.set_visible(False)
        sub = f"{len(ids)} vehículos · sin datos de velocidad: color según el carril"
        titulo = "Trayectorias de los vehículos"
    ax.set_xlim(0, sesion.ancho)
    ax.set_ylim(sesion.alto, 0)
    _titulo(fig, titulo, sub)
    return _guardar(fig, ruta)


# --- 6. Línea virtual ------------------------------------------------------------------

def linea_virtual(sesion, cruces: dict, ruta: Path) -> Path:
    fig = plt.figure(figsize=(14, 6.2))
    ax_img = fig.add_axes([0.015, 0.04, 0.5, 0.78])
    ax = fig.add_axes([0.6, 0.15, 0.38, 0.62])
    _imagen(ax_img, _fondo(sesion, 0.6))
    ax_img.set_xlim(0, sesion.ancho)
    ax_img.set_ylim(sesion.alto, 0)

    (x1, y1), (x2, y2) = cruces["linea"]
    ax_img.plot([x1, x2], [y1, y2], color=AMBAR, lw=3.5, solid_capstyle="round")
    nombres = list(cruces["totales"])
    colores = [TURQUESA, FUCSIA]
    d = np.array([x2 - x1, y2 - y1])
    n = np.array([-d[1], d[0]])
    if abs(d[0]) >= abs(d[1]):
        n = n if n[1] < 0 else -n
    else:
        n = n if n[0] > 0 else -n
    n = n / np.linalg.norm(n)
    largo = 0.11 * sesion.alto
    ev = cruces["eventos"]
    for k, (nombre, col) in enumerate(zip(nombres, colores)):
        signo = 1 if k == 0 else -1
        sel = ev[ev["sentido"] == nombre]
        if len(sel):   # la flecha va donde cruzan, en promedio, los vehículos de ese sentido
            base = sel[["x", "y"]].median().to_numpy()
        else:
            base = np.array([x1, y1]) + d * (0.3 if k == 0 else 0.7)
        ax_img.annotate("", xy=base + signo * n * largo, xytext=base - signo * n * largo,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=3, mutation_scale=22))
        ax_img.text(*(base + signo * n * largo * 1.35), f"{cruces['totales'][nombre]}",
                    ha="center", va="center", fontsize=17, fontweight="bold", color=col, path_effects=CONTORNO)
    ax_img.set_title("Línea de conteo", fontsize=11.5, color=GRIS, loc="left", fontweight="normal")

    pm = cruces["por_minuto"]
    m = np.arange(len(pm))
    for nombre, col in zip(nombres, colores):
        ax.plot(m, pm[nombre], color=col, lw=2.4, marker="o", ms=4.5,
                label=f"{nombre}: {cruces['flujo_por_minuto'][nombre]:.1f} veh/min")
    paso = max(1, int(np.ceil(len(m) / 10)))
    ax.set_xticks(m[::paso])
    if sesion.inicio is not None:
        ax.set_xticklabels([f"{sesion.inicio + timedelta(minutes=int(i)):%H:%M}" for i in m[::paso]])
        ax.set_xlabel("Minuto")
    else:
        ax.set_xlabel("Minuto desde el inicio")
    ax.set_ylabel("Cruces por minuto")
    ax.set_ylim(0, max(pm.to_numpy().max() * 1.4, 1))
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Flujo a lo largo de la sesión", fontsize=12.5, loc="left")
    total = sum(cruces["totales"].values())
    _titulo(fig, f"{total} vehículos cruzaron la línea de conteo",
            " · ".join(f"{k}: {v}" for k, v in cruces["totales"].items()))
    return _guardar(fig, ruta)


# --- 7. Origen-destino ----------------------------------------------------------------

def origen_destino(sesion, od: dict, ruta: Path) -> Path:
    """Flechas de los movimientos sobre la calle + matriz origen × destino."""
    fig = plt.figure(figsize=(14, 6.4))
    ax_img = fig.add_axes([0.015, 0.04, 0.52, 0.78])
    ax_m = fig.add_axes([0.63, 0.14, 0.3, 0.62])
    _imagen(ax_img, _fondo(sesion, 0.55))
    ax_img.set_xlim(0, sesion.ancho)
    ax_img.set_ylim(sesion.alto, 0)

    flechas = od["flechas"]
    tope = max((f["vehiculos"] for f in flechas), default=1)
    total = max(sum(f["vehiculos"] for f in flechas), 1)
    colores = [TURQUESA, FUCSIA, AMBAR, AZUL, VERDE, "#C3A6FF"]
    for i, f in enumerate(flechas[:6]):
        a, b = np.asarray(f["desde"]), np.asarray(f["hasta"])
        col = colores[i % len(colores)]
        ancho = 1.5 + 7 * f["vehiculos"] / tope
        ax_img.annotate("", xy=b, xytext=a, arrowprops=dict(
            arrowstyle="-|>", color=col, lw=ancho, mutation_scale=14 + 2 * ancho, alpha=0.9,
            connectionstyle="arc3,rad=0.12"))
        m = (a + b) / 2
        ax_img.text(*m, f"{100 * f['vehiculos'] / total:.0f}%", ha="center", va="center", fontsize=13,
                    fontweight="bold", color=col, path_effects=CONTORNO)
    ax_img.set_title("Movimientos más frecuentes (grosor = vehículos)", fontsize=11.5, color=GRIS,
                     loc="left", fontweight="normal")

    t = od["tabla"]
    valores = t.to_numpy()
    ax_m.imshow(valores, cmap=LinearSegmentedColormap.from_list("od", [PANEL, TURQUESA]), aspect="auto")
    for (i, j), v in np.ndenumerate(valores):
        ax_m.text(j, i, f"{v}", ha="center", va="center", fontsize=13, fontweight="bold",
                  color=FONDO if v > valores.max() * 0.55 else TEXTO)
    ax_m.set_xticks(range(len(t.columns)))
    ax_m.set_xticklabels(t.columns)
    ax_m.set_yticks(range(len(t.index)))
    ax_m.set_yticklabels(t.index)
    ax_m.set_xlabel("Destino (hacia dónde van)")
    ax_m.set_ylabel("Origen (de dónde vienen)")
    ax_m.tick_params(length=0)
    for s in ax_m.spines.values():
        s.set_visible(False)
    ax_m.set_title("Matriz origen-destino", fontsize=12.5, loc="left")
    fig.text(0.63, 0.04, "Norte = arriba de la imagen · Sur = abajo · Este = derecha · Oeste = izquierda",
             fontsize=9.5, color=GRIS)

    p = od["principal"]
    _titulo(fig, "¿De dónde vienen y hacia dónde van?",
            f"Movimiento principal: de {p['origen'].lower()} a {p['destino'].lower()} "
            f"({int(p['vehiculos'])} vehículos, {p['porcentaje']:.0f}%)" if p else None)
    return _guardar(fig, ruta)


# --- 8. Anomalías ------------------------------------------------------------------------

COLOR_ANOMALIA = {"detenido": AMBAR, "estacionado": GRIS, "contramano": CORAL, "atipica": FUCSIA}


def anomalias(sesion, anom: dict, ruta: Path) -> Path:
    """Trayectorias normales en tenue y las anómalas resaltadas, con la lista al costado."""
    from anomalias import TIPOS
    df = sesion.df
    quietos = sesion.limpieza.get("quietos") if sesion.limpieza else None
    fig = plt.figure(figsize=(14, 6.6))
    ax = fig.add_axes([0.015, 0.04, 0.6, 0.78])
    ax_t = fig.add_axes([0.64, 0.04, 0.35, 0.78])
    _imagen(ax, _fondo(sesion, 0.45))
    ax.set_xlim(0, sesion.ancho)
    ax.set_ylim(sesion.alto, 0)
    tabla = anom["tabla"]
    marcados = set(tabla["track_id"])
    normales = [g[["px", "py"]].to_numpy() for t, g in df.groupby("track_id") if t not in marcados]
    ax.add_collection(LineCollection(normales, colors=GRIS, linewidths=0.7, alpha=0.25))

    for r in tabla.itertuples():
        fuente = quietos if r.tipo == "estacionado" and quietos is not None else df
        g = fuente[fuente["track_id"] == r.track_id]
        col = COLOR_ANOMALIA[r.tipo]
        p = g[["px", "py"]].to_numpy()
        ax.plot(p[:, 0], p[:, 1], color=col, lw=2.6, solid_capstyle="round")
        if len(p) > 1 and r.tipo in ("contramano", "atipica"):
            ax.annotate("", xy=p[-1], xytext=p[max(len(p) - 6, 0)],
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6, mutation_scale=18))
        ax.scatter([r.x], [r.y], s=170, facecolors="none", edgecolors=col, linewidths=2.2, zorder=5)
        ax.text(r.x, r.y - 0.035 * sesion.alto, f"#{r.track_id}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=col, path_effects=CONTORNO)

    # lista a la derecha
    ax_t.axis("off")
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    y = 0.97
    ax_t.text(0, y, "Resumen", fontsize=12.5, fontweight="bold", color=TEXTO, va="top")
    y -= 0.07
    for clave, nombre in TIPOS.items():
        n = int((tabla["tipo"] == clave).sum())
        ax_t.add_patch(plt.Rectangle((0, y - 0.035), 0.03, 0.035, color=COLOR_ANOMALIA[clave]))
        ax_t.text(0.05, y, f"{nombre}: {n}", fontsize=11, color=TEXTO if n else GRIS, va="top")
        y -= 0.055
    y -= 0.03
    ax_t.text(0, y, "Casos", fontsize=12.5, fontweight="bold", color=TEXTO, va="top")
    y -= 0.07
    maximo = 6
    for r in tabla.head(maximo).itertuples():
        m, s = divmod(int(r.t), 60)
        cuando = (f"{sesion.inicio + timedelta(seconds=r.t):%H:%M:%S}" if sesion.inicio is not None
                  else f"{m:02d}:{s:02d}")
        ax_t.text(0, y, f"#{r.track_id}", fontsize=10.5, fontweight="bold", color=COLOR_ANOMALIA[r.tipo], va="top")
        ax_t.text(0.13, y, f"{cuando} · {TIPOS[r.tipo].lower()}", fontsize=10.5, color=TEXTO, va="top")
        ax_t.text(0.13, y - 0.038, r.detalle, fontsize=9.3, color=GRIS, va="top")
        y -= 0.085
    if len(tabla) > maximo:
        ax_t.text(0, y, f"… y {len(tabla) - maximo} más (ver anomalias.csv)", fontsize=10, color=GRIS, va="top")
    if tabla.empty:
        ax_t.text(0, y, "No se detectaron anomalías.", fontsize=11, color=GRIS, va="top")

    n, k = len(tabla), anom["n_patrones"]
    _titulo(fig, f"Anomalías: {n} caso{'s' if n != 1 else ''}" if n else "Anomalías: ninguna",
            f"Detenido = quieto más de {anom['min_detenido_s']:g} s · sentido contrario = más de 120° del "
            f"sentido del carril · atípica = no se parece a ningún patrón de movimiento habitual "
            f"({k} encontrado{'s' if k != 1 else ''})")
    return _guardar(fig, ruta)


# --- 9. Mapa de calor animado (GIF) ---------------------------------------------------------

def mapa_calor_gif(sesion, ruta: Path, n_cuadros: int = 36, ancho_gif: int = 720, sigma: float = 5.0,
                   duracion_ms: int = 160, pausa_final_ms: int = 2500) -> Path:
    """GIF en el que el mapa de calor se acumula a lo largo de la sesión.

    Cada vehículo cuenta una vez por zona (como en mapa_calor.png) y la escala de
    color es la del mapa final, así se ve cómo «se enciende» la calle con el tiempo.
    """
    from PIL import Image, ImageDraw, ImageFont

    df = sesion.df
    escala = ancho_gif / sesion.ancho
    w, h = ancho_gif, int(round(sesion.alto * escala))
    red = 2   # la grilla se calcula a la mitad de la resolución del GIF
    gw, gh = int(np.ceil(w / red)), int(np.ceil(h / red))
    fondo = cv2.resize((_fondo(sesion, 0.8) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)
    fondo = fondo.astype(np.float32) / 255.0

    duracion = max(sesion.duracion_s, float(df["timestamp_s"].max()))
    cortes = np.linspace(0, duracion, n_cuadros + 1)[1:]
    grupos = [(g["timestamp_s"].to_numpy(), np.round(g[["px", "py"]].to_numpy() * escala / red).astype(np.int32))
              for _, g in df.groupby("track_id")]

    # acumulado por cuadro: cada track suma (una vez) el tramo nuevo de su recorrido hasta el corte
    acumulados = []
    acum = np.zeros((gh, gw), np.float32)
    dibujado = [0] * len(grupos)          # cuántos puntos de cada track ya se dibujaron
    mascara = np.zeros((gh, gw), np.uint8)
    for t_corte in cortes:
        for k, (t, p) in enumerate(grupos):
            hasta = int(np.searchsorted(t, t_corte, side="right"))
            if hasta > dibujado[k]:
                desde = max(dibujado[k] - 1, 0)   # se repite el último punto para unir los tramos
                mascara[:] = 0
                cv2.polylines(mascara, [p[desde:hasta]], False, 1, thickness=1)
                if desde > 0:   # ese punto ya se sumó en el cuadro anterior
                    mascara[np.clip(p[desde, 1], 0, gh - 1), np.clip(p[desde, 0], 0, gw - 1)] = 0
                acum += mascara
                dibujado[k] = hasta
        acumulados.append(gaussian_filter(acum, sigma * escala * 2 / red))
    final = acumulados[-1]
    tope = np.percentile(final[final > final.max() * 0.01], 99.5) if final.max() > 0 else 1.0

    inicios = np.sort(df.groupby("track_id")["timestamp_s"].min().to_numpy())
    cuadros = []
    barra_h = max(6, h // 90)
    esc = w / 900
    fuente = ImageFont.load_default(size=int(30 * esc))
    for nombre in ("segoeuib.ttf", "seguisb.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf"):
        try:
            fuente = ImageFont.truetype(nombre, int(30 * esc))
            break
        except OSError:
            pass
    for t_corte, a in zip(cortes, acumulados):
        calor = cv2.resize(np.clip(a / tope, 0, 1) ** 0.6, (w, h), interpolation=cv2.INTER_LINEAR)
        img = (np.clip(_superponer(fondo, calor), 0, 1) * 255).astype(np.uint8)
        # barra de progreso (turquesa) y rótulo con la hora y los vehículos vistos hasta ese momento
        cv2.rectangle(img, (0, h - barra_h), (w, h), (14, 22, 33), -1)
        cv2.rectangle(img, (0, h - barra_h), (int(w * t_corte / duracion), h), (45, 226, 207), -1)
        n = int(np.searchsorted(inicios, t_corte, side="right"))
        if sesion.inicio is not None:
            reloj = f"{sesion.inicio + timedelta(seconds=float(t_corte)):%H:%M:%S}"
        else:
            m, s = divmod(int(t_corte), 60)
            reloj = f"{m:02d}:{s:02d}"
        cuadro = Image.fromarray(img)
        dibujo = ImageDraw.Draw(cuadro)
        dibujo.text((int(16 * esc), int(12 * esc)), f"{reloj}   {n} vehículos", font=fuente,
                    fill=(242, 245, 248), stroke_width=max(2, int(3 * esc)), stroke_fill=(14, 22, 33))
        cuadros.append(cuadro)

    # paleta común para todos los cuadros (evita el parpadeo de colores)
    paleta = cuadros[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = [c.quantize(palette=paleta, dither=Image.Dither.NONE) for c in cuadros]
    duraciones = [duracion_ms] * (len(gif) - 1) + [pausa_final_ms]
    gif[0].save(ruta, save_all=True, append_images=gif[1:], duration=duraciones, loop=0, optimize=True)
    return ruta
