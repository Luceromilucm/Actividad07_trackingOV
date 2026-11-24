"""Carriles: carga de carriles.json o detección automática, y asignación de vehículos.

Formato de carriles.json:
    {"carriles": [{"nombre": "Carril 1", "poligono": [[x,y], ...], "sentido": [dx, dy]}, ...]}

Si no existe el archivo, los carriles se detectan agrupando la posición lateral de
los vehículos (ver `detectar_carriles`). El análisis se genera solo al cerrar el
programa, así que no puede depender de un paso manual.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as Poligono
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FUERA = "Fuera de carril"


def cargar_carriles(carpeta: Path) -> list[dict] | None:
    ruta = Path(carpeta) / "carriles.json"
    if not ruta.exists():
        return None
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    carriles = datos.get("carriles", datos if isinstance(datos, list) else [])
    for i, c in enumerate(carriles):
        c.setdefault("nombre", f"Carril {i + 1}")
        c["poligono"] = [[float(x), float(y)] for x, y in c["poligono"]]
    return carriles or None


def guardar_carriles(carpeta: Path, carriles: list[dict]) -> Path:
    ruta = Path(carpeta) / "carriles.json"
    limpio = [{"nombre": c["nombre"], "poligono": [[round(x), round(y)] for x, y in c["poligono"]],
               **({"sentido": c["sentido"]} if c.get("sentido") is not None else {})}
              for c in carriles]
    ruta.write_text(json.dumps({"carriles": limpio}, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def asignar_por_poligonos(df: pd.DataFrame, carriles: list[dict]) -> pd.Series:
    """Carril de cada track: aquel donde quedó la mayor parte de sus puntos."""
    pts = df[["px", "py"]].to_numpy()
    etiqueta = np.full(len(df), FUERA, dtype=object)
    for c in carriles:
        dentro = Poligono(np.asarray(c["poligono"])).contains_points(pts)
        etiqueta[dentro & (etiqueta == FUERA)] = c["nombre"]
    tmp = pd.DataFrame({"track_id": df["track_id"].to_numpy(), "carril": etiqueta})
    # el carril más frecuente entre los puntos que caen en algún carril
    def moda(s: pd.Series) -> str:
        en_carril = s[s != FUERA]
        return en_carril.value_counts().idxmax() if len(en_carril) else FUERA
    return tmp.groupby("track_id")["carril"].agg(moda)


# --- Detección automática ---------------------------------------------------

def _direccion_dominante(df: pd.DataFrame) -> np.ndarray:
    """Eje principal del movimiento (sin importar el sentido): PCA de los desplazamientos."""
    g = df.groupby("track_id")
    d = (g[["px", "py"]].last() - g[["px", "py"]].first()).to_numpy()
    norma = np.linalg.norm(d, axis=1)
    d = d[norma > 1e-6] / norma[norma > 1e-6, None]
    if len(d) == 0:
        return np.array([0.0, -1.0])
    valores, vectores = np.linalg.eigh(d.T @ d)
    eje = vectores[:, np.argmax(valores)]
    return eje if eje[1] <= 0 else -eje   # por convención apunta "hacia arriba" de la imagen


def detectar_carriles(df: pd.DataFrame, k_max: int = 6, n_franjas: int = 14) -> tuple[pd.Series, list[dict]]:
    """Detecta carriles sin intervención del usuario.

    1. Se estima el eje dominante del tráfico y se rota la imagen para que quede vertical.
    2. La calle se corta en franjas a lo largo de ese eje. En cada franja, la posición
       lateral se normaliza entre los percentiles 1 y 99: así la perspectiva (carriles
       que se angostan al alejarse) no mezcla los carriles.
    3. Cada vehículo queda representado por la mediana de su posición lateral normalizada,
       y esos valores se agrupan con KMeans eligiendo k por silueta.

    Devuelve el carril de cada track y polígonos aproximados para dibujar.
    """
    eje = _direccion_dominante(df)
    perp = np.array([-eje[1], eje[0]])
    if perp[0] < 0:
        perp = -perp                       # lateral crece hacia la derecha de la imagen
    pts = df[["px", "py"]].to_numpy()
    a = pts @ eje                           # coordenada a lo largo del tráfico
    l = pts @ perp                          # coordenada lateral

    bordes = np.quantile(a, np.linspace(0, 1, n_franjas + 1))
    franja = np.clip(np.searchsorted(bordes, a, side="right") - 1, 0, n_franjas - 1)
    lo = np.zeros(n_franjas)
    hi = np.ones(n_franjas)
    for f in range(n_franjas):
        sel = l[franja == f]
        if len(sel) >= 10:
            lo[f], hi[f] = np.percentile(sel, [1, 99])
    lat = (l - lo[franja]) / np.maximum(hi[franja] - lo[franja], 1e-6)

    tmp = pd.DataFrame({"track_id": df["track_id"].to_numpy(), "lat": lat})
    por_track = tmp.groupby("track_id")["lat"].median()
    X = por_track.to_numpy().reshape(-1, 1)

    mejor_k, mejor_s, mejor_modelo = 1, -1.0, None
    for k in range(2, min(k_max, len(X) - 1) + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        s = silhouette_score(X, modelo.labels_, sample_size=min(len(X), 3000), random_state=0)
        if s > mejor_s:
            mejor_k, mejor_s, mejor_modelo = k, s, modelo
    if mejor_modelo is None or mejor_s < 0.5:
        centros = np.array([float(np.median(X))])
        etiquetas = np.zeros(len(X), int)
    else:
        orden = np.argsort(mejor_modelo.cluster_centers_.ravel())
        centros = mejor_modelo.cluster_centers_.ravel()[orden]
        rango = np.empty_like(orden)
        rango[orden] = np.arange(len(orden))
        etiquetas = rango[mejor_modelo.labels_]

    nombres = [f"Carril {i + 1}" for i in range(len(centros))]
    asignacion = pd.Series([nombres[e] for e in etiquetas], index=por_track.index, name="carril")

    # Polígonos para dibujar: fronteras en el punto medio entre centros, por franja
    if len(centros) > 1:
        medios = (centros[:-1] + centros[1:]) / 2
        cortes = np.concatenate([[2 * centros[0] - medios[0]], medios, [2 * centros[-1] - medios[-1]]])
    else:
        cortes = np.array([0.0, 1.0])
    def a_pixel(a_val, lat_val, f):
        l_val = lo[f] + lat_val * (hi[f] - lo[f])
        return (a_val * eje + l_val * perp).tolist()
    sentido_track = (tmp.assign(px=pts[:, 0], py=pts[:, 1]).groupby("track_id")[["px", "py"]]
                     .agg(lambda s: s.iloc[-1] - s.iloc[0]))
    carriles = []
    for i, nombre in enumerate(nombres):
        izq, der = [], []
        for j, a_val in enumerate(bordes):
            f = min(j, n_franjas - 1)
            izq.append(a_pixel(a_val, cortes[i], f))
            der.append(a_pixel(a_val, cortes[i + 1], f))
        d = sentido_track.loc[asignacion.index[asignacion == nombre]].to_numpy()
        d = d[np.linalg.norm(d, axis=1) > 1e-6]
        sentido = None
        if len(d):
            # sentido mayoritario de los vehículos de este carril
            signo = np.sign(d @ eje)
            v = eje * (1 if (signo >= 0).sum() >= (signo < 0).sum() else -1)
            sentido = [round(float(v[0]), 3), round(float(v[1]), 3)]
        carriles.append({"nombre": nombre, "poligono": izq + der[::-1], "sentido": sentido,
                         "centro_lateral": float(centros[i])})
    return asignacion, carriles


def asignar_carriles(df: pd.DataFrame, carpeta: Path) -> tuple[pd.Series, list[dict], str]:
    """Devuelve (carril por track, carriles, origen) con origen 'dibujados' o 'automaticos'."""
    carriles = cargar_carriles(carpeta)
    if carriles:
        return asignar_por_poligonos(df, carriles), carriles, "dibujados"
    asignacion, carriles = detectar_carriles(df)
    return asignacion, carriles, "automaticos"
