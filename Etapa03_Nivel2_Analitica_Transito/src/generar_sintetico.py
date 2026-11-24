"""Genera una sesión de tránsito sintética para desarrollar y probar el Nivel 2.

Crea en la carpeta de salida:
  historial.csv          detecciones por frame, con el formato del Nivel 1
  meta.json              metadatos de la sesión
  frame_referencia.jpg   la calle vacía, vista desde la cámara
  verdad.json            lo que el análisis debería encontrar (para las pruebas)

La escena es una calle de tres carriles vista con una cámara pinhole:
los carriles 1 y 2 se alejan de la cámara y el carril 3 viene hacia ella.

Uso:
    python src/generar_sintetico.py [carpeta] [--semilla N] [--con-carriles]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --- Geometría de la cámara (pinhole) ---------------------------------------
ANCHO, ALTO = 1280, 720
FPS = 30
DURACION_S = 600            # 10 minutos
INICIO = "2026-11-18T15:30:00"
LIMITE_KMH = 50

CX = 640.0                  # centro óptico horizontal
Y_HORIZONTE = 18.0
C_Y = 20910.0               # y = Y_HORIZONTE + C_Y / z
F = 3143.0                  # x = CX + F * X / z
Z_MIN, Z_MAX = 30.0, 90.0   # tramo de calle visible donde se rastrea (m)
LARGO = Z_MAX - Z_MIN

ANCHO_CARRIL = 3.5
CARRILES = [  # X del centro (m), sentido: +1 se aleja, -1 se acerca
    {"nombre": "Carril 1", "x": -3.5, "sentido": +1},
    {"nombre": "Carril 2", "x": 0.0, "sentido": +1},
    {"nombre": "Carril 3", "x": 3.5, "sentido": -1},
]
PROB_CARRIL = [0.24, 0.58, 0.18]   # el carril 2 domina

CLASES = {  # proporción, ancho (m), alto aparente (m), velocidad media y desvío (km/h)
    "car":        (0.80, 1.8, 1.5, 36, 8),
    "motorcycle": (0.10, 0.8, 1.4, 40, 9),
    "truck":      (0.07, 2.5, 3.2, 31, 6),
    "bus":        (0.03, 2.6, 3.3, 30, 5),
}


def a_imagen(X: np.ndarray | float, z: np.ndarray | float):
    """Proyecta un punto del suelo (X lateral, z profundidad) a píxeles."""
    return CX + F * np.asarray(X) / np.asarray(z), Y_HORIZONTE + C_Y / np.asarray(z)


def tasa_llegadas(t: np.ndarray) -> np.ndarray:
    """Vehículos por segundo: flujo base con picos en los minutos 3 y 7."""
    pico = lambda c: 1.0 * np.exp(-((t - c) / 25.0) ** 2)
    return 0.33 + pico(210) + pico(450)


def tiempos_llegada(rng: np.random.Generator) -> np.ndarray:
    """Proceso de Poisson no homogéneo (método de adelgazamiento)."""
    tmax = 1.4
    t, salida = 0.0, []
    while True:
        t += rng.exponential(1 / tmax)
        if t >= DURACION_S - 5:
            return np.array(salida)
        if rng.random() < tasa_llegadas(np.array(t)) / tmax:
            salida.append(t)


def recorrido(t0, v_ms, rng, parada=None):
    """Distancia recorrida (m) por frame desde que el vehículo entra a la vista.

    parada = (distancia_m, segundos) detiene al vehículo un rato.
    """
    frames, dist = [], []
    f = int(np.ceil(t0 * FPS))
    d = (f / FPS - t0) * v_ms
    restante_parada = parada[1] if parada else 0.0
    while d <= LARGO and f < DURACION_S * FPS:
        frames.append(f)
        dist.append(d)
        if parada and d >= parada[0] and restante_parada > 0:
            restante_parada -= 1 / FPS
        else:
            d += v_ms / FPS
        f += 1
    return np.array(frames), np.array(dist)


def detecciones_vehiculo(tid, clase, carril, t0, v_kmh, rng, parada=None, contramano=False):
    """Filas del CSV para un vehículo: caja con ruido, velocidad medida y huecos."""
    _, ancho_m, alto_m, _, _ = CLASES[clase]
    frames, d = recorrido(t0, v_kmh / 3.6, rng, parada)
    if len(frames) == 0:
        return []
    sentido = carril["sentido"] * (-1 if contramano else 1)
    z = Z_MIN + d if sentido > 0 else Z_MAX - d
    X = carril["x"] + rng.normal(0, 0.25) + rng.normal(0, 0.05, len(z)).cumsum() * 0.05
    u, y = a_imagen(X, z)
    escala = F / z
    w = ancho_m * escala
    h = alto_m * escala
    ruido = lambda: rng.normal(0, 1.2, len(z))
    x1, x2 = u - w / 2 + ruido(), u + w / 2 + ruido()
    y2 = y + ruido()
    y1 = y2 - h + ruido()
    # velocidad estimada por el Nivel 1: verdadera + ruido; 0 mientras está detenido
    dd = np.diff(d, prepend=d[0] - v_kmh / 3.6 / FPS)
    v_med = np.where(dd > 0, v_kmh, 0.0) + rng.normal(0, 2.5, len(z))
    v_med = np.clip(v_med, 0, None)
    filas = []
    for i, fr in enumerate(frames):
        if rng.random() < 0.03:          # el detector pierde el vehículo en este frame
            continue
        if y2[i] > ALTO - 1 or y1[i] < 0:
            continue
        filas.append((tid, fr, x1[i], y1[i], x2[i], y2[i], clase, v_med[i]))
    return filas


def dibujar_calle(rng: np.random.Generator) -> np.ndarray:
    """Frame de referencia: calle vacía con carriles, veredas y árboles."""
    s = 2  # se dibuja al doble y se reduce para suavizar bordes
    img = np.zeros((ALTO * s, ANCHO * s, 3), np.uint8)

    def P(X, z):
        u, v = a_imagen(X, z)
        return int(round(u * s)), int(round(v * s))

    # cielo y suelo
    img[: int(Y_HORIZONTE * s) + 2] = (215, 190, 160)
    yy = np.linspace(0, 1, ALTO * s - int(Y_HORIZONTE * s) - 2)[:, None]
    pasto = (np.array([70, 115, 85]) * (0.75 + 0.35 * yy)).astype(np.uint8)
    img[int(Y_HORIZONTE * s) + 2:] = pasto[:, None, :].repeat(ANCHO * s, axis=1)

    zs_lejos, z_cerca = 4000.0, 8.0
    def franja(X0, X1, color):
        pts = np.array([P(X0, z_cerca), P(X1, z_cerca), P(X1, zs_lejos), P(X0, zs_lejos)])
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)

    franja(-9.5, 9.5, (150, 152, 156))        # veredas
    franja(-5.6, 5.6, (95, 92, 90))           # bordillo
    franja(-5.4, 5.4, (72, 70, 70))           # asfalto

    # textura del asfalto
    ruido = rng.normal(0, 6, img.shape[:2]).astype(np.int16)
    mascara = np.zeros(img.shape[:2], np.uint8)
    pts = np.array([P(-5.4, z_cerca), P(5.4, z_cerca), P(5.4, zs_lejos), P(-5.4, zs_lejos)])
    cv2.fillPoly(mascara, [pts], 1)
    for c in range(3):
        canal = img[:, :, c].astype(np.int16) + ruido * mascara
        img[:, :, c] = np.clip(canal, 0, 255).astype(np.uint8)

    def linea(X, z0, z1, color, grosor_m):
        pts = np.array([P(X - grosor_m / 2, z0), P(X + grosor_m / 2, z0),
                        P(X + grosor_m / 2, z1), P(X - grosor_m / 2, z1)])
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)

    blanco, amarillo = (235, 235, 235), (40, 190, 235)
    for X in (-5.25, 5.25):                    # bordes continuos
        linea(X, z_cerca, zs_lejos, blanco, 0.15)
    z = z_cerca                                # línea discontinua entre carriles 1 y 2
    while z < 600:
        linea(-1.75, z, z + 3, blanco, 0.15)
        z += 9
    for X in (1.62, 1.88):                     # doble amarilla entre sentidos
        linea(X, z_cerca, zs_lejos, amarillo, 0.12)

    # árboles a los costados (de lejos a cerca para que se superpongan bien)
    for z in sorted(rng.uniform(25, 300, 22), reverse=True):
        for lado in (-1, 1):
            X = lado * rng.uniform(11, 14)
            u, v = P(X, z)
            r = int(1.5 * F / z * s)
            cv2.line(img, (u, v), (u, v - int(1.2 * r)), (40, 60, 85), max(1, r // 5), cv2.LINE_AA)
            verde = tuple(int(c) for c in rng.integers([45, 95, 50], [70, 135, 80]))
            cv2.circle(img, (u, v - int(1.6 * r)), r, verde, -1, cv2.LINE_AA)

    return cv2.resize(img, (ANCHO, ALTO), interpolation=cv2.INTER_AREA)


def generar(carpeta: Path, semilla: int = 7, con_carriles: bool = False) -> dict:
    rng = np.random.default_rng(semilla)
    carpeta.mkdir(parents=True, exist_ok=True)

    llegadas = tiempos_llegada(rng)
    nombres_clase = list(CLASES)
    prob_clase = [CLASES[c][0] for c in nombres_clase]

    filas, vehiculos = [], []
    tid = 1
    idx_detenido = int(np.argmin(np.abs(llegadas - 120)))
    idx_contramano = int(np.argmin(np.abs(llegadas - 300)))
    for i, t0 in enumerate(llegadas):
        k = rng.choice(3, p=PROB_CARRIL)
        clase = rng.choice(nombres_clase, p=prob_clase)
        _, _, _, media, desvio = CLASES[clase]
        v = float(np.clip(rng.normal(media, desvio), 15, 60))
        parada, contramano = None, False
        if i == idx_detenido:
            k, clase, v, parada = 0, "car", 30.0, (25.0, 45.0)
        if i == idx_contramano:
            k, clase, v, contramano = 1, "car", 35.0, True
        nuevas = detecciones_vehiculo(tid, clase, CARRILES[k], t0, v, rng, parada, contramano)
        if not nuevas:
            continue
        filas += nuevas
        vehiculos.append({"track_id": tid, "carril": CARRILES[k]["nombre"], "clase": clase,
                          "t0": float(t0), "v_kmh": v,
                          "detenido": parada is not None, "contramano": contramano})
        tid += 1

    n_reales = len(vehiculos)

    # Fragmentación: a 8 vehículos el tracker les cambia el ID en los últimos frames
    df = pd.DataFrame(filas, columns=["track_id", "frame", "x1", "y1", "x2", "y2", "clase", "velocidad_kmh"])
    fragmentados = rng.choice(df["track_id"].unique(), 8, replace=False)
    for viejo in fragmentados:
        idx = df.index[df["track_id"] == viejo]
        cola = idx[-int(rng.integers(3, 7)):]
        df.loc[cola, "track_id"] = tid
        tid += 1

    # Detecciones fantasma: tracks de 2 a 6 frames
    extras = []
    for _ in range(12):
        f0 = int(rng.integers(0, DURACION_S * FPS - 10))
        X, z = rng.uniform(-5, 5), rng.uniform(Z_MIN, Z_MAX)
        u, y = a_imagen(X, z)
        w = 1.8 * F / z
        for f in range(f0, f0 + int(rng.integers(2, 7))):
            extras.append((tid, f, u - w / 2, y - w * 0.8, u + w / 2, y, "car", rng.uniform(0, 40)))
        tid += 1
    # Falsos positivos fijos (un cartel confundido con auto): muchos frames, sin moverse
    for X, z in ((6.2, 40.0), (-6.3, 70.0)):
        u, y = a_imagen(X, z)
        w = 1.2 * F / z
        f0 = int(rng.integers(0, 3000))
        for f in range(f0, f0 + 400):
            extras.append((tid, f, u - w / 2 + rng.normal(0, 1), y - w + rng.normal(0, 1),
                           u + w / 2 + rng.normal(0, 1), y + rng.normal(0, 1), "truck", 0.0))
        tid += 1
    df = pd.concat([df, pd.DataFrame(extras, columns=df.columns)], ignore_index=True)

    # Columnas en el formato acordado con el Nivel 1
    df["timestamp_s"] = (df["frame"] / FPS).round(4)
    df["cx"] = ((df["x1"] + df["x2"]) / 2).round(1)
    df["cy"] = df["y2"].round(1)
    for c in ("x1", "y1", "x2", "y2", "velocidad_kmh"):
        df[c] = df[c].round(1)
    df["excede_limite"] = (df["velocidad_kmh"] > LIMITE_KMH).astype(int)
    df = df.sort_values(["frame", "track_id"])[
        ["track_id", "frame", "timestamp_s", "clase", "cx", "cy",
         "x1", "y1", "x2", "y2", "velocidad_kmh", "excede_limite"]]
    df.to_csv(carpeta / "historial.csv", index=False)

    z_linea = 45.0
    _, y_linea = a_imagen(0, z_linea)
    meta = {
        "fuente": "sintetico (generar_sintetico.py)",
        "fps": FPS, "ancho": ANCHO, "alto": ALTO,
        "inicio_datetime": INICIO,
        "duracion_s": DURACION_S,
        "limite_kmh": LIMITE_KMH,
        "px_por_metro": None,
        "linea_conteo": [[250, round(float(y_linea))], [1030, round(float(y_linea))]],
    }
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    cv2.imwrite(str(carpeta / "frame_referencia.jpg"), dibujar_calle(rng), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Polígonos verdaderos de los carriles (para --con-carriles y para las pruebas)
    poligonos = []
    for c in CARRILES:
        X0, X1 = c["x"] - ANCHO_CARRIL / 2, c["x"] + ANCHO_CARRIL / 2
        pts = [a_imagen(X0, Z_MIN - 0.5), a_imagen(X1, Z_MIN - 0.5),
               a_imagen(X1, Z_MAX + 5), a_imagen(X0, Z_MAX + 5)]
        poligonos.append({
            "nombre": c["nombre"],
            "poligono": [[round(float(u)), round(float(v))] for u, v in pts],
            "sentido": [0.0, -1.0] if c["sentido"] > 0 else [0.0, 1.0],
        })
    if con_carriles:
        (carpeta / "carriles.json").write_text(
            json.dumps({"carriles": poligonos}, indent=2, ensure_ascii=False), encoding="utf-8")

    vdf = pd.DataFrame(vehiculos)
    por_carril = vdf["carril"].value_counts().reindex([c["nombre"] for c in CARRILES], fill_value=0)
    por_minuto = (vdf["t0"] // 60).astype(int).value_counts().sort_index()
    verdad = {
        "vehiculos_reales": n_reales,
        "tracks_fragmento": len(fragmentados),
        "tracks_fantasma": 12,
        "tracks_fijos": 2,
        "por_carril": {k: int(v) for k, v in por_carril.items()},
        "carril_mas_usado": str(por_carril.idxmax()),
        "por_minuto": {int(k): int(v) for k, v in por_minuto.items()},
        "minutos_pico": [int(m) for m in por_minuto.nlargest(2).index],
        "por_clase": {k: int(v) for k, v in vdf["clase"].value_counts().items()},
        "detenido_track_id": int(vdf.loc[vdf["detenido"], "track_id"].iloc[0]),
        "contramano_track_id": int(vdf.loc[vdf["contramano"], "track_id"].iloc[0]),
        "carriles": poligonos,
    }
    (carpeta / "verdad.json").write_text(json.dumps(verdad, indent=2, ensure_ascii=False), encoding="utf-8")
    return verdad


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    raiz = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Genera una sesión de tránsito sintética.")
    ap.add_argument("carpeta", nargs="?", default=str(raiz / "data" / "sintetico"))
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--con-carriles", action="store_true",
                    help="escribe también carriles.json con los polígonos verdaderos")
    a = ap.parse_args()
    v = generar(Path(a.carpeta), a.semilla, a.con_carriles)
    print(f"Sesión sintética creada en {a.carpeta}")
    print(f"  Vehículos reales: {v['vehiculos_reales']}  "
          f"(+{v['tracks_fragmento']} fragmentos, {v['tracks_fantasma']} fantasmas, {v['tracks_fijos']} fijos)")
    print(f"  Por carril: {v['por_carril']}  -> más usado: {v['carril_mas_usado']}")
    print(f"  Minutos pico: {v['minutos_pico']}  Por minuto: {v['por_minuto']}")


if __name__ == "__main__":
    main()
