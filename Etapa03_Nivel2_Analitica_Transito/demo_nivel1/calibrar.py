"""Herramienta para calibrar la distancia a lo largo de la calle (para la velocidad).

Uso:
    python demo_nivel1/calibrar.py demo_nivel1/videos/mi_video.mp4 --periodo 9

Muestra un fondo del video sin autos (mediana de varios frames). Haz clic, de
cerca hacia lejos, en el INICIO de líneas discontinuas consecutivas de un mismo
carril (o en cualquier serie de marcas separadas por `--periodo` metros):

    clic izquierdo  agregar punto
    clic derecho    borrar el último
    Enter           guardar          ESC  salir sin guardar

Se guarda <video>.calibracion.json, que demo_yolo.py usa automáticamente.
Periodos habituales de las líneas discontinuas: revisa la norma del país del
video (por ejemplo, en el Reino Unido 1 m de línea + 5 m de espacio = 6 m en vías
de hasta 40 mph, y 2 m + 7 m = 9 m en autopistas).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibracion import Calibracion  # noqa: E402


def fondo_sin_autos(video: Path, n: int = 25) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Calibra la distancia a lo largo de la calle con marcas del carril.")
    ap.add_argument("video")
    ap.add_argument("--periodo", type=float, required=True, help="metros entre marcas consecutivas")
    ap.add_argument("--limite", type=float, default=None, help="límite de velocidad de la vía (km/h)")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    video = Path(a.video)
    print("Calculando el fondo sin autos...")
    fondo = fondo_sin_autos(video)
    alto, ancho = fondo.shape[:2]
    puntos: list[tuple[int, int]] = []
    ventana = "Calibrar: clic en marcas consecutivas (Enter guarda, ESC sale)"

    def raton(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            puntos.append((x, y))
        elif evento == cv2.EVENT_RBUTTONDOWN and puntos:
            puntos.pop()

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, min(1400, ancho), int(min(1400, ancho) * alto / ancho))
    cv2.setMouseCallback(ventana, raton)
    while True:
        img = fondo.copy()
        for i, (x, y) in enumerate(puntos):
            cv2.circle(img, (x, y), 9, (207, 226, 45), 3)
            cv2.putText(img, f"{i * a.periodo:g} m", (x + 14, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (207, 226, 45), 2, cv2.LINE_AA)
        cv2.imshow(ventana, img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            print("Sin cambios.")
            return 1
        if k in (13, 10):
            if len(puntos) < 3:
                print("Hacen falta al menos 3 puntos.")
                continue
            break
    cv2.destroyAllWindows()
    # de cerca (y grande) hacia lejos
    ys = sorted((y for _, y in puntos), reverse=True)
    calib = Calibracion(ancho, alto, [[float(y), i * a.periodo] for i, y in enumerate(ys)], a.limite,
                        f"{len(ys)} marcas cada {a.periodo:g} m, marcadas a mano")
    ruta = video.with_suffix(".calibracion.json")
    calib.guardar(ruta)
    print(f"Guardado {ruta} (error del ajuste: {calib.error_m:.2f} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
