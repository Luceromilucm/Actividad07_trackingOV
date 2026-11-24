"""Herramienta interactiva para dibujar los carriles sobre frame_referencia.jpg.

Uso:
    python src/definir_carriles.py <carpeta_sesion>

Controles (con la ventana de la imagen activa):
    clic izquierdo   agrega un vértice del carril actual
    Enter            cierra el polígono; luego dos clics marcan el sentido
                     del tráfico (origen -> destino). Enter de nuevo lo omite.
    z                deshace el último punto (o el último carril)
    s                guarda carriles.json y sale
    Esc / q          sale sin guardar

Si no se dibuja nada, el análisis detecta los carriles automáticamente.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carriles import cargar_carriles, guardar_carriles  # noqa: E402

COLORES_BGR = [(207, 226, 45), (249, 121, 232), (39, 182, 255), (255, 169, 90), (132, 220, 61)]
VENTANA = "Definir carriles"


class Editor:
    """Estado del dibujo. Separado de la ventana para poder probarlo sin interfaz."""

    def __init__(self, carriles: list[dict] | None = None):
        self.carriles: list[dict] = list(carriles or [])
        self.puntos: list[list[float]] = []
        self.sentido: list[list[float]] = []
        self.modo = "poligono"           # "poligono" o "sentido"

    def clic(self, x: float, y: float):
        if self.modo == "poligono":
            self.puntos.append([float(x), float(y)])
        else:
            self.sentido.append([float(x), float(y)])
            if len(self.sentido) == 2:
                self._terminar_carril()

    def enter(self):
        if self.modo == "poligono" and len(self.puntos) >= 3:
            self.modo = "sentido"
        elif self.modo == "sentido":
            self._terminar_carril()

    def deshacer(self):
        if self.modo == "sentido":
            if self.sentido:
                self.sentido.pop()
            else:
                self.modo = "poligono"
        elif self.puntos:
            self.puntos.pop()
        elif self.carriles:
            self.carriles.pop()

    def _terminar_carril(self):
        sentido = None
        if len(self.sentido) == 2:
            (x0, y0), (x1, y1) = self.sentido
            v = np.array([x1 - x0, y1 - y0])
            if np.linalg.norm(v) > 1:
                v = v / np.linalg.norm(v)
                sentido = [round(float(v[0]), 3), round(float(v[1]), 3)]
        self.carriles.append({"nombre": f"Carril {len(self.carriles) + 1}",
                              "poligono": self.puntos, "sentido": sentido})
        self.puntos, self.sentido, self.modo = [], [], "poligono"

    def instruccion(self) -> str:
        n = len(self.carriles) + 1
        if self.modo == "sentido":
            return f"Carril {n}: 2 clics para el sentido del trafico (Enter = omitir)"
        if len(self.puntos) < 3:
            return f"Carril {n}: clic en los vertices ({len(self.puntos)} puntos)  |  s guardar  z deshacer  Esc salir"
        return f"Carril {n}: Enter para cerrar el poligono  |  z deshacer"


def dibujar(base: np.ndarray, ed: Editor) -> np.ndarray:
    img = base.copy()
    capa = img.copy()
    for i, c in enumerate(ed.carriles):
        col = COLORES_BGR[i % len(COLORES_BGR)]
        pts = np.round(np.asarray(c["poligono"])).astype(np.int32)
        cv2.fillPoly(capa, [pts], col)
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
    img = cv2.addWeighted(capa, 0.3, img, 0.7, 0)
    for i, c in enumerate(ed.carriles):
        pts = np.asarray(c["poligono"])
        centro = pts.mean(axis=0)
        cv2.putText(img, c["nombre"], tuple(int(v) for v in centro - [40, 0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if c.get("sentido"):
            fin = centro + np.asarray(c["sentido"]) * 60
            cv2.arrowedLine(img, tuple(int(v) for v in centro), tuple(int(v) for v in fin),
                            (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
    col = COLORES_BGR[len(ed.carriles) % len(COLORES_BGR)]
    if ed.puntos:
        pts = np.round(np.asarray(ed.puntos)).astype(np.int32)
        cv2.polylines(img, [pts], ed.modo == "sentido", col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(int(v) for v in p), 4, col, -1, cv2.LINE_AA)
    for p in ed.sentido:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (33, 22, 14), -1)
    cv2.putText(img, ed.instruccion(), (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 245, 242), 1, cv2.LINE_AA)
    return img


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Dibuja los carriles sobre frame_referencia.jpg")
    ap.add_argument("carpeta", help="carpeta de la sesión")
    a = ap.parse_args()
    carpeta = Path(a.carpeta)
    base = cv2.imread(str(carpeta / "frame_referencia.jpg"))
    if base is None:
        print(f"Error: no se encontró {carpeta / 'frame_referencia.jpg'}", file=sys.stderr)
        return 1

    existentes = cargar_carriles(carpeta)
    if existentes:
        print(f"Se cargaron {len(existentes)} carriles existentes (z para borrarlos uno a uno).")
    ed = Editor(existentes)
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA, min(base.shape[1], 1400), int(min(base.shape[1], 1400) * base.shape[0] / base.shape[1]))
    cv2.setMouseCallback(VENTANA, lambda ev, x, y, *_: ed.clic(x, y) if ev == cv2.EVENT_LBUTTONDOWN else None)

    while True:
        cv2.imshow(VENTANA, dibujar(base, ed))
        k = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1 or k in (27, ord("q")):
            print("Salida sin guardar.")
            break
        if k in (13, 10):
            ed.enter()
        elif k == ord("z"):
            ed.deshacer()
        elif k == ord("s"):
            if ed.modo == "sentido" or len(ed.puntos) >= 3:
                ed._terminar_carril()
            if not ed.carriles:
                print("No hay carriles para guardar.")
                continue
            ruta = guardar_carriles(carpeta, ed.carriles)
            print(f"Se guardaron {len(ed.carriles)} carriles en {ruta}")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
