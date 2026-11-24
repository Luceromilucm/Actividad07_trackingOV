#!/usr/bin/env python3
"""
Radar de Velocidad — versión de CONSOLA (la app gráfica es app.py).

  python radar_velocidad.py --fuente trafico.mp4 --limite 30 --guardar salida.mp4
  python radar_velocidad.py --fuente 0                      # webcam
  python radar_velocidad.py --fuente http://IP:8080/video   # celular (IP Webcam)

Teclas: q/ESC salir · p pausa · +/- límite
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

from motor import Calibracion, Procesador, cargar_modelo, dibujar_puntos_calibracion, etiqueta, BLANCO, NEGRO


def seleccionar_puntos(frame, n, metodo):
    s = min(1.0, 1280 / frame.shape[1])
    pts = []

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x / s, y / s))

    cv2.namedWindow("Calibracion")
    cv2.setMouseCallback("Calibracion", cb)
    ayuda = ("Clic: 1 sup-izq, 2 sup-der, 3 inf-der, 4 inf-izq del rectangulo" if n == 4
             else "Clic en 2 puntos de distancia conocida")
    while True:
        vis = dibujar_puntos_calibracion(frame.copy(), pts, metodo)
        etiqueta(vis, ayuda + "  |  ENTER confirmar, r reiniciar", (10, 30), NEGRO)
        cv2.imshow("Calibracion", cv2.resize(vis, None, fx=s, fy=s))
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k in (13, 10, 32) and len(pts) == n:
            break
        elif k == 27:
            sys.exit("Calibración cancelada.")
    cv2.destroyWindow("Calibracion")
    return pts


def pedir(msg):
    while True:
        try:
            v = float(input(msg).replace(",", "."))
            if v > 0:
                return v
        except ValueError:
            pass


def calibrar(frame, metodo):
    h, w = frame.shape[:2]
    if metodo == "escala":
        p = seleccionar_puntos(frame, 2, metodo)
        return Calibracion.desde_escala(p[0], p[1], pedir("Distancia real (m): "), [w, h])
    p = seleccionar_puntos(frame, 4, metodo)
    return Calibracion("homografia", p, ancho=pedir("Ancho real 1->2 (m): "),
                       largo=pedir("Largo real 2->3 (m): "), tam_frame=[w, h])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default="0")
    ap.add_argument("--modelo", default="yolo26n.pt")
    ap.add_argument("--limite", type=float, default=30)
    ap.add_argument("--metodo", choices=["homografia", "escala"], default="homografia")
    ap.add_argument("--calibracion", default="calibracion.json")
    ap.add_argument("--recalibrar", action="store_true")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640, help="resolución de YOLO (640, 960, 1280)")
    ap.add_argument("--guardar", default=None)
    ap.add_argument("--evidencias", default="infracciones")
    ap.add_argument("--sin-ventana", action="store_true")
    a = ap.parse_args()

    src = int(a.fuente) if a.fuente.isdigit() else a.fuente
    es_archivo = isinstance(src, str) and not src.lower().startswith(("http", "rtsp"))
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"No se pudo leer: {a.fuente}")
    h, w = frame.shape[:2]

    cal = None
    if Path(a.calibracion).exists() and not a.recalibrar:
        cal = Calibracion.cargar(a.calibracion)
        if cal.tam_frame and cal.tam_frame != [w, h]:
            cal = None
    if cal is None:
        if a.sin_ventana:
            sys.exit("Falta calibración.")
        cal = calibrar(frame, a.metodo)
        cal.guardar(a.calibracion)

    proc = Procesador(cargar_modelo(a.modelo), cal, a.limite, a.conf, imgsz=a.imgsz, carpeta_evidencias=a.evidencias)
    print(f"Calibración: {cal.descripcion()}")
    writer = cv2.VideoWriter(a.guardar, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if a.guardar else None
    vehiculos, idx, t0, pausa = [], 0, time.perf_counter(), False

    while True:
        if not pausa:
            if idx:
                ok, frame = cap.read()
                if not ok:
                    break
            idx += 1
            t = idx / fps if es_archivo else time.perf_counter() - t0
            vis, n, term = proc.procesar(frame, t)
            vehiculos += term
            if writer:
                writer.write(vis)
        if a.sin_ventana:
            continue
        cv2.imshow("Radar de Velocidad", cv2.resize(vis, None, fx=min(1, 1280 / w), fy=min(1, 1280 / w)))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("p"):
            pausa = not pausa
        if k in (ord("+"), ord("=")):
            proc.limite += 5
        if k == ord("-"):
            proc.limite = max(5, proc.limite - 5)

    vehiculos += proc.finalizar()
    ruta = Path(a.evidencias) / "vehiculos.csv"
    with ruta.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(vehiculos[0]) if vehiculos else ["id"])
        wr.writeheader()
        wr.writerows(vehiculos)
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    inf = sum(v["infraccion"] for v in vehiculos)
    print(f"Vehículos medidos: {len(vehiculos)} · Infracciones: {inf} · Detalle: {ruta}")


if __name__ == "__main__":
    main()
