"""Demo del Nivel 1 con video real: YOLO + ByteTrack -> historial.csv -> analítica del Nivel 2.

Simula lo que hará el radar de las compañeras del Nivel 1: abre un video, detecta
y sigue los vehículos, los registra con `RegistroHistorial` y, al presionar Q (o
cuando termina el video), se lanza automáticamente el análisis del Nivel 2.

Uso:
    python demo_nivel1/demo_yolo.py                          # video por defecto
    python demo_nivel1/demo_yolo.py demo_nivel1/videos/roboflow_vehicles.mp4
    python demo_nivel1/demo_yolo.py --sin-ventana            # sin ventana (más rápido)

También se puede usar desde el dashboard (página «Procesar video»), que usa la
misma clase `ProcesadorVideo`.

Velocidad (opcional): si junto al video existe <video>.calibracion.json (ver
calibrar.py), se estima la velocidad en km/h a partir de la posición de cada
vehículo a lo largo de la calle. Sin calibración, el CSV queda sin velocidad.

Este archivo está FUERA de src/ a propósito: el módulo del Nivel 2 no importa YOLO.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import webbrowser
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(AQUI))

from logger import RegistroHistorial  # noqa: E402
from calibracion import Calibracion  # noqa: E402

# Clases de COCO que son vehículos
CLASES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
NOMBRE = {"car": "Auto", "motorcycle": "Moto", "bus": "Bus", "truck": "Camion"}  # cv2 no dibuja tildes
# Colores BGR (paleta del proyecto)
COLOR = {"car": (207, 226, 45), "motorcycle": (249, 121, 232), "bus": (255, 169, 90), "truck": (39, 182, 255)}
CORAL = (115, 93, 255)
TEXTO = (248, 245, 242)
FONDO = (33, 22, 14)

CARPETA_VIDEOS = AQUI / "videos"
VIDEO_DEFECTO = CARPETA_VIDEOS / "pexels_854671_highway.mp4"
MODELO_DEFECTO = AQUI / "modelos" / "yolo11n.pt"
ANCHO_VIDEO_SALIDA = 1280      # el video procesado se guarda a este ancho (como máximo)


def ruta_calibracion(video: Path) -> Path:
    return Path(video).with_suffix(".calibracion.json")


class Velocimetro:
    """Velocidad de cada track a partir de su avance a lo largo de la calle (en metros).

    Se mide en una ventana de ~1 s y se suaviza con la mediana de las últimas
    mediciones: el temblor de la caja frame a frame daría velocidades absurdas.
    """

    def __init__(self, calib: Calibracion, ventana_s: float = 1.0):
        self.calib = calib
        self.ventana = ventana_s
        self.hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=240))
        self.medidas: dict[int, deque] = defaultdict(lambda: deque(maxlen=9))

    def actualizar(self, tid: int, t: float, x: float, y: float) -> float | None:
        s = self.calib.metros(x, y)
        if s is None:                       # fuera del tramo calibrado: no se mide
            self.hist.pop(tid, None)
            return None
        h = self.hist[tid]
        h.append((t, s))
        while len(h) > 2 and t - h[0][0] > self.ventana:
            h.popleft()
        if t - h[0][0] < 0.5 * self.ventana:
            return None
        v = abs(s - h[0][1]) / (t - h[0][0]) * 3.6
        self.medidas[tid].append(v)
        return float(np.median(self.medidas[tid]))


class ProcesadorVideo:
    """Lee un video cuadro a cuadro, detecta y sigue vehículos y los registra.

    Uso:
        proc = ProcesadorVideo("video.mp4")
        while (img := proc.paso()) is not None:   # img = cuadro con cajas, estelas y velocidades
            mostrar(img)
        carpeta = proc.cerrar()                   # guarda CSV, meta.json y frame_referencia.jpg
    """

    def __init__(self, video: str | Path, modelo: str | Path = MODELO_DEFECTO, imgsz: int = 640,
                 conf: float = 0.4, limite: float | None = None, calibracion: str | Path | None = None,
                 inicio: datetime | None = None, max_segundos: float | None = None,
                 carpeta: str | Path | None = None, analizar_al_cerrar: bool = False,
                 guardar_video: bool = True):
        self.video = Path(video)
        self.cap = cv2.VideoCapture(str(self.video))
        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {self.video}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.ancho = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if max_segundos:
            self.n_frames = min(self.n_frames, int(max_segundos * self.fps))

        ruta_calib = Path(calibracion) if calibracion else ruta_calibracion(self.video)
        self.calib = Calibracion.cargar(ruta_calib) if ruta_calib.exists() else None
        self.aviso = None
        if self.calib and (self.calib.ancho, self.calib.alto) != (self.ancho, self.alto):
            self.aviso = (f"La calibración es para {self.calib.ancho}x{self.calib.alto} y el video es "
                          f"{self.ancho}x{self.alto}: se ignora.")
            self.calib = None
        self.limite = limite if limite is not None else (self.calib.limite_kmh if self.calib else None)
        self.velocimetro = Velocimetro(self.calib) if self.calib else None

        from ultralytics import YOLO   # solo la demo importa YOLO; src/ no
        self.modelo = YOLO(str(modelo))
        self.imgsz, self.conf = imgsz, conf

        self.inicio = inicio or datetime.now().replace(microsecond=0)
        self.reg = RegistroHistorial(carpeta=carpeta, fuente=self.video.name, fps=self.fps, ancho=self.ancho,
                                     alto=self.alto, limite_kmh=self.limite, inicio=self.inicio,
                                     analizar_al_cerrar=analizar_al_cerrar)
        if self.calib:
            self.reg.meta["calibracion"] = self.calib.como_dict()
        self.estelas: dict[int, deque] = defaultdict(lambda: deque(maxlen=40))
        self.vistos: set[int] = set()
        self.frame_idx = 0
        self.en_pantalla = 0
        self.excesos: set[int] = set()
        self.t0 = time.perf_counter()
        self.terminado = False
        self.cerrado = False
        self.guardar_video = guardar_video
        self._writer = None
        self.video_salida: Path | None = None

    # --- estado ---
    @property
    def t_video(self) -> float:
        return self.frame_idx / self.fps

    @property
    def progreso(self) -> float:
        return min(self.frame_idx / self.n_frames, 1.0)

    @property
    def fps_proc(self) -> float:
        dt = time.perf_counter() - self.t0
        return self.frame_idx / dt if dt > 0 else 0.0

    @property
    def carpeta(self) -> Path:
        return self.reg.carpeta

    # --- procesamiento ---
    def paso(self, dibujar: bool = True) -> np.ndarray | None:
        """Procesa el cuadro siguiente. Devuelve el cuadro anotado (BGR) o None si terminó el video.

        Si `guardar_video`, el cuadro anotado también se escribe en el video procesado
        (por eso se dibuja aunque `dibujar` sea False).
        """
        dibujar = dibujar or self.guardar_video
        if self.terminado or self.frame_idx >= self.n_frames:
            self.terminado = True
            return None
        ok, frame = self.cap.read()
        if not ok:
            self.terminado = True
            return None
        self.reg.guardar_frame(frame)
        t_video = self.frame_idx / self.fps
        res = self.modelo.track(frame, persist=True, classes=list(CLASES), imgsz=self.imgsz, conf=self.conf,
                                tracker="bytetrack.yaml", verbose=False)[0]
        self.en_pantalla = 0
        if res.boxes is not None and res.boxes.id is not None:
            cajas = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().numpy()
            clases = res.boxes.cls.int().cpu().numpy()
            for (x1, y1, x2, y2), tid, c in zip(cajas, ids, clases):
                clase = CLASES[int(c)]
                tid = int(tid)
                px, py = (x1 + x2) / 2, y2
                v = self.velocimetro.actualizar(tid, t_video, px, py) if self.velocimetro else None
                excede = (v > self.limite) if (v is not None and self.limite) else None
                self.reg.registrar(tid, self.frame_idx, clase, float(x1), float(y1), float(x2), float(y2),
                                   velocidad_kmh=v, excede_limite=excede)
                self.vistos.add(tid)
                if excede:
                    self.excesos.add(tid)
                self.en_pantalla += 1
                if dibujar:
                    self._dibujar_vehiculo(frame, tid, clase, (x1, y1, x2, y2), px, py, v, excede)
        if self.guardar_video:
            self._escribir_video(frame)
        self.frame_idx += 1
        return frame

    # --- video procesado ---
    def _escribir_video(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        escala = min(1.0, ANCHO_VIDEO_SALIDA / w)
        tam = (int(round(w * escala)) // 2 * 2, int(round(h * escala)) // 2 * 2)   # pares, como pide H.264
        if self._writer is None:
            self._tmp_video = self.carpeta / "video_procesado_tmp.mp4"
            self._writer = cv2.VideoWriter(str(self._tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, tam)
            if not self._writer.isOpened():
                print("No se pudo crear el video procesado; se sigue sin guardarlo.")
                self.guardar_video, self._writer = False, None
                return
        salida = cv2.resize(frame, tam, interpolation=cv2.INTER_AREA) if escala < 1 else frame.copy()
        self.dibujar_hud(salida, texto_salida=f"{self.video.name}"
                         + (f"  ·  limite {self.limite:g} km/h" if self.limite else ""), con_fps=False)
        self._writer.write(salida)

    def _cerrar_video(self):
        """Cierra el video y, si hay ffmpeg, lo pasa a H.264 (liviano y reproducible en el navegador)."""
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        final = self.carpeta / "video_procesado.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(self._tmp_video), "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(final)],
                               capture_output=True, stdin=subprocess.DEVNULL)
            if r.returncode == 0 and final.exists():
                self._tmp_video.unlink(missing_ok=True)
                self.video_salida = final
                return
        # sin ffmpeg: queda en MPEG-4 (se abre con VLC o el reproductor de Windows, no siempre en el navegador)
        self._tmp_video.replace(final)
        self.video_salida = final

    def _dibujar_vehiculo(self, frame, tid, clase, caja, px, py, v, excede):
        x1, y1, x2, y2 = (int(c) for c in caja)
        self.estelas[tid].append((int(px), int(py)))
        col = CORAL if excede else COLOR[clase]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
        pts = np.array(self.estelas[tid], np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, col, 3, cv2.LINE_AA)
        rotulo = f"#{tid} {NOMBRE[clase]}" + (f" {v:.0f} km/h" if v is not None else "")
        (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 12, y1), col, -1)
        cv2.putText(frame, rotulo, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, FONDO, 2, cv2.LINE_AA)

    def dibujar_hud(self, img, texto_salida: str = "Q o ESC: terminar y analizar", con_fps: bool = True):
        h, w = img.shape[:2]
        esc = w / 1920
        caja_w, caja_h = int(560 * esc), int(150 * esc)
        sub = img[0:caja_h, 0:caja_w]
        img[0:caja_h, 0:caja_w] = (sub * 0.35 + np.array(FONDO) * 0.65).astype(np.uint8)
        m, s = divmod(int(self.t_video), 60)
        lineas = [(f"Vehiculos registrados: {len(self.vistos)}", 1.05, TEXTO),
                  (f"En pantalla: {self.en_pantalla}   Video {m:02d}:{s:02d}"
                   + (f"   {self.fps_proc:4.1f} fps" if con_fps else ""), 0.8, (196, 179, 163)),
                  (texto_salida, 0.8, (207, 226, 45))]
        y = int(45 * esc)
        for texto, tam, color in lineas:
            cv2.putText(img, texto, (int(18 * esc), y), cv2.FONT_HERSHEY_SIMPLEX, tam * esc, color,
                        max(1, int(2 * esc)), cv2.LINE_AA)
            y += int(42 * esc)
        cv2.rectangle(img, (0, h - int(8 * esc)), (int(w * self.progreso), h), (207, 226, 45), -1)

    def cerrar(self) -> Path:
        """Cierra el video y el registro (CSV, meta.json, frame_referencia.jpg). Se puede llamar varias veces."""
        if not self.cerrado:
            self.cerrado = True
            self.terminado = True
            self.cap.release()
            self._cerrar_video()
            if self.video_salida:
                self.reg.meta["video_procesado"] = self.video_salida.name
            self.reg.cerrar()
        return self.reg.carpeta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Demo YOLO + ByteTrack que registra el historial y lanza la analítica.")
    ap.add_argument("video", nargs="?", default=str(VIDEO_DEFECTO))
    ap.add_argument("--modelo", default=str(MODELO_DEFECTO),
                    help="pesos de YOLO (por defecto yolo11n, el más liviano para CPU)")
    ap.add_argument("--imgsz", type=int, default=640, help="tamaño de entrada de YOLO (menor = más rápido)")
    ap.add_argument("--conf", type=float, default=0.4,
                    help="confianza mínima (con 0,3 el cartel del radar del video por defecto sale como bus)")
    ap.add_argument("--limite", type=float, default=None, help="límite de velocidad en km/h")
    ap.add_argument("--calibracion", default=None,
                    help="archivo de calibración (por defecto <video>.calibracion.json si existe)")
    ap.add_argument("--inicio", default=None,
                    help="hora real del primer frame, AAAA-MM-DDTHH:MM:SS (por defecto, ahora)")
    ap.add_argument("--max-segundos", type=float, default=None, help="procesar solo los primeros N segundos")
    ap.add_argument("--sin-ventana", action="store_true", help="no mostrar la ventana (procesa más rápido)")
    ap.add_argument("--no-abrir", action="store_true", help="no abrir el reporte al terminar")
    ap.add_argument("--carpeta", default=None, help="carpeta de la sesión (por defecto data/sesiones/<fecha>)")
    ap.add_argument("--sin-video", action="store_true",
                    help="no guardar el video procesado (video_procesado.mp4 en la carpeta de la sesión)")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("Cargando YOLO...")
    try:
        proc = ProcesadorVideo(a.video, a.modelo, a.imgsz, a.conf, a.limite, a.calibracion,
                               datetime.fromisoformat(a.inicio) if a.inicio else None, a.max_segundos,
                               a.carpeta, analizar_al_cerrar=True, guardar_video=not a.sin_video)
    except ValueError as e:
        print(e)
        return 1
    if proc.aviso:
        print(proc.aviso)
    print(f"Video: {proc.video.name} · {proc.ancho}x{proc.alto} · {proc.fps:.0f} fps · "
          f"{proc.n_frames / proc.fps:.0f} s")
    print("Velocidad: " + ("sí (calibración encontrada)" if proc.calib else "no (sin calibración)"))

    ventana = "Demo Nivel 1 - YOLO + ByteTrack"
    if not a.sin_ventana:
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ventana, min(1280, proc.ancho), int(min(1280, proc.ancho) * proc.alto / proc.ancho))
    try:
        while (frame := proc.paso(dibujar=not a.sin_ventana)) is not None:
            if a.sin_ventana:
                if (proc.frame_idx - 1) % int(proc.fps * 5) == 0:
                    print(f"  {proc.t_video:5.0f} s de video · {len(proc.vistos)} vehículos · {proc.fps_proc:.1f} fps")
                continue
            proc.dibujar_hud(frame)
            cv2.imshow(ventana, frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27):
                print("Detenido por el usuario.")
                break
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:   # cerró la ventana con la X
                break
    finally:
        if not a.sin_ventana:
            cv2.destroyAllWindows()
        print(f"\nProcesado en {time.perf_counter() - proc.t0:.0f} s. Generando el análisis...\n")
        proc.cerrar()   # guarda el CSV, meta.json, frame_referencia.jpg y el video, y lanza el análisis
        print(f"\nHistorial:        {proc.carpeta / 'historial.csv'}")
        if proc.video_salida:
            print(f"Video procesado:  {proc.video_salida}")

    reporte = proc.carpeta / "salidas" / "reporte.html"
    if reporte.exists() and not a.no_abrir:
        webbrowser.open(reporte.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
