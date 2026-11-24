#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ítem 4 · Tracking 2D con CÁMARA EN MOVIMIENTO compensado con ODOMETRÍA VISUAL
=============================================================================
Extiende el tracker del ítem 2 (YOLO + ByteTrack / BoT-SORT):

- Fuente: webcam o archivo de video (si el video del proyecto está junto al
  script o en la carpeta «videos/», se carga automáticamente).
- Odometría visual 2D (frame a frame):
    1. Puntos del FONDO con goodFeaturesToTrack (se enmascaran las cajas de
       YOLO para no usar objetos que se mueven por sí mismos).
    2. Seguimiento con flujo óptico Lucas-Kanade + chequeo ida y vuelta.
    3. Movimiento de la cámara con RANSAC: similaridad (4 GL) u homografía (8 GL).
    4. Se acumulan las transformaciones -> pose de la cámara respecto al
       1er frame ("mundo").
- Los centros de los objetos, la línea de conteo, las estelas y la velocidad se
  calculan en coordenadas del MUNDO, así el conteo ya no se arruina cuando la
  cámara gira o se desplaza.
- Mosaico (panorama) de la escena con las trayectorias compensadas encima.
- Simulador de cámara en movimiento (paneo + rotación + zoom + temblor) sobre
  cualquier video: permite probar el método y medir el error de la odometría,
  porque el movimiento real de la cámara simulada es conocido.
- Comparador: modelos x trackers x (sin VO / con VO) sobre el mismo video.

Instalación:
    pip install ultralytics opencv-python numpy pandas matplotlib pillow lap

Ejecución:
    python tracking_vo_gui.py
"""

import json
import os
import queue
import re
import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

MIN_POINTS = 5      # mínimo de puntos para considerar válida una trayectoria
SHORT_TRACK = 15    # trayectorias con menos frames que esto = "cortas" (fragmentadas)
# x, y = coordenadas en la IMAGEN; xw, yw = coordenadas en el MUNDO (1er frame)
CSV_COLUMNS = ["id", "frame", "time_s", "x", "y", "xw", "yw", "v_px_s", "class", "conf"]
BASE_COLUMNS = ["id", "frame", "time_s", "x", "y", "class", "conf"]
CAM_COLUMNS = ["frame", "time_s", "cam_x", "cam_y", "rot_deg", "escala", "inliers", "ok",
               "gt_x", "gt_y",
               # pose 3D (solo en el modo «3D: matriz esencial»); escala relativa
               "x3d", "y3d", "z3d", "pitch_deg", "yaw_deg", "roll_deg", "modo3d"]
TRACKER_COLORS = {"bytetrack": "#1f77b4", "botsort": "#ff7f0e"}
# Modelos OBB (entrenados con DOTA, imágenes AÉREAS): sus clases no son las de COCO
DOTA_FROM_COCO = {1: [10], 2: [10], 3: [10], 5: [9], 7: [9]}  # bici/auto/moto -> small vehicle; bus/camión -> large vehicle
IMG_SIZES = ["640", "960", "1024", "1280"]
SIM_LEVELS = {"Desactivada": 0.0, "Suave": 0.5, "Media": 1.0, "Fuerte": 1.5}
VO_MOTIONS = {"Similaridad (4 GL)": "similarity", "Homografía (8 GL)": "homography",
              "3D: matriz esencial (6 GL)": "3d"}
DEFAULT_VIDEO_NAMES = ["12570536_1080_1920_30fps.mp4"]


# ----------------------------------------------------------------------------
# Geometría y utilidades
# ----------------------------------------------------------------------------
def side(p, a, b):
    """Producto cruz: signo = lado de la recta AB en que está el punto p."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed_line(p_prev, p_curr, a, b):
    """+1 si cruzó el segmento AB del lado (-) al (+), -1 en sentido contrario, 0 si no cruzó."""
    d1, d2 = side(p_prev, a, b), side(p_curr, a, b)
    d3, d4 = side(a, p_prev, p_curr), side(b, p_prev, p_curr)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 1 if d1 < 0 else -1
    return 0


def color_for_id(track_id):
    rng = np.random.default_rng(int(track_id) * 9973 + 7)
    return tuple(int(c) for c in rng.integers(60, 255, 3))  # BGR


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def to_ascii(text):
    """cv2.putText solo dibuja ASCII: quita tildes y símbolos (si no, salen «??»)."""
    import unicodedata
    text = text.replace("·", "|").replace("°", " deg").replace("¡", "").replace("−", "-")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def apply_h(H, pts):
    """Aplica una homografía 3x3 a una lista/array de puntos (N,2) -> (N,2)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def find_default_video():
    """Busca el video del proyecto junto al script, en «videos/» o en la carpeta actual."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    dirs = [here, os.path.join(here, "videos"), os.getcwd(), os.path.join(os.getcwd(), "videos")]
    for d in dirs:
        for name in DEFAULT_VIDEO_NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    for d in (os.path.join(here, "videos"), os.path.join(os.getcwd(), "videos")):
        if os.path.isdir(d):
            vids = sorted(f for f in os.listdir(d)
                          if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
            if vids:
                return os.path.join(d, vids[0])
    return ""


# ----------------------------------------------------------------------------
# Simulador de cámara en movimiento (con movimiento real conocido)
# ----------------------------------------------------------------------------
class CameraMotionSimulator:
    """
    Recorta una ventana que se mueve sobre el video original: paneo horizontal y
    vertical, rotación, zoom y un temblor de "cámara en mano". Como la ventana se
    conoce, también se conoce el movimiento real de la cámara (ground truth) y se
    puede medir el error de la odometría visual.
    """

    def __init__(self, W, H, intensity, crop=0.72):
        self.W, self.H, self.k = W, H, float(intensity)
        self.ow = int(W * crop) // 2 * 2
        self.oh = int(H * crop) // 2 * 2
        self.O = np.array([self.ow / 2.0, self.oh / 2.0])
        self.A0 = None

    def matrix(self, t):
        """Matriz 3x3 que lleva un píxel de la imagen simulada al video original."""
        k, W, H = self.k, self.W, self.H
        mx, my = (W - self.ow) / 2.0, (H - self.oh) / 2.0
        ang = np.radians(4.0 * k * np.sin(2 * np.pi * t / 9.0)
                         + 0.25 * k * np.sin(2 * np.pi * t / 0.9))
        z = 1.0 + 0.06 * k * (0.5 - 0.5 * np.cos(2 * np.pi * t / 13.0))
        cx = W / 2.0 + 0.55 * k * mx * np.sin(2 * np.pi * t / 7.0) \
            + 2.0 * k * np.sin(2 * np.pi * t / 0.53)
        cy = H / 2.0 + 0.45 * k * my * np.sin(2 * np.pi * t / 11.0 + 1.0) \
            + 2.0 * k * np.sin(2 * np.pi * t / 0.61)
        c, s = np.cos(ang), np.sin(ang)
        L = np.array([[c, -s], [s, c]]) / z
        # Se limita el centro para que la ventana nunca salga del video (sin bordes negros)
        corners = (np.array([[0, 0], [self.ow, 0], [self.ow, self.oh], [0, self.oh]])
                   - self.O) @ L.T
        cx = np.clip(cx, -corners[:, 0].min(), W - corners[:, 0].max())
        cy = np.clip(cy, -corners[:, 1].min(), H - corners[:, 1].max())
        t_vec = np.array([cx, cy]) - L @ self.O
        A = np.eye(3)
        A[:2, :2], A[:2, 2] = L, t_vec
        return A

    def apply(self, frame, t):
        A = self.matrix(t)
        if self.A0 is None:
            self.A0 = A.copy()
        out = cv2.warpAffine(frame, A[:2], (self.ow, self.oh),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        return out, A

    def gt_cur_to_world(self, A):
        """Movimiento real: imagen actual -> 1er frame simulado (= mundo)."""
        return np.linalg.inv(self.A0) @ A


# ----------------------------------------------------------------------------
# Odometría visual 2D (movimiento de la cámara entre frames)
# ----------------------------------------------------------------------------
class VisualOdometry2D:
    """
    Odometría visual 2D basada en KEYFRAMES (como los sistemas de VO reales):

    - En un keyframe se detectan puntos del FONDO (las cajas de YOLO se enmascaran).
    - Esos puntos se siguen frame a frame con flujo óptico Lucas-Kanade
      (con chequeo ida y vuelta), pero el movimiento se estima SIEMPRE contra el
      keyframe, no contra el frame anterior: así los errores no se acumulan en
      cada frame (con la cámara quieta la deriva es prácticamente cero).
    - RANSAC estima la transformación actual -> keyframe y descarta los puntos que
      se mueven distinto (personas que no detectó YOLO, sombras, etc.).
    - Cuando quedan pocos puntos (la cámara se alejó del keyframe) se crea un
      keyframe nuevo y se encadena: H_actual->mundo = H_kf->mundo · H_actual->kf.
    """

    def __init__(self, motion="similarity", proc_width=480, min_pts=80, keep_ratio=0.45):
        self.motion = motion
        self.proc_width = proc_width
        self.min_pts = min_pts
        self.keep_ratio = keep_ratio
        self.s = None
        self.prev = None
        self.kf_pts = self.cur_pts = None   # mismos puntos: en el keyframe y en el frame actual
        self.n_init = 0
        self.H_kf = np.eye(3)               # keyframe -> mundo
        self.H = np.eye(3)                  # actual -> mundo
        self.last_rel = np.eye(3)           # actual -> keyframe (último válido)
        self.n_keyframes = 0
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def _mask(self, shape, boxes):
        mask = np.full(shape, 255, np.uint8)
        s = self.s
        for x1, y1, x2, y2 in boxes:
            pw, ph = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            xa, ya = int(max((x1 - pw) * s, 0)), int(max((y1 - ph) * s, 0))
            xb, yb = int((x2 + pw) * s) + 1, int((y2 + ph) * s) + 1
            mask[ya:yb, xa:xb] = 0
        return mask

    def _new_keyframe(self, g, boxes):
        p = cv2.goodFeaturesToTrack(g, 800, 0.01, 7, mask=self._mask(g.shape, boxes),
                                    blockSize=7)
        if p is None or len(p) < 30:   # escena llena de objetos: se usa toda la imagen
            p = cv2.goodFeaturesToTrack(g, 800, 0.01, 7, blockSize=7)
        p = np.zeros((0, 1, 2), np.float32) if p is None else p.astype(np.float32)
        self.kf_pts, self.cur_pts = p.copy(), p.copy()
        self.n_init = len(p)
        self.H_kf = self.H.copy()
        self.last_rel = np.eye(3)
        self.n_keyframes += 1

    def _estimate(self, src, dst):
        """Transformación src -> dst con RANSAC. Devuelve (3x3, máscara de inliers)."""
        if self.motion == "homography":
            M, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        else:
            A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=2.0)
            M = None if A is None else np.vstack([A, [0, 0, 1]])
        if M is None or inl is None:
            return None, None
        return M, inl.ravel().astype(bool)

    def update(self, frame, boxes_prev):
        """boxes_prev: cajas (x1,y1,x2,y2) del frame ANTERIOR, en píxeles originales."""
        h, w = frame.shape[:2]
        if self.s is None:
            self.s = min(1.0, self.proc_width / float(w))
        s = self.s
        S, Si = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
        g = cv2.cvtColor(cv2.resize(frame, (max(int(w * s), 1), max(int(h * s), 1)),
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        info = {"inliers": 0, "ok": True, "pts": None, "keyframes": self.n_keyframes}
        if self.prev is None or self.prev.shape != g.shape:
            self.prev = g
            self._new_keyframe(g, boxes_prev)
            info["keyframes"] = self.n_keyframes
            return self.H, info

        rel = None
        if len(self.cur_pts) >= 8:
            # 1) Seguimiento de los puntos: frame anterior -> actual (ida y vuelta)
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.cur_pts, None, **self.lk)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, p1, None, **self.lk)
            fb = np.linalg.norm((self.cur_pts - p0r).reshape(-1, 2), axis=1)
            good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            # se descartan los puntos que ahora caen dentro de un objeto detectado
            if good.any() and len(boxes_prev):
                q = p1.reshape(-1, 2) / s
                for x1, y1, x2, y2 in boxes_prev:
                    good &= ~((q[:, 0] > x1) & (q[:, 0] < x2) & (q[:, 1] > y1) & (q[:, 1] < y2))
            kf, cur = self.kf_pts[good], p1[good]
            if len(kf) >= 8:
                # 2) Movimiento actual -> keyframe con RANSAC
                M, inl = self._estimate(cur.reshape(-1, 2), kf.reshape(-1, 2))
                if M is not None and int(inl.sum()) >= 12:
                    rel = Si @ M @ S
                    self.kf_pts, self.cur_pts = kf[inl], cur[inl]   # quedan solo inliers
                    info["inliers"] = int(inl.sum())
                    info["pts"] = self.cur_pts.reshape(-1, 2) / s
        if rel is None:
            # Sin estimación confiable: se mantiene la última pose y se reinicia el keyframe
            info["ok"] = False
            rel = self.last_rel
        self.last_rel = rel
        # 3) Pose: actual -> mundo
        self.H = self.H_kf @ rel
        self.H /= self.H[2, 2]
        self.prev = g
        # 4) ¿Keyframe nuevo? (pocos puntos vivos o se perdió el seguimiento)
        if not info["ok"] or len(self.cur_pts) < max(self.min_pts, self.keep_ratio * self.n_init):
            self._new_keyframe(g, boxes_prev)
        info["keyframes"] = self.n_keyframes
        return self.H, info


# ----------------------------------------------------------------------------
# Odometría visual 3D monocular (pose 6-DoF con la matriz esencial)
# ----------------------------------------------------------------------------
def load_calibration(path):
    """
    Lee la calibración de la cámara (la del tablero de ajedrez).
    Acepta .npz / .json (claves mtx|K|camera_matrix y dist|dist_coeffs) o
    .yml/.yaml/.xml de cv2.FileStorage. Devuelve (K, dist, (ancho, alto) o None).
    """
    kkeys = ("camera_matrix", "cameraMatrix", "mtx", "K", "k", "intrinsic", "intrinsics")
    dkeys = ("dist_coeffs", "distCoeffs", "dist", "D", "distortion", "dist_coeff")
    skeys = ("image_size", "img_size", "size", "resolution", "imageSize")
    ext = os.path.splitext(path)[1].lower()
    data = {}
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        data = {k: z[k] for k in z.files}
    elif ext == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    else:  # yml / yaml / xml (cv2.FileStorage)
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        for k in kkeys + dkeys + skeys + ("image_width", "image_height"):
            node = fs.getNode(k)
            if node.empty():
                continue
            m = node.mat()
            data[k] = m if m is not None else node.real()
        fs.release()
    K = next((np.asarray(data[k], float).reshape(3, 3) for k in kkeys if k in data), None)
    if K is None:
        raise ValueError("No se encontró la matriz de la cámara (mtx / K / camera_matrix).")
    dist = next((np.asarray(data[k], float).ravel() for k in dkeys if k in data), None)
    size = next((tuple(int(v) for v in np.asarray(data[k]).ravel()[:2]) for k in skeys
                 if k in data), None)
    if size is None and "image_width" in data and "image_height" in data:
        size = (int(np.asarray(data["image_width"]).ravel()[0]),
                int(np.asarray(data["image_height"]).ravel()[0]))
    return K, dist, size


def approx_K(W, H, fov_h_deg):
    """Matriz de cámara aproximada a partir del campo de visión horizontal."""
    f = (W / 2.0) / np.tan(np.radians(fov_h_deg) / 2.0)
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


def rot_to_euler(R):
    """Ángulos (pitch, yaw, roll) en grados. Ejes de cámara: x der., y abajo, z adelante."""
    angles = cv2.RQDecomp3x3(R)[0]
    return float(angles[0]), float(angles[1]), float(angles[2])


class VisualOdometry3D:
    """
    Odometría visual MONOCULAR 3D (6 grados de libertad):

    1. Puntos del fondo seguidos con Lucas-Kanade desde un keyframe (cajas de YOLO
       enmascaradas, igual que en la VO 2D).
    2. Cuando el desplazamiento (paralaje) es suficiente, se estiman dos modelos
       entre el keyframe y el frame actual:
         - Matriz esencial E (escena con profundidad)  -> recoverPose -> R, t
         - Homografía H (escena plana o rotación pura) -> decomposeHomographyMat
       Se usa el que explica más puntos (si H explica casi todo, la escena es plana,
       como el suelo visto por un dron, y E se vuelve inestable).
    3. Se encadenan las poses: R_mundo y centro C de la cámara.

    Limitación fundamental: con UNA cámara la escala no se puede conocer; t sale
    normalizado. Aquí se mantiene una escala relativa (unidades arbitrarias).
    """

    def __init__(self, K, dist=None, proc_width=640, min_parallax_px=10.0):
        self.K_full = np.asarray(K, float)
        self.dist = None if dist is None else np.asarray(dist, float)
        self.proc_width = proc_width
        self.min_parallax = min_parallax_px
        self.s = None
        self.prev = None
        self.kf_pts = self.cur_pts = None
        self.n_init = 0
        self.R = np.eye(3)            # orientación de la cámara actual en el mundo
        self.C = np.zeros(3)          # posición (centro) de la cámara en el mundo
        self.R_kf, self.C_kf = np.eye(3), np.zeros(3)
        self.step_norm = 1.0          # escala relativa usada cuando E no da magnitud
        self.mode = "inicio"
        self.n_keyframes = 0
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def _new_keyframe(self, g, boxes):
        mask = np.full(g.shape, 255, np.uint8)
        s = self.s
        for x1, y1, x2, y2 in boxes:
            mask[int(max(y1 * s, 0)):int(y2 * s) + 1, int(max(x1 * s, 0)):int(x2 * s) + 1] = 0
        p = cv2.goodFeaturesToTrack(g, 1000, 0.01, 8, mask=mask, blockSize=7)
        if p is None or len(p) < 30:
            p = cv2.goodFeaturesToTrack(g, 1000, 0.01, 8, blockSize=7)
        p = np.zeros((0, 1, 2), np.float32) if p is None else p.astype(np.float32)
        self.kf_pts, self.cur_pts = p.copy(), p.copy()
        self.n_init = len(p)
        self.R_kf, self.C_kf = self.R.copy(), self.C.copy()
        self.n_keyframes += 1

    def _undistort(self, pts):
        """Píxeles (escala completa) -> píxeles sin distorsión (misma K)."""
        p = pts.reshape(-1, 1, 2).astype(np.float64)
        if self.dist is not None and np.any(self.dist):
            p = cv2.undistortPoints(p, self.K_full, self.dist, P=self.K_full)
        return p.reshape(-1, 2)

    def _relative_pose(self, a, b):
        """Movimiento keyframe -> actual. Devuelve (R, t, modo, inliers) con X_b = R X_a + t."""
        K = self.K_full
        E, mE = cv2.findEssentialMat(a, b, K, cv2.RANSAC, 0.999, 1.0)
        Hm, mH = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
        nE = 0 if E is None or E.shape != (3, 3) else int(mE.sum())
        nH = 0 if Hm is None else int(mH.sum())
        if max(nE, nH) < 15:
            return None
        if nH >= 0.9 * nE:   # escena plana o rotación pura: la homografía es más estable
            n_sol, Rs, ts, ns = cv2.decomposeHomographyMat(Hm, K)
            idx = list(range(n_sol))
            try:
                inl = mH.ravel().astype(bool)
                vis = cv2.filterHomographyDecompByVisibleRefpoints(
                    Rs, ns, a[inl].reshape(-1, 1, 2), b[inl].reshape(-1, 1, 2))
                if vis is not None and len(vis):
                    idx = [int(i) for i in np.asarray(vis).ravel()]
            except cv2.error:
                pass
            # se prefiere el plano que "mira" a la cámara (normal con z grande)
            i = max(idx, key=lambda k: float(ns[k].ravel()[2]))
            R, t = Rs[i], ts[i].ravel()     # t ya viene dividido por la distancia al plano
            mode = "H (rotación)" if np.linalg.norm(t) < 0.01 else "H (plano)"
            return R, t, mode, nH
        _, R, t, mP = cv2.recoverPose(E, a, b, K, mask=mE.copy())
        t = t.ravel() * self.step_norm      # dirección de E + escala relativa conocida
        return R, t, "E (3D)", int((mP > 0).sum())

    def update(self, frame, boxes_prev):
        h, w = frame.shape[:2]
        if self.s is None:
            self.s = min(1.0, self.proc_width / float(w))
        s = self.s
        g = cv2.cvtColor(cv2.resize(frame, (max(int(w * s), 1), max(int(h * s), 1)),
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        info = {"mode": self.mode, "inliers": 0, "ok": True}
        if self.prev is None or self.prev.shape != g.shape:
            self.prev = g
            self._new_keyframe(g, boxes_prev)
            return self.R, self.C, info

        if len(self.cur_pts) >= 8:
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.cur_pts, None, **self.lk)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, p1, None, **self.lk)
            fb = np.linalg.norm((self.cur_pts - p0r).reshape(-1, 2), axis=1)
            good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            if good.any() and len(boxes_prev):
                q = p1.reshape(-1, 2) / s
                for x1, y1, x2, y2 in boxes_prev:
                    good &= ~((q[:, 0] > x1) & (q[:, 0] < x2) & (q[:, 1] > y1) & (q[:, 1] < y2))
            self.kf_pts, self.cur_pts = self.kf_pts[good], p1[good]
        self.prev = g

        n = len(self.cur_pts)
        if n < 15:   # se perdió el seguimiento: keyframe nuevo, la pose se mantiene
            info.update(ok=False, mode="perdida")
            self.mode = "perdida"
            self._new_keyframe(g, boxes_prev)
            return self.R, self.C, info

        a = self._undistort(self.kf_pts.reshape(-1, 2) / s)
        b = self._undistort(self.cur_pts.reshape(-1, 2) / s)
        parallax = float(np.median(np.linalg.norm(b - a, axis=1)))
        info["inliers"] = n
        need_kf = n < max(60, 0.4 * self.n_init)
        if parallax >= self.min_parallax or (need_kf and parallax > 2.0):
            rel = self._relative_pose(a, b)
            if rel is not None:
                R_rel, t_rel, mode, ninl = rel
                # Pose de la cámara actual: X_cur = R_rel X_kf + t_rel
                self.R = self.R_kf @ R_rel.T
                self.C = self.C_kf - self.R_kf @ R_rel.T @ t_rel
                if mode.startswith("H (plano)"):
                    self.step_norm = max(float(np.linalg.norm(t_rel)), 1e-3)
                self.mode = mode
                info.update(mode=mode, inliers=ninl)
                self._new_keyframe(g, boxes_prev)
            elif need_kf:
                info.update(ok=False, mode="sin modelo")
                self._new_keyframe(g, boxes_prev)
        elif need_kf:
            self._new_keyframe(g, boxes_prev)
        info["mode"] = self.mode
        return self.R, self.C, info


def camera_pose(H, W, Hh):
    """Posición (centro de la imagen en el mundo), rotación y escala de la cámara."""
    cx, cy = apply_h(H, [[W / 2.0, Hh / 2.0]])[0]
    rot = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
    scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
    return float(cx), float(cy), rot, scale


def build_mosaic(keyframes, frame_size, max_side=3000):
    """Une los keyframes en un panorama. Devuelve (imagen, T) con T: mundo -> píxel del mosaico."""
    if not keyframes:
        return None, None
    W, H = frame_size
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    allc = np.vstack([apply_h(Hk, corners) for _, Hk, _ in keyframes])
    xmin, ymin = allc.min(axis=0)
    xmax, ymax = allc.max(axis=0)
    ks = keyframes[0][2]
    s = min(ks, max_side / max(xmax - xmin, ymax - ymin, 1.0))
    T = np.array([[s, 0, -xmin * s], [0, s, -ymin * s], [0, 0, 1.0]])
    size = (int(np.ceil((xmax - xmin) * s)) + 1, int(np.ceil((ymax - ymin) * s)) + 1)
    canvas = np.zeros((size[1], size[0], 3), np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    for img, Hk, kscale in keyframes:
        M = T @ Hk @ np.diag([1 / kscale, 1 / kscale, 1.0])
        warped = cv2.warpPerspective(img, M, size)
        m = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), M, size,
                                flags=cv2.INTER_NEAREST)
        m = cv2.erode(m, kernel) > 0   # sin bordes oscuros entre keyframes
        canvas[m] = warped[m]
    return canvas, T


# ----------------------------------------------------------------------------
# Vista para graficar (imagen o mundo/mosaico)
# ----------------------------------------------------------------------------
def metric_df(df, vo):
    """DataFrame con x,y en las coordenadas que se usan para medir (mundo si hay VO)."""
    d = df.copy()
    if vo and "xw" in d.columns:
        d["x"], d["y"] = d["xw"], d["yw"]
    return d


def make_view(res):
    """
    res: dict con df, vo, bg, mosaic, mosaic_T, line_world.
    Devuelve df con x,y listos para dibujar sobre el fondo, el fondo y la línea en píxeles.
    """
    df, vo = res["df"], res.get("vo", False)
    line = res.get("line_world")
    if vo and res.get("mosaic") is not None:
        T = res["mosaic_T"]
        d = df.copy()
        xy = apply_h(T, d[["xw", "yw"]].to_numpy(float))
        d["x"], d["y"] = xy[:, 0], xy[:, 1]
        line_px = apply_h(T, line) if line is not None else None
        return d, res["mosaic"], line_px, T
    d = metric_df(df, vo)
    return d, res.get("bg"), (np.asarray(line, float) if line is not None else None), None


# ----------------------------------------------------------------------------
# Análisis de trayectorias (una ejecución)
# ----------------------------------------------------------------------------
SUMMARY_COLS = ["id", "clase", "frames", "duracion_s", "distancia_px",
                "desplazamiento_px", "rectitud", "velocidad_px_s", "angulo_deg"]


def compute_summary(df):
    rows = []
    for tid, g in df.sort_values("frame").groupby("id"):
        if len(g) < MIN_POINTS:
            continue
        xy = g[["x", "y"]].to_numpy(dtype=float)
        path_len = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        vec = xy[-1] - xy[0]
        disp = float(np.linalg.norm(vec))
        dur = float(g["time_s"].iloc[-1] - g["time_s"].iloc[0])
        rows.append({
            "id": int(tid),
            "clase": g["class"].mode().iloc[0],
            "frames": len(g),
            "duracion_s": round(dur, 2),
            "distancia_px": round(path_len, 1),
            "desplazamiento_px": round(disp, 1),
            "rectitud": round(disp / path_len, 3) if path_len > 0 else 0.0,
            "velocidad_px_s": round(path_len / dur, 1) if dur > 0 else 0.0,
            # 0° = derecha, 90° = abajo (coordenadas de imagen)
            "angulo_deg": round(float(np.degrees(np.arctan2(vec[1], vec[0]))), 1),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


def _limits(df, bg):
    if bg is not None:
        h, w = bg.shape[:2]
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0 = y0 = 0
        x1 = float(df["x"].max()) + 1 if len(df) else 640.0
        y1 = float(df["y"].max()) + 1 if len(df) else 480.0
    if not len(df):
        return x0, y0, x1, y1
    x0, y0 = min(x0, float(df["x"].min())), min(y0, float(df["y"].min()))
    x1, y1 = max(x1, float(df["x"].max())), max(y1, float(df["y"].max()))
    return x0, y0, x1, y1


def draw_analysis(fig, df, summary, bg_bgr, line_px=None, cam=None, cam_T=None, vo=False):
    fig.clear()
    has_cam = cam is not None and len(cam) > 1
    has3d = has_cam and "x3d" in cam and cam["x3d"].notna().any()
    ncol = 2 + int(has_cam) + int(has3d)
    axs = [fig.add_subplot(1, ncol, i + 1) for i in range(ncol - int(has3d))]
    if has3d:
        axs.append(fig.add_subplot(1, ncol, ncol, projection="3d"))
    ax1, ax2 = axs[0], axs[1]
    bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB) if bg_bgr is not None else None
    x0, y0, x1, y1 = _limits(df, bg)

    cmap = matplotlib.colormaps["tab20"]
    if bg is not None:
        ax1.imshow(bg)
    valid_ids = set(summary["id"]) if not summary.empty else set()
    for tid, g in df.groupby("id"):
        if tid not in valid_ids:
            continue
        c = cmap(int(tid) % 20)
        ax1.plot(g["x"], g["y"], "-", color=c, linewidth=1.2)
        ax1.plot(g["x"].iloc[0], g["y"].iloc[0], "o", color=c, markersize=3)
        ax1.plot(g["x"].iloc[-1], g["y"].iloc[-1], "s", color=c, markersize=3)
    if line_px is not None:
        ax1.plot(line_px[:, 0], line_px[:, 1], "r-", linewidth=2)
    ax1.set_title("Trayectorias " + ("compensadas (mosaico)" if vo else "(imagen)") +
                  "\n● inicio, ■ fin", fontsize=9)
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y1, y0)
    ax1.set_aspect("equal")
    ax1.axis("off" if bg is not None else "on")

    if bg is not None:
        ax2.imshow(bg)
    heat, _, _ = np.histogram2d(df["y"], df["x"], bins=[36, 36], range=[[y0, y1], [x0, x1]])
    ax2.imshow(heat, extent=[x0, x1, y1, y0], cmap="inferno",
               alpha=0.65 if bg is not None else 1.0)
    ax2.set_xlim(x0, x1)
    ax2.set_ylim(y1, y0)
    ax2.set_title("Mapa de calor de posiciones", fontsize=9)
    ax2.axis("off")

    if has_cam:
        ax3 = axs[2]
        est = cam[["cam_x", "cam_y"]].to_numpy(float)
        if cam_T is not None and bg is not None:
            ax3.imshow(bg, alpha=0.45)
            est = apply_h(cam_T, est)
        ax3.plot(est[:, 0], est[:, 1], "-", color="#00a0ff", lw=1.5, label="estimada (VO)")
        gt_ok = "gt_x" in cam and cam["gt_x"].notna().any()
        if gt_ok:
            gt = cam[["gt_x", "gt_y"]].to_numpy(float)
            if cam_T is not None and bg is not None:
                gt = apply_h(cam_T, gt)
            ax3.plot(gt[:, 0], gt[:, 1], "--", color="#ff3030", lw=1.2, label="real (simulada)")
        ax3.plot(est[0, 0], est[0, 1], "go", ms=5)
        ax3.set_title("Trayectoria de la cámara\n(centro de la imagen)", fontsize=9)
        if cam_T is not None and bg is not None:
            ax3.axis("off")
        else:
            ax3.invert_yaxis()
        ax3.set_aspect("equal", adjustable="datalim")
        ax3.legend(fontsize=7, loc="best")
        ax3.tick_params(labelsize=7)
    if has3d:
        ax4 = axs[-1]
        c3 = cam[cam["x3d"].notna()]
        # cámara: x der., y abajo, z adelante  ->  gráfico: X, Z (suelo) y -Y (arriba)
        X, Y, Z = c3["x3d"].to_numpy(), c3["z3d"].to_numpy(), -c3["y3d"].to_numpy()
        ax4.plot(X, Y, Z, "-", color="#7b2cbf", lw=1.5)
        ax4.scatter([X[0]], [Y[0]], [Z[0]], color="green", s=20, label="inicio")
        ax4.scatter([X[-1]], [Y[-1]], [Z[-1]], color="red", s=20, label="fin")
        # misma escala en los 3 ejes (si no, el ruido vertical se ve exagerado)
        mid = np.array([X.mean(), Y.mean(), Z.mean()])
        r = max(np.ptp(X), np.ptp(Y), np.ptp(Z), 1e-3) / 2 * 1.1
        ax4.set_xlim(mid[0] - r, mid[0] + r)
        ax4.set_ylim(mid[1] - r, mid[1] + r)
        ax4.set_zlim(mid[2] - r, mid[2] + r)
        ax4.set_xlabel("X", fontsize=7)
        ax4.set_ylabel("Z (adelante)", fontsize=7)
        ax4.set_zlabel("arriba", fontsize=7)
        ax4.tick_params(labelsize=6)
        ax4.set_title("Pose 3D de la cámara (VO 3D)\nescala relativa", fontsize=9)
        ax4.legend(fontsize=6, loc="upper left")
    fig.tight_layout()


# ----------------------------------------------------------------------------
# Métricas de una ejecución y puntaje comparativo
# ----------------------------------------------------------------------------
def vo_metrics(cam):
    out = {"inliers_medios": None, "pct_fallos_vo": None, "error_vo_px": None,
           "error_vo_final_px": None, "desplaz_camara_px": None}
    if cam is None or len(cam) == 0:
        return out
    out["inliers_medios"] = float(cam["inliers"].iloc[1:].mean()) if len(cam) > 1 else 0.0
    out["pct_fallos_vo"] = float((~cam["ok"].astype(bool)).mean())
    est = cam[["cam_x", "cam_y"]].to_numpy(float)
    out["desplaz_camara_px"] = float(np.linalg.norm(np.diff(est, axis=0), axis=1).sum())
    if "gt_x" in cam and cam["gt_x"].notna().all():
        err = np.linalg.norm(est - cam[["gt_x", "gt_y"]].to_numpy(float), axis=1)
        out["error_vo_px"] = float(err.mean())
        out["error_vo_final_px"] = float(err[-1])
    return out


def compute_run_metrics(df, summary, perf, pos, neg, cam=None):
    """Métricas de desempeño (tiempo) y de calidad del tracking (sin ground truth)."""
    n = max(int(perf["frames"]), 1)
    proc_s = float(perf["proc_s"])

    # Velocidad "en régimen": se excluye el 1er frame (calentamiento de CUDA / modelo)
    steady_n = n - 1
    steady_t = proc_s - float(perf["first_proc_s"])
    if steady_n > 0 and steady_t > 0:
        fps = steady_n / steady_t
    else:
        fps = n / proc_s if proc_s > 0 else 0.0
    infer = perf["infer_s"][1:] or perf["infer_s"]
    ms_frame = 1000.0 * float(np.mean(infer)) if infer else 0.0
    x_rt = fps / perf["src_fps"] if perf["src_fps"] > 0 else 0.0

    # Calidad del tracking
    sizes = df.groupby("id").size()
    pct_cortos = float((sizes < SHORT_TRACK).mean()) if len(sizes) else 0.0
    gaps = pairs = 0
    for _, g in df.groupby("id"):
        fr = np.sort(g["frame"].to_numpy())
        if len(fr) > 1:
            d = np.diff(fr)
            gaps += int((d > 1).sum())
            pairs += len(d)
    continuidad = 1.0 - gaps / pairs if pairs > 0 else 1.0

    m = {
        "frames": int(perf["frames"]),
        "tiempo_s": proc_s,
        "ms_frame": ms_frame,
        "fps": fps,
        "x_rt": x_rt,
        "carga_s": float(perf["load_s"]),
        "cruces_pos": int(pos),
        "cruces_neg": int(neg),
        "cruces_total": int(pos + neg),
        "ids_unicos": int(df["id"].nunique()),
        "tracks_validos": int(len(summary)),
        "dur_media_s": float(summary["duracion_s"].mean()) if not summary.empty else 0.0,
        "vel_media_px_s": float(summary["velocidad_px_s"].mean()) if not summary.empty else 0.0,
        "pct_cortos": pct_cortos,
        "continuidad": continuidad,
        "det_por_frame": len(df) / n,
        "conf_media": float(df["conf"].mean()),
    }
    m.update(vo_metrics(cam))
    return m


def compute_scores(runs, w_prec, w_stab, w_speed, gt_cross, gt_objs):
    """
    Puntaje 0-100 = media ponderada de tres componentes (cada una en 0-1):
      - Precisión : cercanía del conteo / de los IDs únicos a los valores reales
                    (solo si el usuario los ingresó; si no, esta componente se omite)
      - Estabilidad: 0.6*(1 - % trayectorias cortas) + 0.4*continuidad
      - Velocidad : FPS / FPS de la ejecución más rápida
    """
    if not runs:
        return []
    max_fps = max(r["metrics"]["fps"] for r in runs) or 1.0
    out = []
    for r in runs:
        m = r["metrics"]
        comps, err_c, err_i = [], None, None
        if gt_cross is not None:
            err_c = abs(m["cruces_total"] - gt_cross)
            comps.append(max(0.0, 1 - err_c / max(gt_cross, 1.0)))
        if gt_objs is not None:
            err_i = abs(m["ids_unicos"] - gt_objs)
            comps.append(max(0.0, 1 - err_i / max(gt_objs, 1.0)))
        prec = sum(comps) / len(comps) if comps else None
        stab = 0.6 * (1 - m["pct_cortos"]) + 0.4 * m["continuidad"]
        speed = m["fps"] / max_fps
        parts = [(w_stab, stab), (w_speed, speed)]
        if prec is not None:
            parts.append((w_prec, prec))
        wsum = sum(w for w, _ in parts)
        score = 100.0 * sum(w * v for w, v in parts) / wsum if wsum > 0 else 0.0
        out.append({"run": r, "score": score, "prec": prec, "stab": stab, "speed": speed,
                    "err_c": err_c, "err_i": err_i})
    out.sort(key=lambda d: -d["score"])
    for i, d in enumerate(out):
        d["rank"] = i + 1
    return out


def _r(v, n=2):
    return None if v is None else round(v, n)


def rows_to_df(rows):
    data = []
    for d in rows:
        r, m = d["run"], d["run"]["metrics"]
        data.append({
            "ranking": d["rank"], "ejecucion": r["label"], "modelo": r["model"],
            "tracker": r["tracker"], "odometria_visual": r["vo"],
            "simulacion": r["sim"], "fuente": r["source"], "confianza": r["conf"],
            "frames": m["frames"], "tiempo_total_s": round(m["tiempo_s"], 2),
            "ms_por_frame": round(m["ms_frame"], 2), "fps": round(m["fps"], 2),
            "x_tiempo_real": round(m["x_rt"], 2), "carga_modelo_s": round(m["carga_s"], 2),
            "cruces_pos": m["cruces_pos"], "cruces_neg": m["cruces_neg"],
            "cruces_total": m["cruces_total"], "error_cruces": d["err_c"],
            "ids_unicos": m["ids_unicos"], "error_ids": d["err_i"],
            "tracks_validos": m["tracks_validos"], "dur_media_s": round(m["dur_media_s"], 2),
            "vel_media_px_s": round(m["vel_media_px_s"], 1),
            "pct_tracks_cortos": round(m["pct_cortos"], 3),
            "continuidad": round(m["continuidad"], 4),
            "det_por_frame": round(m["det_por_frame"], 2),
            "conf_media": round(m["conf_media"], 3),
            "inliers_medios_vo": _r(m["inliers_medios"], 1),
            "pct_fallos_vo": _r(m["pct_fallos_vo"], 3),
            "error_vo_medio_px": _r(m["error_vo_px"]),
            "error_vo_final_px": _r(m["error_vo_final_px"]),
            "dispositivo": r["device"],
            "puntaje": round(d["score"], 2),
            "score_precision": None if d["prec"] is None else round(d["prec"], 3),
            "score_estabilidad": round(d["stab"], 3), "score_velocidad": round(d["speed"], 3),
        })
    return pd.DataFrame(data)


def draw_comparison(fig, rows, gt_cross, gt_objs):
    fig.clear()
    if not rows:
        fig.text(0.5, 0.5, "Sin ejecuciones todavía", ha="center", va="center", color="#666")
        return
    axs = fig.subplots(2, 2).ravel()
    labels = [d["run"]["label"].replace(" + ", "\n") for d in rows]
    colors = [TRACKER_COLORS.get(d["run"]["tracker"], "#888888") for d in rows]
    hatches = ["//" if d["run"]["vo"] else "" for d in rows]
    x = np.arange(len(rows))

    def bars(ax, vals, title, fmt="%.1f", ref=None):
        b = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        for patch, h in zip(b, hatches):
            patch.set_hatch(h)
        ax.bar_label(b, fmt=fmt, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6)
        ax.tick_params(axis="y", labelsize=7)
        if ref is not None:
            ax.axhline(ref, color="red", ls="--", lw=1, label=f"real = {ref:g}")
            ax.legend(fontsize=6, loc="upper right")

    bars(axs[0], [d["score"] for d in rows], "Puntaje global (0-100)")
    bars(axs[1], [d["run"]["metrics"]["fps"] for d in rows], "Velocidad (FPS, sin 1er frame)")
    bars(axs[2], [d["run"]["metrics"]["cruces_total"] for d in rows],
         "Cruces totales de la línea", "%.0f", gt_cross)
    bars(axs[3], [d["run"]["metrics"]["ids_unicos"] for d in rows],
         "IDs únicos (menos = menos fragmentación)", "%.0f", gt_objs)

    trackers = sorted({d["run"]["tracker"] for d in rows})
    handles = [Patch(facecolor=TRACKER_COLORS.get(t, "#888888"), label=t) for t in trackers]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="con VO"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])


def draw_traj_grid(fig, rows):
    fig.clear()
    if not rows:
        fig.text(0.5, 0.5, "Sin ejecuciones todavía", ha="center", va="center", color="#666")
        return
    n = len(rows)
    cols = min(3, n)
    nr = int(np.ceil(n / cols))
    axs = fig.subplots(nr, cols, squeeze=False)
    cmap = matplotlib.colormaps["tab20"]
    for i, ax in enumerate(axs.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        run = rows[i]["run"]
        df, bg, line_px, _ = make_view(run)
        summary = run["summary"]
        if bg is not None:
            ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        x0, y0, x1, y1 = _limits(df, bg)
        valid = set(summary["id"]) if not summary.empty else set()
        for tid, g in df.groupby("id"):
            if tid in valid:
                ax.plot(g["x"], g["y"], "-", color=cmap(int(tid) % 20), linewidth=1.0)
        if line_px is not None:
            ax.plot(line_px[:, 0], line_px[:, 1], "r-", linewidth=1.5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"#{rows[i]['rank']} " + run["label"].replace(" + sim", "\nsim") +
                     f" · {len(valid)} tray.", fontsize=7)
    fig.tight_layout()


# ----------------------------------------------------------------------------
# Hilo de procesamiento (YOLO + tracker + odometría visual)
# ----------------------------------------------------------------------------
class Worker(threading.Thread):
    def __init__(self, app, cfg):
        super().__init__(daemon=True)
        self.app, self.cfg = app, cfg
        self.stop_ev = threading.Event()
        self.pause_ev = threading.Event()
        # Estado leído por la interfaz
        self.counts_pos = self.counts_neg = 0
        self.counted_ids, self.unique_ids = set(), set()
        self.active = 0
        self.fps = 0.0
        self.frame_idx = 0
        self.total_frames = 0
        self.records = []
        self.cam_records = []
        self.bg_frame = None
        self.frame_size = None
        self.mosaic, self.mosaic_T = None, None
        self.line_world = None
        self.cam_text = ""
        self.k_text = ""
        # Medición de tiempos
        self.infer_s = []          # tiempo de model.track() por frame
        self.proc_time = 0.0       # tiempo de procesamiento total (sin pausas ni esperas)
        self.first_proc_s = 0.0
        self.load_s = 0.0
        self.device = "desconocido"
        self.line_used = tuple(app.line_rel)
        self.conf_used = app.conf
        self.perf = None

    def status(self, text):
        self.app.events.put(("status", text))

    def run(self):
        try:
            self._loop()
        except Exception as e:  # se muestra en la interfaz
            self.app.events.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            self.app.events.put(("finished", None))

    def _loop(self):
        from ultralytics import YOLO

        cfg, app = self.cfg, self.app
        bench = bool(cfg.get("benchmark"))
        max_frames = int(cfg.get("max_frames") or 0)
        use_vo = bool(cfg.get("vo"))
        sim_k = float(cfg.get("sim") or 0.0)

        self.status(f"Cargando {cfg['model']} (la primera vez puede descargarse)…")
        try:
            import torch
            self.device = (f"GPU ({torch.cuda.get_device_name(0)})"
                           if torch.cuda.is_available() else "CPU")
        except Exception:
            pass
        t_load = time.perf_counter()
        model = YOLO(cfg["model"])
        self.load_s = time.perf_counter() - t_load
        names = model.names
        tracker_cfg = "bytetrack.yaml" if cfg["tracker"] == "bytetrack" else "botsort.yaml"
        is_obb = getattr(model, "task", "detect") == "obb"
        classes = cfg["classes"]
        if is_obb and classes:   # traduce las clases COCO elegidas a las de DOTA
            mapped = sorted({d for c in classes for d in DOTA_FROM_COCO.get(c, [])})
            if not mapped:
                self.status("El modelo aéreo (OBB) no tiene «persona»: se siguen vehículos.")
            classes = mapped or [9, 10]

        cap = cv2.VideoCapture(cfg["source"])
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {cfg['source']}")
        is_webcam = isinstance(cfg["source"], int)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        src_fps = src_fps if src_fps and src_fps > 1 else 30.0
        n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not is_webcam and n_src > 0:
            self.total_frames = min(n_src, max_frames) if max_frames else n_src
        elif max_frames:
            self.total_frames = max_frames

        writer = None
        trails = defaultdict(lambda: deque(maxlen=300))    # en coordenadas del MUNDO
        hist = defaultdict(lambda: deque(maxlen=int(src_fps * 1.5)))  # para la velocidad
        last_pos = {}
        sim = None
        motion = VO_MOTIONS.get(cfg.get("vo_motion"), "similarity")
        # En el modo 3D, la compensación del tracking sigue siendo 2D (similaridad): para
        # llevar un píxel al mundo en 3D haría falta su profundidad, que no se conoce.
        vo = VisualOdometry2D("similarity" if motion == "3d" else motion) if use_vo else None
        vo3d = None
        self.k_text = ""
        R = np.eye(3)            # mundo -> imagen de referencia (donde se define la línea)
        P = np.eye(3)            # imagen actual -> mundo
        prev_boxes = []
        keyframes, kf_every, kf_scale = [], 10, 0.5
        t_start = t_prev = time.perf_counter()
        self.status("Procesando…")

        try:
            while not self.stop_ev.is_set():
                if self.pause_ev.is_set():
                    time.sleep(0.03)
                    continue
                if max_frames and self.frame_idx >= max_frames:
                    break
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                self.frame_idx += 1
                time_s = (time.perf_counter() - t_start) if is_webcam else self.frame_idx / src_fps

                # --- Cámara en movimiento simulada (opcional)
                A_sim = None
                if sim_k > 0:
                    if sim is None:
                        sim = CameraMotionSimulator(frame.shape[1], frame.shape[0], sim_k)
                        ref_size = (frame.shape[1], frame.shape[0])
                    frame, A_sim = sim.apply(frame, self.frame_idx / src_fps)
                    if self.frame_idx == 1:
                        R = sim.A0.copy()   # la línea se define sobre el video original
                H, W = frame.shape[:2]
                if self.frame_size is None:
                    self.frame_size = (W, H)
                    if sim is None:
                        ref_size = (W, H)
                if self.bg_frame is None:
                    self.bg_frame = frame.copy()

                if app.reset_event.is_set():
                    app.reset_event.clear()
                    self.counts_pos = self.counts_neg = 0
                    self.counted_ids.clear()

                # --- Odometría visual: movimiento de la cámara (actual -> mundo)
                vo_info = None
                if vo is not None and motion == "3d" and vo3d is None:
                    K, dist = self._camera_matrix(cfg, W, H)
                    vo3d = VisualOdometry3D(K, dist)
                if vo is not None:
                    P, vo_info = vo.update(frame, prev_boxes)
                    P = P.copy()
                    if self.frame_idx == 1 or self.frame_idx % kf_every == 0:
                        small = cv2.resize(frame, None, fx=kf_scale, fy=kf_scale,
                                           interpolation=cv2.INTER_AREA)
                        keyframes.append((small, P.copy(), kf_scale))
                        if len(keyframes) > 200:   # limita memoria (webcam o videos largos)
                            keyframes = keyframes[::2]
                            kf_every *= 2
                    cx_c, cy_c, rot, sc = camera_pose(P, W, H)
                    pose3d, txt3d = (None,) * 7, ""
                    if vo3d is not None:
                        R3, C3, i3 = vo3d.update(frame, prev_boxes)
                        pi, ya, ro = rot_to_euler(R3)
                        pose3d = (round(float(C3[0]), 4), round(float(C3[1]), 4),
                                  round(float(C3[2]), 4), round(pi, 2), round(ya, 2),
                                  round(ro, 2), i3["mode"])
                        txt3d = (f"\nPose 3D [{i3['mode']}]: pos=({C3[0]:+.2f}, {C3[1]:+.2f}, "
                                 f"{C3[2]:+.2f}) u.rel. · yaw={ya:+.1f}° pitch={pi:+.1f}° "
                                 f"roll={ro:+.1f}°")
                    gt = (None, None)
                    if A_sim is not None:
                        gx, gy = apply_h(sim.gt_cur_to_world(A_sim), [[W / 2.0, H / 2.0]])[0]
                        gt = (round(float(gx), 2), round(float(gy), 2))
                    self.cam_records.append((self.frame_idx, round(time_s, 3), round(cx_c, 2),
                                             round(cy_c, 2), round(rot, 3), round(sc, 4),
                                             vo_info["inliers"], bool(vo_info["ok"]), *gt,
                                             *pose3d))
                    dx, dy = cx_c - W / 2.0, cy_c - H / 2.0
                    self.cam_text = (f"Cámara: dx={dx:+.0f} dy={dy:+.0f} px · rot={rot:+.1f}° · "
                                     f"esc={sc:.2f} · inliers={vo_info['inliers']} · "
                                     f"keyframes={vo_info['keyframes']}"
                                     + ("" if vo_info["ok"] else " · ¡VO PERDIDA!") + txt3d)

                # --- Línea: definida en la imagen de referencia -> mundo -> imagen actual
                line = cfg.get("line") or app.line_rel
                self.line_used = tuple(line)
                lx1, ly1, lx2, ly2 = line
                RW, RH = ref_size
                L_ref = np.array([[lx1 * RW, ly1 * RH], [lx2 * RW, ly2 * RH]])
                L_w = apply_h(np.linalg.inv(R), L_ref)
                self.line_world = L_w
                a, b = tuple(L_w[0]), tuple(L_w[1])
                Pinv = np.linalg.inv(P)
                L_img = apply_h(Pinv, L_w)
                with app.lock:
                    app.line_disp_rel = [L_img[0, 0] / W, L_img[0, 1] / H,
                                         L_img[1, 0] / W, L_img[1, 1] / H]
                    app.disp_to_ref = (R @ P, (W, H), ref_size)

                conf = cfg["fixed_conf"] if cfg.get("fixed_conf") is not None else app.conf
                self.conf_used = conf

                ti = time.perf_counter()
                res = model.track(frame, persist=True, tracker=tracker_cfg, conf=conf,
                                  classes=classes, imgsz=cfg["imgsz"], verbose=False)[0]
                self.infer_s.append(time.perf_counter() - ti)

                disp = frame.copy()
                active_ids = []
                prev_boxes = []

                boxes = res.obb if is_obb else res.boxes   # OBB = cajas rotadas (vista aérea)
                if boxes is not None and len(boxes):
                    prev_boxes = boxes.xyxy.cpu().numpy().tolist()
                if boxes is not None and boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.int().cpu().tolist()
                    clss = boxes.cls.int().cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    polys = (boxes.xyxyxyxy.cpu().numpy() if is_obb else [None] * len(ids))
                    centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]
                                        for x1, y1, x2, y2 in xyxy], float)
                    world = apply_h(P, centers) if vo is not None else centers
                    for (x1, y1, x2, y2), tid, c, cf, (cx, cy), (xw, yw), poly in zip(
                            xyxy, ids, clss, confs, centers, world, polys):
                        col = color_for_id(tid)
                        active_ids.append(tid)
                        self.unique_ids.add(tid)

                        # Velocidad en tiempo real (en el mundo, px/s, ventana ~0.5 s)
                        hq = hist[tid]
                        hq.append((time_s, xw, yw))
                        v = None
                        j = 0
                        while j < len(hq) - 1 and time_s - hq[j][0] > 0.5:
                            j += 1
                        if j < len(hq) - 1 and time_s - hq[j][0] > 0.15:
                            v = float(np.hypot(xw - hq[j][1], yw - hq[j][2]) / (time_s - hq[j][0]))
                        self.records.append((tid, self.frame_idx, round(time_s, 3),
                                             int(cx), int(cy), round(float(xw), 1),
                                             round(float(yw), 1),
                                             None if v is None else round(v, 1),
                                             names[c], round(cf, 3)))
                        trails[tid].append((xw, yw))

                        # Conteo: cruce de la línea en coordenadas del MUNDO
                        if tid in last_pos:
                            d = crossed_line(last_pos[tid], (xw, yw), a, b)
                            if d == 1:
                                self.counts_pos += 1
                                self.counted_ids.add(tid)
                            elif d == -1:
                                self.counts_neg += 1
                                self.counted_ids.add(tid)
                        last_pos[tid] = (xw, yw)

                        if poly is not None:
                            cv2.polylines(disp, [poly.astype(np.int32)], True, col, 2)
                        else:
                            cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                        label = f"ID {tid} {names[c]} {cf:.2f}"
                        if v is not None:
                            label += f" | {v:.0f} px/s"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(disp, (int(x1), int(y1) - th - 6),
                                      (int(x1) + tw + 4, int(y1)), col, -1)
                        cv2.putText(disp, label, (int(x1) + 2, int(y1) - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.circle(disp, (int(cx), int(cy)), 4, col, -1)

                # Estelas: guardadas en el mundo y re-proyectadas a la imagen actual,
                # así quedan "pegadas" al suelo aunque la cámara se mueva
                n_trail = int(app.trail_len)
                if n_trail > 1:
                    for tid in active_ids:
                        pts = list(trails[tid])[-n_trail:]
                        if len(pts) > 1:
                            pi = apply_h(Pinv, pts) if vo is not None else np.array(pts)
                            cv2.polylines(disp, [pi.astype(np.int32)], False,
                                          color_for_id(tid), 2, cv2.LINE_AA)

                # Puntos del fondo usados por la odometría (inliers)
                if vo_info is not None and app.show_vo_pts and vo_info["pts"] is not None:
                    for px, py in vo_info["pts"][::2]:
                        cv2.circle(disp, (int(px), int(py)), 2, (0, 255, 0), -1)
                if vo_info is not None:
                    self._hud(disp, to_ascii(self.cam_text))
                elif sim is not None:
                    self._hud(disp, "Camara en movimiento SIN compensar (item 2)")

                self.active = len(active_ids)

                # Frame para la interfaz (sin la línea: la dibuja la GUI encima)
                with app.lock:
                    app.latest_frame = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

                # Video anotado (aquí sí se dibuja la línea y los contadores)
                if cfg.get("save_video"):
                    if writer is None:
                        os.makedirs(cfg["out_dir"], exist_ok=True)
                        writer = cv2.VideoWriter(
                            os.path.join(cfg["out_dir"], "resultado.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))
                    rec = disp.copy()
                    ia, ib = L_img.astype(int)
                    cv2.line(rec, tuple(ia), tuple(ib), (0, 0, 255), 3)
                    cv2.putText(rec, f"(+): {self.counts_pos}  (-): {self.counts_neg}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    writer.write(rec)

                # Tiempo de procesamiento de este frame (sin pausas ni esperas)
                dt = time.perf_counter() - t0
                self.proc_time += dt
                if self.frame_idx == 1:
                    self.first_proc_s = dt
                now = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now

                # Reproducción a velocidad original solo en ejecuciones individuales
                if not is_webcam and not bench:
                    wait = 1.0 / src_fps - dt
                    if wait > 0:
                        time.sleep(wait)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if keyframes and self.frame_size:
                self.status("Construyendo el mosaico de la escena…")
                try:
                    self.mosaic, self.mosaic_T = build_mosaic(keyframes, self.frame_size)
                except Exception:
                    self.mosaic, self.mosaic_T = None, None
            self.perf = {"frames": self.frame_idx, "proc_s": self.proc_time,
                         "first_proc_s": self.first_proc_s, "infer_s": self.infer_s,
                         "load_s": self.load_s, "device": self.device, "src_fps": src_fps}

    @staticmethod
    def _hud(img, text):
        y = 5
        for line in text.split("\n"):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (5, y), (15 + tw, y + 12 + th), (0, 0, 0), -1)
            cv2.putText(img, line, (10, y + 5 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            y += 12 + th

    def _camera_matrix(self, cfg, W, H):
        """K de la calibración (escalada al tamaño del video) o aproximada por el FOV."""
        path = cfg.get("calib")
        if path:
            try:
                K, dist, size = load_calibration(path)
                K = K.copy()
                note = ""
                if size and tuple(size) != (W, H):
                    K[0] *= W / float(size[0])
                    K[1] *= H / float(size[1])
                    note = f" (escalada de {size[0]}x{size[1]})"
                elif not size and abs(K[0, 2] - W / 2) > 0.25 * W:
                    note = " (¡ojo! parece de otra resolución)"
                self.k_text = f"K de {os.path.basename(path)}{note}: f={K[0, 0]:.0f} px"
                return K, dist
            except Exception as e:
                self.status(f"No se pudo leer la calibración ({e}); se usa K aproximada.")
        fov = float(cfg.get("fov") or 65)
        K = approx_K(W, H, fov)
        self.k_text = f"K aproximada (FOV {fov:.0f}°): f={K[0, 0]:.0f} px"
        return K, None


# ----------------------------------------------------------------------------
# Aplicación (interfaz)
# ----------------------------------------------------------------------------
COMP_COLS = [("run", "Ejecución", 270), ("frames", "Frames", 55), ("tiempo", "Tiempo (s)", 70),
             ("ms", "ms/frame", 65), ("fps", "FPS", 55), ("rt", "×T.real", 60),
             ("cruces", "Cruces", 55), ("err_c", "Err. cruces", 75), ("ids", "IDs únicos", 70),
             ("err_i", "Err. IDs", 60), ("dur", "Dur. media (s)", 90), ("cortos", "% cortos", 65),
             ("cont", "Continuidad", 80), ("vel", "Vel. media px/s", 95),
             ("vo_err", "Err. VO (px)", 80), ("score", "Puntaje", 65), ("rank", "#", 35)]


class App(tk.Tk):
    CLASSES = [("Persona", 0), ("Bicicleta", 1), ("Auto", 2), ("Moto", 3),
               ("Bus", 5), ("Camión", 7)]
    DEFAULT_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt",
                      "yolo11n-obb.pt", "yolo11s-obb.pt"]

    def __init__(self):
        super().__init__()
        self.title("Ítem 4 · Tracking con cámara en movimiento · YOLO + tracker + odometría visual")
        self.geometry("1440x900")
        self.minsize(1150, 700)

        # Estado compartido con el hilo
        self.lock = threading.Lock()
        self.events = queue.Queue()
        self.latest_frame = None
        self.reset_event = threading.Event()
        self.conf = 0.30
        self.trail_len = 40
        self.line_rel = [0.0, 0.5, 1.0, 0.5]  # x1,y1,x2,y2 relativos a la imagen de referencia
        self.line_disp_rel = None             # la misma línea proyectada en el frame mostrado
        self.disp_to_ref = None               # (matriz frame mostrado -> referencia, tamaños)
        self.show_vo_pts = True
        self.worker = None

        # Estado de la interfaz
        self.out_dir = os.path.abspath("salida")
        self.video_path = find_default_video()
        self.drawing_line = False
        self.click_pts = []
        self.show_hint = True
        self.img_rect = None
        self.photo = None
        self.last_rgb = np.zeros((480, 640, 3), np.uint8)
        self.records, self.bg_frame = [], None
        self.last_res, self.last_summary = None, None
        self.config_widgets = []

        # Comparación
        self.runs = []
        self.run_counter = 0
        self.cmp_rows = []
        self.bench_queue = deque()
        self.bench_active = False
        self.bench_cancelled = False
        self.bench_errors = []
        self.bench_done = 0
        self.bench_total = 0
        self.bench_params = None
        self._wjob = None

        self._build_ui()
        if self.video_path:
            self.source_mode.set("video")
            self.video_lbl.config(text=os.path.basename(self.video_path) + "  (del proyecto)",
                                  foreground="#000")
            self._preview_first_frame()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, lambda: self._show_frame(self.last_rgb))
        self.after(20, self._poll)
        self._refresh_comparison()

    # ------------------------------------------------------------------ UI
    def _build_left_scroll(self):
        """Panel izquierdo con barra de desplazamiento (hay muchas opciones)."""
        holder = ttk.Frame(self)
        holder.pack(side="left", fill="y")
        cv = tk.Canvas(holder, highlightthickness=0, width=340)
        sb = ttk.Scrollbar(holder, orient="vertical", command=cv.yview)
        inner = ttk.Frame(cv, padding=8)
        cv.create_window((0, 0), window=inner, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="y")
        sb.pack(side="left", fill="y")

        def on_conf(_e):
            cv.configure(scrollregion=cv.bbox("all"), width=inner.winfo_reqwidth())
        inner.bind("<Configure>", on_conf)

        def wheel(e):
            if e.num == 4 or getattr(e, "delta", 0) > 0:
                cv.yview_scroll(-2, "units")
            else:
                cv.yview_scroll(2, "units")
        for w in (cv, inner):
            w.bind("<Enter>", lambda e: (cv.bind_all("<MouseWheel>", wheel),
                                         cv.bind_all("<Button-4>", wheel),
                                         cv.bind_all("<Button-5>", wheel)))
            w.bind("<Leave>", lambda e: (cv.unbind_all("<MouseWheel>"),
                                         cv.unbind_all("<Button-4>"),
                                         cv.unbind_all("<Button-5>")))
        return inner

    def _build_ui(self):
        # Barra de estado (se crea primero para que reserve su espacio abajo)
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value="Listo. Elige una fuente y presiona «Iniciar».")
        ttk.Label(bottom, textvariable=self.status, relief="sunken", anchor="w",
                  padding=3).pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bottom, length=220, mode="determinate", maximum=100)
        self.progress.pack(side="right", padx=4)

        left = self._build_left_scroll()
        right = ttk.Frame(self, padding=(0, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)

        # --- Fuente
        f = ttk.LabelFrame(left, text="1. Fuente", padding=6)
        f.pack(fill="x", pady=3)
        self.source_mode = tk.StringVar(value="webcam")
        self.cam_index = tk.IntVar(value=0)
        r1 = ttk.Radiobutton(f, text="Webcam, índice:", variable=self.source_mode, value="webcam")
        r1.grid(row=0, column=0, sticky="w")
        sp = ttk.Spinbox(f, from_=0, to=9, width=4, textvariable=self.cam_index)
        sp.grid(row=0, column=1, sticky="w", padx=4)
        r2 = ttk.Radiobutton(f, text="Archivo de video", variable=self.source_mode, value="video")
        r2.grid(row=1, column=0, sticky="w", pady=(4, 0))
        b = ttk.Button(f, text="Abrir…", command=self._choose_video)
        b.grid(row=1, column=1, sticky="w", padx=4, pady=(4, 0))
        self.video_lbl = ttk.Label(f, text="(ningún video)", foreground="#666", width=36)
        self.video_lbl.grid(row=2, column=0, columnspan=2, sticky="w")
        self.config_widgets += [(r1, "normal"), (sp, "normal"), (r2, "normal"), (b, "normal")]

        # --- Modelo y tracker
        f = ttk.LabelFrame(left, text="2. Modelo y tracker (ejecución individual)", padding=6)
        f.pack(fill="x", pady=3)
        self.model_name = tk.StringVar(value="yolov8n.pt")
        self.tracker_name = tk.StringVar(value="bytetrack")
        ttk.Label(f, text="Modelo YOLO:").grid(row=0, column=0, sticky="w")
        cb_model = ttk.Combobox(f, textvariable=self.model_name, width=18,
                                values=self.DEFAULT_MODELS)
        cb_model.grid(row=0, column=1, padx=4, pady=2)
        b2 = ttk.Button(f, text="…", width=3, command=self._choose_weights)
        b2.grid(row=0, column=2)
        ttk.Label(f, text="Tracker:").grid(row=1, column=0, sticky="w")
        cb_trk = ttk.Combobox(f, textvariable=self.tracker_name, width=18, state="readonly",
                              values=["bytetrack", "botsort"])
        cb_trk.grid(row=1, column=1, padx=4, pady=2)
        ttk.Label(f, text="Tamaño (imgsz):").grid(row=2, column=0, sticky="w")
        self.imgsz = tk.StringVar(value="640")
        cb_sz = ttk.Combobox(f, textvariable=self.imgsz, width=18, values=IMG_SIZES)
        cb_sz.grid(row=2, column=1, padx=4, pady=2)
        ttk.Label(f, foreground="#666", justify="left", wraplength=310,
                  text="Video aéreo/dron: usa yolo11n-obb.pt (entrenado con imágenes aéreas) "
                       "con imgsz 1024. Objetos pequeños: sube el imgsz."
                  ).grid(row=3, column=0, columnspan=3, sticky="w")
        self.config_widgets += [(cb_model, "normal"), (b2, "normal"), (cb_trk, "readonly"),
                                (cb_sz, "normal")]

        # --- Cámara en movimiento (ítem 4)
        f = ttk.LabelFrame(left, text="3. Cámara en movimiento (ítem 4)", padding=6)
        f.pack(fill="x", pady=3)
        self.vo_on = tk.BooleanVar(value=True)
        c1 = ttk.Checkbutton(f, text="Compensar con odometría visual (VO)", variable=self.vo_on)
        c1.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(f, text="Modelo de movimiento:").grid(row=1, column=0, sticky="w")
        self.vo_motion = tk.StringVar(value="Similaridad (4 GL)")
        cb_m = ttk.Combobox(f, textvariable=self.vo_motion, width=24, state="readonly",
                            values=list(VO_MOTIONS))
        cb_m.grid(row=1, column=1, padx=4, pady=2, sticky="w")
        # Calibración (para el modo 3D)
        self.calib_path = ""
        b_cal = ttk.Button(f, text="Calibración…", command=self._choose_calib)
        b_cal.grid(row=2, column=0, sticky="w", pady=2)
        fr = ttk.Frame(f)
        fr.grid(row=2, column=1, sticky="w")
        ttk.Label(fr, text="o FOV (°):").pack(side="left")
        self.fov = tk.StringVar(value="65")
        sp_fov = ttk.Spinbox(fr, from_=20, to=150, increment=5, width=5, textvariable=self.fov)
        sp_fov.pack(side="left", padx=3)
        self.calib_lbl = ttk.Label(f, text="Modo 3D: sin calibración → K aproximada por el FOV",
                                   foreground="#666", wraplength=310, justify="left")
        self.calib_lbl.grid(row=3, column=0, columnspan=2, sticky="w")
        self.config_widgets += [(b_cal, "normal"), (sp_fov, "normal")]
        ttk.Label(f, text="Simular cámara móvil:").grid(row=4, column=0, sticky="w")
        self.sim_level = tk.StringVar(value="Desactivada")
        cb_s = ttk.Combobox(f, textvariable=self.sim_level, width=17, state="readonly",
                            values=list(SIM_LEVELS))
        cb_s.grid(row=4, column=1, padx=4, pady=2, sticky="w")
        self.show_pts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Mostrar puntos del fondo usados (verde)",
                        variable=self.show_pts_var,
                        command=lambda: setattr(self, "show_vo_pts", self.show_pts_var.get())
                        ).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Label(f, foreground="#666", justify="left", wraplength=310,
                  text="La simulación mueve una ventana (paneo, giro, zoom y temblor) sobre "
                       "el video; así se conoce el movimiento real y se mide el error de la VO."
                  ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.config_widgets += [(c1, "normal"), (cb_m, "readonly"), (cb_s, "readonly")]

        # --- Clases
        f = ttk.LabelFrame(left, text="4. Objetos a seguir", padding=6)
        f.pack(fill="x", pady=3)
        self.class_vars = {}
        for i, (name, cid) in enumerate(self.CLASSES):
            v = tk.BooleanVar(value=(cid == 0))
            self.class_vars[cid] = v
            c = ttk.Checkbutton(f, text=name, variable=v)
            c.grid(row=i // 2, column=i % 2, sticky="w", padx=4)
            self.config_widgets.append((c, "normal"))
        ttk.Label(f, text="(sin marcar = todas las clases)", foreground="#666").grid(
            row=3, column=0, columnspan=2, sticky="w")

        # --- Parámetros en vivo
        f = ttk.LabelFrame(left, text="5. Parámetros", padding=6)
        f.pack(fill="x", pady=3)
        self.conf_lbl = ttk.Label(f, text="Confianza: 0.30")
        self.conf_lbl.pack(anchor="w")
        ttk.Scale(f, from_=0.05, to=0.95, value=0.30, command=self._on_conf).pack(fill="x")
        self.trail_lbl = ttk.Label(f, text="Largo de estela: 40")
        self.trail_lbl.pack(anchor="w", pady=(4, 0))
        ttk.Scale(f, from_=0, to=150, value=40, command=self._on_trail).pack(fill="x")
        self.save_video = tk.BooleanVar(value=False)
        c = ttk.Checkbutton(f, text="Guardar video anotado (salida/resultado.mp4)",
                            variable=self.save_video)
        c.pack(anchor="w", pady=(4, 0))
        self.config_widgets.append((c, "normal"))

        # --- Control
        f = ttk.LabelFrame(left, text="6. Control", padding=6)
        f.pack(fill="x", pady=3)
        self.btn_start = ttk.Button(f, text="▶  Iniciar", command=self._start)
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self.pause_txt = tk.StringVar(value="⏸  Pausar")
        self.btn_pause = ttk.Button(f, textvariable=self.pause_txt, command=self._toggle_pause,
                                    state="disabled")
        self.btn_pause.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        self.btn_stop = ttk.Button(f, text="⏹  Detener", command=self._stop, state="disabled")
        self.btn_stop.grid(row=0, column=2, sticky="ew", padx=2, pady=2)
        self.btn_line = ttk.Button(f, text="✏  Dibujar línea", command=self._toggle_draw)
        self.btn_line.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        self.btn_reset = ttk.Button(f, text="↺ Reiniciar", command=self._reset_counts)
        self.btn_reset.grid(row=1, column=2, sticky="ew", padx=2, pady=2)
        for i in range(3):
            f.columnconfigure(i, weight=1)

        # --- Contadores
        f = ttk.LabelFrame(left, text="Contadores", padding=6)
        f.pack(fill="x", pady=3)
        big = ("Helvetica", 20, "bold")
        self.lbl_pos = tk.StringVar(value="0")
        self.lbl_neg = tk.StringVar(value="0")
        ttk.Label(f, text="Cruces hacia lado (+)").grid(row=0, column=0, sticky="w")
        ttk.Label(f, textvariable=self.lbl_pos, font=big, foreground="#0a7").grid(
            row=0, column=1, padx=12)
        ttk.Label(f, text="Cruces hacia lado (−)").grid(row=1, column=0, sticky="w")
        ttk.Label(f, textvariable=self.lbl_neg, font=big, foreground="#c60").grid(
            row=1, column=1, padx=12)
        self.lbl_info = ttk.Label(f, text="Total: 0 | Activos: 0 | IDs únicos: 0 | FPS: 0.0")
        self.lbl_info.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.lbl_cam = ttk.Label(f, text="", foreground="#06c", wraplength=310, justify="left")
        self.lbl_cam.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(f, text="(+) = lado derecho yendo de A hacia B", foreground="#666").grid(
            row=4, column=0, columnspan=2, sticky="w")

        ttk.Button(left, text="Analizar trayectorias", command=self._analyze_current).pack(
            fill="x", pady=(8, 0))

        # --- Derecha: pestañas
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        tab_video = ttk.Frame(self.nb)
        tab_an = ttk.Frame(self.nb)
        tab_cmp = ttk.Frame(self.nb)
        self.nb.add(tab_video, text="  Video  ")
        self.nb.add(tab_an, text="  Análisis de trayectorias  ")
        self.nb.add(tab_cmp, text="  Comparación  ")

        self.canvas = tk.Canvas(tab_video, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._show_frame(self.last_rgb))
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self._build_analysis_tab(tab_an)
        self._build_compare_tab(tab_cmp)

    def _build_analysis_tab(self, parent):
        bar = ttk.Frame(parent, padding=4)
        bar.pack(fill="x")
        ttk.Button(bar, text="Cargar CSV…", command=self._load_csv).pack(side="left", padx=2)
        ttk.Button(bar, text="Exportar CSV…", command=self._export_csv).pack(side="left", padx=2)
        ttk.Button(bar, text="Exportar resumen…", command=self._export_summary).pack(
            side="left", padx=2)
        ttk.Button(bar, text="Guardar gráfico…", command=self._save_fig).pack(side="left", padx=2)

        self.fig = Figure(figsize=(11, 4.6), dpi=100)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.fig_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.stats_lbl = ttk.Label(parent, text="Sin datos todavía. Ejecuta el tracking y "
                                   "presiona «Analizar trayectorias».", padding=4,
                                   wraplength=900, justify="left")
        self.stats_lbl.pack(fill="x")
        tf = ttk.Frame(parent)
        tf.pack(fill="x")
        self.tree = ttk.Treeview(tf, columns=SUMMARY_COLS, show="headings", height=6)
        for col in SUMMARY_COLS:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=95, anchor="center")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")

    def _build_compare_tab(self, parent):
        cfgf = ttk.LabelFrame(parent, padding=6,
                              text="Configuraciones a comparar (se ejecutan sobre el video de «1. Fuente» "
                                   "con la simulación elegida en «3»)")
        cfgf.pack(fill="x", padx=4, pady=4)

        # Fila 1: modelos
        r1 = ttk.Frame(cfgf)
        r1.pack(fill="x")
        ttk.Label(r1, text="Modelos:").pack(side="left")
        self.bench_models = {}
        self.models_frame = ttk.Frame(r1)
        self.models_frame.pack(side="left")
        for m in self.DEFAULT_MODELS:
            self._add_model_check(m, m == "yolov8n.pt")
        ttk.Button(r1, text="+ Agregar .pt…", command=self._add_model_file).pack(
            side="left", padx=6)

        # Fila 2: trackers, VO, frames, valores reales
        r2 = ttk.Frame(cfgf)
        r2.pack(fill="x", pady=3)
        ttk.Label(r2, text="Trackers:").pack(side="left")
        self.bench_trackers = {"bytetrack": tk.BooleanVar(value=True),
                               "botsort": tk.BooleanVar(value=True)}
        ttk.Checkbutton(r2, text="ByteTrack", variable=self.bench_trackers["bytetrack"]).pack(
            side="left", padx=3)
        ttk.Checkbutton(r2, text="BoT-SORT", variable=self.bench_trackers["botsort"]).pack(
            side="left", padx=3)
        ttk.Label(r2, text="   Odometría visual:").pack(side="left")
        self.bench_vo = {False: tk.BooleanVar(value=True), True: tk.BooleanVar(value=True)}
        ttk.Checkbutton(r2, text="sin VO", variable=self.bench_vo[False]).pack(side="left", padx=3)
        ttk.Checkbutton(r2, text="con VO", variable=self.bench_vo[True]).pack(side="left", padx=3)
        ttk.Label(r2, text="   Frames máx. (0 = todo):").pack(side="left")
        self.max_frames = tk.IntVar(value=300)
        ttk.Spinbox(r2, from_=0, to=1000000, increment=50, width=7,
                    textvariable=self.max_frames).pack(side="left", padx=3)

        r2b = ttk.Frame(cfgf)
        r2b.pack(fill="x", pady=2)
        ttk.Label(r2b, text="Cruces reales (opc.):").pack(side="left")
        self.gt_cross = tk.StringVar()
        e1 = ttk.Entry(r2b, width=6, textvariable=self.gt_cross)
        e1.pack(side="left", padx=3)
        ttk.Label(r2b, text="Objetos reales (opc.):").pack(side="left")
        self.gt_objs = tk.StringVar()
        e2 = ttk.Entry(r2b, width=6, textvariable=self.gt_objs)
        e2.pack(side="left", padx=3)
        ttk.Label(r2b, foreground="#666",
                  text="  Tip: los cruces del video SIN simulación sirven como «reales»."
                  ).pack(side="left")
        for e in (e1, e2):
            e.bind("<FocusOut>", lambda ev: self._refresh_comparison())
            e.bind("<Return>", lambda ev: self._refresh_comparison())

        # Fila 3: pesos del puntaje
        r3 = ttk.Frame(cfgf)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Pesos del puntaje →").pack(side="left")
        self.w_vars = {"prec": tk.DoubleVar(value=40), "stab": tk.DoubleVar(value=35),
                       "speed": tk.DoubleVar(value=25)}
        self.w_lbls = {}
        for key, text in (("prec", "Precisión"), ("stab", "Estabilidad"), ("speed", "Velocidad")):
            ttk.Label(r3, text=text + ":").pack(side="left", padx=(10, 2))
            ttk.Scale(r3, from_=0, to=100, variable=self.w_vars[key], length=110,
                      command=lambda v, k=key: self._on_weights(k)).pack(side="left")
            lbl = ttk.Label(r3, text=str(int(self.w_vars[key].get())), width=4)
            lbl.pack(side="left")
            self.w_lbls[key] = lbl

        # Fila 4: botones
        r4 = ttk.Frame(cfgf)
        r4.pack(fill="x", pady=(4, 0))
        self.btn_bench = ttk.Button(r4, text="▶  Ejecutar comparación", command=self._start_benchmark)
        self.btn_bench.pack(side="left", padx=2)
        ttk.Button(r4, text="Quitar seleccionada", command=self._remove_selected_run).pack(
            side="left", padx=2)
        ttk.Button(r4, text="Limpiar todo", command=self._clear_runs).pack(side="left", padx=2)
        ttk.Button(r4, text="Exportar…", command=self._export_comparison).pack(side="left", padx=2)
        self.auto_accumulate = tk.BooleanVar(value=True)
        ttk.Checkbutton(r4, text="Sumar a la comparación cada ejecución individual",
                        variable=self.auto_accumulate).pack(side="left", padx=12)

        # Tabla
        tf = ttk.Frame(parent)
        tf.pack(fill="x", padx=4)
        self.cmp_tree = ttk.Treeview(tf, columns=[c[0] for c in COMP_COLS], show="headings",
                                     height=6)
        for key, title, wd in COMP_COLS:
            self.cmp_tree.heading(key, text=title)
            self.cmp_tree.column(key, width=wd, anchor="w" if key == "run" else "center",
                                 stretch=False)
        vs = ttk.Scrollbar(tf, orient="vertical", command=self.cmp_tree.yview)
        hs = ttk.Scrollbar(tf, orient="horizontal", command=self.cmp_tree.xview)
        self.cmp_tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.cmp_tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        tf.columnconfigure(0, weight=1)

        self.cmp_info = ttk.Label(parent, text="", wraplength=950, justify="left", padding=4)
        self.cmp_info.pack(fill="x", padx=4)

        # Gráficos
        self.cmp_nb = ttk.Notebook(parent)
        self.cmp_nb.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        t1, t2 = ttk.Frame(self.cmp_nb), ttk.Frame(self.cmp_nb)
        self.cmp_nb.add(t1, text="  Métricas  ")
        self.cmp_nb.add(t2, text="  Trayectorias lado a lado  ")
        self.fig_cmp = Figure(figsize=(10, 4.2), dpi=100)
        self.cv_cmp = FigureCanvasTkAgg(self.fig_cmp, master=t1)
        self.cv_cmp.get_tk_widget().pack(fill="both", expand=True)
        self.fig_trj = Figure(figsize=(10, 4.2), dpi=100)
        self.cv_trj = FigureCanvasTkAgg(self.fig_trj, master=t2)
        self.cv_trj.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------- Eventos de UI
    def _on_conf(self, v):
        self.conf = float(v)
        self.conf_lbl.config(text=f"Confianza: {self.conf:.2f}")

    def _on_trail(self, v):
        self.trail_len = int(float(v))
        self.trail_lbl.config(text=f"Largo de estela: {self.trail_len}")

    def _choose_video(self):
        p = filedialog.askopenfilename(
            title="Selecciona un video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Todos", "*.*")])
        if p:
            self.video_path = p
            self.video_lbl.config(text=os.path.basename(p), foreground="#000")
            self.source_mode.set("video")
            self._preview_first_frame()

    def _preview_first_frame(self):
        """Muestra el 1er frame del video para poder dibujar la línea antes de iniciar."""
        cap = cv2.VideoCapture(self.video_path)
        ok, fr = cap.read()
        cap.release()
        self.line_disp_rel, self.disp_to_ref = None, None
        if ok:
            self.last_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            self._show_frame(self.last_rgb)

    def _choose_weights(self):
        p = filedialog.askopenfilename(title="Selecciona pesos YOLO (.pt)",
                                       filetypes=[("Pesos PyTorch", "*.pt"), ("Todos", "*.*")])
        if p:
            self.model_name.set(p)

    # ------------------------------------------------ Dibujo de la línea
    def _toggle_draw(self):
        self.drawing_line = not self.drawing_line
        self.click_pts = []
        self.btn_line.config(text="✖ Cancelar dibujo" if self.drawing_line else "✏  Dibujar línea")
        self.nb.select(0)
        self._redraw_overlay()

    def _disp_to_ref_rel(self, fx, fy):
        """Punto relativo del frame mostrado -> relativo a la imagen de referencia."""
        with self.lock:
            m = self.disp_to_ref
        if m is None:
            return fx, fy
        M, (W, H), (RW, RH) = m
        x, y = apply_h(M, [[fx * W, fy * H]])[0]
        return x / RW, y / RH

    def _on_canvas_click(self, e):
        if not self.drawing_line or not self.img_rect:
            return
        ox, oy, dw, dh = self.img_rect
        fx = min(max((e.x - ox) / dw, 0.0), 1.0)
        fy = min(max((e.y - oy) / dh, 0.0), 1.0)
        self.click_pts.append((fx, fy))
        if len(self.click_pts) == 2:
            (x1, y1), (x2, y2) = [self._disp_to_ref_rel(*p) for p in self.click_pts]
            self.line_rel = [x1, y1, x2, y2]
            with self.lock:
                self.line_disp_rel = None if self.disp_to_ref is None else [
                    self.click_pts[0][0], self.click_pts[0][1],
                    self.click_pts[1][0], self.click_pts[1][1]]
            self.click_pts = []
            self.drawing_line = False
            self.btn_line.config(text="✏  Dibujar línea")
            self._reset_counts()
        self._redraw_overlay()

    def _reset_counts(self):
        self.reset_event.set()
        self.lbl_pos.set("0")
        self.lbl_neg.set("0")

    # ------------------------------------------------ Mostrar video
    def _show_frame(self, rgb):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        h, w = rgb.shape[:2]
        s = min(cw / w, ch / h)
        dw, dh = max(int(w * s), 1), max(int(h * s), 1)
        img = Image.fromarray(rgb).resize((dw, dh), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(img)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.canvas.delete("img")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo, tags="img")
        self.canvas.tag_lower("img")
        self.img_rect = (ox, oy, dw, dh)
        self._redraw_overlay()

    def _redraw_overlay(self):
        c = self.canvas
        c.delete("ov")
        if not self.img_rect:
            return
        ox, oy, dw, dh = self.img_rect

        def P(fx, fy):
            return ox + fx * dw, oy + fy * dh

        with self.lock:
            ln = self.line_disp_rel or self.line_rel
        x1, y1 = P(ln[0], ln[1])
        x2, y2 = P(ln[2], ln[3])
        c.create_line(x1, y1, x2, y2, fill="#ff3030", width=3, tags="ov")
        for (x, y, t) in ((x1, y1, "A"), (x2, y2, "B")):
            c.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ffd400", outline="", tags="ov")
            c.create_text(x + 12, y - 12, text=t, fill="#ffd400",
                          font=("Helvetica", 12, "bold"), tags="ov")
        for fx, fy in self.click_pts:
            x, y = P(fx, fy)
            c.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ff40ff", outline="", tags="ov")
        msg = None
        if self.drawing_line:
            msg = ("Haz clic en el punto A" if not self.click_pts
                   else "Ahora haz clic en el punto B")
        elif self.show_hint:
            msg = "Elige la fuente y presiona ▶ Iniciar"
        if msg:
            c.create_text(ox + dw / 2, oy + 24, text=msg, fill="white",
                          font=("Helvetica", 14, "bold"), tags="ov")

    # ------------------------------------------------ Control del hilo
    def _set_running(self, running):
        for w, st in self.config_widgets:
            w.configure(state="disabled" if running else st)
        self.btn_start.config(state="disabled" if running else "normal")
        self.btn_bench.config(state="disabled" if running else "normal")
        self.btn_pause.config(state="normal" if running else "disabled")
        self.btn_stop.config(state="normal" if running else "disabled")
        lock_line = running and self.bench_active  # en la comparación la línea queda fija
        self.btn_line.config(state="disabled" if lock_line else "normal")
        self.btn_reset.config(state="disabled" if lock_line else "normal")
        self.pause_txt.set("⏸  Pausar")

    def _fov(self):
        try:
            return min(max(float(self.fov.get()), 10.0), 170.0)
        except (tk.TclError, ValueError):
            return 65.0

    def _choose_calib(self):
        p = filedialog.askopenfilename(
            title="Archivo de calibración de la cámara",
            filetypes=[("Calibración", "*.npz *.json *.yml *.yaml *.xml"), ("Todos", "*.*")])
        if not p:
            return
        try:
            K, dist, size = load_calibration(p)
        except Exception as e:
            messagebox.showerror("Calibración inválida", str(e))
            return
        self.calib_path = p
        res = f" · {size[0]}x{size[1]}" if size else " · resolución no indicada"
        self.calib_lbl.config(
            text=f"Calibración: {os.path.basename(p)} · fx={K[0, 0]:.0f}, fy={K[1, 1]:.0f}{res}. "
                 "Solo es válida si el video se grabó con ESA cámara.", foreground="#060")
        self.vo_motion.set("3D: matriz esencial (6 GL)")

    def _imgsz(self):
        try:
            return max(32, int(float(self.imgsz.get())) // 32 * 32)
        except (tk.TclError, ValueError):
            return 640

    def _selected_classes(self):
        return [cid for cid, v in self.class_vars.items() if v.get()] or None

    def _start(self):
        if self.worker is not None:
            return
        if self.source_mode.get() == "video":
            if not self.video_path:
                messagebox.showwarning("Falta el video", "Primero elige un archivo de video.")
                return
            source = self.video_path
        else:
            try:
                source = int(self.cam_index.get())
            except (tk.TclError, ValueError):
                source = 0
        cfg = {"source": source, "model": self.model_name.get().strip() or "yolov8n.pt",
               "tracker": self.tracker_name.get(), "classes": self._selected_classes(),
               "imgsz": self._imgsz(), "save_video": self.save_video.get(), "out_dir": self.out_dir,
               "benchmark": False, "max_frames": 0, "fixed_conf": None, "line": None,
               "vo": self.vo_on.get(), "vo_motion": self.vo_motion.get(),
               "calib": self.calib_path, "fov": self._fov(),
               "sim": SIM_LEVELS.get(self.sim_level.get(), 0.0),
               "sim_name": self.sim_level.get()}
        self._launch(cfg)
        self.nb.select(0)

    def _launch(self, cfg):
        self.reset_event.clear()
        self.show_hint = False
        with self.lock:
            self.line_disp_rel, self.disp_to_ref = None, None
        self.lbl_cam.config(text="")
        self.worker = Worker(self, cfg)
        self._set_running(True)
        self.worker.start()

    def _toggle_pause(self):
        w = self.worker
        if not w:
            return
        if w.pause_ev.is_set():
            w.pause_ev.clear()
            self.pause_txt.set("⏸  Pausar")
        else:
            w.pause_ev.set()
            self.pause_txt.set("▶  Continuar")
            self.status.set("En pausa.")

    def _stop(self):
        if self.worker:
            if self.bench_active:  # cancelar toda la comparación
                self.bench_queue.clear()
                self.bench_cancelled = True
            self.worker.stop_ev.set()
            self.worker.pause_ev.clear()

    # ------------------------------------------------ Bucle de la GUI
    def _poll(self):
        with self.lock:
            fr, self.latest_frame = self.latest_frame, None
        if fr is not None:
            self.last_rgb = fr
            self._show_frame(fr)

        w = self.worker
        if w is not None:
            self.lbl_pos.set(str(w.counts_pos))
            self.lbl_neg.set(str(w.counts_neg))
            self.lbl_info.config(
                text=f"Total: {w.counts_pos + w.counts_neg} | Activos: {w.active} | "
                     f"IDs únicos: {len(w.unique_ids)} | FPS: {w.fps:.1f}")
            self.lbl_cam.config(text=w.cam_text + (f"\n{w.k_text}" if w.k_text else ""))
            tot = w.total_frames
            self.progress.configure(value=(100.0 * w.frame_idx / tot) if tot > 0 else 0)
            if w.frame_idx > 0 and not w.pause_ev.is_set():
                pre = (f"Comparación {self.bench_done + 1}/{self.bench_total} · "
                       if self.bench_active else "")
                prog = f"frame {w.frame_idx}/{tot}" if tot else f"frame {w.frame_idx}"
                self.status.set(f"{pre}{self._cfg_label(w.cfg)} · {prog} · {w.fps:.1f} FPS")

        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.set(data)
            elif kind == "error":
                if self.bench_active and self.worker is not None:
                    self.bench_errors.append(f"{self._cfg_label(self.worker.cfg)}: {data}")
                else:
                    self.status.set("Error: " + data)
                    messagebox.showerror("Error", data)
            elif kind == "finished":
                self._on_finished()
        self.after(15, self._poll)

    @staticmethod
    def _cfg_label(cfg):
        t = f"{os.path.basename(cfg['model'])} + {cfg['tracker']}"
        if cfg.get("imgsz", 640) != 640:
            t += f" @{cfg['imgsz']}"
        if cfg.get("sim"):
            t += f" + sim {cfg.get('sim_name', '')}".rstrip()
        if cfg.get("vo"):
            t += " + VO3D" if VO_MOTIONS.get(cfg.get("vo_motion")) == "3d" else " + VO"
        return t

    def _result_of(self, w):
        """Agrupa lo producido por una ejecución (para analizar y guardar)."""
        df = pd.DataFrame(w.records, columns=CSV_COLUMNS)
        cam = pd.DataFrame(w.cam_records, columns=CAM_COLUMNS) if w.cam_records else None
        return {"df": df, "vo": bool(w.cfg.get("vo")), "bg": w.bg_frame,
                "mosaic": w.mosaic, "mosaic_T": w.mosaic_T, "line_world": w.line_world,
                "cam": cam}

    def _on_finished(self):
        w = self.worker
        self.worker = None
        self._set_running(False)
        self.progress.configure(value=0)
        if w is None:
            return
        was_bench = bool(w.cfg.get("benchmark"))

        if was_bench and self.bench_cancelled:
            self.bench_active = self.bench_cancelled = False
            self.status.set("Comparación cancelada (la ejecución en curso se descartó; "
                            "las ya terminadas se conservan).")
            return

        self.records, self.bg_frame = list(w.records), w.bg_frame

        run = None
        if self.records and (was_bench or self.auto_accumulate.get()):
            run = self._make_run(w)
            if run:
                self.runs.append(run)
                self._persist_run(run)

        if was_bench:
            self.bench_done += 1
            if self.bench_queue:
                self._start_next_bench()
                return
            self.bench_active = False
            self._refresh_comparison(save=True)
            self.nb.select(2)
            self.status.set(f"Comparación terminada: {len(self.runs)} ejecuciones acumuladas. "
                            f"Resultados en {os.path.join(self.out_dir, 'comparacion')}")
            if self.bench_errors:
                messagebox.showwarning("Algunas configuraciones fallaron",
                                       "\n\n".join(self.bench_errors))
            return

        total = w.counts_pos + w.counts_neg
        msg = (f"Finalizado · cruces (+): {w.counts_pos} | (−): {w.counts_neg} | "
               f"total: {total} | IDs únicos: {len(w.unique_ids)}")
        if self.records:
            res = self._result_of(w)
            self._save_result(res, self.out_dir)
            msg += f" · {len(res['df'])} puntos guardados en {self.out_dir}"
            if run:
                msg += " · sumada a la comparación"
                self._refresh_comparison(save=True)
            self.status.set(msg)
            self._show_analysis(res)
        elif w.cam_records:   # sin objetos, pero sí trayectoria de la cámara
            res = self._result_of(w)
            self._save_result(res, self.out_dir)
            self.status.set(msg + " · no se detectaron objetos (se muestra solo la cámara)")
            self._show_analysis(res)
        else:
            self.status.set(msg + " · no se detectaron objetos")

    @staticmethod
    def _save_result(res, d):
        os.makedirs(d, exist_ok=True)
        res["df"].to_csv(os.path.join(d, "trayectorias.csv"), index=False)
        if res["bg"] is not None:
            cv2.imwrite(os.path.join(d, "fondo.jpg"), res["bg"])
        for f in ("camara.csv", "mosaico.jpg", "mosaico.json"):  # evita mezclar con corridas viejas
            p = os.path.join(d, f)
            if os.path.exists(p):
                os.remove(p)
        if res["cam"] is not None:
            res["cam"].to_csv(os.path.join(d, "camara.csv"), index=False)
        if res["mosaic"] is not None:
            cv2.imwrite(os.path.join(d, "mosaico.jpg"), res["mosaic"])
            with open(os.path.join(d, "mosaico.json"), "w", encoding="utf-8") as fh:
                json.dump({"T_mundo_a_mosaico": np.asarray(res["mosaic_T"]).tolist(),
                           "linea_mundo": None if res["line_world"] is None
                           else np.asarray(res["line_world"]).tolist()}, fh, indent=1)

    # ------------------------------------------------ Análisis (una ejecución)
    def _analyze_current(self):
        w = self.worker
        if w is not None:
            if not w.records:
                messagebox.showinfo("Sin datos", "Aún no hay trayectorias.")
                return
            res = self._result_of(w)
            res["df"] = pd.DataFrame(list(w.records), columns=CSV_COLUMNS)
            self._show_analysis(res)
            return
        if self.last_res is None:
            messagebox.showinfo("Sin datos", "Aún no hay trayectorias. Ejecuta el tracking primero.")
            return
        self._show_analysis(self.last_res)

    def _show_analysis(self, res):
        df = res["df"]
        if df.empty and res.get("cam") is None:
            messagebox.showinfo("Sin datos", "El CSV no contiene trayectorias.")
            return
        summary = compute_summary(metric_df(df, res["vo"]))
        self.last_res, self.last_summary = res, summary
        d_plot, bg, line_px, T = make_view(res)
        draw_analysis(self.fig, d_plot, summary, bg, line_px, res.get("cam"), T, res["vo"])
        self.fig_canvas.draw()

        self.tree.delete(*self.tree.get_children())
        for _, r in summary.iterrows():
            self.tree.insert("", "end", values=[r[c] for c in SUMMARY_COLS])
        coord = "MUNDO (compensadas con VO)" if res["vo"] else "imagen"
        if summary.empty:
            txt = (f"IDs únicos: {df['id'].nunique()} · ninguna trayectoria con "
                   f"≥ {MIN_POINTS} puntos.")
        else:
            txt = (f"Coordenadas: {coord} · IDs únicos: {df['id'].nunique()} · "
                   f"analizados (≥{MIN_POINTS} pts): {len(summary)} · "
                   f"duración media: {summary['duracion_s'].mean():.2f} s · "
                   f"distancia media: {summary['distancia_px'].mean():.0f} px · "
                   f"velocidad media: {summary['velocidad_px_s'].mean():.0f} px/s · "
                   f"rectitud media: {summary['rectitud'].mean():.2f}")
        vm = vo_metrics(res.get("cam"))
        if vm["inliers_medios"] is not None:
            txt += (f"\nOdometría visual · inliers medios: {vm['inliers_medios']:.0f} · "
                    f"frames sin estimación: {100 * vm['pct_fallos_vo']:.1f}% · "
                    f"recorrido de la cámara: {vm['desplaz_camara_px']:.0f} px")
            if vm["error_vo_px"] is not None:
                txt += (f" · error vs. movimiento real: medio {vm['error_vo_px']:.1f} px, "
                        f"final (deriva) {vm['error_vo_final_px']:.1f} px")
        cam = res.get("cam")
        if cam is not None and "x3d" in cam and cam["x3d"].notna().any():
            c3 = cam[cam["x3d"].notna()]
            last = c3.iloc[-1]
            pos = c3[["x3d", "y3d", "z3d"]].to_numpy(float)
            modos = c3["modo3d"].value_counts(normalize=True)
            txt += (f"\nPose 3D final: pos=({last['x3d']:+.2f}, {last['y3d']:+.2f}, "
                    f"{last['z3d']:+.2f}) u.rel. · yaw={last['yaw_deg']:+.1f}° "
                    f"pitch={last['pitch_deg']:+.1f}° roll={last['roll_deg']:+.1f}° · "
                    f"recorrido 3D: {np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.2f} u.rel. · "
                    "modelo usado: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in modos.items()))
        self.stats_lbl.config(text=txt)
        self.nb.select(1)

    def _load_csv(self):
        p = filedialog.askopenfilename(title="Cargar trayectorias",
                                       filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if not p:
            return
        try:
            df = pd.read_csv(p)
            missing = set(BASE_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"Faltan columnas: {', '.join(sorted(missing))}")
        except Exception as e:
            messagebox.showerror("CSV inválido", str(e))
            return
        d = os.path.dirname(p)
        vo = "xw" in df.columns and not (np.allclose(df["xw"], df["x"]) and
                                         np.allclose(df["yw"], df["y"]))
        if "xw" not in df.columns:
            df["xw"], df["yw"] = df["x"], df["y"]
        bgp = os.path.join(d, "fondo.jpg")
        res = {"df": df, "vo": vo, "bg": cv2.imread(bgp) if os.path.exists(bgp) else None,
               "mosaic": None, "mosaic_T": None, "line_world": None, "cam": None}
        mj, mi = os.path.join(d, "mosaico.json"), os.path.join(d, "mosaico.jpg")
        if vo and os.path.exists(mj) and os.path.exists(mi):
            with open(mj, encoding="utf-8") as fh:
                meta = json.load(fh)
            res["mosaic"], res["mosaic_T"] = cv2.imread(mi), np.array(meta["T_mundo_a_mosaico"])
            if meta.get("linea_mundo") is not None:
                res["line_world"] = np.array(meta["linea_mundo"])
        cp = os.path.join(d, "camara.csv")
        if vo and os.path.exists(cp):
            res["cam"] = pd.read_csv(cp)
        self._show_analysis(res)

    def _export_csv(self):
        if self.last_res is None:
            messagebox.showinfo("Sin datos", "No hay trayectorias para exportar.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="trayectorias.csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            self.last_res["df"].to_csv(p, index=False)
            self.status.set(f"Trayectorias exportadas a {p}")

    def _export_summary(self):
        if self.last_res is None:
            messagebox.showinfo("Sin datos", "No hay resumen para exportar.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         initialfile="resumen_trayectorias.csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            self.last_summary.to_csv(p, index=False)
            self.status.set(f"Resumen exportado a {p}")

    def _save_fig(self):
        if self.last_res is None:
            messagebox.showinfo("Sin datos", "No hay gráfico para guardar.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         initialfile="analisis_trayectorias.png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            self.fig.savefig(p, dpi=130)
            self.status.set(f"Gráfico guardado en {p}")

    # ------------------------------------------------ Comparación: configuración
    def _add_model_check(self, name, checked=True):
        if name in self.bench_models:
            self.bench_models[name].set(True)
            return
        v = tk.BooleanVar(value=checked)
        self.bench_models[name] = v
        ttk.Checkbutton(self.models_frame, text=os.path.basename(name), variable=v).pack(
            side="left", padx=3)

    def _add_model_file(self):
        p = filedialog.askopenfilename(title="Selecciona pesos YOLO (.pt)",
                                       filetypes=[("Pesos PyTorch", "*.pt"), ("Todos", "*.*")])
        if p:
            self._add_model_check(p, True)

    def _on_weights(self, key):
        self.w_lbls[key].config(text=str(int(self.w_vars[key].get())))
        if self._wjob:
            self.after_cancel(self._wjob)
        self._wjob = self.after(250, lambda: self._refresh_comparison(save=False))

    def _start_benchmark(self):
        if self.worker is not None:
            return
        if not self.video_path:
            messagebox.showwarning(
                "Falta el video",
                "La comparación necesita un video (todas las configuraciones deben procesar "
                "exactamente lo mismo).\n\nEn «1. Fuente» elige «Archivo de video» y presiona «Abrir…».")
            return
        models = [m for m, v in self.bench_models.items() if v.get()]
        trackers = [t for t, v in self.bench_trackers.items() if v.get()]
        vos = [k for k, v in self.bench_vo.items() if v.get()]
        if not models or not trackers or not vos:
            messagebox.showwarning("Faltan opciones",
                                   "Marca al menos un modelo, un tracker y una opción de VO.")
            return
        try:
            max_frames = max(int(self.max_frames.get()), 0)
        except (tk.TclError, ValueError):
            max_frames = 0
        self.bench_queue = deque((m, t, v) for m in models for t in trackers for v in vos)
        self.bench_total = len(self.bench_queue)
        self.bench_done = 0
        self.bench_errors = []
        self.bench_cancelled = False
        self.bench_active = True
        # Condiciones congeladas: todas las configuraciones usan exactamente lo mismo
        self.bench_params = {"source": self.video_path, "conf": self.conf,
                             "classes": self._selected_classes(),
                             "line": list(self.line_rel), "max_frames": max_frames,
                             "sim": SIM_LEVELS.get(self.sim_level.get(), 0.0),
                             "sim_name": self.sim_level.get(),
                             "vo_motion": self.vo_motion.get(), "imgsz": self._imgsz(),
                             "calib": self.calib_path, "fov": self._fov()}
        self.nb.select(0)
        self._start_next_bench()

    def _start_next_bench(self):
        model, tracker, vo = self.bench_queue.popleft()
        p = self.bench_params
        cfg = {"source": p["source"], "model": model, "tracker": tracker,
               "classes": p["classes"], "imgsz": p["imgsz"], "save_video": False,
               "out_dir": self.out_dir, "benchmark": True, "max_frames": p["max_frames"],
               "fixed_conf": p["conf"], "line": p["line"], "vo": vo,
               "vo_motion": p["vo_motion"], "sim": p["sim"], "sim_name": p["sim_name"],
               "calib": p["calib"], "fov": p["fov"]}
        self._launch(cfg)

    # ------------------------------------------------ Comparación: datos acumulados
    def _unique_label(self, base):
        n = sum(1 for r in self.runs if r["label"] == base or r["label"].startswith(base + " ("))
        return base if n == 0 else f"{base} ({n + 1})"

    def _make_run(self, w):
        if w.perf is None or not w.records:
            return None
        res = self._result_of(w)
        summary = compute_summary(metric_df(res["df"], res["vo"]))
        metrics = compute_run_metrics(res["df"], summary, w.perf, w.counts_pos, w.counts_neg,
                                      res["cam"])
        src = w.cfg["source"]
        self.run_counter += 1
        res.update({
            "id": self.run_counter,
            "label": self._unique_label(self._cfg_label(w.cfg)),
            "model": os.path.basename(w.cfg["model"]), "tracker": w.cfg["tracker"],
            "sim": w.cfg.get("sim_name", "Desactivada") if w.cfg.get("sim") else "Desactivada",
            "source": f"webcam {src}" if isinstance(src, int) else os.path.basename(src),
            "conf": round(float(w.conf_used), 2),
            "classes": tuple(w.cfg["classes"]) if w.cfg["classes"] else (),
            "line": tuple(round(float(v), 2) for v in w.line_used),
            "summary": summary, "metrics": metrics, "device": w.perf["device"],
        })
        return res

    def _persist_run(self, run):
        d = os.path.join(self.out_dir, "comparacion",
                         f"{run['id']:02d}_{safe_name(run['label'])}")
        self._save_result(run, d)
        run["summary"].to_csv(os.path.join(d, "resumen_trayectorias.csv"), index=False)

    def _remove_selected_run(self):
        sel = {int(i) for i in self.cmp_tree.selection()}
        if not sel:
            messagebox.showinfo("Selecciona una fila", "Selecciona en la tabla la ejecución a quitar.")
            return
        self.runs = [r for r in self.runs if r["id"] not in sel]
        self._refresh_comparison(save=True)

    def _clear_runs(self):
        if self.runs and messagebox.askyesno("Limpiar comparación",
                                             "¿Quitar todas las ejecuciones acumuladas?"):
            self.runs = []
            self._refresh_comparison(save=False)

    @staticmethod
    def _parse_gt(var):
        s = var.get().strip().replace(",", ".")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _conditions_text(self):
        if len(self.runs) < 2:
            return ""
        keys = {"source": "el video", "conf": "la confianza", "classes": "las clases",
                "line": "la línea", "sim": "la simulación de cámara"}
        diff = [name for k, name in keys.items() if len({r[k] for r in self.runs}) > 1]
        if diff:
            return ("ATENCIÓN: las ejecuciones no comparten " + ", ".join(diff) +
                    " (si es a propósito, p. ej. cámara fija vs. simulada, está bien). ")
        return ""

    def _info_text(self, rows, gt_c, gt_o):
        if not rows:
            return ("Marca modelos, trackers y opciones de VO, elige un video en «1. Fuente» y "
                    "presiona «Ejecutar comparación». Cada ejecución se acumula aquí.")
        best = rows[0]
        fast = max(rows, key=lambda d: d["run"]["metrics"]["fps"])
        stab = max(rows, key=lambda d: d["stab"])
        t = (f"Mejor puntaje: {best['run']['label']} ({best['score']:.1f}) · "
             f"Más rápido: {fast['run']['label']} ({fast['run']['metrics']['fps']:.1f} FPS) · "
             f"Más estable: {stab['run']['label']} ({stab['stab']:.2f}).  ")
        if gt_c is None and gt_o is None:
            t += ("Sin valores reales, el puntaje solo mide estabilidad y velocidad "
                  "(no se puede evaluar la precisión). ")
        t += self._conditions_text()
        devices = ", ".join(sorted({r["device"] for r in self.runs}))
        t += (f"Dispositivo: {devices}. FPS y ms/frame excluyen el primer frame "
              "(calentamiento). Nota: BoT-SORT ya trae compensación de movimiento propia (GMC) "
              "para asociar IDs; la VO de este programa además corrige el conteo, las "
              "trayectorias y la velocidad.")
        return t

    def _refresh_comparison(self, save=False):
        gt_c, gt_o = self._parse_gt(self.gt_cross), self._parse_gt(self.gt_objs)
        rows = compute_scores(self.runs, self.w_vars["prec"].get(), self.w_vars["stab"].get(),
                              self.w_vars["speed"].get(), gt_c, gt_o)
        self.cmp_rows = rows

        self.cmp_tree.delete(*self.cmp_tree.get_children())
        for d in rows:
            r, m = d["run"], d["run"]["metrics"]
            vals = [r["label"], m["frames"], f"{m['tiempo_s']:.1f}", f"{m['ms_frame']:.1f}",
                    f"{m['fps']:.1f}", f"{m['x_rt']:.2f}", m["cruces_total"],
                    "—" if d["err_c"] is None else f"{d['err_c']:g}", m["ids_unicos"],
                    "—" if d["err_i"] is None else f"{d['err_i']:g}",
                    f"{m['dur_media_s']:.2f}", f"{m['pct_cortos'] * 100:.0f}%",
                    f"{m['continuidad']:.3f}", f"{m['vel_media_px_s']:.0f}",
                    "—" if m["error_vo_px"] is None else f"{m['error_vo_px']:.1f}",
                    f"{d['score']:.1f}", d["rank"]]
            self.cmp_tree.insert("", "end", iid=str(r["id"]), values=vals)

        self.cmp_info.config(text=self._info_text(rows, gt_c, gt_o))
        draw_comparison(self.fig_cmp, rows, gt_c, gt_o)
        draw_traj_grid(self.fig_trj, rows)
        self.cv_cmp.draw()
        self.cv_trj.draw()

        if save and rows:
            d = os.path.join(self.out_dir, "comparacion")
            os.makedirs(d, exist_ok=True)
            rows_to_df(rows).to_csv(os.path.join(d, "comparacion_metricas.csv"), index=False)

    def _export_comparison(self):
        if not self.cmp_rows:
            messagebox.showinfo("Sin datos", "Aún no hay ejecuciones para exportar.")
            return
        d = filedialog.askdirectory(title="Carpeta donde exportar la comparación")
        if not d:
            return
        rows_to_df(self.cmp_rows).to_csv(os.path.join(d, "comparacion_metricas.csv"), index=False)
        for r in self.runs:
            name = f"{r['id']:02d}_{safe_name(r['label'])}"
            r["df"].to_csv(os.path.join(d, f"trayectorias_{name}.csv"), index=False)
            if r.get("cam") is not None:
                r["cam"].to_csv(os.path.join(d, f"camara_{name}.csv"), index=False)
        self.fig_cmp.savefig(os.path.join(d, "comparacion_metricas.png"), dpi=130)
        self.fig_trj.savefig(os.path.join(d, "comparacion_trayectorias.png"), dpi=130)
        self.status.set(f"Comparación exportada en {d}")

    # ------------------------------------------------ Cierre
    def _on_close(self):
        if self.worker:
            self.bench_queue.clear()
            self.worker.stop_ev.set()
            self.worker.pause_ev.clear()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
