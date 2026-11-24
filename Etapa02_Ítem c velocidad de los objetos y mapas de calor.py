#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocidad en tiempo real, mapas de calor y conversión a metros por segundo
==========================================================================
Actividad 07 · Etapa 02 · Ítem 3

Qué hace
  1. Sigue personas/vehículos con YOLO + ByteTrack / BoT-SORT sobre un video.
  2. Calcula la velocidad de cada ID en tiempo real (px/s y, si hay
     calibración, m/s y km/h).
  3. Calibra la escala real con una HOMOGRAFÍA (4 puntos del suelo con medidas
     reales) o con una ESCALA SIMPLE (2 puntos + distancia).
  4. Genera mapas de calor: posiciones y VELOCIDAD MEDIA POR ZONA.
  5. También analiza un CSV de trayectorias ya guardado (id, frame, x, y ...).

Instalación
    pip install ultralytics opencv-python numpy pandas matplotlib pillow

Ejecución
    python velocidad_mapas_calor.py
"""

import json
import os
import queue
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
from PIL import Image, ImageTk  # noqa: E402

# ----------------------------------------------------------------------------
# Paleta y constantes
# ----------------------------------------------------------------------------
BG, PANEL, CARD, BORDER = "#0B1220", "#0F1A30", "#16233F", "#28406B"
TEXT, MUTED = "#E8EEF8", "#93A4C3"
ACCENT, AMBER, DANGER = "#2DD4BF", "#F5B84B", "#F87171"
FONT = "Segoe UI"

MIN_POINTS = 5          # puntos mínimos para analizar un ID
CELL = 40               # tamaño de celda (px) de los mapas de zonas
V_MAX_PLAUSIBLE = 15.0  # m/s: por encima se considera error de tracking
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibracion.json")
CLASES = [("Persona", 0), ("Bicicleta", 1), ("Auto", 2), ("Moto", 3), ("Bus", 5), ("Camión", 7)]


# ----------------------------------------------------------------------------
# Calibración píxeles -> metros
# ----------------------------------------------------------------------------
class Calibration:
    """Convierte coordenadas de imagen (px) a metros sobre el plano del suelo."""

    def __init__(self):
        self.mode = "homografia"      # "homografia" | "escala"
        self.pts = []                 # puntos clicados (px de la imagen original)
        self.width_m, self.length_m, self.dist_m = 6.0, 4.0, 1.0
        self.H = None
        self.scale = None             # metros por píxel (modo escala simple)
        self.size = None              # (ancho, alto) del video

    @property
    def ready(self):
        return self.H is not None or self.scale is not None

    def reset(self):
        self.pts, self.H, self.scale = [], None, None

    def build(self):
        if self.mode == "homografia":
            if len(self.pts) != 4:
                raise ValueError("Marca 4 puntos en la imagen (arriba-izq, arriba-der, abajo-der, abajo-izq).")
            if self.width_m <= 0 or self.length_m <= 0:
                raise ValueError("El ancho y el largo deben ser mayores que 0.")
            src = np.float32(self.pts)
            dst = np.float32([[0, 0], [self.width_m, 0], [self.width_m, self.length_m], [0, self.length_m]])
            self.H = cv2.getPerspectiveTransform(src, dst)
            self.scale = None
        else:
            if len(self.pts) != 2:
                raise ValueError("Marca 2 puntos y escribe la distancia real entre ellos.")
            d = float(np.hypot(*(np.subtract(self.pts[1], self.pts[0]))))
            if d < 1 or self.dist_m <= 0:
                raise ValueError("Los puntos están muy juntos o la distancia es inválida.")
            self.scale = self.dist_m / d
            self.H = None

    def to_world(self, xy):
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        if self.H is not None:
            return cv2.perspectiveTransform(xy.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        if self.scale is not None:
            return xy * self.scale
        return xy

    def to_dict(self):
        return {"mode": self.mode, "pts": self.pts, "width_m": self.width_m,
                "length_m": self.length_m, "dist_m": self.dist_m, "size": self.size}

    def from_dict(self, d):
        self.mode = d.get("mode", "homografia")
        self.pts = [tuple(p) for p in d.get("pts", [])]
        self.width_m = float(d.get("width_m", 6.0))
        self.length_m = float(d.get("length_m", 4.0))
        self.dist_m = float(d.get("dist_m", 1.0))
        self.size = tuple(d["size"]) if d.get("size") else None
        self.build()


# ----------------------------------------------------------------------------
# Velocidades (análisis completo, sobre todas las trayectorias)
# ----------------------------------------------------------------------------
def prepare_tracks(df, fps):
    """Normaliza el CSV: asegura time_s, fx (x del pie) y fy (y del pie)."""
    df = df.copy()
    if "time_s" not in df.columns:
        df["time_s"] = df["frame"] / fps
    if "class" not in df.columns:
        df["class"] = "objeto"
    df["fx"] = df["x"].astype(float)
    has_feet = "y2" in df.columns and df["y2"].notna().any()
    df["fy"] = df["y2"].astype(float) if has_feet else df["y"].astype(float)
    return df, has_feet


def compute_speeds(df, calib, fps, window_s=0.4, smooth=7):
    """Añade wx, wy (m), v_px_s y v_m_s por punto. Suaviza y usa una ventana de tiempo."""
    k = max(1, int(round(fps * window_s)))
    parts = []
    for _, g in df.sort_values(["id", "frame"]).groupby("id"):
        g = g.copy()
        n = len(g)
        g["wx"], g["wy"] = np.nan, np.nan
        g["v_px_s"], g["v_m_s"] = np.nan, np.nan
        if n >= MIN_POINTS:
            sx = g["fx"].rolling(smooth, center=True, min_periods=1).mean().to_numpy()
            sy = g["fy"].rolling(smooth, center=True, min_periods=1).mean().to_numpy()
            t = g["time_s"].to_numpy(float)
            idx = np.arange(n)
            j = np.maximum(idx - k, 0)
            dt = t[idx] - t[j]
            ok = dt > 0
            safe = np.where(ok, dt, 1.0)
            px = np.c_[sx, sy]
            g["v_px_s"] = np.where(ok, np.linalg.norm(px[idx] - px[j], axis=1) / safe, np.nan)
            if calib.ready:
                w = calib.to_world(px)
                g["wx"], g["wy"] = w[:, 0], w[:, 1]
                v = np.linalg.norm(w[idx] - w[j], axis=1) / safe
                v = np.where(ok & (v <= V_MAX_PLAUSIBLE), v, np.nan)
                g["v_m_s"] = v
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df


def summarize(df):
    rows = []
    for tid, g in df.sort_values("frame").groupby("id"):
        if len(g) < MIN_POINTS:
            continue
        v, vp = g["v_m_s"].dropna(), g["v_px_s"].dropna()
        dist_m = desp_m = np.nan
        if g["wx"].notna().all():
            w = g[["wx", "wy"]].to_numpy()
            dist_m = float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())
            desp_m = float(np.linalg.norm(w[-1] - w[0]))
        mv = v[v > 0.3]  # solo los frames en que realmente se mueve
        rows.append({
            "id": int(tid), "clase": g["class"].mode().iloc[0], "frames": len(g),
            "duracion_s": round(float(g["time_s"].iloc[-1] - g["time_s"].iloc[0]), 2),
            "dist_m": round(dist_m, 2) if not np.isnan(dist_m) else np.nan,
            "desp_m": round(desp_m, 2) if not np.isnan(desp_m) else np.nan,
            "v_media_m_s": round(float(v.mean()), 2) if len(v) else np.nan,
            "v_mov_m_s": round(float(mv.median()), 2) if len(mv) >= 5 else np.nan,
            "v_p95_m_s": round(float(v.quantile(0.95)), 2) if len(v) else np.nan,
            "v_media_km_h": round(float(v.mean()) * 3.6, 1) if len(v) else np.nan,
            "v_media_px_s": round(float(vp.mean()), 1) if len(vp) else np.nan,
        })
    cols = ["id", "clase", "frames", "duracion_s", "dist_m", "desp_m", "v_media_m_s", "v_mov_m_s",
            "v_p95_m_s", "v_media_km_h", "v_media_px_s"]
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------
# Visualización
# ----------------------------------------------------------------------------
def render_speed_overlay(bg, s, c, vmax, cell=CELL):
    """Mapa de velocidad media por zona sobre el frame (para la vista en vivo)."""
    h, w = bg.shape[:2]
    mean = np.where(c > 0, s / np.maximum(c, 1), 0).astype(np.float32)
    size = (s.shape[1] * cell, s.shape[0] * cell)
    big = cv2.resize(mean, size, interpolation=cv2.INTER_NEAREST)[:h, :w]
    mask = cv2.resize((c > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)[:h, :w].astype(bool)
    col = cv2.applyColorMap(np.clip(big / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col = cv2.GaussianBlur(col, (0, 0), cell / 3)
    base = (bg * 0.45).astype(np.uint8)
    blend = cv2.addWeighted(base, 0.25, col, 0.75, 0)
    base[mask] = blend[mask]
    return base


def style_axes(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def draw_analysis(fig, df, summary, bg_bgr, calib):
    fig.clear()
    bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB) if bg_bgr is not None else None
    H, W = (bg.shape[:2]) if bg is not None else (int(df["fy"].max()) + 1, int(df["fx"].max()) + 1)
    gh, gw = int(np.ceil(H / CELL)), int(np.ceil(W / CELL))
    use_m = calib.ready and df["v_m_s"].notna().any()
    vcol, unit = ("v_m_s", "m/s") if use_m else ("v_px_s", "px/s")
    if H > W * 1.1:   # video vertical: dos mapas altos a la izquierda y dos gráficas a la derecha
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.7])
        a_pos, a_spd = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
        a_hist, a_per = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    else:
        gs = fig.add_gridspec(2, 2)
        a_pos, a_spd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        a_hist, a_per = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # 1) Mapa de calor de posiciones
    ax = a_pos
    style_axes(ax)
    if bg is not None:
        ax.imshow(bg)
    heat, _, _ = np.histogram2d(df["fy"], df["fx"], bins=[gh, gw], range=[[0, gh * CELL], [0, gw * CELL]])
    heat = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), 1.0)
    ax.imshow(np.ma.masked_less(heat, heat.max() * 0.04), extent=[0, gw * CELL, gh * CELL, 0],
              cmap="inferno", alpha=0.75)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title("Mapa de calor de posiciones")

    # 2) Velocidad media por zona
    ax = a_spd
    style_axes(ax)
    if bg is not None:
        ax.imshow(bg)
    d = df.dropna(subset=[vcol])
    if len(d):
        S, _, _ = np.histogram2d(d["fy"], d["fx"], bins=[gh, gw], range=[[0, gh * CELL], [0, gw * CELL]], weights=d[vcol])
        C, _, _ = np.histogram2d(d["fy"], d["fx"], bins=[gh, gw], range=[[0, gh * CELL], [0, gw * CELL]])
        mean = np.where(C >= 3, S / np.maximum(C, 1), np.nan)
        if np.isfinite(mean).any():
            im = ax.imshow(np.ma.masked_invalid(mean), extent=[0, gw * CELL, gh * CELL, 0], cmap="turbo",
                           alpha=0.85, vmin=0, vmax=np.nanpercentile(mean, 95), interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(f"Velocidad media ({unit})", color=MUTED, fontsize=8)
            cb.ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title(f"Velocidad media por zona ({unit})")

    # 3) Histograma de velocidades por ID
    ax = a_hist
    style_axes(ax)
    if use_m and len(summary):
        vals = summary.loc[summary["desp_m"] >= 2, "v_mov_m_s"].dropna()
    else:
        vals = summary["v_media_px_s"].dropna() if len(summary) else pd.Series(dtype=float)
    if len(vals):
        ax.hist(vals, bins=30, color=ACCENT, alpha=0.85, edgecolor=BG)
        ax.axvline(vals.median(), color=AMBER, lw=2, label=f"mediana {vals.median():.2f} {unit}")
        if use_m:
            ax.axvspan(1.2, 1.5, color=AMBER, alpha=0.18, label="caminata típica 1,2–1,5 m/s")
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    ax.set_xlabel(f"Velocidad al caminar por ID ({unit}) · solo IDs que avanzan ≥ 2 m" if use_m else f"Velocidad media por ID ({unit})"); ax.set_ylabel("N.º de IDs")
    ax.set_title("Distribución de velocidades")

    # 4) Por qué convertir: px/s depende de la distancia a la cámara
    ax = a_per
    style_axes(ax)
    edges = np.linspace(0, H, 9)
    cen, mpx, mm = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = df[(df["fy"] >= a) & (df["fy"] < b)]
        if s["v_px_s"].notna().sum() >= 30:
            cen.append((a + b) / 2)
            mpx.append(s["v_px_s"].median())
            mm.append(s["v_m_s"].median() if use_m else np.nan)
    if cen:
        ax.plot(cen, mpx, "-o", color=AMBER, label="px/s")
        ax.set_ylabel("Mediana (px/s)", color=AMBER)
        if use_m:
            ax2 = ax.twinx()
            ax2.plot(cen, mm, "-s", color=ACCENT, label="m/s")
            ax2.set_ylabel("Mediana (m/s)", color=ACCENT)
            ax2.tick_params(colors=MUTED, labelsize=8)
            ax2.set_ylim(0, max(np.nanmax(mm) * 1.6, 2))
            for sp in ax2.spines.values():
                sp.set_color(BORDER)
    ax.set_xlabel("Posición vertical en la imagen (px, abajo = cerca de la cámara)")
    ax.set_title("px/s cambia con la distancia; m/s no")


def fit_to_box(img_bgr, max_w, max_h):
    h, w = img_bgr.shape[:2]
    s = max(0.05, min(max_w / w, max_h / h))
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb)), s


# ----------------------------------------------------------------------------
# Hilo de procesamiento (YOLO + tracker + velocidad en vivo)
# ----------------------------------------------------------------------------
def live_speed(hist, calib):
    if len(hist) < 4:
        return None, None
    t1, x1, y1 = hist[-1]
    ref = hist[0]
    for h in hist:                      # el punto más antiguo dentro de 0,5 s
        if t1 - h[0] <= 0.5:
            ref = h
            break
    t0, x0, y0 = ref
    dt = t1 - t0
    if dt <= 0.05:
        return None, None
    vpx = float(np.hypot(x1 - x0, y1 - y0)) / dt
    vm = None
    if calib.ready:
        w = calib.to_world([[x0, y0], [x1, y1]])
        vm = float(np.hypot(*(w[1] - w[0]))) / dt
    return vpx, vm


class Worker(threading.Thread):
    def __init__(self, app, cfg):
        super().__init__(daemon=True)
        self.app, self.cfg = app, cfg

    def run(self):
        try:
            self._loop()
        except Exception as e:  # noqa: BLE001
            self.app.q.put(("error", f"{type(e).__name__}: {e}"))

    def _loop(self):
        from ultralytics import YOLO  # import tardío: la interfaz abre rápido

        app, cfg, calib = self.app, self.cfg, self.cfg["calib"]
        cap = cv2.VideoCapture(cfg["video"])
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1 else 30.0
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        app.q.put(("status", "Cargando modelo…"))
        model = YOLO(cfg["weights"])
        names = model.names

        gh, gw = int(np.ceil(H / CELL)), int(np.ceil(W / CELL))
        spd_sum, spd_cnt = np.zeros((gh, gw), np.float32), np.zeros((gh, gw), np.float32)
        hist = defaultdict(lambda: deque(maxlen=30))
        sm_v, trails, records = {}, defaultdict(lambda: deque(maxlen=40)), []
        unit = "m/s" if calib.ready else "px/s"
        vmax = 3.0 if calib.ready else 200.0
        fidx, fps_ema, t_prev, bg, heat = 0, 0.0, time.time(), app.bg_frame, None
        ok_col, alert_col = (191, 212, 45), (113, 113, 248)

        app.q.put(("status", "Procesando video…"))
        while not app.stop_event.is_set():
            while app.pause_event.is_set() and not app.stop_event.is_set():
                time.sleep(0.05)
            ok, frame = cap.read()
            if not ok:
                break
            fidx += 1
            if bg is None:
                bg = frame.copy()
            if (fidx - 1) % cfg["skip"]:
                continue
            t_s = fidx / fps

            res = model.track(frame, persist=True, tracker=cfg["tracker"], conf=cfg["conf"],
                              classes=cfg["classes"], imgsz=cfg["imgsz"], verbose=False)[0]
            disp, vals, active = frame.copy(), [], 0
            b = res.boxes
            if b is not None and b.id is not None:
                xyxy = b.xyxy.cpu().numpy()
                ids, clss, confs = b.id.int().cpu().tolist(), b.cls.int().cpu().tolist(), b.conf.cpu().tolist()
                for (x1, y1, x2, y2), tid, c, cf in zip(xyxy, ids, clss, confs):
                    active += 1
                    cx, cy, fx, fy = int((x1 + x2) / 2), int((y1 + y2) / 2), (x1 + x2) / 2, float(y2)
                    records.append((tid, fidx, round(t_s, 3), cx, cy, int(x2), int(y2), names[c], round(cf, 3)))
                    h = hist[tid]
                    sx, sy = (0.4 * fx + 0.6 * h[-1][1], 0.4 * fy + 0.6 * h[-1][2]) if h else (fx, fy)
                    h.append((t_s, sx, sy))
                    trails[tid].append((int(sx), int(sy)))
                    vpx, vm = live_speed(h, calib)
                    v = vm if (calib.ready and vm is not None) else vpx
                    if v is not None:
                        prev = sm_v.get(tid)
                        v = v if prev is None else 0.6 * prev + 0.4 * v
                        sm_v[tid] = v
                        gx, gy = min(gw - 1, int(sx // CELL)), min(gh - 1, int(sy // CELL))
                        spd_sum[gy, gx] += v
                        spd_cnt[gy, gx] += 1
                        vals.append(v)
                    alert = calib.ready and v is not None and v > cfg["alert"]
                    col = alert_col if alert else ok_col
                    cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
                    cv2.circle(disp, (int(fx), int(fy)), 6, col, -1)
                    if len(trails[tid]) > 1:
                        cv2.polylines(disp, [np.array(trails[tid], np.int32)], False, col, 2, cv2.LINE_AA)
                    if v is None:
                        label = f"#{tid}"
                    elif calib.ready:
                        label = f"#{tid} {v:.1f} m/s"
                    else:
                        label = f"#{tid} {v:.0f} px/s"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    cv2.rectangle(disp, (int(x1), int(y1) - th - 12), (int(x1) + tw + 8, int(y1)), col, -1)
                    cv2.putText(disp, label, (int(x1) + 4, int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (10, 20, 30), 3, cv2.LINE_AA)

            now = time.time()
            fps_ema = 0.9 * fps_ema + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            if heat is None or (fidx // cfg["skip"]) % 4 == 0:
                heat = render_speed_overlay(bg, spd_sum, spd_cnt, vmax)
            if app.q.qsize() < 3:
                sc = min(1.0, 900 / H)
                small = lambda im: cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)  # noqa: E731
                app.q.put(("frame", small(disp), small(heat), {
                    "active": active, "mean": float(np.mean(vals)) if vals else 0.0,
                    "max": float(np.max(vals)) if vals else 0.0, "fps": fps_ema,
                    "unit": unit, "progress": 100.0 * fidx / total}))
        cap.release()
        app.q.put(("done", records, fps))


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
class ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.cv = tk.Canvas(self, bg=BG, highlightthickness=0, width=330)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.cv.yview)
        self.inner = tk.Frame(self.cv, bg=BG)
        self.inner.bind("<Configure>", lambda e: self.cv.configure(scrollregion=self.cv.bbox("all")))
        self.win = self.cv.create_window((0, 0), window=self.inner, anchor="nw")
        self.cv.configure(yscrollcommand=sb.set)
        self.cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.cv.bind("<Configure>", lambda e: self.cv.itemconfigure(self.win, width=e.width))
        self.cv.bind("<Enter>", lambda e: self.cv.bind_all("<MouseWheel>", self._wheel))
        self.cv.bind("<Leave>", lambda e: self.cv.unbind_all("<MouseWheel>"))

    def _wheel(self, e):
        self.cv.yview_scroll(int(-e.delta / 120), "units")


def make_card(parent, title):
    outer = tk.Frame(parent, bg=BORDER)
    inner = tk.Frame(outer, bg=CARD)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    tk.Label(inner, text=title.upper(), bg=CARD, fg=ACCENT, font=(FONT, 9, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
    body = tk.Frame(inner, bg=CARD)
    body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
    return outer, body


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Velocidad y mapas de calor · Tracking 2D")
        self.geometry("1560x900")
        self.minsize(1300, 760)
        self.configure(bg=BG)

        self.q = queue.Queue()
        self.stop_event, self.pause_event = threading.Event(), threading.Event()
        self.worker = None
        self.video_path, self.bg_frame, self.frame_size, self.fps = None, None, None, 30.0
        self.calib = Calibration()
        self.raw = self.tracks = self.summary = None
        self.has_feet = False
        self._cal_geom = (0, 0, 1.0)
        self._meas = []
        self._photos = {}

        self._style()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(30, self._poll)

    # ---------------- estilo ----------------
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        self.option_add("*TCombobox*Listbox.background", CARD)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", BG)
        s.configure(".", background=BG, foreground=TEXT, font=(FONT, 10))
        s.configure("TNotebook", background=BG, borderwidth=0, bordercolor=BORDER, lightcolor=BG, darkcolor=BG)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(18, 9), font=(FONT, 10, "bold"), borderwidth=0, bordercolor=BG, lightcolor=PANEL, darkcolor=PANEL)
        s.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", ACCENT)])
        s.configure("Accent.TButton", background=ACCENT, foreground="#06231F", font=(FONT, 10, "bold"), padding=9, borderwidth=0)
        s.map("Accent.TButton", background=[("active", "#5EEAD4"), ("disabled", BORDER)], foreground=[("disabled", MUTED)])
        s.configure("Ghost.TButton", background=BORDER, foreground=TEXT, font=(FONT, 10), padding=8, borderwidth=0)
        s.map("Ghost.TButton", background=[("active", "#35527F"), ("disabled", PANEL)], foreground=[("disabled", MUTED)])
        s.configure("TCombobox", fieldbackground=PANEL, background=BORDER, foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=5)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        s.configure("TEntry", fieldbackground=PANEL, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER, padding=5)
        s.configure("TSpinbox", fieldbackground=PANEL, foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER, padding=4)
        s.configure("Card.TCheckbutton", background=CARD, foreground=TEXT)
        s.configure("Card.TCheckbutton", indicatorcolor=PANEL, upperbordercolor=BORDER, lowerbordercolor=BORDER)
        s.map("Card.TCheckbutton", background=[("active", CARD)], indicatorcolor=[("selected", ACCENT), ("!selected", PANEL)])
        s.configure("Card.TRadiobutton", background=CARD, foreground=TEXT)
        s.configure("Card.TRadiobutton", indicatorcolor=PANEL)
        s.map("Card.TRadiobutton", background=[("active", CARD)], indicatorcolor=[("selected", ACCENT), ("!selected", PANEL)])
        s.configure("Horizontal.TScale", background=ACCENT, troughcolor=BORDER, bordercolor=CARD, lightcolor=ACCENT, darkcolor=ACCENT)
        s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BORDER, borderwidth=0)
        s.configure("Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT, rowheight=26, borderwidth=0)
        s.configure("Treeview.Heading", background=BORDER, foreground=TEXT, font=(FONT, 9, "bold"), borderwidth=0)
        s.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", BG)])
        s.configure("Vertical.TScrollbar", background=BORDER, troughcolor=BG, arrowcolor=MUTED, borderwidth=0, bordercolor=BG, lightcolor=BORDER, darkcolor=BORDER)

    # ---------------- construcción ----------------
    def _lbl(self, parent, text, fg=TEXT, size=10, bold=False, bg=CARD, **kw):
        return tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, size, "bold" if bold else "normal"), **kw)

    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=22, pady=(16, 8))
        tk.Label(head, text="Velocidad y mapas de calor", bg=BG, fg=TEXT, font=(FONT, 20, "bold")).pack(side="left")
        tk.Label(head, text="   Actividad 07 · Etapa 02 · Ítem 3", bg=BG, fg=MUTED, font=(FONT, 11)).pack(side="left", pady=(8, 0))
        self.pill = tk.Label(head, text="●  Listo", bg=CARD, fg=ACCENT, font=(FONT, 10, "bold"), padx=14, pady=6)
        self.pill.pack(side="right")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=22, pady=(0, 6))
        side = ScrollFrame(body)
        side.pack(side="left", fill="y", padx=(0, 16))
        self._build_sidebar(side.inner)
        self.nb = ttk.Notebook(body)
        self.nb.pack(side="left", fill="both", expand=True)
        self._build_live()
        self._build_calib()
        self._build_analysis()
        self._build_theory()
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._cal_render())

        self.status = tk.Label(self, text="Abre un video para comenzar.", bg=BG, fg=MUTED, font=(FONT, 9), anchor="w")
        self.status.pack(fill="x", padx=22, pady=(0, 10))

    def _build_sidebar(self, sb):
        c, b = make_card(sb, "1 · Fuente")
        c.pack(fill="x", pady=(0, 12))
        ttk.Button(b, text="Abrir video…", style="Accent.TButton", command=self._open_video).pack(fill="x")
        self.lbl_video = self._lbl(b, "Ningún video seleccionado", MUTED, 9, wraplength=280, justify="left")
        self.lbl_video.pack(anchor="w", pady=(6, 8))
        ttk.Button(b, text="Cargar CSV de trayectorias…", style="Ghost.TButton", command=self._load_csv).pack(fill="x")

        c, b = make_card(sb, "2 · Detección y tracking")
        c.pack(fill="x", pady=(0, 12))
        self.var_weights, self.var_tracker, self.var_imgsz = tk.StringVar(value="yolov8n.pt"), tk.StringVar(value="bytetrack.yaml"), tk.StringVar(value="960")
        for txt, var, vals in (("Modelo YOLO", self.var_weights, ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"]),
                               ("Tracker", self.var_tracker, ["bytetrack.yaml", "botsort.yaml"]),
                               ("Tamaño de imagen", self.var_imgsz, ["640", "960", "1280"])):
            self._lbl(b, txt, MUTED, 9).pack(anchor="w")
            ttk.Combobox(b, textvariable=var, values=vals, state="readonly" if var is not self.var_weights else "normal").pack(fill="x", pady=(0, 8))
        self.var_conf = tk.DoubleVar(value=0.30)
        self.lbl_conf = self._lbl(b, "Confianza: 0.30", MUTED, 9)
        self.lbl_conf.pack(anchor="w")
        ttk.Scale(b, from_=0.1, to=0.9, variable=self.var_conf, command=lambda v: self.lbl_conf.config(text=f"Confianza: {float(v):.2f}")).pack(fill="x", pady=(0, 8))
        self.var_skip = tk.IntVar(value=1)
        row = tk.Frame(b, bg=CARD); row.pack(fill="x", pady=(0, 8))
        self._lbl(row, "Procesar 1 de cada", MUTED, 9).pack(side="left")
        ttk.Spinbox(row, from_=1, to=6, width=4, textvariable=self.var_skip).pack(side="left", padx=6)
        self._lbl(row, "frames", MUTED, 9).pack(side="left")
        grid = tk.Frame(b, bg=CARD); grid.pack(fill="x")
        self.var_cls = {}
        for i, (name, cid) in enumerate(CLASES):
            v = tk.BooleanVar(value=(cid == 0))
            self.var_cls[cid] = v
            ttk.Checkbutton(grid, text=name, variable=v, style="Card.TCheckbutton").grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 8))

        c, b = make_card(sb, "3 · Calibración (px → metros)")
        c.pack(fill="x", pady=(0, 12))
        self.lbl_cal = self._lbl(b, "● Sin calibrar: se mostrará px/s", AMBER, 10, True, wraplength=280, justify="left")
        self.lbl_cal.pack(anchor="w", pady=(0, 8))
        ttk.Button(b, text="Ir a calibración", style="Ghost.TButton", command=lambda: self.nb.select(1)).pack(fill="x")

        c, b = make_card(sb, "4 · Control")
        c.pack(fill="x", pady=(0, 12))
        self.btn_start = ttk.Button(b, text="▶  Iniciar", style="Accent.TButton", command=self._start)
        self.btn_start.pack(fill="x")
        row = tk.Frame(b, bg=CARD); row.pack(fill="x", pady=8)
        self.btn_pause = ttk.Button(row, text="Pausar", style="Ghost.TButton", command=self._toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_stop = ttk.Button(row, text="Detener", style="Ghost.TButton", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.var_alert = tk.DoubleVar(value=2.5)
        self.lbl_alert = self._lbl(b, "Alerta roja si v > 2.5 m/s", MUTED, 9)
        self.lbl_alert.pack(anchor="w")
        ttk.Scale(b, from_=0.5, to=10, variable=self.var_alert, command=lambda v: self.lbl_alert.config(text=f"Alerta roja si v > {float(v):.1f} m/s")).pack(fill="x", pady=(0, 8))
        self.progress = ttk.Progressbar(b, maximum=100)
        self.progress.pack(fill="x")

    def _kpi(self, parent, caption):
        f = tk.Frame(parent, bg=CARD, padx=16, pady=10)
        v = tk.Label(f, text="—", bg=CARD, fg=ACCENT, font=(FONT, 22, "bold"))
        v.pack(anchor="w")
        tk.Label(f, text=caption, bg=CARD, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
        return f, v

    def _build_live(self):
        tab = tk.Frame(self.nb, bg=PANEL)
        self.nb.add(tab, text="  En vivo  ")
        strip = tk.Frame(tab, bg=PANEL)
        strip.pack(fill="x", padx=14, pady=14)
        self.kpi = {}
        for key, cap in (("active", "Objetos activos"), ("mean", "Velocidad media"), ("max", "Velocidad máxima"), ("fps", "Cuadros por segundo")):
            f, v = self._kpi(strip, cap)
            f.pack(side="left", fill="x", expand=True, padx=(0, 10))
            self.kpi[key] = v
        area = tk.Frame(tab, bg=PANEL)
        area.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.live_imgs = []
        for title in ("Tracking con velocidad por ID", "Mapa de velocidad en vivo (por zona)"):
            col = tk.Frame(area, bg=PANEL)
            col.pack(side="left", fill="both", expand=True, padx=6)
            tk.Label(col, text=title, bg=PANEL, fg=MUTED, font=(FONT, 10, "bold")).pack(pady=(0, 6))
            box = tk.Frame(col, bg=CARD)
            box.pack(fill="both", expand=True)
            lab = tk.Label(box, bg=CARD, text="Abre un video y pulsa Iniciar", fg=MUTED, font=(FONT, 10))
            lab.pack(fill="both", expand=True)
            self.live_imgs.append((box, lab))

    def _build_calib(self):
        tab = tk.Frame(self.nb, bg=PANEL)
        self.nb.add(tab, text="  Calibración  ")
        self.cal_cv = tk.Canvas(tab, bg=CARD, highlightthickness=0, cursor="crosshair")
        self.cal_cv.pack(side="left", fill="both", expand=True, padx=14, pady=14)
        self.cal_cv.bind("<Configure>", lambda e: self._cal_render())
        self.cal_cv.bind("<Button-1>", self._cal_click)
        self.cal_cv.bind("<Button-3>", self._cal_undo)
        c, b = make_card(tab, "Parámetros de calibración")
        c.pack(side="left", fill="y", padx=(0, 14), pady=14)
        c.config(width=340)
        self.var_mode = tk.StringVar(value="homografia")
        ttk.Radiobutton(b, text="Homografía · 4 puntos (recomendado)", variable=self.var_mode, value="homografia", style="Card.TRadiobutton", command=self._mode_changed).pack(anchor="w")
        ttk.Radiobutton(b, text="Escala simple · 2 puntos", variable=self.var_mode, value="escala", style="Card.TRadiobutton", command=self._mode_changed).pack(anchor="w", pady=(2, 10))
        self.entries = {}
        for key, txt, val in (("w", "Ancho real 1→2 (m)", "6.0"), ("l", "Largo real 2→3 (m)", "4.0"), ("d", "Distancia real (m) · solo escala simple", "1.0")):
            self._lbl(b, txt, MUTED, 9).pack(anchor="w")
            e = ttk.Entry(b)
            e.insert(0, val)
            e.pack(fill="x", pady=(0, 8))
            self.entries[key] = e
        self.lbl_help = self._lbl(b, "", MUTED, 9, wraplength=300, justify="left")
        self.lbl_help.pack(anchor="w", pady=(2, 10))
        self.var_grid, self.var_meas = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        ttk.Checkbutton(b, text="Mostrar cuadrícula de 1 m", variable=self.var_grid, style="Card.TCheckbutton", command=self._cal_render).pack(anchor="w")
        ttk.Checkbutton(b, text="Modo medir (2 clics)", variable=self.var_meas, style="Card.TCheckbutton", command=self._meas_toggle).pack(anchor="w", pady=(2, 4))
        self._lbl(b, "Comprobación: una persona de pie debe ocupar medio metro de ancho en la cuadrícula.", MUTED, 9, wraplength=300, justify="left").pack(anchor="w", pady=(0, 10))
        ttk.Button(b, text="Aplicar calibración", style="Accent.TButton", command=self._apply_cal).pack(fill="x")
        ttk.Button(b, text="Borrar puntos", style="Ghost.TButton", command=self._clear_cal).pack(fill="x", pady=(8, 0))
        self.lbl_cal2 = self._lbl(b, "", AMBER, 9, True, wraplength=300, justify="left")
        self.lbl_cal2.pack(anchor="w", pady=(10, 0))
        self._mode_changed()

    def _build_analysis(self):
        tab = tk.Frame(self.nb, bg=PANEL)
        self.nb.add(tab, text="  Análisis  ")
        bar = tk.Frame(tab, bg=PANEL)
        bar.pack(fill="x", padx=14, pady=(14, 6))
        for txt, cmd in (("Exportar velocidades CSV…", self._export_points), ("Exportar resumen CSV…", self._export_summary), ("Guardar gráfico…", self._save_fig)):
            ttk.Button(bar, text=txt, style="Ghost.TButton", command=cmd).pack(side="left", padx=(0, 8))
        self.lbl_sum = tk.Label(tab, text="Aún no hay resultados. Procesa un video o carga un CSV.", bg=PANEL, fg=MUTED, font=(FONT, 10), anchor="w", justify="left", wraplength=1000)
        self.lbl_sum.pack(fill="x", padx=14, pady=(0, 6))
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor=PANEL, layout="constrained")
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.fig_canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        cols = ("id", "clase", "frames", "duracion_s", "dist_m", "desp_m", "v_media_m_s", "v_mov_m_s", "v_p95_m_s", "v_media_km_h", "v_media_px_s")
        wrap = tk.Frame(tab, bg=PANEL); wrap.pack(side="bottom", fill="x", padx=14, pady=14)
        self.table = ttk.Treeview(wrap, columns=cols, show="headings", height=5)
        for cname in cols:
            self.table.heading(cname, text=cname)
            self.table.column(cname, width=88, anchor="center")
        self.table.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.fig_canvas.get_tk_widget().pack(fill="both", expand=True, padx=14)

    def _build_theory(self):
        tab = tk.Frame(self.nb, bg=PANEL)
        self.nb.add(tab, text="  ¿Se puede pasar a m/s?  ")
        t = tk.Text(tab, bg=CARD, fg=TEXT, font=(FONT, 11), wrap="word", relief="flat", padx=28, pady=22, spacing3=6, borderwidth=0)
        t.pack(fill="both", expand=True, padx=14, pady=14)
        t.tag_configure("h", font=(FONT, 15, "bold"), foreground=ACCENT, spacing3=8)
        t.tag_configure("b", font=(FONT, 11, "bold"), foreground=AMBER)
        txt = [
            ("h", "¿Es posible convertir píxeles por segundo a metros por segundo reales?\n"),
            ("", "Sí, siempre que se conozca la relación entre píxeles y metros en el plano por donde se mueven los objetos.\n\n"),
            ("b", "Por qué los px/s no sirven por sí solos\n"),
            ("", "Con perspectiva, un píxel cercano a la cámara equivale a pocos centímetros y uno lejano a muchos. La misma caminata da muchos px/s abajo de la imagen y pocos arriba (ver la gráfica «px/s cambia con la distancia»).\n\n"),
            ("b", "Método 1 · Escala simple\n"),
            ("", "metros_por_píxel = distancia_real / distancia_en_píxeles. Solo vale si la cámara mira casi perpendicular al suelo.\n\n"),
            ("b", "Método 2 · Homografía (el usado aquí)\n"),
            ("", "Se marcan 4 puntos del suelo que forman un rectángulo de medidas conocidas y se calcula una matriz H de 3×3 que lleva cada píxel (x, y) a coordenadas (X, Y) en metros. La velocidad es la distancia en metros entre posiciones sucesivas dividida entre el tiempo (v = Δd / Δt, con Δt = Δframes / fps).\n\n"),
            ("b", "Limitaciones\n"),
            ("", "• Solo es exacto para puntos sobre el plano del suelo; por eso se usa el punto de los pies (borde inferior de la caja).\n"
                  "• Depende de la precisión de las medidas reales y de los 4 puntos clicados.\n"
                  "• La distorsión del lente y el temblor de las cajas de YOLO introducen error (se suaviza la posición).\n"
                  "• Los fps deben ser los reales del video.\n"
                  "• Si la cámara se mueve, la calibración deja de valer: se necesita odometría visual (ítem 4).\n\n"),
            ("b", "Cómo comprobar que está bien\n"),
            ("", "Una persona camina en promedio entre 1,2 y 1,5 m/s. Si la mediana de tus IDs cae cerca de ese rango, la escala es coherente.\n"),
        ]
        for tag, s in txt:
            t.insert("end", s, tag)
        t.config(state="disabled")

    # ---------------- utilidades ----------------
    def _status(self, text, ok=True):
        self.status.config(text=text)
        self.pill.config(text=("●  " + text[:34]) if text else "●  Listo", fg=ACCENT if ok else DANGER)

    def _on_close(self):
        self.stop_event.set()
        self.destroy()

    # ---------------- video ----------------
    def _open_video(self):
        p = filedialog.askopenfilename(title="Video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")])
        if not p:
            return
        cap = cv2.VideoCapture(p)
        ok, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not ok:
            messagebox.showerror("Video", "No se pudo leer el video.")
            return
        self.video_path, self.bg_frame = p, frame
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.fps = fps if fps and fps > 1 else 30.0
        self.lbl_video.config(text=f"{os.path.basename(p)}\n{self.frame_size[0]}×{self.frame_size[1]} · {self.fps:.0f} fps", fg=TEXT)
        self.calib.size = self.frame_size
        self._try_load_calib()
        self._cal_render()
        self._status("Video cargado. Calibra la escala o inicia el tracking.")

    def _try_load_calib(self):
        try:
            if os.path.exists(CALIB_FILE):
                d = json.load(open(CALIB_FILE, encoding="utf-8"))
                if tuple(d.get("size") or ()) == tuple(self.frame_size):
                    self.calib.from_dict(d)
                    self.var_mode.set(self.calib.mode)
                    for k, v in (("w", self.calib.width_m), ("l", self.calib.length_m), ("d", self.calib.dist_m)):
                        self.entries[k].delete(0, "end"); self.entries[k].insert(0, str(v))
                    self._mode_changed()
                    self._refresh_cal_labels()
        except Exception:  # noqa: BLE001
            self.calib.reset()

    # ---------------- calibración ----------------
    def _mode_changed(self):
        self.calib.mode = self.var_mode.get()
        self.calib.reset()
        n = 4 if self.calib.mode == "homografia" else 2
        self.lbl_help.config(text=("Haz clic en las 4 esquinas de un rectángulo sobre el suelo, en este orden: "
                                   "arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda. "
                                   "Mide el ancho (1→2) y el largo (2→3) reales, por ejemplo con Google Maps. Clic derecho deshace el último punto.")
                             if n == 4 else
                             "Haz clic en 2 puntos del suelo cuya distancia real conozcas y escríbela en «Distancia real».")
        self._refresh_cal_labels()
        self._cal_render()

    def _refresh_cal_labels(self):
        if self.calib.ready:
            txt = "● Homografía lista · se mostrará m/s y km/h" if self.calib.H is not None else f"● Escala lista · {self.calib.scale:.4f} m/px"
            self.lbl_cal.config(text=txt, fg=ACCENT)
            self.lbl_cal2.config(text=txt, fg=ACCENT)
        else:
            self.lbl_cal.config(text="● Sin calibrar: se mostrará px/s", fg=AMBER)
            self.lbl_cal2.config(text=f"Puntos marcados: {len(self.calib.pts)}", fg=AMBER)

    def _meas_toggle(self):
        self._meas = []
        self._cal_render()

    def _cal_click(self, e):
        if self.bg_frame is None:
            return
        if self.var_meas.get():
            if not self.calib.ready:
                messagebox.showinfo("Medir", "Primero aplica la calibración.")
                return
            ox, oy, s = self._cal_geom
            if len(self._meas) >= 2:
                self._meas = []
            self._meas.append(((e.x - ox) / s, (e.y - oy) / s))
            if len(self._meas) == 2:
                w = self.calib.to_world(self._meas)
                self.lbl_cal2.config(text=f"Distancia medida: {float(np.hypot(*(w[1] - w[0]))):.2f} m", fg=ACCENT)
            self._cal_render()
            return
        need = 4 if self.var_mode.get() == "homografia" else 2
        if len(self.calib.pts) >= need:
            return
        ox, oy, s = self._cal_geom
        x, y = (e.x - ox) / s, (e.y - oy) / s
        if 0 <= x < self.frame_size[0] and 0 <= y < self.frame_size[1]:
            self.calib.pts.append((float(x), float(y)))
            self.calib.H = self.calib.scale = None
            self._refresh_cal_labels()
            self._cal_render()

    def _cal_undo(self, _e=None):
        if self.var_meas.get():
            self._meas = self._meas[:-1]
            self._cal_render()
        elif self.calib.pts:
            self.calib.pts.pop()
            self.calib.H = self.calib.scale = None
            self._refresh_cal_labels()
            self._cal_render()

    def _cal_render(self):
        cv_ = self.cal_cv
        cv_.delete("all")
        cw, ch = cv_.winfo_width(), cv_.winfo_height()
        if self.bg_frame is None or cw < 50:
            cv_.create_text(cw // 2, ch // 2, text="Abre un video para calibrar", fill=MUTED, font=(FONT, 12))
            return
        photo, s = fit_to_box(self.bg_frame, cw - 10, ch - 10)
        ox, oy = (cw - photo.width()) // 2, (ch - photo.height()) // 2
        self._photos["cal"], self._cal_geom = photo, (ox, oy, s)
        cv_.create_image(ox, oy, anchor="nw", image=photo)
        if self.var_grid.get() and self.calib.H is not None:
            self._draw_grid(ox, oy, s)
        if len(self._meas) == 2:
            (x1, y1), (x2, y2) = self._meas
            cv_.create_line(ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s, fill=DANGER, width=3)
            w = self.calib.to_world(self._meas)
            cv_.create_text(ox + (x1 + x2) / 2 * s + 16, oy + (y1 + y2) / 2 * s, anchor="w", fill=DANGER,
                            font=(FONT, 12, "bold"), text=f"{float(np.hypot(*(w[1] - w[0]))):.2f} m")
        for x, y in self._meas:
            cv_.create_oval(ox + x * s - 5, oy + y * s - 5, ox + x * s + 5, oy + y * s + 5, fill=DANGER, outline=BG)
        pts = [(ox + x * s, oy + y * s) for x, y in self.calib.pts]
        if len(pts) >= 2:
            flat = [c for p in pts for c in p]
            if len(pts) == 4:
                cv_.create_polygon(*flat, outline=ACCENT, fill="", width=2)
            else:
                cv_.create_line(*flat, fill=ACCENT, width=2)
        for i, (x, y) in enumerate(pts, 1):
            cv_.create_oval(x - 7, y - 7, x + 7, y + 7, fill=AMBER, outline=BG, width=2)
            cv_.create_text(x + 14, y - 14, text=str(i), fill=AMBER, font=(FONT, 12, "bold"))

    def _draw_grid(self, ox, oy, s):
        """Cuadrícula de 1 m sobre el suelo: sirve para comprobar visualmente la escala."""
        Hi = np.linalg.inv(self.calib.H)
        W, L = self.calib.width_m, self.calib.length_m
        fw, fh = self.frame_size

        def project(pts):
            p = np.c_[pts, np.ones(len(pts))] @ Hi.T
            return p[:, :2] / p[:, 2:3], p[:, 2]

        _, w0 = project(np.array([[W / 2, L / 2]]))
        sign = np.sign(w0[0])
        xs, ys = np.arange(-2 * W, 3 * W + 1), np.arange(-3 * L, 3 * L + 1)
        for fixed, vary, is_x in ((xs, np.arange(-3 * L, 3 * L + 0.01, 0.5), True), (ys, np.arange(-2 * W, 3 * W + 0.01, 0.5), False)):
            for f in fixed:
                pts = np.c_[np.full(len(vary), f), vary] if is_x else np.c_[vary, np.full(len(vary), f)]
                px, w = project(pts)
                ok = (w * sign > 0) & (px[:, 0] >= 0) & (px[:, 0] <= fw) & (px[:, 1] >= 0) & (px[:, 1] <= fh)
                run = []
                for p, good in zip(px, ok):
                    if good:
                        run.append((ox + p[0] * s, oy + p[1] * s))
                    else:
                        if len(run) > 1:
                            self.cal_cv.create_line(*[c for q in run for c in q], fill=ACCENT, width=1)
                        run = []
                if len(run) > 1:
                    self.cal_cv.create_line(*[c for q in run for c in q], fill=ACCENT, width=1)

    def _clear_cal(self):
        self.calib.reset()
        self._refresh_cal_labels()
        self._cal_render()

    def _apply_cal(self):
        try:
            self.calib.mode = self.var_mode.get()
            self.calib.width_m = float(self.entries["w"].get().replace(",", "."))
            self.calib.length_m = float(self.entries["l"].get().replace(",", "."))
            self.calib.dist_m = float(self.entries["d"].get().replace(",", "."))
            self.calib.build()
        except ValueError as e:
            messagebox.showwarning("Calibración", str(e))
            return
        self.calib.size = self.frame_size
        try:
            json.dump(self.calib.to_dict(), open(CALIB_FILE, "w", encoding="utf-8"))
        except OSError:
            pass
        self._refresh_cal_labels()
        self._cal_render()
        self._status("Calibración aplicada.")
        if self.raw is not None:
            self._analyze()

    # ---------------- ejecución ----------------
    def _start(self):
        if not self.video_path:
            messagebox.showinfo("Video", "Primero abre un video.")
            return
        if self.worker and self.worker.is_alive():
            return
        if not self.calib.ready and not messagebox.askyesno("Sin calibrar", "No hay calibración: la velocidad saldrá en píxeles por segundo.\n¿Continuar de todos modos?"):
            return
        classes = [cid for cid, v in self.var_cls.items() if v.get()] or None
        cfg = {"video": self.video_path, "weights": self.var_weights.get(), "tracker": self.var_tracker.get(),
               "imgsz": int(self.var_imgsz.get()), "conf": float(self.var_conf.get()), "classes": classes,
               "skip": max(1, int(self.var_skip.get())), "alert": float(self.var_alert.get()), "calib": self.calib}
        self.stop_event.clear(); self.pause_event.clear()
        self.worker = Worker(self, cfg)
        self.worker.start()
        self._set_running(True)
        self.nb.select(0)

    def _set_running(self, run):
        self.btn_start.config(state="disabled" if run else "normal")
        self.btn_pause.config(state="normal" if run else "disabled", text="Pausar")
        self.btn_stop.config(state="normal" if run else "disabled")

    def _toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear(); self.btn_pause.config(text="Pausar")
        else:
            self.pause_event.set(); self.btn_pause.config(text="Reanudar")

    def _stop(self):
        self.stop_event.set()
        self.pause_event.clear()

    def _poll(self):
        last = None
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "frame":
                    last = m
                elif m[0] == "status":
                    self._status(m[1])
                elif m[0] == "error":
                    self._set_running(False)
                    self._status("Error", ok=False)
                    messagebox.showerror("Error", m[1])
                elif m[0] == "done":
                    self._on_done(m[1], m[2])
        except queue.Empty:
            pass
        if last:
            self._show_live(last)
        self.after(30, self._poll)

    def _show_live(self, m):
        _, disp, heat, st = m
        for (box, lab), img in zip(self.live_imgs, (disp, heat)):
            photo, _ = fit_to_box(img, max(120, box.winfo_width() - 12), max(120, box.winfo_height() - 12))
            lab.config(image=photo, text="")
            lab.image = photo
        u = st["unit"]
        fmt = (lambda v: f"{v:.1f} {u}") if u == "m/s" else (lambda v: f"{v:.0f} {u}")
        self.kpi["active"].config(text=str(st["active"]))
        self.kpi["mean"].config(text=fmt(st["mean"]))
        self.kpi["max"].config(text=fmt(st["max"]))
        self.kpi["fps"].config(text=f"{st['fps']:.1f}")
        self.progress["value"] = st["progress"]

    def _on_done(self, records, fps):
        self._set_running(False)
        self.progress["value"] = 100
        if not records:
            self._status("Sin detecciones.", ok=False)
            return
        df = pd.DataFrame(records, columns=["id", "frame", "time_s", "x", "y", "x2", "y2", "class", "conf"])
        self.fps = fps
        self.raw, self.has_feet = prepare_tracks(df, fps)
        self._analyze()
        self.nb.select(2)

    # ---------------- CSV ----------------
    def _load_csv(self):
        p = filedialog.askopenfilename(title="CSV de trayectorias", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        try:
            df = pd.read_csv(p)
            need = {"id", "frame", "x", "y"}
            if not need.issubset(df.columns):
                raise ValueError(f"El CSV debe tener las columnas {sorted(need)}.")
            fps = self.fps
            if "time_s" in df.columns and not self.video_path:
                d = df.sort_values(["id", "frame"]).groupby("id")[["frame", "time_s"]].diff().dropna()
                d = d[(d["time_s"] > 0)]
                if len(d):
                    fps = float(np.median(d["frame"] / d["time_s"]))
            self.fps = fps
            self.raw, self.has_feet = prepare_tracks(df, fps)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("CSV", str(e))
            return
        if not self.has_feet:
            messagebox.showinfo("CSV", "El CSV no trae la columna y2 (borde inferior de la caja).\n"
                                "Se usará el centro de la caja: la conversión a metros será menos precisa.\n"
                                "Los CSV generados por esta aplicación sí la incluyen.")
        self._status(f"CSV cargado: {len(self.raw)} puntos, {self.raw['id'].nunique()} IDs.")
        self._analyze()
        self.nb.select(2)

    # ---------------- análisis ----------------
    def _analyze(self):
        if self.raw is None:
            return
        self.tracks = compute_speeds(self.raw, self.calib, self.fps)
        self.summary = summarize(self.tracks)
        draw_analysis(self.fig, self.tracks, self.summary, self.bg_frame, self.calib)
        self.fig_canvas.draw_idle()
        self.table.delete(*self.table.get_children())
        for _, r in self.summary.iterrows():
            self.table.insert("", "end", values=[("" if pd.isna(v) else v) for v in r.tolist()])
        n_all, n_ok = self.tracks["id"].nunique(), len(self.summary)
        if self.calib.ready and self.summary["v_media_m_s"].notna().any():
            mov = self.summary.loc[self.summary["desp_m"] >= 2, "v_mov_m_s"].dropna()
            if len(mov):
                med = float(mov.median())
                verdict = ("Coherente con una caminata normal (1,0–1,8 m/s)." if 1.0 <= med <= 1.8
                           else "Fuera del rango típico de una caminata: verifica la calibración con la cuadrícula de 1 m.")
                txt = (f"IDs únicos: {n_all} · analizados: {n_ok} · que avanzan ≥ 2 m: {len(mov)} · "
                       f"mediana al caminar: {med:.2f} m/s ({med * 3.6:.1f} km/h). {verdict}")
            else:
                txt = (f"IDs únicos: {n_all} · analizados: {n_ok}. Ninguno avanza ≥ 2 m con esta calibración: "
                       "verifica las medidas reales y el orden de los 4 puntos.")
            self.lbl_sum.config(text=txt, fg=TEXT)
        else:
            self.lbl_sum.config(text=f"IDs únicos: {n_all} · analizados: {n_ok}. Sin calibración: velocidades en px/s. "
                                     "Calibra en la pestaña «Calibración» para obtener m/s y km/h.", fg=AMBER)
        self._status("Análisis listo.")

    # ---------------- exportar ----------------
    def _export_points(self):
        if self.tracks is None:
            return messagebox.showinfo("Exportar", "No hay resultados todavía.")
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="velocidades.csv")
        if p:
            self.tracks.drop(columns=[c for c in ("fx", "fy") if c in self.tracks.columns]).to_csv(p, index=False)
            self._status(f"Guardado: {p}")

    def _export_summary(self):
        if self.summary is None:
            return messagebox.showinfo("Exportar", "No hay resultados todavía.")
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="resumen_velocidades.csv")
        if p:
            self.summary.to_csv(p, index=False)
            self._status(f"Guardado: {p}")

    def _save_fig(self):
        if self.tracks is None:
            return messagebox.showinfo("Exportar", "No hay resultados todavía.")
        p = filedialog.asksaveasfilename(defaultextension=".png", initialfile="analisis_velocidad.png")
        if p:
            self.fig.savefig(p, dpi=150, facecolor=self.fig.get_facecolor())
            self._status(f"Guardado: {p}")


if __name__ == "__main__":
    App().mainloop()
