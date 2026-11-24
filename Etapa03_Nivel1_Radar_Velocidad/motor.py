"""
motor.py — Núcleo del Radar de Velocidad
YOLO26 + ByteTrack + calibración métrica (escala px/m u homografía) + estimación robusta de velocidad.
Lo usan la aplicación (app.py) y la versión de consola (radar_velocidad.py).
"""
import json
import math
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CLASES_VEHICULO = {2: "auto", 3: "moto", 5: "bus", 7: "camion"}  # IDs COCO
VERDE, NARANJA, ROJO = (60, 200, 60), (0, 165, 255), (0, 0, 255)
BLANCO, NEGRO, CIAN, GRIS = (255, 255, 255), (0, 0, 0), (255, 200, 0), (200, 200, 200)

MODELOS = {
    "Rápido (YOLO26n)": "yolo26n.pt",
    "Balanceado (YOLO26s)": "yolo26s.pt",
    "Preciso (YOLO26m)": "yolo26m.pt",
}


# ─────────────────────────────── Calibración ───────────────────────────────
class Calibracion:
    """escala: 2 puntos + distancia -> píxeles/metro.
    homografia: 4 puntos de un rectángulo real (sup-izq, sup-der, inf-der, inf-izq) + ancho y largo -> vista cenital."""

    def __init__(self, metodo, puntos, ppm=None, ancho=None, largo=None, tam_frame=None):
        self.metodo = metodo
        self.puntos = np.array(puntos, dtype=np.float32)
        self.ppm, self.ancho, self.largo = ppm, ancho, largo
        self.tam_frame = list(tam_frame) if tam_frame else None
        self.H = None
        if metodo == "homografia":
            dst = np.array([[0, 0], [ancho, 0], [ancho, largo], [0, largo]], dtype=np.float32)
            self.H = cv2.getPerspectiveTransform(self.puntos, dst)

    @staticmethod
    def desde_escala(p1, p2, distancia_m, tam_frame):
        ppm = math.dist(p1, p2) / distancia_m
        return Calibracion("escala", [p1, p2], ppm=ppm, tam_frame=tam_frame)

    def a_metros(self, pts_px):
        pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        if self.H is not None:
            return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return pts.reshape(-1, 2) / self.ppm

    def dentro_zona(self, p):
        if self.metodo != "homografia":
            return True
        return cv2.pointPolygonTest(self.puntos.reshape(-1, 1, 2), (float(p[0]), float(p[1])), False) >= 0

    def es_valida(self):
        if self.metodo == "homografia":
            return bool(cv2.isContourConvex(self.puntos.reshape(-1, 1, 2).astype(np.int32)))
        return self.ppm is not None and self.ppm > 0

    def descripcion(self):
        if self.metodo == "homografia":
            return f"Homografía · zona {self.ancho:.1f} m × {self.largo:.1f} m · escala {self.rango_ppm()}"
        return f"Escala · {self.ppm:.1f} píxeles/metro"

    def rango_ppm(self):
        """Factor de escala (píxeles/metro) en cada lado del rectángulo. Con perspectiva no es uno solo:
        es mayor cerca de la cámara y menor lejos; la homografía usa el valor correcto en cada punto."""
        p = self.puntos
        lados = [(0, 1, self.ancho), (3, 2, self.ancho), (1, 2, self.largo), (0, 3, self.largo)]
        v = [math.dist(p[a], p[b]) / m for a, b, m in lados]
        return f"{min(v):.1f}–{max(v):.1f} px/m"

    def guardar(self, ruta):
        Path(ruta).write_text(json.dumps({
            "metodo": self.metodo, "puntos": self.puntos.tolist(), "ppm": self.ppm,
            "ancho": self.ancho, "largo": self.largo, "tam_frame": self.tam_frame}, indent=2))

    @staticmethod
    def cargar(ruta):
        d = json.loads(Path(ruta).read_text())
        return Calibracion(d["metodo"], d["puntos"], d.get("ppm"), d.get("ancho"), d.get("largo"), d.get("tam_frame"))


# ─────────────────────────── Estimador de velocidad ─────────────────────────
class EstimadorVelocidad:
    """1) Regresión lineal posición(m) vs tiempo(s) en ventana deslizante -> velocidad instantánea.
    2) Mediana de las últimas N estimaciones -> velocidad mostrada (robusta a picos).
    3) Mediana de todas las estimaciones del vehículo -> velocidad registrada."""

    def __init__(self, ventana_s=1.0, min_s=0.5, min_muestras=6, n_mediana=10, min_estimaciones=4, vmax=250):
        self.ventana_s, self.min_s, self.min_muestras = ventana_s, min_s, min_muestras
        self.n_mediana, self.min_est, self.vmax = n_mediana, min_estimaciones, vmax
        self.hist = defaultdict(deque)
        self.crudas = defaultdict(list)
        self.vel = {}
        self.ultimo_visto = {}
        self.votos = defaultdict(Counter)

    def actualizar(self, tid, t, xy_m):
        h = self.hist[tid]
        h.append((t, float(xy_m[0]), float(xy_m[1])))
        while h and t - h[0][0] > self.ventana_s:
            h.popleft()
        if len(h) < self.min_muestras or h[-1][0] - h[0][0] < self.min_s:
            return self.vel.get(tid)
        a = np.array(h)
        ts = a[:, 0] - a[:, 0].mean()
        vx = np.polyfit(ts, a[:, 1], 1)[0]
        vy = np.polyfit(ts, a[:, 2], 1)[0]
        kmh = math.hypot(vx, vy) * 3.6
        if kmh > self.vmax:  # salto del tracker / cambio de ID
            return self.vel.get(tid)
        self.crudas[tid].append(kmh)
        if len(self.crudas[tid]) >= self.min_est:
            self.vel[tid] = float(np.median(self.crudas[tid][-self.n_mediana:]))
        return self.vel.get(tid)

    def velocidad_oficial(self, tid):
        c = self.crudas.get(tid)
        return float(np.median(c)) if c and len(c) >= self.min_est else None

    def clase(self, tid, c):
        self.votos[tid][c] += 1
        return self.votos[tid].most_common(1)[0][0]

    def marcar_visto(self, tid, t):
        self.ultimo_visto[tid] = t

    def olvidar(self, tid):
        for d in (self.hist, self.crudas, self.vel, self.ultimo_visto, self.votos):
            d.pop(tid, None)


# ─────────────────────────────── Dibujo ───────────────────────────────
def etiqueta(img, txt, org, color, escala=0.6):
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
    x, y = int(org[0]), int(max(org[1], th + 8))
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y), color, -1)
    cv2.putText(img, txt, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, escala, BLANCO, 2)


def dibujar_calibracion(img, cal):
    p = cal.puntos.astype(int)
    if cal.metodo == "homografia":
        over = img.copy()
        cv2.fillPoly(over, [p], CIAN)
        cv2.addWeighted(over, 0.15, img, 0.85, 0, img)
        cv2.polylines(img, [p], True, CIAN, 2)
    else:
        cv2.line(img, tuple(p[0]), tuple(p[1]), CIAN, 2)
        for q in p:
            cv2.circle(img, tuple(q), 5, CIAN, -1)


def dibujar_puntos_calibracion(img, puntos, metodo):
    """Vista previa mientras el usuario hace clic."""
    pts = [tuple(map(int, p)) for p in puntos]
    n = 4 if metodo == "homografia" else 2
    if len(pts) == n and n == 4:
        over = img.copy()
        cv2.fillPoly(over, [np.array(pts)], CIAN)
        cv2.addWeighted(over, 0.2, img, 0.8, 0, img)
    for i, p in enumerate(pts):
        if i:
            cv2.line(img, pts[i - 1], p, NARANJA, 2)
        if i == n - 1 and n == 4:
            cv2.line(img, p, pts[0], NARANJA, 2)
    r = max(6, img.shape[1] // 200)
    for i, p in enumerate(pts):
        cv2.circle(img, p, r, ROJO, -1)
        cv2.circle(img, p, r, BLANCO, 2)
        etiqueta(img, str(i + 1), (p[0] + r + 2, p[1] - r), ROJO, 0.7)
    return img


def minimapa(cal, posiciones, limite, lado=200):
    W, L = cal.ancho, cal.largo
    esc = (lado - 20) / max(W, L)
    alto, ancho_px = int(L * esc) + 36, max(int(W * esc) + 20, 100)
    mm = np.full((alto, ancho_px, 3), 40, np.uint8)
    cv2.rectangle(mm, (10, 10), (10 + int(W * esc), 10 + int(L * esc)), (90, 90, 90), -1)
    for xm, ym, v in posiciones.values():
        c = ROJO if v and v > limite else VERDE
        cv2.circle(mm, (10 + int(xm * esc), 10 + int(ym * esc)), 5, c, -1)
    cv2.putText(mm, "vista cenital", (4, alto - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLANCO, 1)
    return mm


# ──────────────────────────────── Modelo ─────────────────────────────────
def cargar_modelo(nombre="yolo26n.pt"):
    from ultralytics import YOLO
    try:
        return YOLO(nombre)
    except Exception as e:  # ultralytics antiguo sin YOLO26
        print(f"[!] No se pudo cargar {nombre} ({e}). Usando yolo11n.pt — actualiza: pip install -U ultralytics")
        return YOLO("yolo11n.pt")


def formato_tiempo(s):
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


# ─────────────────────────────── Procesador ───────────────────────────────
class Procesador:
    """Procesa frame a frame. Devuelve el frame anotado y los vehículos que terminaron de pasar,
    cada uno con su velocidad registrada, tipo, sentido e indicador de infracción."""

    TIMEOUT_S = 2.0

    def __init__(self, modelo, cal, limite=30.0, conf=0.35, imgsz=640, tracker="bytetrack.yaml",
                 carpeta_evidencias="infracciones", minimapa=True):
        self.modelo, self.cal, self.limite = modelo, cal, limite
        self.conf, self.imgsz, self.tracker = conf, imgsz, tracker
        self.dir = Path(carpeta_evidencias)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.usar_minimapa = minimapa and cal.metodo == "homografia"
        self.est = EstimadorVelocidad()
        self.fotos, self.vmax, self.t_entrada = {}, {}, {}
        self.primera, self.ultima = {}, {}
        self.n_frame = 0

    def procesar(self, frame, t):
        self.n_frame += 1
        h, w = frame.shape[:2]
        res = self.modelo.track(frame, persist=True, classes=list(CLASES_VEHICULO), conf=self.conf,
                                imgsz=self.imgsz, tracker=self.tracker, verbose=False)[0]
        vis = frame.copy()
        dibujar_calibracion(vis, self.cal)
        pos_mm, en_escena = {}, 0

        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()
            clases = res.boxes.cls.int().cpu().tolist()
            anclas = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)  # contacto con el piso
            metros = self.cal.a_metros(anclas)
            for box, tid, c, anc, m in zip(xyxy, ids, clases, anclas, metros):
                en_escena += 1
                x1, y1, x2, y2 = box
                c = self.est.clase(tid, c)
                self.est.marcar_visto(tid, t)
                self.t_entrada.setdefault(tid, t)
                toca_borde = x1 < 3 or y1 < 3 or x2 > w - 3 or y2 > h - 3
                vel = self.est.vel.get(tid)
                if self.cal.dentro_zona(anc) and not toca_borde:
                    vel = self.est.actualizar(tid, t, m)
                    self.primera.setdefault(tid, (float(m[0]), float(m[1])))
                    self.ultima[tid] = (float(m[0]), float(m[1]))
                    pos_mm[tid] = (m[0], m[1], vel)
                if vel is not None:
                    self.vmax[tid] = max(self.vmax.get(tid, 0.0), vel)

                nombre = CLASES_VEHICULO[c]
                if vel is None:
                    color, txt = GRIS, f"#{tid} {nombre}"
                else:
                    color = ROJO if vel > self.limite else (NARANJA if vel > 0.9 * self.limite else VERDE)
                    txt = f"#{tid} {nombre} {vel:.0f} km/h"
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 4 if color == ROJO else 2)
                cv2.circle(vis, (int(anc[0]), int(anc[1])), 4, color, -1)
                etiqueta(vis, txt, (x1, y1), color)
                if vel is not None and vel > self.limite:
                    if tid not in self.fotos:
                        self.fotos[tid] = self._foto(frame, box, tid, nombre)
                    if (self.n_frame // 6) % 2 == 0:
                        etiqueta(vis, "EXCESO DE VELOCIDAD", (x1, y2 + 30), ROJO)

        terminados = self._cerrar([i for i, tv in self.est.ultimo_visto.items() if t - tv > self.TIMEOUT_S])

        if self.usar_minimapa:
            mm = minimapa(self.cal, pos_mm, self.limite)
            mh, mw = mm.shape[:2]
            if mh < h - 20 and mw < w - 20:
                vis[h - mh - 10:h - 10, w - mw - 10:w - 10] = mm
        return vis, en_escena, terminados

    def finalizar(self):
        return self._cerrar(list(self.est.ultimo_visto))

    # ── internos ──
    def _foto(self, frame, box, tid, nombre):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        m = 20
        crop = frame[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
        ruta = self.dir / f"{datetime.now():%Y%m%d_%H%M%S}_id{tid}_{nombre}.jpg"
        cv2.imwrite(str(ruta), crop)
        return ruta

    def _cerrar(self, ids):
        out = []
        for tid in ids:
            v = self.est.velocidad_oficial(tid)
            foto = self.fotos.pop(tid, None)
            if v is not None:
                votos = self.est.votos.get(tid)
                clase = CLASES_VEHICULO[votos.most_common(1)[0][0]] if votos else "auto"
                infr = v > self.limite
                if foto and not infr:  # la lectura final no superó el límite -> no es infracción
                    foto.unlink(missing_ok=True)
                    foto = None
                out.append({
                    "id": tid, "clase": clase, "vel": round(v, 1),
                    "vel_max": round(self.vmax.get(tid, v), 1),
                    "t": round(self.t_entrada.get(tid, 0.0), 2),
                    "hora": datetime.now().strftime("%H:%M:%S"),
                    "sentido": self._sentido(tid), "limite": self.limite, "infraccion": infr,
                    "foto": str(foto) if (foto and infr) else "",
                })
            elif foto:
                foto.unlink(missing_ok=True)
            self.est.olvidar(tid)
            for d in (self.vmax, self.t_entrada, self.primera, self.ultima):
                d.pop(tid, None)
        return out

    def _sentido(self, tid):
        if tid not in self.primera or tid not in self.ultima:
            return "-"
        dx = self.ultima[tid][0] - self.primera[tid][0]
        dy = self.ultima[tid][1] - self.primera[tid][1]
        if abs(dx) >= abs(dy):
            return "→" if dx > 0 else "←"
        return "↓" if dy > 0 else "↑"
