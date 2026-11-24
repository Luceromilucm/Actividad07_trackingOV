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
from types import SimpleNamespace

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
            return f"Homografía · zona {self.ancho:.1f} m × {self.largo:.1f} m"
        return f"Escala · {self.ppm:.1f} píxeles/metro"

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


# ─────────────────────── Odometría visual (cámara móvil) ───────────────────────
class OdometriaVisual:
    """Compensa el movimiento de la cámara respecto al suelo.
    ORB (FAST + BRIEF rotado) -> emparejamiento Hamming -> homografía con RANSAC.
    Mejoras frente a encadenar solo frame a frame:
      · se empareja contra un KEYFRAME (menos deriva) y se renueva cuando quedan pocos inliers;
      · solo mira el suelo (sin cielo) y excluye vehículos y PERSONAS (objetos que se mueven con la cámara);
      · rechaza homografías implausibles, así la zona calibrada no se deforma;
      · trabaja a ~640 px de ancho (mucho más rápido) y mejora el contraste con CLAHE."""

    def __init__(self, n_feat=2000, ratio=0.75, min_inliers=20, kf_inliers=60, roi_top=0.45, ancho_proc=640):
        self.orb = cv2.ORB_create(nfeatures=n_feat, fastThreshold=12, scoreType=cv2.ORB_HARRIS_SCORE)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio, self.min_inliers, self.kf_inliers = ratio, min_inliers, kf_inliers
        self.roi_top, self.ancho_proc = roi_top, ancho_proc
        self.H_ref = np.eye(3)  # frame de calibración <- frame actual (coordenadas a resolución completa)
        self.kf = None
        self.prev = None
        self.ok = False
        self.inliers = 0
        self._sz = (1, 1)

    def _plausible(self, H):
        w, h = self._sz
        esq = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(esq, H).reshape(-1, 2)
        if not np.isfinite(p).all() or np.abs(p).max() > 1e5:
            return False
        if not cv2.isContourConvex(p.astype(np.int32).reshape(-1, 1, 2)):
            return False
        if not 0.4 < cv2.contourArea(p.astype(np.float32)) / (w * h) < 2.5:
            return False
        return float(np.abs(p - esq.reshape(-1, 2)).max()) < 0.6 * w

    def _estimar(self, act, ref):
        if act["des"] is None or ref["des"] is None or len(act["pts"]) < 8 or len(ref["pts"]) < 8:
            return None, 0
        pares = self.bf.knnMatch(act["des"], ref["des"], k=2)
        buenos = [p[0] for p in pares if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(buenos) < self.min_inliers:
            return None, 0
        src = act["pts"][[g.queryIdx for g in buenos]].reshape(-1, 1, 2)
        dst = ref["pts"][[g.trainIdx for g in buenos]].reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or inl is None:
            return None, 0
        n = int(inl.sum())
        if n < self.min_inliers or not self._plausible(H):
            return None, 0
        return H, n

    def actualizar(self, frame, cajas):
        """cajas: todo lo que NO es suelo estático (vehículos y personas), en píxeles del frame."""
        h, w = frame.shape[:2]
        s = min(1.0, self.ancho_proc / w)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if s < 1.0:
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = self.clahe.apply(gray)
        hs, ws = gray.shape
        self._sz = (ws, hs)
        mask = np.full(gray.shape, 255, np.uint8)
        mask[:int(hs * self.roi_top)] = 0  # sin cielo / edificios lejanos
        for x1, y1, x2, y2 in cajas:
            pad = 12
            mask[max(0, int(y1 * s) - pad):int(y2 * s) + pad, max(0, int(x1 * s) - pad):int(x2 * s) + pad] = 0
        kp, des = self.orb.detectAndCompute(gray, mask)
        pts = np.float32([k.pt for k in kp]).reshape(-1, 2) if kp else np.zeros((0, 2), np.float32)
        act = {"pts": pts, "des": des}
        S = np.diag([s, s, 1.0])
        Sinv = np.linalg.inv(S)

        self.ok, self.inliers = False, 0
        if self.kf is None:
            if des is not None and len(pts) >= self.min_inliers:
                self.kf = {**act, "H": self.H_ref.copy()}
                self.ok = True
        else:
            H, n = self._estimar(act, self.kf)
            if H is not None:
                self.H_ref = self.kf["H"] @ Sinv @ H @ S
                self.ok, self.inliers = True, n
                if n < self.kf_inliers:  # el keyframe se está quedando sin coincidencias: renovarlo
                    self.kf = {**act, "H": self.H_ref / self.H_ref[2, 2]}
            elif self.prev is not None:
                H, n = self._estimar(act, self.prev)
                if H is not None:
                    self.H_ref = self.prev["H"] @ Sinv @ H @ S
                    self.ok, self.inliers = True, n
                    self.kf = {**act, "H": self.H_ref / self.H_ref[2, 2]}
        self.H_ref = self.H_ref / self.H_ref[2, 2]
        self.prev = {**act, "H": self.H_ref.copy()}
        return self.ok

    def a_referencia(self, pts_px):
        """Píxeles del frame actual -> píxeles del frame de calibración."""
        p = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H_ref).reshape(-1, 2)

    def a_actual(self, pts_ref):
        """Píxeles del frame de calibración -> píxeles del frame actual (para dibujar la zona)."""
        p = np.asarray(pts_ref, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, np.linalg.inv(self.H_ref)).reshape(-1, 2)


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
                 carpeta_evidencias="infracciones", minimapa=True, camara_movil=False, infracciones=True):
        self.modelo, self.cal, self.limite = modelo, cal, limite
        self.conf, self.imgsz, self.tracker = conf, imgsz, tracker
        self.dir = Path(carpeta_evidencias)
        if infracciones:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.usar_minimapa = minimapa and cal.metodo == "homografia" and not camara_movil
        self.odo = OdometriaVisual() if camara_movil else None
        self.infracciones = infracciones
        self.vivos = {}  # vehículos en escena en el último frame (para la tabla en vivo)
        if camara_movil:  # ventana más larga y más muestras: el ruido de la odometría pesa más
            self.est = EstimadorVelocidad(ventana_s=1.5, min_s=1.0, min_muestras=8, n_mediana=15,
                                          min_estimaciones=6, vmax=120)
        else:
            self.est = EstimadorVelocidad()
        self.v_parado = 4.0 if camara_movil else 0.0  # piso de ruido: por debajo se reporta 0 km/h
        self.fotos, self.vmax, self.t_entrada = {}, {}, {}
        self.primera, self.ultima = {}, {}
        self.n_frame = 0

    @staticmethod
    def _fraccion_tapada(box, otros):
        """Fracción del vehículo cubierta por la persona que más lo tapa."""
        if len(otros) == 0:
            return 0.0
        x1, y1, x2, y2 = box
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        ix = np.clip(np.minimum(x2, otros[:, 2]) - np.maximum(x1, otros[:, 0]), 0, None)
        iy = np.clip(np.minimum(y2, otros[:, 3]) - np.maximum(y1, otros[:, 1]), 0, None)
        return float((ix * iy).max() / area)

    def procesar(self, frame, t):
        self.n_frame += 1
        h, w = frame.shape[:2]
        # con cámara móvil también se detectan personas (clase 0), solo para enmascararlas en la odometría
        clases_det = list(CLASES_VEHICULO) + ([0] if self.odo is not None else [])
        res = self.modelo.track(frame, persist=True, classes=clases_det, conf=self.conf,
                                imgsz=self.imgsz, tracker=self.tracker, verbose=False)[0]
        hay = res.boxes is not None and len(res.boxes) > 0
        xyxy_all = res.boxes.xyxy.cpu().numpy() if hay else np.zeros((0, 4), np.float32)
        cls_all = res.boxes.cls.int().cpu().numpy() if hay else np.zeros(0, int)
        id_all = res.boxes.id.int().cpu().numpy() if (hay and res.boxes.id is not None) else None
        es_veh = np.isin(cls_all, list(CLASES_VEHICULO))
        personas = xyxy_all[~es_veh]

        if self.odo is not None:
            self.odo.actualizar(frame, xyxy_all)  # siempre, haya o no detecciones
        self.vivos = {}
        vis = frame.copy()
        if self.odo is not None:  # proyecta la zona calibrada sobre el frame actual
            zona = self.odo.a_actual(self.cal.puntos)
            if self.odo.ok and cv2.isContourConvex(np.clip(zona, -1e5, 1e5).astype(np.int32).reshape(-1, 1, 2)):
                dibujar_calibracion(vis, SimpleNamespace(metodo=self.cal.metodo, puntos=zona))
            if not self.odo.ok:
                etiqueta(vis, "ODOMETRIA: sin referencia", (10, 40), NARANJA)
        else:
            dibujar_calibracion(vis, self.cal)
        pos_mm, en_escena = {}, 0

        if id_all is not None and es_veh.any():
            xyxy = xyxy_all[es_veh]
            ids = id_all[es_veh].tolist()
            clases = cls_all[es_veh].tolist()
            anclas = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)  # contacto con el piso
            anclas_ref = self.odo.a_referencia(anclas) if self.odo is not None else anclas
            metros = self.cal.a_metros(anclas_ref)
            for box, tid, c, anc, m in zip(xyxy, ids, clases, anclas, metros):
                en_escena += 1
                x1, y1, x2, y2 = box
                c = self.est.clase(tid, c)
                self.est.marcar_visto(tid, t)
                self.t_entrada.setdefault(tid, t)
                toca_borde = x1 < 3 or y1 < 3 or x2 > w - 3 or y2 > h - 3
                vel = self.est.vel.get(tid)
                if self.odo is not None:  # cámara móvil: solo suelo cercano (lejos el error métrico explota)
                    dentro = anc[1] > h * (self.odo.roi_top + 0.08)
                    tapado = self._fraccion_tapada(box, personas) > 0.35  # el pie del auto no es fiable
                    fiable = self.odo.ok and not tapado
                else:
                    dentro, fiable = self.cal.dentro_zona(anc), True
                if dentro and fiable and not toca_borde:
                    vel = self.est.actualizar(tid, t, m)
                    self.primera.setdefault(tid, (float(m[0]), float(m[1])))
                    self.ultima[tid] = (float(m[0]), float(m[1]))
                if vel is not None and vel < self.v_parado:
                    vel = 0.0
                if dentro and fiable and not toca_borde:
                    pos_mm[tid] = (m[0], m[1], vel)
                if vel is not None and vel >= self.v_parado:
                    self.vmax[tid] = max(self.vmax.get(tid, 0.0), vel)

                nombre = CLASES_VEHICULO[c]
                self.vivos[tid] = {"id": tid, "clase": nombre, "vel": vel, "vel_max": self.vmax.get(tid),
                                   "sentido": self._sentido(tid), "t": round(self.t_entrada.get(tid, t), 2)}
                if vel is None:
                    color, txt = GRIS, f"#{tid} {nombre}"
                else:
                    if self.infracciones:
                        color = ROJO if vel > self.limite else (NARANJA if vel > 0.9 * self.limite else VERDE)
                    else:
                        color = VERDE
                    txt = f"#{tid} {nombre} " + ("parado" if (self.v_parado and vel == 0.0) else f"{vel:.0f} km/h")
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 4 if color == ROJO else 2)
                cv2.circle(vis, (int(anc[0]), int(anc[1])), 4, color, -1)
                etiqueta(vis, txt, (x1, y1), color)
                if self.infracciones and vel is not None and vel > self.limite:
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
            if v is not None and v < self.v_parado:
                v = 0.0
            foto = self.fotos.pop(tid, None)
            if v is not None:
                votos = self.est.votos.get(tid)
                clase = CLASES_VEHICULO[votos.most_common(1)[0][0]] if votos else "auto"
                infr = self.infracciones and v > self.limite
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