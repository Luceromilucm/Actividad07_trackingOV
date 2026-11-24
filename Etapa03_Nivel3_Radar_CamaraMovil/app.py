#!/usr/bin/env python3
"""
Radar de Velocidad — Aplicación de escritorio
=============================================
1. Cargar video / webcam / cámara del celular
2. Calibrar haciendo clic sobre la imagen (medidas de Google Maps)
3. Analizar: YOLO26 + tracking + velocidad en km/h con alerta de exceso
4. Estadísticas en vivo, tabla de vehículos en tiempo real y reporte HTML

Ejecutar:  python app.py
"""
import base64
import csv
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSplitter, QStatusBar, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
    QVBoxLayout, QWidget,
)

from motor import (MODELOS, Calibracion, Procesador, cargar_modelo, dibujar_calibracion,
                   dibujar_puntos_calibracion, formato_tiempo)

ACENTO = "#3d8bfd"
VERDE_HEX, ROJO_HEX, NARANJA_HEX = "#3ccf6a", "#ff4d4f", "#ffa500"
LIMITES = [("Calle / jirón — 30 km/h", 30), ("Avenida — 50 km/h", 50), ("Zona escolar / hospital — 30 km/h", 30),
           ("Vía expresa — 80 km/h", 80), ("Personalizado", None)]
NOMBRES_PUNTOS = {
    "homografia": ["1 · esquina SUPERIOR IZQUIERDA", "2 · esquina SUPERIOR DERECHA",
                   "3 · esquina INFERIOR DERECHA", "4 · esquina INFERIOR IZQUIERDA"],
    "escala": ["1 · primer punto de referencia", "2 · segundo punto de referencia"],
}


def a_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ───────────────────────────── Hilo de análisis ─────────────────────────────
class Analizador(QThread):
    frame_listo = Signal(QImage)
    estado = Signal(str)
    en_vivo = Signal(dict)
    vehiculo = Signal(dict)
    activos = Signal(list)
    progreso = Signal(int, int)
    error = Signal(str)

    def __init__(self, fuente, es_archivo, cal, cfg):
        super().__init__()
        self.fuente, self.es_archivo, self.cal, self.cfg = fuente, es_archivo, cal, cfg
        self.limite = cfg["limite"]
        self.pausado = False
        self.detener = False

    def run(self):
        cap = writer = None
        try:
            self.estado.emit(f"Cargando modelo {self.cfg['modelo']} (la primera vez se descarga)…")
            modelo = cargar_modelo(self.cfg["modelo"])
            movil = self.cfg["camara_movil"]
            proc = Procesador(modelo, self.cal, self.limite, self.cfg["conf"],
                              tracker="botsort.yaml" if movil else "bytetrack.yaml",
                              carpeta_evidencias=self.cfg["carpeta"] / "infracciones",
                              camara_movil=movil, infracciones=False)
            cap = cv2.VideoCapture(self.fuente)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.es_archivo else 0
            self.estado.emit("Analizando…")
            idx, t0, mejor = 0, time.perf_counter(), 0
            t_prev, fps_proc = t0, 0.0
            while not self.detener:
                if self.pausado:
                    self.msleep(50)
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                t = idx / fps if self.es_archivo else time.perf_counter() - t0
                proc.limite = self.limite
                vis, n, terminados = proc.procesar(frame, t)
                for v in terminados:
                    self.vehiculo.emit(v)
                if idx % 3 == 0:
                    self.activos.emit(list(proc.vivos.values()))
                if n > mejor:  # captura representativa para el reporte
                    mejor = n
                    cv2.imwrite(str(self.cfg["carpeta"] / "captura.jpg"), vis)
                if self.cfg["guardar_video"]:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(str(self.cfg["carpeta"] / "video_anotado.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(vis)
                ahora = time.perf_counter()
                fps_proc = 0.9 * fps_proc + 0.1 / max(ahora - t_prev, 1e-6)
                t_prev = ahora
                self.frame_listo.emit(a_qimage(vis))
                self.en_vivo.emit({"en_escena": n, "fps": fps_proc, "t": t})
                if total:
                    self.progreso.emit(idx, total)
            for v in proc.finalizar():
                self.vehiculo.emit(v)
            self.estado.emit("Análisis detenido" if self.detener else "Análisis terminado ✔")
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()


# ───────────────────────────── Widgets auxiliares ─────────────────────────────
class VistaVideo(QLabel):
    """Muestra el video escalado y traduce los clics a coordenadas de la imagen original."""
    clic = Signal(float, float)
    clic_derecho = Signal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background:#0e0f12; border-radius:8px; color:#8a90a0; font-size:15px;")
        self.setText("📂  Abre un video, la webcam o la cámara del celular para empezar")
        self._img, self._esc, self._off = None, 1.0, (0, 0)

    def mostrar(self, qimg):
        self._img = qimg
        self._render()

    def resizeEvent(self, e):
        self._render()
        super().resizeEvent(e)

    def _render(self):
        if self._img is None:
            return
        pix = QPixmap.fromImage(self._img).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._esc = pix.width() / self._img.width()
        self._off = ((self.width() - pix.width()) / 2, (self.height() - pix.height()) / 2)
        self.setPixmap(pix)

    def mousePressEvent(self, e):
        if self._img is None:
            return
        if e.button() == Qt.RightButton:
            self.clic_derecho.emit()
            return
        x = (e.position().x() - self._off[0]) / self._esc
        y = (e.position().y() - self._off[1]) / self._esc
        if 0 <= x < self._img.width() and 0 <= y < self._img.height():
            self.clic.emit(x, y)


class Tarjeta(QFrame):
    def __init__(self, titulo, icono=""):
        super().__init__()
        self.setObjectName("tarjeta")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(0)
        self.t = QLabel(f"{icono}  {titulo}")
        self.t.setObjectName("tarjetaTitulo")
        self.v = QLabel("—")
        self.v.setObjectName("tarjetaValor")
        self.s = QLabel("")
        self.s.setObjectName("tarjetaSub")
        lay.addWidget(self.t)
        lay.addWidget(self.v)
        lay.addWidget(self.s)

    def poner(self, valor, sub="", color=None):
        self.v.setText(valor)
        self.s.setText(sub)
        self.v.setStyleSheet(f"color:{color};" if color else "")


class DialogoMedidas(QDialog):
    def __init__(self, metodo, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Medidas reales (Google Maps)")
        lay = QVBoxLayout(self)
        info = QLabel(
            "Ingresa las distancias reales que mediste en <b>Google Maps</b><br>"
            "(clic derecho sobre el mapa → <i>Medir distancia</i>).")
        info.setWordWrap(True)
        lay.addWidget(info)
        form = QFormLayout()
        self.a = self._spin(10.0)
        self.b = self._spin(20.0)
        if metodo == "homografia":
            form.addRow("Ancho real  punto 1 → 2 (m):", self.a)
            form.addRow("Largo real  punto 2 → 3 (m):", self.b)
        else:
            form.addRow("Distancia real  punto 1 → 2 (m):", self.a)
            self.b.hide()
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @staticmethod
    def _spin(v):
        s = QDoubleSpinBox()
        s.setRange(0.1, 1000)
        s.setDecimals(2)
        s.setSuffix(" m")
        s.setValue(v)
        return s


AYUDA_HTML = f"""
<h2 style='color:{ACENTO}'>Cómo usar el Radar de Velocidad</h2>
<h3>① Elegir la fuente</h3>
<ul>
<li><b>Abrir video</b>: un video de una cámara fija que mire la calle (mp4, avi, mov…).</li>
<li><b>Webcam</b>: la cámara de la laptop apuntando a la calle.</li>
<li><b>Celular</b>: instala <i>IP Webcam</i> (Android), pulsa <i>Iniciar servidor</i> y escribe la URL que aparece,
por ejemplo <code>http://192.168.1.5:8080/video</code> (celular y PC en la misma red Wi-Fi).</li>
</ul>
<p><b>Un buen video:</b> cámara inmóvil, grabado desde un lugar alto, tramo de calle recto, de día y con
referencias visibles en el piso (paso peatonal, postes, bordes de la calzada).</p>
<h3>② Calibrar (convertir píxeles a metros)</h3>
<ol>
<li>En Google Maps busca la calle del video y activa la vista satélite.</li>
<li>Clic derecho → <b>Medir distancia</b>, y mide entre referencias que se vean en el video.</li>
<li>Pulsa <b>Calibrar sobre la imagen</b> y haz clic en esos puntos. Clic derecho deshace el último punto.</li>
</ol>
<p><b>Homografía (recomendado):</b> 4 esquinas de un rectángulo real sobre la pista, en este orden:
superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda. Corrige la perspectiva,
así que funciona aunque la cámara mire en diagonal. Mientras más larga sea la zona (20–30 m), más estable la medición.</p>
<p><b>Escala (px/m):</b> 2 puntos y su distancia. Solo es exacta si la cámara mira la calle de costado.</p>
<p>La calibración se guarda junto al video y se carga sola la próxima vez.</p>
<h3>③ Analizar</h3>
<p>Pulsa <b>Iniciar</b>. Cada vehículo detectado se muestra con su ID y su velocidad en km/h.
Si grabas con la cámara en movimiento, marca <b>Cámara en movimiento</b>: la app compensa el desplazamiento
de la cámara con odometría visual (ORB) y mide la velocidad <b>absoluta</b> de cada vehículo. Calibra con la cámara quieta
en el primer frame y usa el método de homografía.</p>
<h3>④ Resultados</h3>
<ul>
<li><b>Gráficos</b>: distribución de velocidades, velocidad en el tiempo y resumen por tipo de vehículo.</li>
<li><b>Vehículos</b>: tabla en vivo con cada vehículo detectado, su velocidad actual, máxima y estado (se puede ordenar).</li>
<li><b>Exportar reporte</b>: genera un reporte HTML y un CSV en la carpeta de resultados.</li>
</ul>
<h3>¿Qué significa cada indicador?</h3>
<ul>
<li><b>Velocidad P85</b>: velocidad que el 85 % de los vehículos no supera. Es el indicador estándar en ingeniería de tránsito.</li>
<li><b>Flujo</b>: vehículos por minuto que pasan por la zona.</li>
<li><b>Velocidad registrada</b>: mediana de todas las mediciones del vehículo dentro de la zona (robusta a errores puntuales).</li>
</ul>
<p style='color:#8a90a0'>Límites urbanos en Perú (D.S. 025-2021-MTC): 30 km/h en calles y jirones, 50 km/h en avenidas.</p>
"""


# ───────────────────────────── Ventana principal ─────────────────────────────
class Ventana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar de Velocidad · YOLO26")
        self.resize(1600, 940)
        self.fuente = None
        self.es_archivo = False
        self.frame0 = None
        self.cal = None
        self.calibrando = False
        self.puntos = []
        self.hilo = None
        self.vehiculos = []
        self.filas_id = {}
        self.carpeta = None
        self.t_video = 0.0
        self._construir()
        self._actualizar_botones()

    # ── construcción de la interfaz ──
    def _construir(self):
        raiz = QSplitter(Qt.Horizontal)
        self.setCentralWidget(raiz)
        raiz.addWidget(self._panel_lateral())

        centro = QWidget()
        lc = QVBoxLayout(centro)
        lc.setContentsMargins(6, 6, 6, 6)
        self.banner = QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.hide()
        lc.addWidget(self.banner)
        self.video = VistaVideo()
        self.video.clic.connect(self._clic_video)
        self.video.clic_derecho.connect(self._deshacer_punto)
        lc.addWidget(self.video, 1)
        self.vivo = QLabel("")
        self.vivo.setObjectName("vivo")
        lc.addWidget(self.vivo)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.k_total = Tarjeta("Vehículos medidos", "🚗")
        self.k_prom = Tarjeta("Velocidad promedio", "📈")
        self.k_p85 = Tarjeta("Velocidad P85", "🎯")
        self.k_max = Tarjeta("Velocidad máxima", "⚡")
        self.k_inf = Tarjeta("En movimiento", "🚗")
        self.k_flujo = Tarjeta("Flujo", "🔁")
        for i, k in enumerate([self.k_total, self.k_prom, self.k_p85, self.k_max, self.k_inf, self.k_flujo]):
            grid.addWidget(k, 0, i)
        lc.addLayout(grid)
        raiz.addWidget(centro)

        raiz.addWidget(self._panel_derecho())
        raiz.setSizes([340, 800, 480])
        raiz.setStretchFactor(1, 1)

        self.barra = QStatusBar()
        self.setStatusBar(self.barra)
        self.barra.showMessage("Listo. Empieza abriendo un video.")

    def _panel_lateral(self):
        cont = QWidget()
        lay = QVBoxLayout(cont)
        lay.setContentsMargins(10, 10, 6, 10)
        lay.setSpacing(10)
        titulo = QLabel("🚦 Radar de Velocidad")
        titulo.setObjectName("titulo")
        lay.addWidget(titulo)

        g1 = QGroupBox("① Fuente de video")
        l1 = QVBoxLayout(g1)
        self.b_video = QPushButton("📂  Abrir video…")
        self.b_cam = QPushButton("💻  Webcam")
        self.b_ip = QPushButton("📱  Cámara del celular (URL)")
        self.b_video.clicked.connect(self._abrir_video)
        self.b_cam.clicked.connect(lambda: self._abrir_fuente(0, False, "Webcam"))
        self.b_ip.clicked.connect(self._abrir_ip)
        for b in (self.b_video, self.b_cam, self.b_ip):
            l1.addWidget(b)
        self.info_fuente = QLabel("Sin fuente")
        self.info_fuente.setObjectName("info")
        self.info_fuente.setWordWrap(True)
        l1.addWidget(self.info_fuente)
        lay.addWidget(g1)

        g2 = QGroupBox("② Calibración")
        l2 = QVBoxLayout(g2)
        self.c_metodo = QComboBox()
        self.c_metodo.addItem("Homografía · 4 puntos (recomendado)", "homografia")
        self.c_metodo.addItem("Escala px/m · 2 puntos", "escala")
        l2.addWidget(self.c_metodo)
        self.b_calibrar = QPushButton("🎯  Calibrar sobre la imagen")
        self.b_calibrar.clicked.connect(self._iniciar_calibracion)
        l2.addWidget(self.b_calibrar)
        fila = QHBoxLayout()
        self.b_cargar_cal = QPushButton("Cargar…")
        self.b_guardar_cal = QPushButton("Guardar…")
        self.b_cargar_cal.clicked.connect(self._cargar_cal)
        self.b_guardar_cal.clicked.connect(self._guardar_cal)
        fila.addWidget(self.b_cargar_cal)
        fila.addWidget(self.b_guardar_cal)
        l2.addLayout(fila)
        self.info_cal = QLabel("⚠ Sin calibrar")
        self.info_cal.setObjectName("info")
        self.info_cal.setWordWrap(True)
        l2.addWidget(self.info_cal)
        lay.addWidget(g2)

        g3 = QGroupBox("③ Configuración")
        f3 = QFormLayout(g3)
        self.c_tipo_via = QComboBox()
        for txt, v in LIMITES:
            self.c_tipo_via.addItem(txt, v)
        self.c_tipo_via.currentIndexChanged.connect(self._preset_limite)
        self.s_limite = QDoubleSpinBox()
        self.s_limite.setRange(5, 200)
        self.s_limite.setDecimals(0)
        self.s_limite.setSuffix(" km/h")
        self.s_limite.setValue(30)
        self.s_limite.valueChanged.connect(self._cambio_limite)
        self.c_modelo = QComboBox()
        for k in MODELOS:
            self.c_modelo.addItem(k, MODELOS[k])
        self.s_conf = QDoubleSpinBox()
        self.s_conf.setRange(0.1, 0.9)
        self.s_conf.setSingleStep(0.05)
        self.s_conf.setValue(0.35)
        self.chk_video = QCheckBox("Guardar video anotado")
        self.chk_video.setChecked(True)
        self.chk_movil = QCheckBox("Cámara en movimiento (odometría visual)")
        f3.addRow("Tipo de vía:", self.c_tipo_via)
        f3.addRow("Límite:", self.s_limite)
        f3.addRow("Modelo:", self.c_modelo)
        f3.addRow("Confianza:", self.s_conf)
        f3.addRow(self.chk_video)
        f3.addRow(self.chk_movil)
        lay.addWidget(g3)

        g4 = QGroupBox("④ Análisis")
        l4 = QVBoxLayout(g4)
        fila = QHBoxLayout()
        self.b_iniciar = QPushButton("▶  Iniciar")
        self.b_iniciar.setObjectName("primario")
        self.b_pausa = QPushButton("⏸ Pausa")
        self.b_detener = QPushButton("⏹ Detener")
        self.b_pausa.setToolTip("Pausar / continuar")
        self.b_detener.setToolTip("Detener")
        self.b_iniciar.clicked.connect(self._iniciar)
        self.b_pausa.clicked.connect(self._pausar)
        self.b_detener.clicked.connect(self._detener)
        l4.addWidget(self.b_iniciar)
        fila.addWidget(self.b_pausa)
        fila.addWidget(self.b_detener)
        l4.addLayout(fila)
        self.prog = QProgressBar()
        self.prog.setTextVisible(True)
        self.prog.setValue(0)
        l4.addWidget(self.prog)
        lay.addWidget(g4)

        g5 = QGroupBox("⑤ Resultados")
        l5 = QVBoxLayout(g5)
        self.b_reporte = QPushButton("📄  Exportar reporte")
        self.b_carpeta = QPushButton("📁  Abrir carpeta de resultados")
        self.b_reporte.clicked.connect(self._exportar)
        self.b_carpeta.clicked.connect(lambda: self.carpeta and QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.carpeta.resolve()))))
        l5.addWidget(self.b_reporte)
        l5.addWidget(self.b_carpeta)
        lay.addWidget(g5)
        lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(cont)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def _panel_derecho(self):
        tabs = QTabWidget()
        self.tabs = tabs

        # Gráficos
        g = QWidget()
        lg = QVBoxLayout(g)
        self.p_hist = pg.PlotWidget(title="Distribución de velocidades")
        self.p_hist.setLabel("bottom", "km/h")
        self.p_hist.setLabel("left", "vehículos")
        self.p_hist.showGrid(y=True, alpha=0.2)
        self.p_tiempo = pg.PlotWidget(title="Velocidad de cada vehículo en el tiempo")
        self.p_tiempo.setLabel("bottom", "tiempo del video (s)")
        self.p_tiempo.setLabel("left", "km/h")
        self.p_tiempo.showGrid(x=True, y=True, alpha=0.2)
        self.scatter = pg.ScatterPlotItem(size=9, pen=pg.mkPen(None))
        self.p_tiempo.addItem(self.scatter)
        self.lin_lim_t = pg.InfiniteLine(angle=0, pen=pg.mkPen(ROJO_HEX, width=2, style=Qt.DashLine))
        self.lin_lim_h = pg.InfiniteLine(angle=90, pen=pg.mkPen(ROJO_HEX, width=2, style=Qt.DashLine))
        self.p_tiempo.addItem(self.lin_lim_t)
        self.p_hist.addItem(self.lin_lim_h)
        self.barras = None
        self.t_tipos = QTableWidget(0, 4)
        self.t_tipos.setHorizontalHeaderLabels(["Tipo", "Cantidad", "Vel. promedio", "En movimiento"])
        self.t_tipos.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.t_tipos.verticalHeader().hide()
        self.t_tipos.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.t_tipos.setMaximumHeight(150)
        lg.addWidget(self.p_hist, 1)
        lg.addWidget(self.p_tiempo, 1)
        lg.addWidget(QLabel("Resumen por tipo de vehículo"))
        lg.addWidget(self.t_tipos)
        tabs.addTab(g, "📊 Gráficos")

        # Vehículos
        v = QWidget()
        lv = QVBoxLayout(v)
        self.tabla = QTableWidget(0, 7)
        self.tabla.setHorizontalHeaderLabels(["ID", "Tipo", "km/h", "Máx.", "Sentido", "Tiempo", "Estado"])
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.verticalHeader().hide()
        self.tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabla.setSortingEnabled(True)
        lv.addWidget(self.tabla)
        tabs.addTab(v, "🚗 Vehículos")

        ayuda = QTextBrowser()
        ayuda.setHtml(AYUDA_HTML)
        ayuda.setOpenExternalLinks(True)
        tabs.addTab(ayuda, "❓ Ayuda")
        tabs.setMinimumWidth(470)
        self._limpiar_resultados()
        return tabs

    # ── estado de botones ──
    def _actualizar_botones(self):
        corriendo = self.hilo is not None and self.hilo.isRunning()
        for b in (self.b_video, self.b_cam, self.b_ip, self.b_calibrar, self.b_cargar_cal, self.c_metodo,
                  self.c_modelo, self.s_conf, self.chk_video, self.chk_movil):
            b.setEnabled(not corriendo)
        self.b_calibrar.setEnabled(not corriendo and self.frame0 is not None)
        self.b_cargar_cal.setEnabled(not corriendo and self.frame0 is not None)
        self.b_guardar_cal.setEnabled(self.cal is not None)
        self.b_iniciar.setEnabled(not corriendo and self.frame0 is not None and self.cal is not None)
        self.b_pausa.setEnabled(corriendo)
        self.b_detener.setEnabled(corriendo)
        self.b_reporte.setEnabled(bool(self.vehiculos) and not corriendo)
        self.b_carpeta.setEnabled(self.carpeta is not None)

    # ── fuente ──
    def _abrir_video(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Elegir video de tráfico", "",
                                              "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;Todos (*.*)")
        if ruta:
            self._abrir_fuente(ruta, True, Path(ruta).name)

    def _abrir_ip(self):
        url, ok = QInputDialog.getText(self, "Cámara del celular",
                                       "URL de la app IP Webcam (ej. http://192.168.1.5:8080/video):",
                                       text="http://192.168.1.5:8080/video")
        if ok and url.strip():
            self._abrir_fuente(url.strip(), False, "Celular")

    def _abrir_fuente(self, fuente, es_archivo, nombre):
        self.barra.showMessage("Abriendo fuente…")
        QApplication.processEvents()
        cap = cv2.VideoCapture(fuente)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok:
            QMessageBox.warning(self, "Error", f"No se pudo abrir la fuente:\n{fuente}")
            self.barra.showMessage("No se pudo abrir la fuente.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.fuente, self.es_archivo, self.frame0 = fuente, es_archivo, frame
        h, w = frame.shape[:2]
        txt = f"<b>{nombre}</b><br>{w}×{h} · {fps:.0f} fps"
        if es_archivo and n > 0:
            txt += f" · {formato_tiempo(n / fps)} min"
        self.info_fuente.setText(txt)
        self.cal = None
        ruta_cal = self._ruta_cal_auto()
        if ruta_cal.exists():
            try:
                c = Calibracion.cargar(ruta_cal)
                if not c.tam_frame or c.tam_frame == [w, h]:
                    self._poner_cal(c, f"(cargada automáticamente)")
            except Exception:
                pass
        if self.cal is None:
            self.info_cal.setText("⚠ Sin calibrar — pulsa <b>Calibrar sobre la imagen</b>")
        self._mostrar_frame0()
        self.barra.showMessage(f"Fuente lista: {nombre}")
        self._actualizar_botones()

    def _ruta_cal_auto(self):
        if self.es_archivo:
            return Path(self.fuente).with_suffix(".calib.json")
        return Path("calibracion_camara.json")

    def _mostrar_frame0(self):
        img = self.frame0.copy()
        if self.calibrando:
            dibujar_puntos_calibracion(img, self.puntos, self.c_metodo.currentData())
        elif self.cal is not None:
            dibujar_calibracion(img, self.cal)
        self.video.mostrar(a_qimage(img))

    # ── calibración ──
    def _iniciar_calibracion(self):
        self.calibrando = True
        self.puntos = []
        self.tabs.setCurrentIndex(2)
        self._texto_calibracion()
        self._mostrar_frame0()

    def _texto_calibracion(self):
        metodo = self.c_metodo.currentData()
        nombres = NOMBRES_PUNTOS[metodo]
        i = len(self.puntos)
        extra = ("Marca un rectángulo REAL sobre la pista (p. ej. ancho de calzada × distancia entre 2 postes)."
                 if metodo == "homografia" else "Marca 2 puntos cuya distancia real conoces (p. ej. dos postes).")
        self.banner.setText(f"🎯 <b>Calibración</b> — haz clic en el punto <b>{nombres[i]}</b>.  {extra}"
                            f"  <i>(clic derecho = deshacer)</i>")
        self.banner.show()

    def _clic_video(self, x, y):
        if not self.calibrando:
            return
        metodo = self.c_metodo.currentData()
        n = 4 if metodo == "homografia" else 2
        self.puntos.append((x, y))
        self._mostrar_frame0()
        if len(self.puntos) < n:
            self._texto_calibracion()
            return
        dlg = DialogoMedidas(metodo, self)
        if dlg.exec() != QDialog.Accepted:
            self.puntos.pop()
            self._texto_calibracion()
            self._mostrar_frame0()
            return
        h, w = self.frame0.shape[:2]
        if metodo == "homografia":
            c = Calibracion("homografia", self.puntos, ancho=dlg.a.value(), largo=dlg.b.value(), tam_frame=[w, h])
        else:
            c = Calibracion.desde_escala(self.puntos[0], self.puntos[1], dlg.a.value(), [w, h])
        if not c.es_valida():
            QMessageBox.warning(self, "Calibración inválida",
                                "Los 4 puntos no forman un cuadrilátero válido.\n"
                                "Respeta el orden: sup-izq → sup-der → inf-der → inf-izq.")
            self.puntos = []
            self._texto_calibracion()
            self._mostrar_frame0()
            return
        self.calibrando = False
        self.banner.hide()
        c.guardar(self._ruta_cal_auto())
        self._poner_cal(c, "(guardada)")
        self._mostrar_frame0()

    def _deshacer_punto(self):
        if self.calibrando and self.puntos:
            self.puntos.pop()
            self._texto_calibracion()
            self._mostrar_frame0()

    def _poner_cal(self, c, nota=""):
        self.cal = c
        self.info_cal.setText(f"✔ {c.descripcion()} <span style='color:#8a90a0'>{nota}</span>")
        self._actualizar_botones()

    def _cargar_cal(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Cargar calibración", "", "Calibración (*.json)")
        if ruta:
            try:
                self._poner_cal(Calibracion.cargar(ruta), "(cargada)")
                self._mostrar_frame0()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Archivo de calibración inválido:\n{e}")

    def _guardar_cal(self):
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar calibración", "calibracion.json", "JSON (*.json)")
        if ruta:
            self.cal.guardar(ruta)

    # ── configuración ──
    def _preset_limite(self):
        v = self.c_tipo_via.currentData()
        if v is not None:
            self.s_limite.blockSignals(True)
            self.s_limite.setValue(v)
            self.s_limite.blockSignals(False)
            self._cambio_limite()

    def _cambio_limite(self):
        lim = self.s_limite.value()
        if self.hilo and self.hilo.isRunning():
            self.hilo.limite = lim
        self.lin_lim_t.setValue(lim)
        self.lin_lim_h.setValue(lim)
        self._refrescar_graficos()

    # ── análisis ──
    def _iniciar(self):
        if self.chk_movil.isChecked() and self.cal.metodo != "homografia":
            QMessageBox.warning(self, "Calibración", "Con cámara en movimiento hay que calibrar con el método de "
                                "homografía (4 puntos): la escala simple no corrige la perspectiva.")
            return
        self.calibrando = False
        self.banner.hide()
        nombre = Path(self.fuente).stem if self.es_archivo else "camara"
        self.carpeta = Path("resultados") / f"{datetime.now():%Y%m%d_%H%M%S}_{nombre}"
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.vehiculos = []
        self._limpiar_resultados()
        cfg = {"modelo": self.c_modelo.currentData(), "conf": self.s_conf.value(), "limite": self.s_limite.value(),
               "guardar_video": self.chk_video.isChecked(), "carpeta": self.carpeta,
               "camara_movil": self.chk_movil.isChecked()}
        self.hilo = Analizador(self.fuente, self.es_archivo, self.cal, cfg)
        self.hilo.frame_listo.connect(self.video.mostrar)
        self.hilo.estado.connect(self.barra.showMessage)
        self.hilo.en_vivo.connect(self._en_vivo)
        self.hilo.vehiculo.connect(self._nuevo_vehiculo)
        self.hilo.activos.connect(self._actualizar_activos)
        self.hilo.progreso.connect(lambda i, n: (self.prog.setMaximum(n), self.prog.setValue(i)))
        self.hilo.error.connect(lambda e: QMessageBox.critical(self, "Error en el análisis", e))
        self.hilo.finished.connect(self._fin)
        self.prog.setMaximum(0 if not self.es_archivo else 100)
        self.hilo.start()
        self.tabs.setCurrentIndex(0)
        self._actualizar_botones()

    def _pausar(self):
        if self.hilo:
            self.hilo.pausado = not self.hilo.pausado
            self.b_pausa.setText("▶ Continuar" if self.hilo.pausado else "⏸ Pausa")
            self.barra.showMessage("En pausa" if self.hilo.pausado else "Analizando…")

    def _detener(self):
        if self.hilo:
            self.hilo.detener = True
            self.hilo.pausado = False

    def _fin(self):
        self.b_pausa.setText("⏸ Pausa")
        self.prog.setMaximum(max(self.prog.maximum(), 1))
        self._actualizar_botones()
        n = len(self.vehiculos)
        self.barra.showMessage(f"Análisis terminado ✔  {n} vehículos medidos · resultados en {self.carpeta}")

    def _en_vivo(self, d):
        self.t_video = d["t"]
        self.vivo.setText(f"🟢 En escena: <b>{d['en_escena']}</b>  ·  Tiempo: <b>{formato_tiempo(d['t'])}</b>"
                          f"  ·  Procesamiento: <b>{d['fps']:.1f} FPS</b>  ·  Límite: <b>{self.s_limite.value():.0f} km/h</b>")
        if self.vehiculos:
            self._kpi_flujo()

    # ── resultados ──
    def _limpiar_resultados(self):
        self.tabla.setRowCount(0)
        self.t_tipos.setRowCount(0)
        self.filas_id = {}
        self.scatter.clear()
        if self.barras is not None:
            self.p_hist.removeItem(self.barras)
            self.barras = None
        for k in (self.k_total, self.k_prom, self.k_p85, self.k_max, self.k_inf, self.k_flujo):
            k.poner("—")
        lim = self.s_limite.value() if hasattr(self, "s_limite") else 30
        self.lin_lim_t.setValue(lim)
        self.lin_lim_h.setValue(lim)

    def _nuevo_vehiculo(self, v):  # vehículo que ya salió de escena: velocidad registrada final
        self.vehiculos.append(v)
        self._actualizar_activos([{**v, "final": True}])
        self._refrescar_graficos()

    def _actualizar_activos(self, lista):
        """Una fila por ID; se actualiza en vivo mientras el vehículo está en escena."""
        self.tabla.setSortingEnabled(False)
        for v in lista:
            it = self.filas_id.get(v["id"])
            if it is None:
                r = self.tabla.rowCount()
                self.tabla.insertRow(r)
                for c in range(7):
                    x = QTableWidgetItem()
                    x.setTextAlignment(Qt.AlignCenter)
                    self.tabla.setItem(r, c, x)
                it = self.filas_id[v["id"]] = self.tabla.item(r, 0)
            r = it.row()
            vel = v["vel"]
            if v.get("final"):
                estado = "Finalizado"
            elif vel is None:
                estado = "Midiendo…"
            else:
                estado = "Parado" if vel < 3 else "En movimiento"
            vmax = v.get("vel_max")
            vals = [v["id"], v["clase"], None if vel is None else round(vel, 1),
                    None if vmax is None else round(vmax, 1), v["sentido"],
                    formato_tiempo(v["t"]) if v.get("t") is not None else "", estado]
            for c, val in enumerate(vals):
                self.tabla.item(r, c).setData(Qt.DisplayRole, "" if val is None else val)
        self.tabla.setSortingEnabled(True)

    def _kpi_flujo(self):
        t = max(self.t_video, max(v["t"] for v in self.vehiculos), 1)
        self.k_flujo.poner(f"{len(self.vehiculos) / (t / 60):.1f}", "vehículos / minuto")

    def _refrescar_graficos(self):
        if not self.vehiculos:
            return
        vs = np.array([v["vel"] for v in self.vehiculos])
        mov = int((vs >= 3).sum())
        lim = self.s_limite.value()
        self.k_total.poner(str(len(vs)), "en la zona calibrada")
        self.k_prom.poner(f"{vs.mean():.1f} km/h", f"mediana {np.median(vs):.1f}")
        p85 = np.percentile(vs, 85)
        self.k_p85.poner(f"{p85:.1f} km/h", "85 % va más lento", ROJO_HEX if p85 > lim else VERDE_HEX)
        self.k_max.poner(f"{vs.max():.1f} km/h", f"mínima {vs.min():.1f}", ROJO_HEX if vs.max() > lim else None)
        self.k_inf.poner(f"{mov}", f"{100 * mov / len(vs):.0f} % de los vehículos")
        self._kpi_flujo()

        # histograma (bins de 5 km/h)
        tope = max(vs.max(), lim) + 10
        bordes = np.arange(0, tope + 5, 5)
        cuenta, _ = np.histogram(vs, bins=bordes)
        centros = bordes[:-1] + 2.5
        colores = [pg.mkBrush(ROJO_HEX if c > lim else VERDE_HEX) for c in centros]
        if self.barras is not None:
            self.p_hist.removeItem(self.barras)
        self.barras = pg.BarGraphItem(x=centros, height=cuenta, width=4.4, brushes=colores, pen=None)
        self.p_hist.addItem(self.barras)
        paso = max(1, int(np.ceil(cuenta.max() / 6)))
        self.p_hist.getAxis("left").setTickSpacing(paso, paso)
        self.p_hist.setYRange(0, cuenta.max() + paso * 0.5)

        # dispersión en el tiempo
        self.scatter.setData([{"pos": (v["t"], v["vel"]),
                               "brush": pg.mkBrush(ROJO_HEX if v["vel"] > lim else VERDE_HEX)}
                              for v in self.vehiculos])

        # por tipo
        tipos = {}
        for v in self.vehiculos:
            tipos.setdefault(v["clase"], []).append(v)
        self.t_tipos.setRowCount(len(tipos))
        for r, (k, lst) in enumerate(sorted(tipos.items(), key=lambda kv: -len(kv[1]))):
            fila = [k, len(lst), f"{np.mean([x['vel'] for x in lst]):.1f} km/h", sum(x["vel"] >= 3 for x in lst)]
            for c, val in enumerate(fila):
                it = QTableWidgetItem(str(val))
                it.setTextAlignment(Qt.AlignCenter)
                self.t_tipos.setItem(r, c, it)

    # ── reporte ──
    def _exportar(self):
        if not self.vehiculos:
            return
        d = self.carpeta
        with (d / "vehiculos.csv").open("w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=["id", "clase", "vel", "vel_max", "sentido", "t", "hora"], extrasaction="ignore")
            wr.writeheader()
            wr.writerows(self.vehiculos)

        def img64(widget):
            ruta = d / f"_{id(widget)}.png"
            widget.grab().save(str(ruta))
            b = base64.b64encode(ruta.read_bytes()).decode()
            ruta.unlink()
            return f"<img src='data:image/png;base64,{b}' style='max-width:100%;border-radius:8px'>"

        tarjetas = "".join(
            f"<div class='k'><div class='kt'>{k.t.text()}</div><div class='kv'>{k.v.text()}</div>"
            f"<div class='ks'>{k.s.text()}</div></div>"
            for k in (self.k_total, self.k_prom, self.k_p85, self.k_max, self.k_inf, self.k_flujo))
        filas = "".join(
            f"<tr><td>{v['id']}</td><td>{v['clase']}</td>"
            f"<td>{v['vel']:.1f}</td><td>{v['vel_max']:.1f}</td><td>{v['sentido']}</td>"
            f"<td>{formato_tiempo(v['t'])}</td><td>{'Parado' if v['vel'] < 3 else 'En movimiento'}</td></tr>"
            for v in self.vehiculos)
        fuente = Path(self.fuente).name if self.es_archivo else str(self.fuente)
        cap_ruta = d / "captura.jpg"
        captura = (f"<img src='data:image/jpeg;base64,{base64.b64encode(cap_ruta.read_bytes()).decode()}' "
                   f"style='max-width:100%;border-radius:8px'>" if cap_ruta.exists() else img64(self.video))
        html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Reporte Radar de Velocidad</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#f4f6fa;color:#1d2330;max-width:1100px;margin:auto;padding:24px}}
h1{{margin-bottom:0}}.meta{{color:#667}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}
.k{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 3px #0002}}.kt{{color:#667;font-size:13px}}
.kv{{font-size:26px;font-weight:700}}.ks{{color:#889;font-size:12px}}.g{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden}}
th,td{{padding:6px 10px;text-align:center;border-bottom:1px solid #eee}}th{{background:#1d2330;color:#fff}}
tr.x td{{background:#ffe5e6;color:#a8071a;font-weight:600}}.f{{display:flex;flex-wrap:wrap;gap:12px}}
figure{{margin:0;background:#fff;padding:8px;border-radius:10px;box-shadow:0 1px 3px #0002}}
figure img{{height:150px;border-radius:6px}}figcaption{{font-size:13px;text-align:center;color:#a8071a}}</style></head><body>
<h1>🚦 Reporte — Radar de Velocidad</h1>
<p class="meta">Fuente: <b>{fuente}</b> · Generado: {datetime.now():%d/%m/%Y %H:%M} · Límite: <b>{self.s_limite.value():.0f} km/h</b>
 · Calibración: {self.cal.descripcion()} · Modelo: {self.c_modelo.currentText()}</p>
<div class="grid">{tarjetas}</div>
<h2>Gráficos</h2><div class="g">{img64(self.p_hist)}{img64(self.p_tiempo)}</div>
<h2>Vista del análisis</h2>{captura}
<h2>Detalle de vehículos</h2><table><tr><th>ID</th><th>Tipo</th><th>Velocidad (km/h)</th><th>Máx. mostrada</th>
<th>Sentido</th><th>Tiempo</th><th>Estado</th></tr>{filas}</table>
<p class="meta">Velocidad = mediana de las mediciones del vehículo dentro de la zona calibrada. YOLO26 + ByteTrack.</p>
</body></html>"""
        ruta = d / "reporte.html"
        ruta.write_text(html, encoding="utf-8")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta.resolve())))
        self.barra.showMessage(f"Reporte exportado: {ruta}")

    def closeEvent(self, e):
        if self.hilo and self.hilo.isRunning():
            self.hilo.detener = True
            self.hilo.wait(3000)
        super().closeEvent(e)


ESTILO = f"""
* {{ font-size: 13px; }}
QMainWindow, QWidget {{ background:#17181c; color:#e6e8ee; }}
QLabel#titulo {{ font-size:20px; font-weight:700; padding:4px 0 8px 0; }}
QGroupBox {{ border:1px solid #2a2d35; border-radius:8px; margin-top:14px; padding:10px 8px 8px 8px; font-weight:600; }}
QGroupBox::title {{ subcontrol-origin: margin; left:10px; padding:0 4px; color:{ACENTO}; }}
QPushButton {{ background:#262932; border:1px solid #333744; border-radius:6px; padding:7px 10px; }}
QPushButton:hover {{ background:#2f3340; border-color:{ACENTO}; }}
QPushButton:disabled {{ color:#5c6170; background:#1d1f25; border-color:#24262d; }}
QPushButton#primario {{ background:{ACENTO}; border:none; font-weight:700; color:white; }}
QPushButton#primario:disabled {{ background:#2a3a55; color:#7d8799; }}
QComboBox, QDoubleSpinBox, QLineEdit {{ background:#21232a; border:1px solid #333744; border-radius:5px; padding:4px 6px; }}
QLabel#info {{ color:#a9afbd; font-size:12px; }}
QLabel#banner {{ background:#3a2f0b; color:#ffd666; border:1px solid #614700; border-radius:6px; padding:8px 10px; }}
QLabel#vivo {{ color:#a9afbd; padding:2px 4px; }}
QFrame#tarjeta {{ background:#1f2128; border:1px solid #2a2d35; border-radius:10px; }}
QLabel#tarjetaTitulo {{ color:#8a90a0; font-size:12px; background:transparent; }}
QLabel#tarjetaValor {{ font-size:22px; font-weight:700; background:transparent; }}
QLabel#tarjetaSub {{ color:#6f7585; font-size:11px; background:transparent; }}
QTabWidget::pane {{ border:1px solid #2a2d35; border-radius:6px; }}
QTabBar::tab {{ background:#1f2128; padding:7px 12px; border-top-left-radius:6px; border-top-right-radius:6px; margin-right:2px; }}
QTabBar::tab:selected {{ background:#2a2d35; color:white; }}
QTableWidget {{ background:#1b1d22; gridline-color:#2a2d35; border:none; }}
QHeaderView::section {{ background:#262932; color:#cfd3dc; border:none; padding:5px; font-weight:600; }}
QProgressBar {{ background:#21232a; border:1px solid #333744; border-radius:5px; text-align:center; height:18px; }}
QProgressBar::chunk {{ background:{ACENTO}; border-radius:4px; }}
QListWidget, QTextBrowser {{ background:#1b1d22; border:none; }}
QStatusBar {{ background:#121317; color:#a9afbd; }}
QScrollArea {{ border:none; }}
"""


def main():
    pg.setConfigOptions(antialias=True, background="#1b1d22", foreground="#cfd3dc")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(ESTILO)
    w = Ventana()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()