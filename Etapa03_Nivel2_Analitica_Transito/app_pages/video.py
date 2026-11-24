"""Página «Procesar video»: YOLO + ByteTrack dentro del dashboard.

Hace lo mismo que demo_nivel1/demo_yolo.py, pero muestra el video en la página en
lugar de una ventana de OpenCV. El procesamiento avanza en un fragmento que se
vuelve a ejecutar solo cada 0,1 s y procesa unos pocos cuadros por vez: así el
botón «Detener» siempre responde, y si se cambia de página el proceso queda en
pausa (el procesador vive en st.session_state) y sigue al volver.

Esta página es la única del dashboard que importa YOLO (a través de demo_nivel1/).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "demo_nivel1"))

import reporte as rp  # noqa: E402
from datos import ErrorSesion  # noqa: E402
from main_analitica import analizar  # noqa: E402

ANCHO_VISTA = 1280          # el cuadro se reduce a este ancho antes de enviarlo al navegador
PRESUPUESTO_S = 0.15        # tiempo de procesamiento por ejecución del fragmento

st.title("Procesar video")
st.caption("Detecta y sigue los vehículos con YOLO + ByteTrack (lo que hará el radar del Nivel 1), guarda "
           "el historial y genera la analítica del Nivel 2 al terminar.")

try:
    import ultralytics  # noqa: F401
except ImportError:
    st.error("Falta ultralytics. Instálalo con:\n\n```\npip install -r demo_nivel1/requirements.txt\n```",
             title="YOLO no está instalado", icon=":material/error:")
    st.stop()

from demo_yolo import CARPETA_VIDEOS, MODELO_DEFECTO, VIDEO_DEFECTO, ProcesadorVideo, ruta_calibracion  # noqa: E402


def mmss(seg: float) -> str:
    m, s = divmod(int(seg), 60)
    return f"{m}:{s:02d}"


@st.cache_data(max_entries=16, show_spinner=False)
def info_video(ruta: str, _mtime: float) -> dict:
    """Resolución, duración y un cuadro de muestra (a los 2 s) del video."""
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(2 * fps), max(n - 1, 0)))
    ok, frame = cap.read()
    cap.release()
    muestra = cv2.resize(frame, (640, int(640 * alto / ancho)), interpolation=cv2.INTER_AREA) if ok else None
    return {"ancho": ancho, "alto": alto, "fps": fps, "duracion": n / fps, "muestra": muestra}


def ruta_corta(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(RAIZ).as_posix() if p.is_relative_to(RAIZ) else str(p)


def detener():
    st.session_state["video_detener"] = True


def finalizar(proc: ProcesadorVideo):
    """Cierra el registro y corre la analítica; deja el resultado en session_state."""
    with st.spinner("Guardando el historial y generando la analítica…"):
        carpeta = proc.cerrar()
        resultado = {"carpeta": carpeta, "video": proc.video.name, "segundos": proc.t_video,
                     "ids": len(proc.vistos)}
        try:
            resultado["resumen"] = analizar(carpeta, silencioso=True)
        except (ErrorSesion, ValueError) as e:
            resultado["error"] = str(e)
    st.session_state["video_resultado"] = resultado
    st.session_state.pop("proc_video", None)
    st.session_state.pop("video_detener", None)


# --- 1. Si había un proceso que terminó o se detuvo, se cierra primero -------------------

proc: ProcesadorVideo | None = st.session_state.get("proc_video")
if proc is not None and (proc.terminado or st.session_state.get("video_detener")):
    finalizar(proc)
    proc = None


# --- 2. Procesando: video en vivo ---------------------------------------------------------

if proc is not None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f"**{proc.video.name}** · {proc.ancho}×{proc.alto} · "
                    + ("velocidad calibrada" if proc.calib else "sin velocidad (no hay calibración)"))
        st.space("stretch")
        st.button("Detener y analizar", on_click=detener, type="primary", icon=":material/stop_circle:",
                  key="btn_detener")

    @st.fragment(run_every=0.1)
    def reproducir():
        p: ProcesadorVideo | None = st.session_state.get("proc_video")
        if p is None:
            return
        limite_t = time.perf_counter() + PRESUPUESTO_S
        while not p.terminado and time.perf_counter() < limite_t:
            cuadro = p.paso()
            if cuadro is not None:
                p.ultimo_cuadro = cuadro
        if p.terminado:
            st.rerun()          # rerun completo: el paso 1 cierra el registro y analiza

        with st.container(horizontal=True):
            st.metric("Vehículos registrados", len(p.vistos), border=True,
                      help="IDs distintos del tracker. La analítica luego descarta los tracks cortos o quietos.")
            st.metric("En pantalla", p.en_pantalla, border=True)
            st.metric("Video", f"{mmss(p.t_video)} / {mmss(p.n_frames / p.fps)}", border=True)
            st.metric("Procesamiento", f"{p.fps_proc:.1f} fps", border=True,
                      help=f"El video original va a {p.fps:.0f} fps; en CPU, YOLO procesa menos.")
            if p.calib:
                st.metric("Sobre el límite", len(p.excesos), border=True,
                          help=f"Vehículos que superaron {p.limite:g} km/h en algún momento.")
        st.progress(p.progreso)
        img = getattr(p, "ultimo_cuadro", None)
        if img is not None:
            h, w = img.shape[:2]
            if w > ANCHO_VISTA:
                img = cv2.resize(img, (ANCHO_VISTA, int(ANCHO_VISTA * h / w)), interpolation=cv2.INTER_AREA)
            st.image(img, channels="BGR", output_format="JPEG", width="stretch",
                     alt="Video con los vehículos detectados, su número, su estela y su velocidad")

    reproducir()
    st.stop()


# --- 3. Resultado del último video ----------------------------------------------------------

res = st.session_state.get("video_resultado")
if res:
    with st.container(border=True):
        st.subheader(f"Resultado: {res['video']}")
        st.caption(f"{mmss(res['segundos'])} de video procesados · {res['ids']} IDs del tracker")
        carpeta_res = Path(res["carpeta"])
        st.table({"Archivo": ["Historial (CSV)", "Gráficos y reporte", "Video procesado"],
                  "Ruta": [ruta_corta(carpeta_res / "historial.csv"), ruta_corta(carpeta_res / "salidas") + "/",
                           ruta_corta(carpeta_res / "video_procesado.mp4")
                           if (carpeta_res / "video_procesado.mp4").exists() else "no se guardó"]},
                 hide_index=True, alt="Dónde quedaron guardados el historial, los gráficos y el video")
        if "error" in res:
            st.warning(res["error"], title="No se pudo generar la analítica", icon=":material/warning:")
        else:
            resumen = res["resumen"]
            st.success(resumen["frase_carril"] + ".", icon=":material/add_road:")
            with st.container(horizontal=True):
                for k in rp.kpis(resumen):
                    st.metric(k["titulo"], k["valor"], border=True, help=k["nota"])
            with st.container(horizontal=True):
                if st.button("Ver en Analítica", type="primary", icon=":material/insights:"):
                    st.session_state["sesion_nueva"] = Path(res["carpeta"])
                    st.switch_page("app_pages/analitica.py")
                rep = carpeta_res / "salidas" / "reporte.html"
                if rep.exists():
                    st.download_button("Descargar reporte.html", rep.read_bytes(),
                                       file_name=f"reporte_{Path(res['carpeta']).name}.html", mime="text/html",
                                       icon=":material/download:")
        if (carpeta_res / "video_procesado.mp4").exists():
            with st.expander("Ver el video procesado", icon=":material/movie:"):
                st.video(str(carpeta_res / "video_procesado.mp4"),
                         alt=f"Video procesado de {res['video']} con las detecciones y las velocidades")


# --- 4. Elegir video y opciones -------------------------------------------------------------

CARPETA_VIDEOS.mkdir(parents=True, exist_ok=True)
izq, der = st.columns([3, 2])
with izq, st.container(border=True):
    st.subheader("Video")
    subido = st.file_uploader("Agregar un video", type=["mp4", "avi", "mov", "mkv"],
                              help="Se copia a demo_nivel1/videos/ y queda disponible en la lista.")
    if subido is not None:
        destino = CARPETA_VIDEOS / Path(subido.name).name
        if not destino.exists():
            destino.write_bytes(subido.getbuffer())
            st.toast(f"Video agregado: {destino.name}", icon=":material/check_circle:")
        st.session_state["video_elegido"] = destino
    videos = sorted(CARPETA_VIDEOS.glob("*.mp4")) + sorted(
        p for ext in ("*.avi", "*.mov", "*.mkv") for p in CARPETA_VIDEOS.glob(ext))
    if not videos:
        st.info("No hay videos en demo_nivel1/videos/. Sube uno arriba.", title="Sin videos")
        st.stop()
    if st.session_state.get("video_elegido") not in videos:
        st.session_state["video_elegido"] = VIDEO_DEFECTO if VIDEO_DEFECTO in videos else videos[0]
    video = st.selectbox("Video", videos, key="video_elegido",
                         format_func=lambda p: p.name + ("  · con velocidad" if ruta_calibracion(p).exists() else ""))
    info = info_video(str(video), video.stat().st_mtime)
    if info["muestra"] is not None:
        st.image(info["muestra"], channels="BGR", width="stretch", alt=f"Cuadro de muestra de {video.name}")
    st.caption(f"{info['ancho']}×{info['alto']} · {info['fps']:.0f} fps · {mmss(info['duracion'])} min")

with der, st.container(border=True):
    st.subheader("Opciones")
    calib = ruta_calibracion(video).exists()
    imgsz = st.segmented_control("Resolución de YOLO", [480, 640, 960], default=640, required=True,
                                 format_func=lambda v: f"{v} px", key="video_imgsz",
                                 help="Mayor = detecta mejor los autos lejanos, pero va más lento.")
    conf = st.slider("Confianza mínima", 0.1, 0.9, 0.4, 0.05, key="video_conf",
                     help="Con 0,3 el cartel del radar del video de la autopista sale como bus.")
    max_s = st.slider("Procesar hasta el segundo", 1, max(int(info["duracion"]), 1), max(int(info["duracion"]), 1),
                      key=f"video_max_{video.name}", help="Para pruebas rápidas, procesa solo el comienzo.")
    if calib:
        st.badge("Velocidad calibrada para este video", icon=":material/speed:", color="green")
    else:
        st.caption("Este video no tiene calibración: se registra sin velocidad. Para calibrarlo: "
                   f"`python demo_nivel1/calibrar.py demo_nivel1/videos/{video.name} --periodo <m>`")
    guardar = st.toggle("Guardar el video procesado", value=True, key="video_guardar",
                        help="Escribe video_procesado.mp4 (con cajas, velocidades y contadores) en la carpeta "
                             "de la sesión.")
    st.caption("Modelo: " + MODELO_DEFECTO.name + " (CPU). Al terminar, la sesión se guarda en data/sesiones/.")
    if st.button("Iniciar", type="primary", icon=":material/play_arrow:", width="stretch"):
        with st.spinner("Cargando YOLO…"):
            try:
                st.session_state["proc_video"] = ProcesadorVideo(
                    video, imgsz=int(imgsz), conf=float(conf), guardar_video=bool(guardar),
                    max_segundos=None if max_s >= int(info["duracion"]) else float(max_s))
            except ValueError as e:
                st.error(str(e), icon=":material/error:")
                st.stop()
        st.session_state.pop("video_resultado", None)
        st.session_state.pop("video_detener", None)
        st.rerun()
