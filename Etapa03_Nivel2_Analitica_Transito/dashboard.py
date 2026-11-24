"""Dashboard del Monitor de tráfico (Nivel 2).

    python -m streamlit run dashboard.py      (o doble clic en iniciar_dashboard.bat)

Dos páginas:
- Analítica: lee las sesiones guardadas (historial.csv) y muestra la analítica.
- Procesar video: corre YOLO + ByteTrack sobre un video dentro del navegador, guarda
  la sesión y la abre en Analítica (lo mismo que demo_nivel1/demo_yolo.py, sin ventana aparte).
"""
from __future__ import annotations

import sys

import streamlit as st

st.set_page_config(page_title="Analítica de tránsito", page_icon=":material/traffic:", layout="wide")

VERSION_MINIMA = (1, 67)
if tuple(int(x) for x in st.__version__.split(".")[:2]) < VERSION_MINIMA:
    st.error(
        f"Este dashboard necesita Streamlit {'.'.join(map(str, VERSION_MINIMA))} o más nuevo, y se abrió con "
        f"Streamlit {st.__version__} (Python {sys.version.split()[0]}).\n\n"
        "Probablemente el comando `streamlit` de la consola apunta a otra instalación de Python. "
        "Ciérralo (Ctrl+C) y ábrelo con:\n\n"
        "```\npython -m streamlit run dashboard.py\n```\n\no con doble clic en `iniciar_dashboard.bat`.",
        icon=":material/error:")
    st.stop()

pagina = st.navigation([
    st.Page("app_pages/analitica.py", title="Analítica", icon=":material/insights:", default=True,
            help="Mapa de calor, carril más usado, vehículos por intervalo, origen-destino y anomalías "
                 "de una sesión guardada. Filtra por tipo, carril y tramo del video."),
    st.Page("app_pages/video.py", title="Procesar video", icon=":material/videocam:",
            help="Detecta y sigue los vehículos de un video con YOLO + ByteTrack, mostrando el resultado "
                 "en vivo, y genera una sesión nueva lista para analizar."),
], position="top")
pagina.run()
