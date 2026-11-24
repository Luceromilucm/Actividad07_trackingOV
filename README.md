# Actividad 07 · Tracking y Odometría Visual

**Universidad Nacional del Altiplano – Puno**
Facultad de Ingeniería Mecánica Eléctrica, Electrónica y Sistemas · Escuela Profesional de Ingeniería de Sistemas
**Curso:** Visión Artificial · **Docente:** Ing. FERNANDEZ CHAMBI, Mayenka · **Año:** 2026

**Objetivo:** comprender la diferencia entre el *tracking* y la *odometría visual*, y demostrar con ejemplos que son complementarias.

## Integrantes
- CENTENO MAMANI, Mirian Lucero
- COLQUEHUANCA COLQUEHUANCA, Jhosemy Brissett
- HANCCO VILCA, Yennybel Rocio
- MAMANI APAZA, Sadith Leidy

## Estructura del repositorio
Cada parte práctica de la actividad está en su propia carpeta, con su código, un `README.md` y los resultados que aparecen en el informe. El README de cada carpeta indica quién la subió.

| Etapa | Parte | Carpeta | Estado |
|---|---|---|---|
| 01 · Conceptos | Pregunta 6: simulación de odometría visual con KLT | `Etapa01_P6_Simulacion_KLT/` | pendiente |
| 02 · Aplicaciones | Ítem a: OV vs. tracking 2D (animaciones y presentación) | `Etapa02_ItemA_OV_vs_Tracking2D/` | pendiente |
| 02 · Aplicaciones | Ítem b: tracking 2D con YOLO + ByteTrack, conteo y trayectorias | `Etapa02_ItemB_Tracking2D_YOLO/` | pendiente |
| 02 · Aplicaciones | Ítem c: velocidad (px/s → m/s) y mapas de calor | `Etapa02_ItemC_Velocidad_MapaCalor/` | pendiente |
| 02 · Aplicaciones | **Ítem d: tracking con cámara en movimiento mediante odometría visual** | [`Etapa02_ItemD_Tracking_CamaraMovil_OV/`](Etapa02_ItemD_Tracking_CamaraMovil_OV/) | ✅ |
| 03 · Proyecto integrador | **Nivel 1: radar de velocidad (cámara fija)** | [`Etapa03_Nivel1_Radar_Velocidad/`](Etapa03_Nivel1_Radar_Velocidad/) | ✅ |
| 03 · Proyecto integrador | Nivel 2: analítica de tránsito (CSV, mapa de calor, barras por hora) | `Etapa03_Nivel2_Analitica_Transito/` | pendiente |
| 03 · Proyecto integrador | Nivel 3: radar con cámara en movimiento (ORB: FAST + BRIEF) | `Etapa03_Nivel3_Radar_CamaraMovil/` | pendiente |

> **Para las integrantes que suban su parte:** usen el nombre de carpeta de la tabla, incluyan un `README.md` con la línea «Subido por: APELLIDOS, Nombres» y cambien el estado a ✅.

## Qué no se sube
Para mantener el repositorio liviano **no se incluyen** los videos de entrada (pesan de 35 a 120 MB), los modelos YOLO `.pt` (Ultralytics los descarga solo la primera vez), los videos anotados de salida ni las corridas de prueba. Cada README indica qué video usar y dónde colocarlo.

## Requisitos generales
Python 3.10 o superior.
```bash
pip install ultralytics opencv-python numpy pandas matplotlib pillow lap
```
Cada carpeta puede tener además su propio `requirements.txt`.
