# Etapa 02 · Ítem d — Tracking 2D con cámara en movimiento compensado con odometría visual

**Subido por:** CENTENO MAMANI, Mirian Lucero

> *«Si la cámara estuviera en movimiento y se necesitaría realizar una tarea de tracking, la implementación del ítem b no funcionará. Encuentre la solución en función a la odometría visual.»*

## Problema
Con la cámara en movimiento, el desplazamiento que se ve en la imagen es la suma de dos movimientos:

```
Δp_observado = Δp_objeto (real) + Δp_cámara (ego-movimiento)
```

El tracker del ítem b mide todo en píxeles de la imagen. Por eso los autos estacionados parecen moverse, la línea de conteo «barre» la escena y el mapa de calor se dispersa.

## Solución
La aplicación `tracking_vo_gui.py` (interfaz en Tkinter) estima el movimiento de la cámara con **odometría visual** y expresa todas las posiciones en un sistema de coordenadas fijo al mundo, el del primer frame:

1. Detecta puntos del **fondo** con `goodFeaturesToTrack`, enmascarando las cajas de YOLO.
2. Los sigue con flujo óptico **Lucas-Kanade** piramidal y verificación de ida y vuelta.
3. Estima el movimiento de la cámara con **RANSAC**: similaridad (4 GL) u homografía (8 GL). Incluye también un modo 3D con la matriz esencial.
4. Lo estima contra un **keyframe**, no frame a frame, para reducir la deriva.
5. Calcula el conteo con la línea virtual, la velocidad, las trayectorias y el mapa de calor en **coordenadas del mundo**.
6. Construye un **mosaico** de la escena y dibuja encima las trayectorias compensadas.

Funciona con **ByteTrack** y con **BoT-SORT**. Además incluye un **simulador de cámara móvil**, que permite medir el error de la odometría con *ground truth*, y un **comparador** sin VO / con VO.

## Ejecución
```bash
pip install ultralytics opencv-python numpy pandas matplotlib pillow lap
python tracking_vo_gui.py
```
1. En **1. Fuente** elige la webcam o un archivo de video. Si colocas un video en una carpeta `videos/` junto al script, se carga automáticamente.
2. Elige el modelo y el tracker. Para video aéreo (dron) usa `yolo11s-obb.pt`.
3. Activa **Odometría visual** y elige el modelo de movimiento.
4. Dibuja la línea de conteo y pulsa **Iniciar**. Al final pulsa **Analizar trayectorias** o exporta el CSV.

**Video usado en el informe:** `carros arriba.mp4` (dron, 1920×1080, 30 fps, 529 frames, unos 17,6 s). No se sube por su tamaño (44 MB). Cualquier video aéreo o grabado con la cámara en movimiento sirve.

## Resultados (`resultados/`)
| Archivo | Contenido |
|---|---|
| `analisis_trayectorias_sin_vo.png` | Ítem b sin compensación: arcos falsos y mapa de calor disperso |
| `analisis_trayectorias_con_vo.png` | Con odometría visual: trayectorias sobre el mosaico, mapa de calor y pose de la cámara |
| `comparacion_configuraciones.png` | Pestaña de comparación (FPS, cruces, IDs, puntaje) |
| `resumen_trayectorias_sin_vo.csv` / `resumen_trayectorias_con_vo.csv` | Resumen por ID (desplazamiento, rectitud, velocidad) |
| `trayectorias_sin_vo.csv` | Trayectorias crudas: `id, frame, time_s, x, y, xw, yw, v_px_s, class, conf` |

**Resultado principal:** el desplazamiento mediano de las 50 trayectorias baja de **278,8 px a 8,8 px** con odometría visual, y los autos identificados como estacionarios suben de 5 a 34.
