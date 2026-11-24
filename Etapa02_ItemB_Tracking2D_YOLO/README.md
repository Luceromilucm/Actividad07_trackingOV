# Tracking 2D YOLO + ByteTrack/BoT-SORT

Aplicación GUI para tracking 2D en tiempo real con YOLOv8/yolo11, ByteTrack/BoT-SORT, conteo de cruces de línea y análisis de trayectorias. Incluye modo comparación de varios modelos/trackers.

## Características

- **Detección y tracking** en tiempo real con webcam o archivo de vídeo.
- **Contador de cruces de línea** editable con el ratón (cruces positivas/negativas).
- **Guardado de trayectorias** en CSV: ID, frame, tiempo, posición, clase, confianza.
- **Análisis de trayectorias**: duración, distancia, desplazamiento, rectitud, velocidad, ángulo.
- **Modo comparación**: ejecuta varios modelos y trackers sobre el mismo vídeo, acumula métricas y calcula un **puntaje global 0‑100** (ponderación de estabilidad, velocidad y precisión si se proveen valores reales).
- **Soporte para cajas orientadas (OBB)**: modelos `yolo11n-obb.pt` y `yolo11s-obb.pt` (clases DOTA: small/large vehicle).
- **Interfaz moderna** con modo oscuro/claro (botón en la barra superior derecha).
- **Mapa de calor** de posiciones de objetos.
- **Exportación de video** anotado (opcional).

## Instalación

```bash
pip install -U ultralytics opencv-python numpy pandas matplotlib pillow
```

> Los modelos OBB requieren **Ultralytics >= 8.3**. La versión actual de pip debería satisfacer esadependencia.

Coloca los archivos de pesos YOLO en la carpeta del proyecto o en la misma ubicación que el script. Los modelos compatibles (según la interfaz):

- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` (COCO estándar)
- `yolo11n.pt`, `yolo11s.pt` (COCO versión 11)
- `yolo11n-obb.pt`, `yolo11s-obb.pt` (cajas orientadas, DOTA)

## Ejecución

```bash
python tracking_yolo_gui.py
```

### Modo individual

1. Selecciona la fuente (webcam o archivo de vídeo).
2. Elige el modelo YOLO y el tracker (`bytetrack` o `botsort`).
3. Marca las clases a seguir (Persona, Bicicleta, Auto, Moto, Bus, Camión).
4. Ajusta la confianza y largo de estela.
5. Define la línea de conteo arrastrando con el ratón sobre la imagen.
6. Presiona **Iniciar**.

Al finalizar, se generan en `salida/`:
- `trayectorias.csv` – registro completo por cada detección.
- `resumen_trayectorias.csv` – métricas resumidas por ID.
- `resultado.mp4` (solo si está activado el guardado de video).

### Modo comparación

1. Carga el mismo vídeo para todas las ejecuciones.
2. Añade múltiples configuraciones (modelo + tracker + clases + confianza) en la pestaña de comparación.
3. Presiona **Ejecutar comparativa**.
4. La aplicación procesará cada combinación, medirá tiempos, contará cruces y calculará un **puntaje 0‑100** basado en:
   - **Estabilidad**: 0.6 × (1 – % tracks cortos) + 0.4 × continuidad.
   - **Velocidad**: FPS normalizado respecto al más rápido.
   - **Precisión** (opcional): error respecto a conteo/counts reales ingresados.
5. Al terminar, aparece una tabla clasificada y gráficos comparativos (puntaje, FPS, cruces, IDs únicos).

## Parámetros clave

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `conf` | Umbral de confianza YOLO | `0.30` |
| `trail_len` | Longitud de la estela visual | `40` |
| `line_rel` | Línea de conteo [x1,y1,x2,y2] en fracción (0‑1) | `[0.0, 0.5, 1.0, 0.5]` (vertical central) |
| `max_frames` | Límite de frames a procesar (0 = todos) | `0` |
| `save_video` | Grabar video anotado | `False` |
| `imgsz` | Tamaño de entrada YOLO (OBB usaron 1024) | `640` |

## Salidas generadas

- `salida/trayectorias.csv` – Todas las detecciones (`id, frame, time_s, x, y, class, conf`).
- `salida/resumen_trayectorias.csv` – Resumen por objeto (`id, clase, frames, duracion_s, distancia_px, desplazamiento_px, rectitud, velocidad_px_s, angulo_deg`).
- `salida/resultado.mp4` – Video annotado (si `save_video=True`).
- En modo comparación, carpetas adicionales bajo `salida/comparacion/` con resúmenes por configuración.

## Dependencias principales

- Python ≥ 3.9
- **ultralytics** – modelos YOLOv8/yolo11
- **opencv-python** – video y procesamiento de imágenes
- **numpy** – cálculos numéricos
- **pandas** – manipulación de datos y CSV
- **matplotlib** – gráficos de análisis
- **Pillow** – imágenes para la interfaz
- **tkinter** – GUI (incluido con Python)

## Licencia

MIT – libre para usar, modificar y distribuir citando la fuente.

## Referencias

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [BoT-SORT](https://github.com/nmtsarw/BoT-SORT)
- [DOTA dataset](https://captain-whu.github.io/DOTA_dataset/)