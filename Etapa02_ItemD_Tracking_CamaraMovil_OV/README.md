# Etapa 02, ítem d: tracking 2D con cámara en movimiento mediante odometría visual

## Planteamiento

> «Si la cámara estuviera en movimiento y se necesitaría realizar una tarea de tracking, la implementación del ítem b no funcionará. Encuentre la solución en función a la odometría visual.»

El tracker del ítem b mide todas las magnitudes en coordenadas de la imagen. Con la cámara en movimiento, el desplazamiento observado de un objeto es:

```
Δp_observado = Δp_objeto + Δp_cámara
```

Como consecuencia, los objetos estáticos parecen moverse, la línea de conteo se desplaza respecto a la escena, las velocidades se sobrestiman y el mapa de calor se dispersa.

## Solución

`tracking_vo_gui.py` estima el movimiento de la cámara con odometría visual y expresa las posiciones de los objetos en un sistema de referencia fijo: el del primer frame.

1. **Puntos del fondo:** se detectan esquinas con `cv2.goodFeaturesToTrack`, excluyendo las cajas detectadas por YOLO.
2. **Seguimiento:** flujo óptico Lucas-Kanade piramidal con verificación de ida y vuelta.
3. **Estimación del movimiento:** RANSAC con un modelo de similaridad (4 grados de libertad) o una homografía (8 grados de libertad).
4. **Keyframes:** cada frame se estima contra un keyframe y no contra el frame anterior, para limitar la acumulación de error.
5. **Coordenadas del mundo:** el conteo con la línea virtual, la velocidad, las trayectorias y el mapa de calor se calculan sobre el sistema de referencia fijo.
6. **Mosaico:** se construye una vista panorámica de la escena con las trayectorias compensadas.

El método funciona con ByteTrack y con BoT-SORT. La aplicación incluye además:

- un simulador de cámara en movimiento, que permite medir el error de la odometría frente a un movimiento conocido;
- un comparador de configuraciones con y sin odometría visual.

## Ejecución

```bash
pip install ultralytics opencv-python numpy pandas matplotlib pillow lap
python tracking_vo_gui.py
```

1. Seleccione la fuente (webcam o archivo de video). Un video ubicado en la carpeta `videos/` junto al script se carga automáticamente.
2. Seleccione el modelo y el tracker. Para video aéreo se recomienda `yolo11s-obb.pt`.
3. Active la odometría visual y elija el modelo de movimiento.
4. Dibuje la línea de conteo y presione **Iniciar**. Al finalizar, use **Analizar trayectorias** o exporte el CSV.

El video del experimento (`carros arriba.mp4`) no se incluye por su tamaño. Es una grabación aérea con dron de 1920 × 1080 px, 30 fps y 17,6 s de duración.

## Resultados

| Archivo (`resultados/`) | Contenido |
|---|---|
| `analisis_trayectorias_sin_vo.png` | Trayectorias y mapa de calor sin compensación |
| `analisis_trayectorias_con_vo.png` | Trayectorias compensadas sobre el mosaico, mapa de calor y pose de la cámara |
| `comparacion_configuraciones.png` | Comparación de configuraciones: FPS, cruces e identificadores |
| `resumen_trayectorias_sin_vo.csv`, `resumen_trayectorias_con_vo.csv` | Resumen por identificador: desplazamiento, rectitud y velocidad |
| `trayectorias_sin_vo.csv` | Trayectorias crudas: `id, frame, time_s, x, y, xw, yw, v_px_s, class, conf` |

Resultados sobre 50 trayectorias del video de prueba:

| Métrica | Sin odometría visual | Con odometría visual |
|---|---|---|
| Desplazamiento mediano | 278,8 px | 8,8 px |
| Trayectorias identificadas como estacionarias (< 20 px) | 5 | 34 |
| Velocidad media mediana | 39,4 px/s | 16,8 px/s |
| Rectitud mediana de los vehículos en circulación | 0,872 | 0,953 |
