# Nivel 2: Analítica de Tránsito

Lee el historial de vehículos que guarda el radar (Nivel 1) y genera el mapa de calor,
el carril más utilizado y los vehículos por hora/minuto. No depende de YOLO ni del tracker:
solo lee `historial.csv`.

## Instalación

```
pip install -r requirements.txt
```

## Probarlo sin el Nivel 1 (datos sintéticos)

```
python src/generar_sintetico.py               # crea data/sintetico/ (288 vehículos, 10 min)
python src/main_analitica.py data/sintetico --abrir
pytest tests/                                 # 33 pruebas
```

Resultado esperado: pico a las 15:37 (y otro a las 15:33) y
**"El carril más utilizado fue Carril 2 (178 vehículos, 62%)"**.

## Interfaces para ver los resultados

**Reporte HTML** (se genera solo en cada análisis): `<sesion>/salidas/reporte.html`, un único archivo
con indicadores, conclusiones automáticas, todos los gráficos y las tablas. `--abrir` lo abre en el navegador.
Para regenerarlo sin volver a analizar: `python src/reporte.py data/sintetico`.

**Dashboard interactivo** (Streamlit):

```
python -m streamlit run dashboard.py
```

(o doble clic en `iniciar_dashboard.bat`). Usa `python -m streamlit` y no solo `streamlit`: en este equipo el
comando `streamlit` apunta a una instalación vieja (1.57, Python 3.12) que no tiene las funciones que usa el
dashboard. Se necesita Streamlit 1.67 o más nuevo.

Se abre en http://localhost:8501. En la barra lateral se elige la sesión (cualquier carpeta de `data/` con
`historial.csv`), el intervalo y la limpieza; se filtra por tipo de vehículo, carril y tramo del video; y con
**En vivo** relee el CSV cada pocos segundos mientras el Nivel 1 sigue grabando. Vistas: resumen, tráfico en el
tiempo, mapa de calor, carriles, trayectorias, línea virtual, origen-destino (diagrama de flujo), anomalías y
la lista de vehículos (descargable en CSV).
El botón **Generar reporte HTML** corre el análisis completo y permite descargar el reporte.

El dashboard tiene dos páginas (menú de arriba): **Analítica** (lo anterior) y **Procesar video**. En
**Procesar video** se elige un video de `demo_nivel1/videos/` (o se sube uno), se ajustan la resolución de
YOLO, la confianza y hasta qué segundo procesar, y al presionar **Iniciar** el video se muestra en la página
con las cajas, las estelas, las velocidades y los contadores (lo mismo que la ventana de `demo_yolo.py`).
**Detener y analizar** (o el final del video) guarda la sesión, genera la analítica y ofrece **Ver en
Analítica**, que abre esa sesión. Si se cambia de página a mitad del video, el proceso queda en pausa y
sigue al volver. Necesita `demo_nivel1/requirements.txt` instalado.

## Con una sesión real

La carpeta de sesión debe tener `historial.csv` y, si es posible, `meta.json` y
`frame_referencia.jpg`.

```
python src/definir_carriles.py data/sesiones/<sesion>     # opcional: dibujar los carriles
python src/main_analitica.py data/sesiones/<sesion>
```

Sin `carriles.json`, los carriles se detectan automáticamente.

Opciones: `--intervalo auto|hora|minuto|10s`, `--por-deteccion` (mapa de calor por tiempo de
permanencia), `--min-frames 8`, `--min-desplazamiento 40`, `--linea x1,y1,x2,y2`,
`--min-detenido 10` (segundos quieto para marcar un vehículo como detenido), `--sin-gif`.

Las detecciones cuya caja toca el borde de la imagen se ignoran para ubicar al vehículo: al entrar o
salir del cuadro, la caja se recorta y su punto inferior se desliza por el borde.

## Demo con video real (YOLO + ByteTrack)

`demo_nivel1/` simula el radar del Nivel 1 con un video real y prueba todo el flujo: detecta y sigue los
vehículos, los registra con `RegistroHistorial` y, al presionar **Q** (o al terminar el video), genera el
análisis y abre el reporte. Está fuera de `src/` a propósito: el módulo del Nivel 2 no importa YOLO.

```
pip install -r demo_nivel1/requirements.txt          # ultralytics + lap (una sola vez)
python demo_nivel1/demo_yolo.py                      # o doble clic en demo_nivel1/iniciar_demo.bat
python demo_nivel1/demo_yolo.py demo_nivel1/videos/roboflow_vehicles.mp4
python demo_nivel1/demo_yolo.py --sin-ventana        # sin ventana: ~18 fps en CPU
```

- Modelo: `demo_nivel1/modelos/yolo11n.pt` (el más liviano; en este equipo no hay GPU). Clases de COCO:
  auto, moto, bus y camión. `--conf 0.4` por defecto: con 0,3 el cartel del radar del video sale como bus.
- Videos (no van a git, ~97 MB): `pexels_854671_highway.mp4` (por defecto: autopista, 1 min, cámara fija),
  `roboflow_vehicles.mp4` (autopista con los dos sentidos, 21 s, 4K) y `pexels_6269483.mp4` (vista aérea
  lejana; los autos salen muy chicos).
- **Velocidad (km/h):** con perspectiva no sirve un solo `px_por_metro`. `demo_nivel1/calibracion.py` ajusta
  `s(y) = (a·y + b) / (c·y + 1)` (distancia a lo largo de la calle según la fila de la imagen) con marcas a
  distancia conocida, y la demo mide solo dentro del tramo calibrado. El video por defecto ya trae
  `pexels_854671_highway.calibracion.json`: 12 líneas discontinuas detectadas automáticamente, periodo
  supuesto de 9 m (norma británica de vía rápida: 2 m de línea + 7 m de espacio) y límite de 70 mph = 112 km/h.
  Para otro video: `python demo_nivel1/calibrar.py <video> --periodo <m>` y clic en las marcas.
- Resultado de referencia en `data/sesiones/demo_pexels_autopista/`: 152 IDs del tracker → 38 vehículos
  válidos (los IDs cortos son autos lejanos que el tracker fragmenta), dos carriles al 50 %, velocidad
  promedio 111 km/h.

## Integración con el Nivel 1

```python
import sys; sys.path.insert(0, "ruta/a/nivel2_analitica/src")
from logger import RegistroHistorial

with RegistroHistorial(fuente="video.mp4", fps=fps, limite_kmh=40, analizar_al_cerrar=True) as reg:
    for i, frame in enumerate(frames):
        reg.guardar_frame(frame)
        for t in tracks:                       # salida de ByteTrack / BoT-SORT
            reg.registrar(t.id, i, t.clase, *t.caja_xyxy, velocidad_kmh=t.v, excede_limite=t.v > 40)
```

Al salir del `with` se cierra el CSV y se generan las salidas en `data/sesiones/<fecha_hora>/salidas/`.
Si se cierra la consola con la X, el análisis se lanza a mano con `main_analitica.py`.

## Dónde queda cada cosa

Cada corrida (demo por consola, página «Procesar video» o el Nivel 1 con el logger) crea su carpeta:

```
data/sesiones/AAAA-MM-DD_HH-MM-SS/
├── historial.csv          <- el historial de todos los vehículos (una fila por detección)
├── meta.json              fps, resolución, hora de inicio, límite, calibración
├── frame_referencia.jpg   la calle sin autos (mediana de varios cuadros)
├── video_procesado.mp4    el video con cajas, velocidades y contadores (solo la demo)
└── salidas/               gráficos, tablas y reporte.html (se generan solos al cerrar)
```

El video procesado se graba a 1280 px de ancho; si hay `ffmpeg` en el PATH se convierte a H.264 (~12 MB por
minuto, se ve en el navegador y en el dashboard), y si no queda en MPEG-4 (se abre con VLC). `--sin-video`
lo desactiva.

## Salidas (`<sesion>/salidas/`)

| Archivo | Contenido |
|---|---|
| `barras_por_hora.png` | vehículos por hora del día: «¿a qué hora pasaron más autos?» (siempre) |
| `barras_por_intervalo.png` | vehículos únicos por hora/minuto/10 s, con el pico resaltado |
| `mapa_calor.png` | presencia de vehículos sobre la calle, con el % de cada carril |
| `carriles.png` | carriles y vehículos por carril: "El carril más utilizado fue …" |
| `clases_por_intervalo.png` | autos, motos, camiones y buses por intervalo |
| `trayectorias.png` | trayectorias coloreadas por velocidad |
| `conteo_linea.png` | cruces de la línea virtual por sentido y flujo por minuto |
| `mapa_calor_animado.gif` | el mapa de calor acumulándose a lo largo de la sesión |
| `origen_destino.png` / `.csv` | matriz origen-destino por zonas (norte, sur, este, oeste) |
| `anomalias.png` / `.csv` | vehículos detenidos, estacionados, en sentido contrario y trayectorias atípicas |
| `reporte.html` | reporte autocontenido con KPIs, conclusiones, gráficos y tablas |
| `resumen.txt` / `resumen.json` | resumen en texto y en datos (incluye las conclusiones) |
| `*.csv` | tablas por intervalo, carril, clase y vehículo |

### Cómo se calculan los extras

- **Origen-destino:** desde donde aparece cada vehículo se lanza un rayo en contra de su rumbo de entrada;
  el borde de la zona de tráfico por el que sale es el origen (igual para el destino, con el rumbo de
  salida). Así funciona con perspectiva (la calle es un trapecio) y con calles en diagonal.
- **Detenido:** quieto (menos de ~1 % de la altura de la imagen por segundo) durante `--min-detenido` s
  seguidos. **Estacionado:** un track que nunca se movió, dentro de la zona de tráfico; los objetos fijos
  fuera de ella (carteles confundidos con autos) se descartan.
- **Sentido contrario:** avanza a más de 120° del sentido de su carril.
- **Trayectoria atípica:** cada trayectoria se remuestrea en 10 puntos y se agrupa con DBSCAN (`eps`
  automático); las que no caen en ningún grupo son atípicas.
