# Actividad 07: Tracking y Odometría Visual

Repositorio de código de la Actividad 07 (segunda unidad) del curso **Visión Artificial**, Escuela Profesional de Ingeniería de Sistemas, Universidad Nacional del Altiplano – Puno, 2026.

## Descripción

La actividad estudia la diferencia entre dos problemas de la visión artificial y demuestra con implementaciones que son complementarios:

- **Tracking:** seguir objetos a lo largo de un video y mantener su identidad. Responde a la pregunta *¿dónde está el objeto?*
- **Odometría visual:** estimar el movimiento de la propia cámara a partir de las imágenes que captura. Responde a la pregunta *¿cómo se mueve la cámara?*

Cuando la cámara está fija basta con el tracking. Cuando la cámara se mueve, el desplazamiento observado de cada objeto mezcla su propio movimiento con el de la cámara, y hace falta la odometría visual para separarlos.

## Contenido

El trabajo se organiza en tres etapas. Cada parte práctica tiene su propia carpeta, con el código fuente, un `README.md` de uso y los resultados que se presentan en el informe.

| Etapa | Parte | Carpeta |
|---|---|---|
| 01. Conceptos | Simulación de odometría visual con KLT (Pregunta 6) | `Etapa01_P6_Simulacion_KLT/` |
| 02. Aplicaciones | Ítem a: odometría visual frente a tracking 2D | `Etapa02_ItemA_OV_vs_Tracking2D/` |
| 02. Aplicaciones | Ítem b: tracking 2D con YOLO + ByteTrack, conteo y trayectorias | `Etapa02_ItemB_Tracking2D_YOLO/` |
| 02. Aplicaciones | Ítem c: velocidad de los objetos y mapas de calor | `Etapa02_ItemC_Velocidad_MapaCalor/` |
| 02. Aplicaciones | Ítem d: tracking con cámara en movimiento mediante odometría visual | [`Etapa02_ItemD_Tracking_CamaraMovil_OV/`](Etapa02_ItemD_Tracking_CamaraMovil_OV/) |
| 03. Proyecto integrador | Nivel 1: radar de velocidad con cámara fija | [`Etapa03_Nivel1_Radar_Velocidad/`](Etapa03_Nivel1_Radar_Velocidad/) |
| 03. Proyecto integrador | Nivel 2: analítica de tránsito | `Etapa03_Nivel2_Analitica_Transito/` |
| 03. Proyecto integrador | Nivel 3: radar de velocidad con cámara en movimiento | `Etapa03_Nivel3_Radar_CamaraMovil/` |

Las carpetas sin enlace se incorporarán a medida que se suba cada parte. Se mantiene la convención de nombres `EtapaXX_<parte>_<tema>/`.

## Tecnologías

- Python 3.10 o superior
- Ultralytics YOLO (YOLOv8, YOLO11 y YOLO26) para la detección de objetos
- ByteTrack y BoT-SORT para el seguimiento multiobjeto
- OpenCV para el flujo óptico Lucas-Kanade, ORB (FAST + BRIEF), RANSAC y la homografía
- NumPy, pandas y Matplotlib para el análisis y las gráficas

## Instalación

```bash
git clone https://github.com/Luceromilucm/Actividad07_trackingOV.git
cd Actividad07_trackingOV
pip install ultralytics opencv-python numpy pandas matplotlib pillow lap
```

Cada carpeta puede tener dependencias adicionales en su propio `requirements.txt`. Las instrucciones de ejecución de cada parte están en su `README.md`.

## Datos no incluidos

Para mantener el repositorio liviano no se versionan:

- **Videos de entrada y videos anotados:** pesan entre 35 y 120 MB. Cada README indica qué video se usó y dónde colocarlo.
- **Pesos de los modelos YOLO (`.pt`):** Ultralytics los descarga automáticamente la primera vez que se ejecuta el programa.
