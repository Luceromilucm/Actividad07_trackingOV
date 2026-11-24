"""Pruebas con datos sintéticos: verifican los criterios de aceptación (sección 7)."""
from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.path import Path as Poligono

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import analitica as an  # noqa: E402
import generar_sintetico  # noqa: E402
from carriles import detectar_carriles  # noqa: E402
from datos import ErrorSesion, cargar_sesion, limpiar, resumen_por_vehiculo  # noqa: E402
from graficos import calcular_mapa_calor  # noqa: E402
from logger import RegistroHistorial  # noqa: E402
from main_analitica import analizar  # noqa: E402

SALIDAS = ["barras_por_intervalo.png", "mapa_calor.png", "carriles.png", "clases_por_intervalo.png",
           "trayectorias.png", "conteo_linea.png", "resumen.json", "resumen.txt",
           "vehiculos_por_intervalo.csv", "vehiculos_por_carril.csv", "vehiculos.csv",
           "origen_destino.png", "origen_destino.csv", "anomalias.png", "anomalias.csv",
           "mapa_calor_animado.gif", "barras_por_hora.png", "vehiculos_por_hora.csv"]


@pytest.fixture(scope="module")
def sesion_sintetica(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("sintetico")
    verdad = generar_sintetico.generar(carpeta, semilla=7)
    return carpeta, verdad


@pytest.fixture(scope="module")
def resultado(sesion_sintetica):
    carpeta, verdad = sesion_sintetica
    return analizar(carpeta, silencioso=True), verdad, carpeta


def copia(origen: Path, destino: Path, sin=()) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    for f in origen.iterdir():
        if f.is_file() and f.name not in sin:
            shutil.copy(f, destino / f.name)
    return destino


# --- Criterio: genera todas las salidas sin errores --------------------------------

def test_genera_todas_las_salidas(resultado):
    _, _, carpeta = resultado
    for nombre in SALIDAS:
        archivo = carpeta / "salidas" / nombre
        assert archivo.exists() and archivo.stat().st_size > 0, f"falta {nombre}"


def test_cli_devuelve_cero(sesion_sintetica, tmp_path, capsys):
    from main_analitica import main
    carpeta = copia(sesion_sintetica[0], tmp_path / "cli")
    assert main([str(carpeta), "--intervalo", "minuto"]) == 0
    assert "El carril más utilizado fue" in capsys.readouterr().out


# --- Criterio: limpieza de tracks cortos o fragmentados ---------------------------------

def test_filtra_tracks_cortos_y_quietos(resultado):
    res, verdad, _ = resultado
    lim = res["limpieza"]
    assert lim["descartados_cortos"] == verdad["tracks_fragmento"] + verdad["tracks_fantasma"]
    assert lim["descartados_quietos"] == verdad["tracks_fijos"]
    assert res["total_vehiculos"] == verdad["vehiculos_reales"]


def test_cuenta_vehiculos_no_detecciones(sesion_sintetica):
    s = limpiar(cargar_sesion(sesion_sintetica[0]))
    v = resumen_por_vehiculo(s.df)
    tabla = an.vehiculos_por_intervalo(v, s.duracion_s, "minuto", s.inicio)
    assert tabla["vehiculos"].sum() == s.df["track_id"].nunique()
    assert tabla["vehiculos"].sum() < len(s.df) / 50   # hay muchas detecciones por vehículo


# --- Criterio: carril más utilizado (con y sin carriles.json) --------------------------------

def test_sin_carriles_json_detecta_automaticamente(resultado):
    res, verdad, carpeta = resultado
    assert not (carpeta / "carriles.json").exists()
    assert res["carriles_origen"] == "automaticos"
    assert res["carril_mas_usado"]["carril"] == verdad["carril_mas_usado"]
    assert res["frase_carril"].startswith(f"El carril más utilizado fue {verdad['carril_mas_usado']}")
    conteo = {r["carril"]: r["vehiculos"] for r in res["por_carril"]}
    for carril, n in verdad["por_carril"].items():
        assert abs(conteo[carril] - n) <= 2


def test_con_carriles_dibujados(sesion_sintetica, tmp_path):
    carpeta, verdad = sesion_sintetica
    destino = copia(carpeta, tmp_path / "dibujados")
    (destino / "carriles.json").write_text(json.dumps({"carriles": verdad["carriles"]}), encoding="utf-8")
    res = analizar(destino, silencioso=True)
    assert res["carriles_origen"] == "dibujados"
    assert {r["carril"]: r["vehiculos"] for r in res["por_carril"]} == verdad["por_carril"]


def test_imprime_frase_del_carril(sesion_sintetica, tmp_path, capsys):
    destino = copia(sesion_sintetica[0], tmp_path / "frase")
    analizar(destino)
    salida = capsys.readouterr().out
    assert "El carril más utilizado fue Carril 2" in salida
    texto = (destino / "salidas" / "resumen.txt").read_text(encoding="utf-8")
    assert "El carril más utilizado fue Carril 2" in texto


def test_deteccion_automatica_con_otro_numero_de_carriles():
    # 2 carriles verticales sin perspectiva, 70 / 30
    rng = np.random.default_rng(0)
    filas = []
    for tid in range(100):
        x = 300 if tid < 70 else 700
        for f in range(30):
            filas.append((tid, f, x + rng.normal(0, 8), 600 - 15 * f))
    df = pd.DataFrame(filas, columns=["track_id", "frame", "px", "py"])
    asignacion, carriles = detectar_carriles(df)
    assert len(carriles) == 2
    assert asignacion.value_counts().to_dict() == {"Carril 1": 70, "Carril 2": 30}


# --- Criterio: mapa de calor superpuesto sobre la calle ---------------------------------

def test_mapa_calor_sobre_la_calle(sesion_sintetica):
    carpeta, verdad = sesion_sintetica
    s = limpiar(cargar_sesion(carpeta))
    calor = calcular_mapa_calor(s.df, s.ancho, s.alto)
    assert calor.shape == (s.alto, s.ancho) == s.fondo.shape[:2]
    assert 0 <= calor.min() and calor.max() <= 1
    # casi todo el calor cae dentro de los carriles verdaderos
    yy, xx = np.mgrid[0:s.alto:4, 0:s.ancho:4]
    pts = np.c_[xx.ravel(), yy.ravel()]
    dentro = np.zeros(len(pts), bool)
    for c in verdad["carriles"]:
        dentro |= Poligono(np.asarray(c["poligono"])).contains_points(pts)
    muestra = calor[::4, ::4].ravel()
    assert muestra[dentro].sum() / muestra.sum() > 0.9


def test_un_auto_detenido_no_domina_el_mapa():
    # 20 vehículos pasan por x=100; uno se queda detenido 2000 frames en x=500
    filas = [(t, f, 100.0, 50.0 + f * 10) for t in range(20) for f in range(30)]
    filas += [(99, f, 500.0, 200.0) for f in range(2000)]
    df = pd.DataFrame(filas, columns=["track_id", "frame", "px", "py"])
    por_vehiculo = calcular_mapa_calor(df, 640, 400)
    por_deteccion = calcular_mapa_calor(df, 640, 400, por_deteccion=True)
    assert por_vehiculo[200, 100] > por_vehiculo[200, 500]
    assert por_deteccion[200, 500] > por_deteccion[200, 100]


# --- Criterio: barras con videos cortos y largos ---------------------------------------------

def test_intervalo_automatico():
    assert an.elegir_intervalo(90) == "10s"
    assert an.elegir_intervalo(600) == "minuto"
    assert an.elegir_intervalo(5 * 3600) == "hora"
    assert an.elegir_intervalo(600, "hora") == "hora"


def test_picos_en_los_minutos_correctos(resultado):
    res, verdad, carpeta = resultado
    assert res["intervalo"] == "minuto"
    tabla = pd.read_csv(carpeta / "salidas" / "vehiculos_por_intervalo.csv")
    assert len(tabla) == 10
    top2 = set(tabla.nlargest(2, "vehiculos").index)
    assert top2 == set(verdad["minutos_pico"])
    assert res["pico"]["intervalo"] in ("15:33", "15:37")


def test_video_corto_agrupa_cada_10_segundos(sesion_sintetica, tmp_path):
    destino = copia(sesion_sintetica[0], tmp_path / "corto")
    df = pd.read_csv(destino / "historial.csv")
    df[df["timestamp_s"] < 90].to_csv(destino / "historial.csv", index=False)
    meta = json.loads((destino / "meta.json").read_text(encoding="utf-8"))
    meta["duracion_s"] = 90
    (destino / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    res = analizar(destino, silencioso=True)
    assert res["intervalo"] == "10s"
    tabla = pd.read_csv(destino / "salidas" / "vehiculos_por_intervalo.csv")
    assert len(tabla) == 9 and tabla["intervalo"].iloc[0] == "15:30:00"


def test_video_largo_agrupa_por_hora_y_marca_parciales(sesion_sintetica, tmp_path):
    # se estira la sesión a 3 h 20 min empezando 15:30: el primer intervalo (15:30-16:00)
    # y el último (18:00-18:50) son parciales
    destino = copia(sesion_sintetica[0], tmp_path / "largo")
    df = pd.read_csv(destino / "historial.csv")
    df["timestamp_s"] = df["timestamp_s"] * 20
    df.to_csv(destino / "historial.csv", index=False)
    meta = json.loads((destino / "meta.json").read_text(encoding="utf-8"))
    meta["duracion_s"] = 600 * 20
    (destino / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    res = analizar(destino, silencioso=True)
    assert res["intervalo"] == "hora"
    tabla = pd.read_csv(destino / "salidas" / "vehiculos_por_intervalo.csv")
    assert list(tabla["intervalo"]) == ["15:00–16:00", "16:00–17:00", "17:00–18:00", "18:00–19:00"]
    assert list(tabla["parcial"]) == [True, False, False, True]
    assert tabla["vehiculos"].sum() == res["total_vehiculos"]


def test_sin_meta_json_usa_tiempo_relativo(sesion_sintetica, tmp_path):
    destino = copia(sesion_sintetica[0], tmp_path / "sin_meta", sin=("meta.json",))
    res = analizar(destino, silencioso=True)
    assert res["inicio"] is None
    tabla = pd.read_csv(destino / "salidas" / "vehiculos_por_intervalo.csv")
    assert tabla["intervalo"].iloc[0] == "00:00"
    assert res["linea_conteo"]["origen"] == "automática"


# --- Línea virtual y velocidades -------------------------------------------------------------

def test_linea_virtual_cuenta_ambos_sentidos(resultado):
    res, verdad, _ = resultado
    t = res["linea_conteo"]["totales"]
    arriba, abajo = t["Hacia arriba (se alejan)"], t["Hacia abajo (se acercan)"]
    esperado_abajo = verdad["por_carril"]["Carril 3"] + 1          # + el auto a contramano
    assert abs(abajo - esperado_abajo) <= 2
    assert arriba + abajo <= res["total_vehiculos"]
    assert arriba + abajo >= 0.95 * res["total_vehiculos"]


def test_velocidad_promedio_por_vehiculo(resultado):
    res, _, carpeta = resultado
    v = pd.read_csv(carpeta / "salidas" / "vehiculos.csv")
    assert res["velocidades"]["promedio_kmh"] == pytest.approx(v["velocidad_kmh"].mean(), abs=0.1)
    assert res["velocidades"]["excesos"] == int(v["excedio"].sum())


# --- Criterio: el CSV sobrevive a un cierre inesperado; carpeta por sesión --------------------

def test_logger_vacia_a_disco_antes_de_cerrar(tmp_path):
    reg = RegistroHistorial(tmp_path / "s1", fps=30, ancho=640, alto=480, flush_cada=50)
    for f in range(120):
        reg.registrar(1, f, "car", 10, 10 + f, 50, 40 + f)
    # sin cerrar: lo ya vaciado debe estar en disco
    en_disco = pd.read_csv(tmp_path / "s1" / "historial.csv")
    assert len(en_disco) >= 100
    reg.cerrar()
    assert len(pd.read_csv(tmp_path / "s1" / "historial.csv")) == 120


def test_logger_cierra_aunque_haya_excepcion(tmp_path):
    with pytest.raises(RuntimeError):
        with RegistroHistorial(tmp_path / "s2", fps=25, ancho=640, alto=480) as reg:
            for f in range(10):
                reg.registrar(3, f, "bus", 0, 0, 10, 10, velocidad_kmh=20, excede_limite=False)
            raise RuntimeError("el tracker falló")
    df = pd.read_csv(tmp_path / "s2" / "historial.csv")
    meta = json.loads((tmp_path / "s2" / "meta.json").read_text(encoding="utf-8"))
    assert len(df) == 10 and df["timestamp_s"].iloc[-1] == pytest.approx(9 / 25)
    assert df["cy"].iloc[0] == 10 and df["cx"].iloc[0] == 5   # punto inferior central
    assert meta["duracion_s"] == pytest.approx(10 / 25)


def test_logger_crea_carpeta_por_sesion_y_analiza_al_cerrar(sesion_sintetica, tmp_path):
    """Reproduce la sesión sintética a través del logger, como lo haría el Nivel 1."""
    import cv2
    from datetime import datetime
    origen = sesion_sintetica[0]
    df = pd.read_csv(origen / "historial.csv")
    fondo = cv2.imread(str(origen / "frame_referencia.jpg"))
    base = tmp_path / "sesiones"
    carpetas = []
    for _ in range(2):
        carpeta = __import__("logger").crear_carpeta_sesion(base, datetime(2026, 11, 18, 15, 30))
        carpetas.append(carpeta)
    assert carpetas[0] != carpetas[1]

    with RegistroHistorial(carpetas[0], fuente="sintetico", fps=30, limite_kmh=50,
                           inicio=datetime(2026, 11, 18, 15, 30), analizar_al_cerrar=True) as reg:
        for f in range(0, 300, 10):
            reg.guardar_frame(fondo)
        for r in df.itertuples():
            reg.registrar(r.track_id, r.frame, r.clase, r.x1, r.y1, r.x2, r.y2,
                          r.velocidad_kmh, r.excede_limite)
    for nombre in ("historial.csv", "meta.json", "frame_referencia.jpg"):
        assert (carpetas[0] / nombre).exists()
    resumen = json.loads((carpetas[0] / "salidas" / "resumen.json").read_text(encoding="utf-8"))
    assert resumen["carril_mas_usado"]["carril"] == "Carril 2"


# --- Criterio: el código lee solo el CSV ----------------------------------------------------

def test_no_importa_yolo_ni_tracker():
    prohibidos = {"ultralytics", "yolo", "torch", "supervision", "bytetrack", "deep_sort", "boxmot"}
    for archivo in SRC.glob("*.py"):
        arbol = ast.parse(archivo.read_text(encoding="utf-8"))
        for nodo in ast.walk(arbol):
            if isinstance(nodo, (ast.Import, ast.ImportFrom)):
                nombres = [a.name for a in nodo.names] if isinstance(nodo, ast.Import) else [nodo.module or ""]
                for n in nombres:
                    assert n.split(".")[0].lower() not in prohibidos, f"{archivo.name} importa {n}"


def test_error_claro_si_faltan_columnas(tmp_path):
    (tmp_path / "historial.csv").write_text("track_id,frame\n1,1\n", encoding="utf-8")
    with pytest.raises(ErrorSesion, match="faltan columnas"):
        cargar_sesion(tmp_path)


# --- Nivel C: reporte HTML y dashboard -----------------------------------------------

def test_reporte_html_autocontenido(resultado):
    res, verdad, carpeta = resultado
    html = (carpeta / "salidas" / "reporte.html").read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") == 9          # las 9 figuras embebidas
    assert html.count("data:image/gif;base64,") == 1          # y el mapa de calor animado
    assert 'src="http' not in html and "<script" not in html  # no depende de nada externo
    assert res["frase_carril"] in html
    assert any("15:37 a 15:38" in c for c in res["conclusiones"])
    assert any("Carril 2 concentró" in c for c in res["conclusiones"])


def test_dashboard_carga_sin_errores(sesion_sintetica, tmp_path):
    from streamlit.testing.v1 import AppTest
    raiz = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(raiz / "dashboard.py"), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Analítica de tránsito"
    for vista in ["Tráfico en el tiempo", "Mapa de calor", "Carriles", "Trayectorias", "Línea virtual",
                  "Origen-destino", "Anomalías", "Vehículos"]:
        at.segmented_control(key="vista").set_value(vista).run()
        assert not at.exception, vista


# --- Nivel C: anomalías, origen-destino y GIF ------------------------------------------

def test_detecta_el_vehiculo_detenido_y_el_de_contramano(resultado):
    res, verdad, _ = resultado
    casos = res["anomalias"]["casos"]
    detenidos = {c["track_id"] for c in casos if c["tipo"] == "Detenido"}
    contramano = {c["track_id"] for c in casos if c["tipo"] == "Sentido contrario"}
    assert detenidos == {verdad["detenido_track_id"]}
    # los 2 «carteles» fijos del sintético están fuera de la calzada: no son estacionados
    assert not [c for c in casos if c["tipo"] == "Estacionado"]
    assert res["anomalias"]["fijos_fuera_de_la_via"] == verdad["tracks_fijos"]
    assert contramano == {verdad["contramano_track_id"]}
    # pocas falsas alarmas: las atípicas son una fracción mínima del total
    atipicas = [c for c in casos if c["tipo"] == "Trayectoria atípica"]
    assert len(atipicas) <= 0.02 * res["total_vehiculos"]


def test_origen_destino_sigue_el_sentido_de_los_carriles(resultado):
    res, verdad, _ = resultado
    od = res["origen_destino"]
    assert (od["principal"]["origen"], od["principal"]["destino"]) == ("Sur", "Norte")
    pares = {(p["origen"], p["destino"]): p["vehiculos"] for p in od["pares"]}
    # el carril de bajada (Carril 3 en el sintético, sentido [0, 1]) va de norte a sur
    bajada = verdad["por_carril"]["Carril 3"]
    assert abs(pares.get(("Norte", "Sur"), 0) - bajada) <= 0.1 * bajada


def test_origen_destino_en_calle_diagonal():
    """Calle en diagonal (como un video real con curva): de arriba-izquierda a abajo-derecha."""
    import origen_destino as od
    filas = []
    for tid in range(30):
        x0 = 300 + 5 * tid
        for k in range(40):
            filas.append({"track_id": tid, "px": x0 + 22 * k, "py": 380 + 17 * k})
    m = od.matriz(pd.DataFrame(filas))
    assert (m["principal"]["origen"], m["principal"]["destino"]) == ("Norte", "Sur")


def test_cajas_cortadas_por_el_borde_no_se_usan(tmp_path):
    """Un auto que sale por abajo no debe parecer que se mueve en horizontal."""
    filas = []
    for k in range(30):
        y2 = min(400 + 30 * k, 1080)          # al final la caja queda recortada en el borde inferior
        x = 900 + (0 if y2 < 1080 else 25 * (k - 23))
        filas.append({"track_id": 1, "frame": k, "timestamp_s": k / 25, "clase": "car",
                      "x1": x - 80, "y1": y2 - 120, "x2": x + 80, "y2": y2})
    pd.DataFrame(filas).to_csv(tmp_path / "historial.csv", index=False)
    (tmp_path / "meta.json").write_text(json.dumps({"ancho": 1920, "alto": 1080, "fps": 25}))
    sesion = cargar_sesion(tmp_path)
    assert sesion.df["py"].max() < 1080 - 3
    assert sesion.df["px"].nunique() == 1


def test_gif_se_acumula(resultado):
    from PIL import Image
    _, _, carpeta = resultado
    gif = Image.open(carpeta / "salidas" / "mapa_calor_animado.gif")
    assert gif.n_frames >= 20
    gif.seek(2)
    temprano = np.asarray(gif.convert("RGB"), float)
    gif.seek(gif.n_frames - 1)
    final = np.asarray(gif.convert("RGB"), float)
    assert np.abs(final - temprano).mean() > 1.0     # la calle se «enciende» con el tiempo


# --- Dashboard: página «Procesar video» ------------------------------------------------------

VIDEOS = Path(__file__).resolve().parents[1] / "demo_nivel1" / "videos"


def _yolo_disponible() -> bool:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return any(VIDEOS.glob("*.mp4"))


def test_pagina_video_carga():
    from streamlit.testing.v1 import AppTest
    raiz = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(raiz / "dashboard.py"), default_timeout=120).run()
    at.switch_page("app_pages/video.py").run()
    assert not at.exception
    assert at.title[0].value == "Procesar video"


@pytest.mark.skipif(not _yolo_disponible(), reason="sin ultralytics o sin videos de la demo")
def test_pagina_video_procesa_y_analiza(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import logger
    monkeypatch.setattr(logger, "RAIZ", tmp_path)        # la sesión se guarda en tmp, no en data/
    raiz = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(raiz / "dashboard.py"), default_timeout=180).run()
    at.switch_page("app_pages/video.py").run()
    at.slider(key="video_max_pexels_854671_highway.mp4").set_value(2)
    at.button[-1].click().run()                           # «Iniciar»
    assert not at.exception
    assert "proc_video" in at.session_state
    for _ in range(4):                                    # cada ejecución procesa algunos cuadros
        at.run()
    assert at.session_state["proc_video"].frame_idx > 0
    assert at.image                                       # se muestra el cuadro anotado
    at.button(key="btn_detener").click().run()            # «Detener y analizar»
    assert not at.exception
    res = at.session_state["video_resultado"]
    assert (Path(res["carpeta"]) / "historial.csv").exists()


def test_barras_por_hora_siempre(resultado):
    """El enunciado pide «a qué hora pasaron más autos»: hay tabla por hora aunque la sesión sea corta."""
    res, _, _ = resultado
    assert res["intervalo"] != "hora"                    # la sesión sintética dura 10 min
    assert sum(r["vehiculos"] for r in res["por_hora"]) == res["total_vehiculos"]
    assert res["hora_pico"]["intervalo"] == "15:00–16:00"


@pytest.mark.skipif(not _yolo_disponible(), reason="sin ultralytics o sin videos de la demo")
def test_demo_guarda_video_procesado(tmp_path):
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo_nivel1"))
    from demo_yolo import VIDEO_DEFECTO, ProcesadorVideo
    proc = ProcesadorVideo(VIDEO_DEFECTO, max_segundos=1, carpeta=tmp_path)
    while proc.paso(dibujar=False) is not None:
        pass
    carpeta = proc.cerrar()
    video = carpeta / "video_procesado.mp4"
    assert video.exists() and not (carpeta / "video_procesado_tmp.mp4").exists()
    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == proc.frame_idx
    assert json.loads((carpeta / "meta.json").read_text(encoding="utf-8"))["video_procesado"] == "video_procesado.mp4"
