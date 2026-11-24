"""Página «Analítica» del dashboard (Nivel 2). Se abre desde dashboard.py.

Lee las mismas sesiones que main_analitica.py (carpetas con historial.csv) y usa el
mismo motor de cálculo. Permite filtrar por tipo de vehículo, carril y tiempo, y con
«En vivo» relee el CSV cada pocos segundos mientras el Nivel 1 sigue grabando.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import analitica as an  # noqa: E402
import anomalias as anom  # noqa: E402
import graficos as gr  # noqa: E402
import origen_destino as od  # noqa: E402
import reporte as rp  # noqa: E402
import vistas  # noqa: E402
from carriles import FUERA  # noqa: E402
from datos import ErrorSesion, nombre_clase  # noqa: E402
from main_analitica import analizar, procesar  # noqa: E402

COLOR_CLASE = {nombre_clase(k): v for k, v in gr.COLORES_CLASE.items()}

# --- Datos ----------------------------------------------------------------------

def buscar_sesiones() -> list[Path]:
    """Carpetas con historial.csv dentro de data/ (la más reciente primero)."""
    encontradas = {p.parent for p in (RAIZ / "data").glob("**/historial.csv") if "salidas" not in p.parts}
    return sorted(encontradas, key=lambda p: (p / "historial.csv").stat().st_mtime, reverse=True)


def huella(carpeta: Path) -> tuple:
    """Cambia cuando cambia algún archivo de la sesión: invalida la caché."""
    return tuple((n, (carpeta / n).stat().st_mtime_ns, (carpeta / n).stat().st_size)
                 for n in ("historial.csv", "meta.json", "carriles.json", "frame_referencia.jpg")
                 if (carpeta / n).exists())


@st.cache_data(max_entries=8, show_spinner="Analizando la sesión…")
def cargar(carpeta: str, intervalo: str, min_frames: int, min_desp: float, _huella: tuple):
    return procesar(carpeta, intervalo, min_frames, min_desp)


@st.cache_data(max_entries=16, show_spinner="Calculando el mapa de calor…")
def imagen_calor(carpeta: str, ids: tuple, por_deteccion: bool, _r, _huella: tuple, _clave: tuple):
    df = _r.sesion.df[_r.sesion.df["track_id"].isin(ids)]
    tabla = an.conteo_por_carril(_r.asignacion.loc[list(ids)], _r.carriles_def)
    mas = an.carril_mas_usado(tabla)["carril"] if tabla["vehiculos"].sum() else None
    return vistas.mapa_calor(_r.sesion, df, _r.carriles_def, tabla, mas, por_deteccion)


def mmss(seg: float) -> str:
    m, s = divmod(int(round(seg)), 60)
    return f"{m}:{s:02d}"


# --- Barra lateral ----------------------------------------------------------------

sesiones = buscar_sesiones()
with st.sidebar:
    st.header("Sesión", divider="gray")
    if not sesiones:
        st.warning("No hay sesiones en data/. Genera una con `python src/generar_sintetico.py`.",
                   title="Sin sesiones")
        st.stop()
    # Al terminar de procesar un video, la página «Procesar video» deja aquí su sesión
    nueva = st.session_state.pop("sesion_nueva", None)
    if nueva is not None and Path(nueva) in sesiones:
        st.session_state["carpeta"] = Path(nueva)
    carpeta = st.selectbox("Carpeta", sesiones, format_func=lambda p: str(p.relative_to(RAIZ / "data")),
                           key="carpeta")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        vivo = st.toggle("En vivo", key="vivo", help="Relee historial.csv mientras el Nivel 1 sigue grabando.")
        cada = st.number_input("cada (s)", 2, 60, 5, key="cada", disabled=not vivo, width=110)

    st.header("Análisis", divider="gray")
    intervalo = st.segmented_control("Agrupar por", ["auto", "hora", "minuto", "10s"], default="auto",
                                     required=True, key="intervalo")
    with st.container(horizontal=True):
        min_frames = st.number_input("Mín. detecciones", 1, 200, 8, key="min_frames")
        min_desp = st.number_input("Mín. recorrido (px)", 0, 2000, 40, step=10, key="min_desp")
    por_deteccion = st.toggle("Mapa de calor por permanencia", key="por_det",
                              help="Suma cada detección (tiempo detenido) en vez de cada vehículo una vez.")


def cargar_seguro():
    """Carga la sesión; si falla, muestra el motivo y devuelve None."""
    try:
        return cargar(str(carpeta), intervalo, int(min_frames), float(min_desp), huella(carpeta))
    except ErrorSesion as e:
        st.error(str(e), title="No se pudo analizar la sesión")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, ValueError) as e:
        st.warning(f"El CSV se está escribiendo; reintentando… ({e})", title="Esperando datos")
    return None


# Los widgets de la barra lateral van en el script principal (no dentro del fragmento):
# Streamlit no deja que un fragmento cree widgets fuera de su propio contenedor.
r0 = cargar_seguro()
if r0 is None:
    if vivo:
        st.button("Reintentar", icon=":material/refresh:")
    st.stop()

v0 = r0.vehiculos
dur0 = r0.sesion.duracion_s
with st.sidebar:
    st.header("Filtros", divider="gray")
    clases = sorted(v0["clase"].unique(), key=lambda c: -int((v0["clase"] == c).sum()))
    sel_clases = st.pills("Tipo de vehículo", clases, selection_mode="multi", default=clases,
                          format_func=nombre_clase, key=f"f_clase_{carpeta.name}")
    nombres = [c["nombre"] for c in r0.carriles_def] + ([FUERA] if (v0["carril"] == FUERA).any() else [])
    sel_carr = st.pills("Carril", nombres, selection_mode="multi", default=nombres, key=f"f_carril_{carpeta.name}")
    t_max = round(dur0 / 60, 2)
    t_rango = st.slider("Tramo del video (min)", 0.0, t_max, (0.0, t_max),
                        step=max(round(t_max / 100, 2), 0.01), format="%.2f", key=f"f_t_{carpeta.name}")
    # En vivo el video sigue creciendo: si el tramo llega al final, no se corta.
    t_fin = float("inf") if t_rango[1] >= t_max else t_rango[1] * 60

    st.header("Reporte", divider="gray")
    if st.button("Generar reporte HTML", icon=":material/description:", width="stretch", key="generar",
                 help="Corre el análisis completo (sin filtros) y escribe salidas/ y reporte.html."):
        with st.spinner("Generando gráficos y reporte…"):
            analizar(str(carpeta), intervalo, por_deteccion, int(min_frames), float(min_desp), silencioso=True)
        st.toast("Reporte generado en salidas/reporte.html", icon=":material/check_circle:")
    ruta_rep = carpeta / "salidas" / "reporte.html"
    if ruta_rep.exists():
        st.download_button("Descargar reporte.html", ruta_rep.read_bytes(), file_name=f"reporte_{carpeta.name}.html",
                           mime="text/html", icon=":material/download:", width="stretch")
        st.caption(f"Archivo: `{ruta_rep.relative_to(RAIZ).as_posix()}`")


# --- Contenido principal (fragmento: se refresca solo en modo en vivo) -------------

@st.fragment(run_every=cada if vivo else None)
def tablero():
    r = cargar_seguro()
    if r is None:
        return
    sesion, v = r.sesion, r.vehiculos
    dur = sesion.duracion_s
    carriles_ok = [c for c in sel_carr if c in set(v["carril"])] if sel_carr is not None else None

    ids = vistas.vehiculos_filtrados(sesion.df, v, sel_clases, carriles_ok, (t_rango[0] * 60, t_fin))
    filtrado = len(ids) < len(v)

    # Encabezado
    st.title("Analítica de tránsito")
    partes = [f"**{carpeta.name}**"]
    if sesion.inicio:
        partes.append(f"{sesion.inicio:%d/%m/%Y %H:%M}")
    partes += [f"{dur / 60:.1f} min de video", f"fuente: {sesion.meta.get('fuente', 'no indicada')}",
               "carriles " + ("dibujados" if r.origen == "dibujados" else "detectados automáticamente")]
    st.caption(" · ".join(partes))
    if vivo:
        st.badge(f"En vivo · actualiza cada {cada} s · {len(sesion.df):,} detecciones".replace(",", "."),
                 icon=":material/sensors:", color="green")

    if len(ids) == 0:
        st.info("Ningún vehículo cumple los filtros. Amplía la selección en la barra lateral.",
                title="Sin vehículos")
        return

    # Recalcular sobre el subconjunto filtrado
    vf = v.loc[ids]
    dff = sesion.df[sesion.df["track_id"].isin(ids)]
    tabla_iv = an.vehiculos_por_intervalo(vf, dur, r.iv, sesion.inicio)
    pico = an.pico(tabla_iv)
    pico["hasta"] = an.fin_intervalo(tabla_iv, pico["indice"], r.iv, sesion.inicio)
    tabla_cl = an.clases_por_intervalo(vf, tabla_iv, r.iv)
    tabla_carr = an.conteo_por_carril(r.asignacion.loc[ids], r.carriles_def)
    mas = an.carril_mas_usado(tabla_carr)
    cruces = an.cruces_linea(dff, r.linea, dur)
    vel = an.resumen_velocidades(vf)
    resumen = {
        "carpeta": str(carpeta), "duracion_s": dur, "total_vehiculos": len(vf), "intervalo": r.iv,
        "pico": {k: pico[k] for k in ("intervalo", "hasta", "vehiculos")},
        "por_intervalo": tabla_iv.to_dict(orient="records"),
        "carriles_origen": r.origen, "carril_mas_usado": mas,
        "por_carril": tabla_carr.to_dict(orient="records"),
        "por_clase": {nombre_clase(k): int(n) for k, n in vf["clase"].value_counts().items()},
        "linea_conteo": {"totales": cruces["totales"]}, "velocidades": vel, "limite_kmh": r.limite,
        "limpieza": {} if filtrado else {k: x for k, x in sesion.limpieza.items() if k != "quietos"},
    }

    # KPIs
    with st.grid(5 if vel and "excesos" in vel else (4 if vel else 3), gap="small", min_column_width=150):
        st.metric("Vehículos", len(vf), border=True, height="stretch",
                  chart_data=tabla_iv["vehiculos"].tolist(), chart_type="bar",
                  help="Vehículos únicos (track_id) después de la limpieza y los filtros.")
        st.metric("Pico", pico["intervalo"], border=True, height="stretch",
                  help=f"{pico['vehiculos']} vehículos en ese {an.NOMBRE_INTERVALO[r.iv]}.")
        st.metric("Carril más usado", mas["carril"], border=True, height="stretch",
                  help=f"{mas['vehiculos']} vehículos ({mas['porcentaje']:.0f}%).")
        if vel:
            st.metric("Velocidad promedio", f"{vel['promedio_kmh']} km/h", border=True, height="stretch",
                      help=f"Por vehículo. Máxima: {vel['maxima_kmh']} km/h.")
            if "excesos" in vel:
                st.metric("Excesos de velocidad", vel["excesos"], border=True, height="stretch",
                          help=f"Vehículos sobre {r.limite:g} km/h." if r.limite else None)

    st.success(an.frase_carril(mas) + ".", icon=":material/add_road:")

    video_proc = carpeta / "video_procesado.mp4"
    vista = st.segmented_control(
        "Vista", ["Resumen", "Tráfico en el tiempo", "Mapa de calor", "Carriles", "Trayectorias",
                  "Línea virtual", "Origen-destino", "Anomalías", "Vehículos"]
                 + (["Video"] if video_proc.exists() else []),
        default="Resumen", required=True, key="vista", label_visibility="collapsed")

    if vista == "Resumen":
        izq, der = st.columns([3, 2])
        with izq, st.container(border=True):
            st.subheader("Vehículos por " + an.NOMBRE_INTERVALO[r.iv])
            grafico_intervalos(tabla_iv, pico)
        with der, st.container(border=True):
            st.subheader("Por carril")
            grafico_carriles(tabla_carr, mas["carril"])
        with st.container(border=True):
            st.subheader("Conclusiones")
            st.markdown("\n".join(f"- {c}" for c in rp.conclusiones(resumen)))
            if filtrado:
                st.caption(f"Calculadas sobre {len(vf)} de {len(v)} vehículos (hay filtros activos).")

    elif vista == "Tráfico en el tiempo":
        with st.container(border=True):
            st.subheader("Tipos de vehículo por " + an.NOMBRE_INTERVALO[r.iv])
            grafico_clases(tabla_cl)
        parciales = tabla_iv[tabla_iv["parcial"]]["intervalo"].tolist()
        if parciales:
            st.caption("Intervalos parciales (la sesión no los cubre completos): " + ", ".join(parciales))
        with st.container(border=True):
            st.subheader("Tabla por intervalo")
            t = tabla_cl.rename(columns=nombre_clase).reset_index(names="Intervalo")
            t.insert(1, "Total", tabla_iv["vehiculos"].to_numpy())
            st.dataframe(t, hide_index=True, alt="Vehículos por intervalo y tipo",
                         column_config={"Total": st.column_config.ProgressColumn(
                             "Total", format="%d", min_value=0, max_value=int(tabla_iv["vehiculos"].max()))})

    elif vista == "Mapa de calor":
        with st.container(border=True):
            st.subheader("¿Por dónde pasan los vehículos?")
            img = imagen_calor(str(carpeta), tuple(ids), por_deteccion, r, huella(carpeta),
                               (intervalo, min_frames, min_desp))
            st.image(img, width="stretch", alt="Mapa de calor de presencia de vehículos sobre la calle")
            st.caption("Cada vehículo cuenta una vez por zona." if not por_deteccion
                       else "Cada detección suma: muestra dónde se quedan más tiempo.")

    elif vista == "Carriles":
        izq, der = st.columns([3, 2])
        with izq, st.container(border=True):
            st.subheader("Carriles")
            st.image(vistas.carriles(sesion, r.carriles_def, tabla_carr, mas["carril"]), width="stretch",
                     alt="Carriles sobre la calle con el conteo de cada uno")
        with der, st.container(border=True):
            st.subheader("Vehículos por carril")
            grafico_carriles(tabla_carr, mas["carril"])
            st.dataframe(tabla_carr.rename(columns={"carril": "Carril", "vehiculos": "Vehículos",
                                                    "porcentaje": "%"}),
                         hide_index=True, alt="Tabla de vehículos por carril")
            if r.origen != "dibujados":
                st.caption("Detectados automáticamente. Para dibujarlos: "
                           f"`python src/definir_carriles.py {carpeta.relative_to(RAIZ).as_posix()}`")

    elif vista == "Trayectorias":
        izq, der = st.columns([3, 2])
        with izq, st.container(border=True):
            st.subheader("Trayectorias")
            st.image(vistas.trayectorias(sesion, dff, vf, r.limite), width="stretch",
                     alt="Trayectorias de los vehículos coloreadas por velocidad")
            st.caption("Verde: lento · ámbar · coral: rápido. Los puntos coral marcan vehículos sobre el límite.")
        with der, st.container(border=True):
            st.subheader("Distribución de velocidades")
            if vel:
                grafico_velocidades(vf, r.limite)
            else:
                st.info("El CSV no trae velocidad_kmh.")

    elif vista == "Línea virtual":
        izq, der = st.columns([3, 2])
        with izq, st.container(border=True):
            st.subheader("Cruces de la línea de conteo")
            st.image(vistas.linea_virtual(sesion, dff, cruces), width="stretch",
                     alt="Línea virtual y puntos de cruce por sentido")
            st.caption(f"Origen de la línea: {r.origen_linea}.")
        with der, st.container(border=True):
            st.subheader("Flujo por minuto")
            with st.container(horizontal=True):
                for k, n in cruces["totales"].items():
                    st.metric(k, n, border=True)
            grafico_linea(cruces)

    elif vista == "Origen-destino":
        mod = od.matriz(dff)
        izq, der = st.columns([3, 2])
        with izq, st.container(border=True):
            st.subheader("¿De dónde vienen y hacia dónde van?")
            grafico_od(mod)
            st.caption("Norte = arriba de la imagen · Sur = abajo · Este = derecha · Oeste = izquierda.")
        with der, st.container(border=True):
            st.subheader("Matriz origen-destino")
            if mod["principal"]:
                st.success(od.frase(mod) + ".", icon=":material/alt_route:")
            st.dataframe(mod["tabla"].rename_axis(index="Origen ↓ / Destino →"),
                         alt="Matriz de vehículos por zona de origen y de destino")
            st.dataframe(mod["pares"].rename(columns={"origen": "Origen", "destino": "Destino",
                                                      "vehiculos": "Vehículos", "porcentaje": "%"}),
                         hide_index=True, alt="Movimientos ordenados de más a menos frecuente")

    elif vista == "Anomalías":
        tabla = r.anomalias["tabla"]
        # los estacionados no pasan la limpieza (no se mueven), así que no dependen de los filtros
        tabla = tabla[tabla["track_id"].isin(ids) | (tabla["tipo"] == "estacionado")]
        with st.container(horizontal=True):
            for clave, nombre in anom.TIPOS.items():
                st.metric(nombre, int((tabla["tipo"] == clave).sum()), border=True)
        izq, der = st.columns([3, 2])
        with izq, st.container(border=True):
            st.subheader("Dónde ocurrieron")
            st.image(vistas.anomalias(sesion, dff, tabla), width="stretch",
                     alt="Trayectorias anómalas resaltadas sobre la calle")
            st.caption(f"Ámbar: detenido más de {r.anomalias['min_detenido_s']:g} s · gris: estacionado · "
                       "coral: sentido contrario · fucsia: trayectoria atípica.")
        with der, st.container(border=True):
            st.subheader(f"{len(tabla)} caso{'s' if len(tabla) != 1 else ''}")
            if tabla.empty:
                st.info("No se detectaron anomalías con los filtros actuales.")
            else:
                t = tabla.assign(tipo=tabla["tipo"].map(anom.TIPOS), momento=tabla["t"].map(mmss))
                st.dataframe(t[["track_id", "tipo", "momento", "detalle"]], hide_index=True,
                             alt="Lista de anomalías",
                             column_config={"track_id": "ID", "tipo": "Tipo", "momento": "Momento",
                                            "detalle": "Detalle"})
            k = r.anomalias["n_patrones"]
            st.caption(f"Atípica = no se parece a ningún patrón de movimiento habitual ({k} encontrado"
                       f"{'s' if k != 1 else ''} con DBSCAN sobre las trayectorias remuestreadas).")

    elif vista == "Video":
        with st.container(border=True):
            st.subheader("Video procesado")
            st.video(str(video_proc), alt="Video de la sesión con las detecciones, los números y las velocidades")
            st.caption(f"Archivo: `{video_proc.relative_to(RAIZ).as_posix()}`" if video_proc.is_relative_to(RAIZ)
                       else f"Archivo: `{video_proc}`")

    elif vista == "Vehículos":
        with st.container(border=True):
            st.subheader(f"{len(vf)} vehículos")
            t = vf.reset_index().assign(
                clase=lambda d: d["clase"].map(nombre_clase),
                aparece=lambda d: d["t_inicio"].map(mmss),
                segundos_en_cuadro=lambda d: (d["t_fin"] - d["t_inicio"]).round(1))
            cols = ["track_id", "clase", "carril", "origen", "destino", "aparece", "segundos_en_cuadro",
                    "n_detecciones"]
            cols += [c for c in ("velocidad_kmh", "excedio") if c in t.columns]
            st.dataframe(t[cols], hide_index=True, alt="Lista de vehículos",
                         column_config={
                             "track_id": "ID", "clase": "Tipo", "carril": "Carril", "origen": "Origen",
                             "destino": "Destino", "aparece": "Aparece",
                             "segundos_en_cuadro": st.column_config.NumberColumn("En cuadro", format="%.1f s"),
                             "n_detecciones": "Detecciones",
                             "velocidad_kmh": st.column_config.NumberColumn("Velocidad", format="%.1f km/h"),
                             "excedio": st.column_config.CheckboxColumn("Excedió"),
                         })
            st.download_button("Descargar CSV", t[cols].to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"vehiculos_{carpeta.name}.csv", icon=":material/download:")


# --- Gráficos (ECharts) --------------------------------------------------------------

EJE = {"axisLine": {"lineStyle": {"color": gr.BORDE}}, "axisLabel": {"color": gr.GRIS},
       "splitLine": {"lineStyle": {"color": gr.BORDE, "opacity": 0.5}}}


def grafico_intervalos(tabla_iv: pd.DataFrame, pico: dict):
    d = pd.DataFrame({"intervalo": tabla_iv["intervalo"],
                      "Vehículos": tabla_iv["vehiculos"].where(tabla_iv.index != pico["indice"]),
                      "Pico": tabla_iv["vehiculos"].where(tabla_iv.index == pico["indice"])})
    st.echarts_chart({
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 40, "right": 16, "top": 24, "bottom": 40},
        "dataset": {"source": d},
        "xAxis": {"type": "category", **EJE},
        "yAxis": {"type": "value", "minInterval": 1, **EJE},
        "series": [
            {"type": "bar", "stack": "t", "itemStyle": {"color": gr.TURQUESA, "borderRadius": [3, 3, 0, 0]}},
            {"type": "bar", "stack": "t", "itemStyle": {"color": gr.CORAL, "borderRadius": [3, 3, 0, 0]},
             "label": {"show": True, "position": "top", "color": gr.TEXTO, "fontWeight": "bold"}},
        ],
    }, alt=f"Vehículos por intervalo; el pico es {pico['intervalo']} con {pico['vehiculos']}", key="g_iv")


def grafico_carriles(tabla_carr: pd.DataFrame, mas_usado: str):
    d = tabla_carr.rename(columns={"carril": "Carril", "vehiculos": "Vehículos"})[["Carril", "Vehículos"]]
    colores = [gr.COLORES_CARRIL[i % len(gr.COLORES_CARRIL)] if c != FUERA else gr.GRIS
               for i, c in enumerate(d["Carril"])]
    st.echarts_chart({
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 90, "right": 50, "top": 10, "bottom": 24},
        "xAxis": {"type": "value", "minInterval": 1, **EJE},
        "yAxis": {"type": "category", "inverse": True, "data": d["Carril"].tolist(), **EJE},
        "series": [{
            "type": "bar", "name": "Vehículos",
            "data": [{"value": int(n), "itemStyle": {"color": col, "borderRadius": [0, 4, 4, 0],
                                                     "opacity": 1 if c == mas_usado else 0.65}}
                     for c, n, col in zip(d["Carril"], d["Vehículos"], colores)],
            "label": {"show": True, "position": "right", "color": gr.TEXTO},
        }],
    }, alt=f"Vehículos por carril; el más usado es {mas_usado}", key="g_carr")


def grafico_clases(tabla_cl: pd.DataFrame):
    d = tabla_cl.rename(columns=nombre_clase).reset_index(names="intervalo")
    series = [{"type": "bar", "stack": "c", "name": c, "itemStyle": {"color": COLOR_CLASE.get(c, gr.GRIS)}}
              for c in d.columns[1:]]
    st.echarts_chart({
        "tooltip": {"trigger": "axis"}, "legend": {"textStyle": {"color": gr.TEXTO}},
        "grid": {"left": 40, "right": 16, "top": 40, "bottom": 56},
        "dataset": {"source": d},
        "xAxis": {"type": "category", **EJE}, "yAxis": {"type": "value", "minInterval": 1, **EJE},
        "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 18, "bottom": 8}] if len(d) > 30 else [],
        "series": series,
    }, alt="Vehículos por intervalo apilados por tipo", key="g_cl")


def grafico_velocidades(vf: pd.DataFrame, limite: float | None):
    v = vf["velocidad_kmh"].dropna()
    bordes = np.arange(np.floor(v.min() / 5) * 5, np.ceil(v.max() / 5) * 5 + 5, 5)
    n, _ = np.histogram(v, bordes)
    d = pd.DataFrame({"rango": [f"{a:.0f}–{b:.0f}" for a, b in zip(bordes[:-1], bordes[1:])], "Vehículos": n})
    colores = [gr.CORAL if limite and a >= limite else gr.TURQUESA for a in bordes[:-1]]
    st.echarts_chart({
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 40, "right": 16, "top": 16, "bottom": 50},
        "xAxis": {"type": "category", "data": d["rango"].tolist(), "name": "km/h", "nameLocation": "middle",
                  "nameGap": 30, **EJE},
        "yAxis": {"type": "value", "minInterval": 1, **EJE},
        "series": [{"type": "bar", "name": "Vehículos",
                    "data": [{"value": int(x), "itemStyle": {"color": c}} for x, c in zip(n, colores)]}],
    }, alt="Histograma de velocidades por vehículo; en coral, sobre el límite", key="g_vel")
    if limite:
        st.caption(f"En coral, los rangos desde el límite de {limite:g} km/h.")


def grafico_linea(cruces: dict):
    d = cruces["por_minuto"].copy()
    d.insert(0, "minuto", [f"{i}–{i + 1}" for i in range(len(d))])
    st.echarts_chart({
        "tooltip": {"trigger": "axis"}, "legend": {"textStyle": {"color": gr.TEXTO}},
        "grid": {"left": 40, "right": 16, "top": 40, "bottom": 44},
        "dataset": {"source": d},
        "xAxis": {"type": "category", "name": "minuto", "nameLocation": "middle", "nameGap": 28, **EJE},
        "yAxis": {"type": "value", "minInterval": 1, **EJE},
        "series": [{"type": "line", "smooth": True, "symbolSize": 6, "itemStyle": {"color": gr.TURQUESA}},
                   {"type": "line", "smooth": True, "symbolSize": 6, "itemStyle": {"color": gr.FUCSIA}}],
    }, alt="Cruces de la línea por minuto y sentido", key="g_linea")


def grafico_od(mod: dict):
    """Sankey: zonas de origen a la izquierda y de destino a la derecha."""
    colores = {"Norte": gr.TURQUESA, "Sur": gr.FUCSIA, "Este": gr.AMBAR, "Oeste": gr.AZUL}
    pares = mod["pares"]
    nodos = ([{"name": f"{z} (origen)", "itemStyle": {"color": colores[z]}} for z in pares["origen"].unique()]
             + [{"name": f"{z} (destino)", "itemStyle": {"color": colores[z]}} for z in pares["destino"].unique()])
    enlaces = [{"source": f"{p.origen} (origen)", "target": f"{p.destino} (destino)", "value": int(p.vehiculos)}
               for p in pares.itertuples()]
    st.echarts_chart({
        "tooltip": {"trigger": "item"},
        "series": [{"type": "sankey", "data": nodos, "links": enlaces, "nodeWidth": 18, "nodeGap": 14,
                    "left": 8, "right": 110, "top": 10, "bottom": 10,
                    "label": {"color": gr.TEXTO, "fontSize": 13},
                    "lineStyle": {"color": "source", "opacity": 0.45, "curveness": 0.5},
                    "emphasis": {"focus": "adjacency"}}],
    }, height=380, alt="Diagrama de flujo de vehículos desde su zona de origen hasta su zona de destino",
        key="g_od")


tablero()
