"""Analítica de tránsito (Nivel 2): punto de entrada.

Uso:
    python src/main_analitica.py <carpeta_sesion> [opciones]

La carpeta debe tener historial.csv (y, si existen, meta.json, frame_referencia.jpg
y carriles.json). Las salidas quedan en <carpeta_sesion>/salidas/.

Desde el programa de tracking (Nivel 1), al cerrar:
    from main_analitica import analizar
    analizar("data/sesiones/2026-11-18_15-30-00")
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analitica as an  # noqa: E402
import anomalias as anom  # noqa: E402
import graficos as gr  # noqa: E402
import origen_destino as od  # noqa: E402
import reporte as rp  # noqa: E402
from carriles import _direccion_dominante, asignar_carriles  # noqa: E402
from datos import ErrorSesion, cargar_sesion, limpiar, nombre_clase, resumen_por_vehiculo  # noqa: E402


def linea_automatica(df, ancho, alto):
    """Si no hay línea definida: una perpendicular al tráfico en la mitad del recorrido."""
    eje = _direccion_dominante(df)
    pts = df[["px", "py"]].to_numpy()
    if abs(eje[1]) >= abs(eje[0]):           # tráfico vertical → línea horizontal
        y = float(np.median(pts[:, 1]))
        cerca = pts[np.abs(pts[:, 1] - y) < 0.05 * alto, 0]
        x0, x1 = np.percentile(cerca if len(cerca) > 10 else pts[:, 0], [1, 99])
        margen = 0.06 * (x1 - x0)
        return [[x0 - margen, y], [x1 + margen, y]]
    x = float(np.median(pts[:, 0]))
    cerca = pts[np.abs(pts[:, 0] - x) < 0.05 * ancho, 1]
    y0, y1 = np.percentile(cerca if len(cerca) > 10 else pts[:, 1], [1, 99])
    margen = 0.06 * (y1 - y0)
    return [[x, y0 - margen], [x, y1 + margen]]


@dataclass
class Resultado:
    """Todo lo que calcula el análisis, sin escribir nada a disco (lo usan el CLI y el dashboard)."""
    sesion: object
    vehiculos: pd.DataFrame
    limite: float | None
    iv: str
    tabla_iv: pd.DataFrame
    pico: dict
    tabla_clases: pd.DataFrame
    asignacion: pd.Series
    carriles_def: list
    origen: str
    tabla_carr: pd.DataFrame
    mas_usado: dict
    frase: str
    linea: list
    origen_linea: str
    cruces: dict
    velocidades: dict | None
    od: dict
    anomalias: dict


def procesar(carpeta, intervalo: str = "auto", min_frames: int = 8,
             min_desplazamiento: float = 40.0, linea=None, min_detenido_s: float = 10.0) -> Resultado:
    """Carga, limpia y calcula todas las tablas de una sesión."""
    sesion = limpiar(cargar_sesion(carpeta), min_frames, min_desplazamiento)
    if sesion.df.empty:
        raise ErrorSesion("Después de la limpieza no quedó ningún vehículo válido. "
                          "Revisa --min-frames y --min-desplazamiento.")

    limite = sesion.meta.get("limite_kmh")
    vehiculos = resumen_por_vehiculo(sesion.df, limite)

    # Intervalos
    iv = an.elegir_intervalo(sesion.duracion_s, intervalo)
    tabla_iv = an.vehiculos_por_intervalo(vehiculos, sesion.duracion_s, iv, sesion.inicio)
    pico = an.pico(tabla_iv)
    pico["hasta"] = an.fin_intervalo(tabla_iv, pico["indice"], iv, sesion.inicio)
    tabla_clases = an.clases_por_intervalo(vehiculos, tabla_iv, iv)

    # Carriles
    asignacion, carriles_def, origen = asignar_carriles(sesion.df, sesion.carpeta)
    vehiculos["carril"] = asignacion.reindex(vehiculos.index)
    tabla_carr = an.conteo_por_carril(asignacion, carriles_def)
    mas_usado = an.carril_mas_usado(tabla_carr)
    frase = an.frase_carril(mas_usado)

    # Línea virtual
    origen_linea = "parámetro --linea" if linea else "meta.json"
    linea = linea or sesion.meta.get("linea_conteo")
    if not linea:
        linea = linea_automatica(sesion.df, sesion.ancho, sesion.alto)
        origen_linea = "automática"
    cruces = an.cruces_linea(sesion.df, linea, sesion.duracion_s)

    # Origen-destino y anomalías (Nivel C)
    matriz_od = od.matriz(sesion.df)
    vehiculos["origen"] = matriz_od["por_vehiculo"]["origen"].reindex(vehiculos.index)
    vehiculos["destino"] = matriz_od["por_vehiculo"]["destino"].reindex(vehiculos.index)
    anomalias = anom.detectar(sesion, asignacion, carriles_def, min_detenido_s)

    return Resultado(sesion, vehiculos, limite, iv, tabla_iv, pico, tabla_clases, asignacion,
                     carriles_def, origen, tabla_carr, mas_usado, frase, linea, origen_linea,
                     cruces, an.resumen_velocidades(vehiculos), matriz_od, anomalias)


def analizar(carpeta, intervalo: str = "auto", por_deteccion: bool = False, min_frames: int = 8,
             min_desplazamiento: float = 40.0, linea=None, silencioso: bool = False,
             min_detenido_s: float = 10.0, gif: bool = True) -> dict:
    """Corre todo el análisis sobre una carpeta de sesión, escribe las salidas y devuelve el resumen."""
    log = (lambda *a: None) if silencioso else print
    r = procesar(carpeta, intervalo, min_frames, min_desplazamiento, linea, min_detenido_s)
    sesion, vehiculos, limite, iv = r.sesion, r.vehiculos, r.limite, r.iv
    tabla_iv, pico, tabla_clases = r.tabla_iv, r.pico, r.tabla_clases
    asignacion, carriles_def, origen = r.asignacion, r.carriles_def, r.origen
    tabla_carr, mas_usado, frase = r.tabla_carr, r.mas_usado, r.frase
    cruces, origen_linea, velocidades = r.cruces, r.origen_linea, r.velocidades
    total = len(vehiculos)
    lim = sesion.limpieza
    salidas = Path(carpeta) / "salidas"
    salidas.mkdir(exist_ok=True)

    # Figuras
    calor = gr.calcular_mapa_calor(sesion.df, sesion.ancho, sesion.alto, por_deteccion)
    archivos = [
        gr.barras_intervalo(tabla_iv, iv, pico, total, salidas / "barras_por_intervalo.png"),
        gr.mapa_calor(sesion, calor, carriles_def, tabla_carr, mas_usado, por_deteccion, salidas / "mapa_calor.png"),
        gr.carriles(sesion, carriles_def, tabla_carr, mas_usado, origen, frase, salidas / "carriles.png"),
        gr.clases_intervalo(tabla_clases, iv, salidas / "clases_por_intervalo.png"),
        gr.trayectorias(sesion, vehiculos, asignacion, carriles_def, limite, salidas / "trayectorias.png"),
        gr.linea_virtual(sesion, cruces, salidas / "conteo_linea.png"),
        gr.origen_destino(sesion, r.od, salidas / "origen_destino.png"),
        gr.anomalias(sesion, r.anomalias, salidas / "anomalias.png"),
    ]
    if gif:
        archivos.append(gr.mapa_calor_gif(sesion, salidas / "mapa_calor_animado.gif"))
    # El enunciado pide «a qué hora pasaron más autos»: la gráfica por hora se genera siempre,
    # además de la del intervalo automático (con un video corto, por hora sale una sola barra).
    tabla_hora, pico_hora = tabla_iv, pico
    if iv != "hora":
        tabla_hora = an.vehiculos_por_intervalo(vehiculos, sesion.duracion_s, "hora", sesion.inicio)
        pico_hora = an.pico(tabla_hora)
        archivos.append(gr.barras_intervalo(tabla_hora, "hora", pico_hora, total, salidas / "barras_por_hora.png"))

    # Tablas
    tabla_iv.drop(columns="inicio_s").to_csv(salidas / "vehiculos_por_intervalo.csv", index=False)
    tabla_hora.drop(columns="inicio_s").to_csv(salidas / "vehiculos_por_hora.csv", index=False)
    tabla_carr.to_csv(salidas / "vehiculos_por_carril.csv", index=False)
    tabla_clases.rename(columns=nombre_clase).to_csv(salidas / "clases_por_intervalo.csv", index_label="intervalo")
    vehiculos.reset_index().to_csv(salidas / "vehiculos.csv", index=False)
    r.od["tabla"].to_csv(salidas / "origen_destino.csv", index_label="origen/destino")
    tabla_anom = r.anomalias["tabla"].assign(tipo=lambda t: t["tipo"].map(anom.TIPOS))
    tabla_anom.round(2).to_csv(salidas / "anomalias.csv", index=False)

    por_clase = {nombre_clase(k): int(v) for k, v in vehiculos["clase"].value_counts().items()}
    resumen = {
        "carpeta": str(Path(carpeta)),
        "fuente": sesion.meta.get("fuente"),
        "inicio": sesion.inicio.isoformat() if sesion.inicio else None,
        "duracion_s": round(sesion.duracion_s, 1),
        "total_vehiculos": total,
        "limpieza": {k: v for k, v in lim.items() if k != "quietos"},
        "intervalo": iv,
        "pico": {"intervalo": pico["intervalo"], "hasta": pico["hasta"], "vehiculos": pico["vehiculos"]},
        "por_intervalo": tabla_iv.drop(columns="inicio_s").to_dict(orient="records"),
        "por_hora": tabla_hora.drop(columns="inicio_s").to_dict(orient="records"),
        "hora_pico": {"intervalo": pico_hora["intervalo"], "vehiculos": pico_hora["vehiculos"]},
        "carriles_origen": origen,
        "carril_mas_usado": mas_usado,
        "frase_carril": frase,
        "por_carril": tabla_carr.to_dict(orient="records"),
        "por_clase": por_clase,
        "linea_conteo": {"origen": origen_linea, "linea": cruces["linea"],
                         "totales": cruces["totales"], "flujo_por_minuto": cruces["flujo_por_minuto"]},
        "velocidades": velocidades,
        "limite_kmh": limite,
        "origen_destino": {
            "pares": r.od["pares"].to_dict(orient="records"),
            "principal": r.od["principal"],
            "frase": od.frase(r.od),
        },
        "anomalias": {
            "conteo": r.anomalias["conteo"],
            "min_detenido_s": min_detenido_s,
            "n_patrones": r.anomalias["n_patrones"],
            "fijos_fuera_de_la_via": r.anomalias["fijos_fuera_de_la_via"],
            "casos": tabla_anom.round(1).to_dict(orient="records"),
        },
        "parametros": {"intervalo": intervalo, "por_deteccion": por_deteccion, "min_frames": min_frames,
                       "min_desplazamiento": min_desplazamiento, "min_detenido_s": min_detenido_s},
        "fecha_analisis": datetime.now().isoformat(timespec="seconds"),
        "archivos": [p.name for p in archivos],
    }
    resumen["conclusiones"] = rp.conclusiones(resumen)
    resumen["archivos"].append(rp.generar_reporte(salidas, resumen).name)
    (salidas / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [
        "=" * 64,
        "  ANALÍTICA DE TRÁNSITO",
        "=" * 64,
        f"  Sesión:      {Path(carpeta).resolve().name}"
        + (f"  ({sesion.inicio:%d/%m/%Y %H:%M})" if sesion.inicio else ""),
        f"  Duración:    {sesion.duracion_s / 60:.1f} min",
        f"  Tracks:      {lim['tracks_totales']} en el CSV -> {lim['tracks_validos']} vehículos válidos "
        f"({lim['descartados_cortos']} cortos y {lim['descartados_quietos']} quietos descartados)",
        f"  Pico:        {pico['intervalo']} con {pico['vehiculos']} vehículos (por {an.NOMBRE_INTERVALO[iv]})",
        f"  Hora pico:   {pico_hora['intervalo']} con {pico_hora['vehiculos']} vehículos"
        + ("  (la sesión no cubre la hora completa)" if tabla_hora["parcial"].all() else ""),
        "  Carriles " + ("dibujados:" if origen == "dibujados" else "(detección automática):"),
    ]
    for fila in tabla_carr.itertuples():
        marca = "  <- más usado" if fila.carril == mas_usado["carril"] else ""
        lineas.append(f"     {fila.carril:<16}{fila.vehiculos:>5}  ({fila.porcentaje:4.1f}%){marca}")
    lineas.append("  Clases:      " + ", ".join(f"{k} {v}" for k, v in por_clase.items()))
    lineas.append("  Línea:       " + " | ".join(f"{k}: {v}" for k, v in cruces["totales"].items()))
    if velocidades:
        txt = f"  Velocidad:   promedio {velocidades['promedio_kmh']} km/h, máxima {velocidades['maxima_kmh']} km/h"
        if "excesos" in velocidades:
            txt += f", {velocidades['excesos']} vehículos sobre el límite"
        lineas.append(txt)
    if r.od["principal"]:
        lineas.append(f"  Movimiento:  {od.frase(r.od)}")
    conteo_anom = {k: v for k, v in r.anomalias["conteo"].items() if v}
    lineas.append("  Anomalías:   " + (", ".join(f"{k.lower()} {v}" for k, v in conteo_anom.items())
                                      if conteo_anom else "ninguna"))
    lineas += ["-" * 64, f"  >> {frase}.", "-" * 64, f"  Salidas en: {salidas.resolve()}",
               f"  Reporte:     {(salidas / 'reporte.html').resolve()}"]
    texto = "\n".join(lineas)
    (salidas / "resumen.txt").write_text(texto + "\n", encoding="utf-8")
    log(texto)
    return resumen


def _parsear_linea(texto: str):
    v = [float(x) for x in texto.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("La línea debe ser x1,y1,x2,y2")
    return [[v[0], v[1]], [v[2], v[3]]]


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Analítica de tránsito: mapa de calor, carril más usado "
                                             "y vehículos por intervalo a partir de historial.csv.")
    ap.add_argument("carpeta", help="carpeta de la sesión (con historial.csv)")
    ap.add_argument("--intervalo", choices=["auto", "hora", "minuto", "10s"], default="auto",
                    help="agrupación de la gráfica de barras (auto según la duración)")
    ap.add_argument("--por-deteccion", action="store_true",
                    help="el mapa de calor suma cada detección en vez de cada vehículo")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--min-desplazamiento", type=float, default=40.0)
    ap.add_argument("--linea", type=_parsear_linea, help="línea de conteo x1,y1,x2,y2 en píxeles")
    ap.add_argument("--min-detenido", type=float, default=10.0,
                    help="segundos quieto para considerar un vehículo detenido (anomalías)")
    ap.add_argument("--sin-gif", action="store_true", help="no generar el mapa de calor animado")
    ap.add_argument("--abrir", action="store_true", help="abre el reporte HTML en el navegador al terminar")
    a = ap.parse_args(argv)
    try:
        analizar(a.carpeta, a.intervalo, a.por_deteccion, a.min_frames, a.min_desplazamiento, a.linea,
                 min_detenido_s=a.min_detenido, gif=not a.sin_gif)
    except ErrorSesion as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if a.abrir:
        import webbrowser
        webbrowser.open((Path(a.carpeta) / "salidas" / "reporte.html").resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
