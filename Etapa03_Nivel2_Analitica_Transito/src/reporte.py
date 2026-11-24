"""Reporte HTML autocontenido (imágenes en base64) y conclusiones automáticas en español.

Todo sale de `resumen.json`, así el reporte se puede regenerar sin volver a analizar:
    python src/reporte.py data/sintetico
"""
from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from jinja2 import Environment

from carriles import FUERA

TITULOS = {
    "barras_por_hora.png": ("Vehículos por hora", "¿A qué hora pasaron más autos? (lo que pide el enunciado)."),
    "barras_por_intervalo.png": ("Vehículos por intervalo", "Vehículos únicos según el momento en que aparecen; en coral, el pico."),
    "mapa_calor.png": ("Mapa de calor", "Por dónde pasan los vehículos, superpuesto sobre la calle."),
    "mapa_calor_animado.gif": ("Mapa de calor animado", "Cómo se acumula el tráfico a lo largo de la sesión."),
    "carriles.png": ("Carriles", "Vehículos asignados a cada carril y el carril más utilizado."),
    "clases_por_intervalo.png": ("Tipos de vehículo", "Autos, motos, camiones y buses por intervalo."),
    "trayectorias.png": ("Trayectorias", "Recorrido de cada vehículo, coloreado por velocidad."),
    "conteo_linea.png": ("Línea virtual", "Cruces de la línea de conteo por sentido y flujo por minuto."),
    "origen_destino.png": ("Origen-destino", "De qué zona vienen y hacia qué zona van los vehículos."),
    "anomalias.png": ("Anomalías", "Vehículos detenidos, en sentido contrario y trayectorias atípicas."),
}

TIPO_MIME = {".png": "image/png", ".gif": "image/gif"}

NOMBRE_INTERVALO = {"hora": "hora", "minuto": "minuto", "10s": "10 segundos"}


def _pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else 0.0


def _duracion(seg: float) -> str:
    seg = int(round(seg))
    h, r = divmod(seg, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h} h {m} min"
    if m:
        return f"{m} min" + (f" {s} s" if s else "")
    return f"{s} s"


def _rango_pico(resumen: dict) -> str:
    p = resumen["pico"]
    if resumen["intervalo"] == "hora" or not p.get("hasta"):
        return f"la franja {p['intervalo']}"
    return f"de {p['intervalo']} a {p['hasta']}"


def conclusiones(resumen: dict) -> list[str]:
    """Frases en lenguaje natural a partir del resumen del análisis."""
    total = resumen["total_vehiculos"]
    dur = resumen["duracion_s"]
    frases = []

    por_min = total / max(dur / 60, 1e-9)
    frases.append(f"En {_duracion(dur)} de video se contaron {total} vehículos únicos "
                  f"(unos {por_min:.1f} por minuto).")

    p = resumen["pico"]
    conteos = [r["vehiculos"] for r in resumen.get("por_intervalo", [])]
    texto = f"El intervalo de mayor flujo fue {_rango_pico(resumen)}, con {p['vehiculos']} vehículos"
    if len(conteos) > 1:
        prom = sum(conteos) / len(conteos)
        if prom > 0 and p["vehiculos"] / prom >= 1.3:
            texto += f", {p['vehiculos'] / prom:.1f} veces el promedio por {NOMBRE_INTERVALO[resumen['intervalo']]}"
    frases.append(texto + ".")

    m = resumen["carril_mas_usado"]
    origen = "dibujados a mano" if resumen["carriles_origen"] == "dibujados" else "detectados automáticamente"
    carriles = [c for c in resumen["por_carril"] if c["carril"] != FUERA]
    texto = f"{m['carril']} concentró el {m['porcentaje']:.0f}% del tráfico"
    if m["porcentaje"] > 50:
        texto += ", más de la mitad de todos los vehículos"
    texto += f" (de {len(carriles)} carriles {origen})."
    frases.append(texto)
    menos = min(carriles, key=lambda c: c["vehiculos"]) if len(carriles) > 1 else None
    if menos and menos["carril"] != m["carril"]:
        frases.append(f"El carril menos usado fue {menos['carril']}, con {menos['vehiculos']} vehículos "
                      f"({menos['porcentaje']:.0f}%).")

    clases = resumen.get("por_clase") or {}
    if clases:
        principal, n = max(clases.items(), key=lambda kv: kv[1])
        pesados = clases.get("Camión", 0) + clases.get("Bus", 0)
        texto = f"Predominan los vehículos de tipo {principal.lower()} ({_pct(n, total):.0f}%)"
        if pesados:
            texto += f"; los pesados (camiones y buses) son el {_pct(pesados, total):.0f}%"
        frases.append(texto + ".")

    tot = resumen.get("linea_conteo", {}).get("totales") or {}
    cruces = sum(tot.values())
    if cruces:
        sentido, n = max(tot.items(), key=lambda kv: kv[1])
        frases.append(f"De los {cruces} vehículos que cruzaron la línea de conteo, el {_pct(n, cruces):.0f}% "
                      f"lo hizo {sentido[0].lower() + sentido[1:]}.")

    v = resumen.get("velocidades")
    if v:
        texto = f"La velocidad promedio por vehículo fue {v['promedio_kmh']} km/h y la máxima {v['maxima_kmh']} km/h"
        if "excesos" in v:
            lim = resumen.get("limite_kmh")
            if v["excesos"]:
                texto += (f"; {v['excesos']} vehículos ({_pct(v['excesos'], total):.0f}%) superaron el límite"
                          + (f" de {lim:g} km/h" if lim else ""))
            else:
                texto += "; ningún vehículo superó el límite"
        frases.append(texto + ".")

    frase_od = (resumen.get("origen_destino") or {}).get("frase")
    if frase_od:
        frases.append(frase_od + ".")

    an = resumen.get("anomalias")
    if an:
        c = an["conteo"]
        partes = []
        if c.get("Detenido"):
            partes.append(f"{c['Detenido']} vehículo{'s' if c['Detenido'] > 1 else ''} detenido{'s' if c['Detenido'] > 1 else ''} "
                          f"más de {an['min_detenido_s']:g} s")
        if c.get("Estacionado"):
            partes.append(f"{c['Estacionado']} estacionado{'s' if c['Estacionado'] > 1 else ''}")
        if c.get("Sentido contrario"):
            partes.append(f"{c['Sentido contrario']} en sentido contrario")
        if c.get("Trayectoria atípica"):
            partes.append(f"{c['Trayectoria atípica']} con trayectoria atípica")
        if partes:
            texto = ", ".join(partes[:-1]) + (" y " if len(partes) > 1 else "") + partes[-1]
            frases.append(f"Se detectaron anomalías: {texto}.")
        else:
            frases.append("No se detectaron anomalías (vehículos detenidos, en sentido contrario ni trayectorias atípicas).")

    lim = resumen.get("limpieza") or {}
    if lim.get("descartados_cortos") or lim.get("descartados_quietos"):
        frases.append(f"Antes de analizar se descartaron {lim['descartados_cortos']} tracks demasiado cortos "
                      f"y {lim['descartados_quietos']} que no se movían (posibles errores del tracker o "
                      f"vehículos estacionados).")
    return frases


def kpis(resumen: dict) -> list[dict]:
    """Tarjetas de indicadores para el reporte y el dashboard."""
    p, m = resumen["pico"], resumen["carril_mas_usado"]
    tarjetas = [
        {"titulo": "Vehículos", "valor": f"{resumen['total_vehiculos']}", "nota": f"en {_duracion(resumen['duracion_s'])}"},
        {"titulo": "Pico", "valor": p["intervalo"], "nota": f"{p['vehiculos']} vehículos por {NOMBRE_INTERVALO[resumen['intervalo']]}"},
        {"titulo": "Carril más usado", "valor": m["carril"], "nota": f"{m['vehiculos']} vehículos · {m['porcentaje']:.0f}%"},
    ]
    v = resumen.get("velocidades")
    if v:
        tarjetas.append({"titulo": "Velocidad promedio", "valor": f"{v['promedio_kmh']} km/h",
                         "nota": f"máxima {v['maxima_kmh']} km/h"})
        if "excesos" in v:
            lim = resumen.get("limite_kmh")
            tarjetas.append({"titulo": "Excesos de velocidad", "valor": f"{v['excesos']}", "alerta": v["excesos"] > 0,
                             "nota": f"vehículos sobre {lim:g} km/h" if lim else "vehículos sobre el límite"})
    an = resumen.get("anomalias")
    if an:
        n = sum(an["conteo"].values())
        graves = an["conteo"].get("Sentido contrario", 0) + an["conteo"].get("Detenido", 0)
        tarjetas.append({"titulo": "Anomalías", "valor": f"{n}", "alerta": graves > 0,
                         "nota": ", ".join(f"{v} {k.lower()}" for k, v in an["conteo"].items() if v) or "ninguna"})
    return tarjetas


PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analítica de tránsito · {{ nombre }}</title>
<style>
  :root {
    --fondo: #0E1621; --panel: #131F2D; --tarjeta: #172433; --borde: #2C4560;
    --texto: #F2F5F8; --gris: #A3B3C4; --turquesa: #2DE2CF; --fucsia: #E879F9; --coral: #FF5D73;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--fondo); color: var(--texto);
         font: 16px/1.55 "Segoe UI", Calibri, system-ui, sans-serif; }
  main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
  header { border-bottom: 1px solid var(--borde); padding-bottom: 20px; margin-bottom: 28px; }
  .etiqueta { color: var(--turquesa); font-size: 13px; letter-spacing: .12em; text-transform: uppercase; font-weight: 600; }
  h1 { font-size: 34px; margin: 6px 0 4px; line-height: 1.15; }
  h2 { font-size: 22px; margin: 44px 0 14px; }
  h3 { font-size: 18px; margin: 0 0 2px; }
  .sub { color: var(--gris); margin: 0; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
  .kpi { background: var(--tarjeta); border: 1px solid var(--borde); border-radius: 12px; padding: 16px 18px; }
  .kpi .t { color: var(--gris); font-size: 13px; }
  .kpi .v { font-size: 28px; font-weight: 700; color: var(--turquesa); line-height: 1.25; margin: 4px 0 2px;
            overflow-wrap: anywhere; }
  .kpi.alerta .v { color: var(--coral); }
  .kpi .n { color: var(--gris); font-size: 13px; }
  .destacado { background: linear-gradient(90deg, rgba(45,226,207,.14), rgba(232,121,249,.10));
               border: 1px solid var(--borde); border-left: 4px solid var(--turquesa); border-radius: 12px;
               padding: 16px 20px; font-size: 20px; font-weight: 600; margin-top: 18px; }
  ul.conc { background: var(--panel); border: 1px solid var(--borde); border-radius: 12px;
            padding: 18px 22px 18px 40px; margin: 0; }
  ul.conc li { margin: 6px 0; }
  ul.conc li::marker { color: var(--fucsia); }
  figure { margin: 0 0 26px; background: var(--panel); border: 1px solid var(--borde); border-radius: 12px; padding: 16px; }
  figure img { width: 100%; height: auto; display: block; border-radius: 6px; margin-top: 10px; }
  figcaption { color: var(--gris); font-size: 14px; }
  .tablas { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
  .caja { background: var(--panel); border: 1px solid var(--borde); border-radius: 12px; padding: 14px 16px; overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
  th { text-align: left; color: var(--gris); font-weight: 600; border-bottom: 1px solid var(--borde); padding: 6px 8px; }
  td { padding: 5px 8px; border-bottom: 1px solid rgba(44,69,96,.45); }
  td.num, th.num { text-align: right; }
  tr.marca td { color: var(--turquesa); font-weight: 700; }
  tr.pico td { color: var(--coral); font-weight: 700; }
  .barra { height: 6px; border-radius: 3px; background: var(--turquesa); min-width: 2px; }
  .nota { color: var(--gris); font-size: 13.5px; margin-top: 44px; border-top: 1px solid var(--borde); padding-top: 16px; }
  .nota dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 8px 0 0; }
  .nota dt { color: var(--texto); }
  .nota dd { margin: 0; overflow-wrap: anywhere; }
  @media (max-width: 600px) { h1 { font-size: 26px; } .kpi .v { font-size: 23px; } .destacado { font-size: 17px; } }
  @media print { body { background: #fff; color: #000; } }
</style>
</head>
<body>
<main>
  <header>
    <div class="etiqueta">Monitor inteligente de tráfico · Nivel 2</div>
    <h1>Analítica de tránsito</h1>
    <p class="sub">Sesión <strong>{{ nombre }}</strong>{% if inicio %} · {{ inicio }}{% endif %} · {{ duracion }} de video</p>
  </header>

  <section class="kpis">
    {% for k in kpis %}
    <div class="kpi{% if k.alerta %} alerta{% endif %}"><div class="t">{{ k.titulo }}</div><div class="v">{{ k.valor }}</div><div class="n">{{ k.nota }}</div></div>
    {% endfor %}
  </section>
  <div class="destacado">{{ frase }}.</div>

  <h2>Conclusiones</h2>
  <ul class="conc">
    {% for c in conclusiones %}<li>{{ c }}</li>{% endfor %}
  </ul>

  <h2>Gráficos</h2>
  {% for f in figuras %}
  <figure>
    <h3>{{ f.titulo }}</h3>
    <figcaption>{{ f.descripcion }}</figcaption>
    <img src="data:{{ f.mime }};base64,{{ f.b64 }}" alt="{{ f.titulo }}">
  </figure>
  {% endfor %}

  <h2>Tablas</h2>
  <div class="tablas">
    <div class="caja">
      <h3>Por carril</h3>
      <table>
        <tr><th>Carril</th><th class="num">Vehículos</th><th class="num">%</th><th style="width:35%"></th></tr>
        {% for c in por_carril %}
        <tr{% if c.carril == mas_usado %} class="marca"{% endif %}><td>{{ c.carril }}</td><td class="num">{{ c.vehiculos }}</td><td class="num">{{ "%.1f"|format(c.porcentaje) }}</td><td><div class="barra" style="width: {{ c.porcentaje }}%"></div></td></tr>
        {% endfor %}
      </table>
    </div>
    <div class="caja">
      <h3>Por tipo de vehículo</h3>
      <table>
        <tr><th>Tipo</th><th class="num">Vehículos</th><th class="num">%</th></tr>
        {% for nombre, n in por_clase %}
        <tr><td>{{ nombre }}</td><td class="num">{{ n }}</td><td class="num">{{ "%.1f"|format(100 * n / total) }}</td></tr>
        {% endfor %}
      </table>
      {% if linea %}
      <h3 style="margin-top:18px">Línea de conteo</h3>
      <table>
        <tr><th>Sentido</th><th class="num">Cruces</th><th class="num">por min</th></tr>
        {% for s, n in linea.totales.items() %}
        <tr><td>{{ s }}</td><td class="num">{{ n }}</td><td class="num">{{ linea.flujo_por_minuto[s] }}</td></tr>
        {% endfor %}
      </table>
      {% endif %}
    </div>
    <div class="caja">
      <h3>Por {{ nombre_intervalo }}</h3>
      <table>
        <tr><th>Intervalo</th><th class="num">Vehículos</th><th></th></tr>
        {% for r in por_intervalo %}
        <tr{% if r.intervalo == pico %} class="pico"{% endif %}><td>{{ r.intervalo }}</td><td class="num">{{ r.vehiculos }}</td><td>{% if r.parcial %}parcial{% endif %}</td></tr>
        {% endfor %}
      </table>
    </div>
  </div>

  {% if anomalias %}
  <h2>Anomalías</h2>
  <div class="caja">
    <table>
      <tr><th>Vehículo</th><th>Tipo</th><th>Momento</th><th>Detalle</th></tr>
      {% for a in anomalias %}
      <tr><td>#{{ a.track_id }}</td><td>{{ a.tipo }}</td><td>{{ a.momento }}</td><td>{{ a.detalle }}</td></tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}

  <div class="nota">
    <strong>Datos del análisis</strong>
    <dl>
      <dt>Fuente</dt><dd>{{ fuente or "no indicada" }}</dd>
      <dt>Carpeta</dt><dd>{{ carpeta }}</dd>
      <dt>Tracks</dt><dd>{{ limpieza.tracks_totales }} en el CSV → {{ limpieza.tracks_validos }} vehículos válidos ({{ limpieza.descartados_cortos }} cortos y {{ limpieza.descartados_quietos }} quietos descartados)</dd>
      <dt>Parámetros</dt><dd>intervalo {{ parametros.intervalo }} (usado: {{ nombre_intervalo }}) · mínimo {{ limpieza.min_frames }} detecciones · desplazamiento mínimo {{ limpieza.min_desplazamiento|round|int }} px · mapa de calor {{ "por detección" if parametros.por_deteccion else "por vehículo" }}</dd>
      <dt>Carriles</dt><dd>{{ "dibujados a mano (carriles.json)" if carriles_origen == "dibujados" else "detectados automáticamente" }}</dd>
      {% if linea %}<dt>Línea de conteo</dt><dd>{{ linea.origen }} · ({{ linea.linea[0]|map("round")|map("int")|join(", ") }}) → ({{ linea.linea[1]|map("round")|map("int")|join(", ") }})</dd>{% endif %}
      <dt>Generado</dt><dd>{{ generado }}</dd>
    </dl>
  </div>
</main>
</body>
</html>
"""


def generar_reporte(salidas: Path, resumen: dict) -> Path:
    """Escribe `reporte.html` en la carpeta de salidas, con las figuras embebidas."""
    salidas = Path(salidas)
    figuras = []
    for nombre, (titulo, desc) in TITULOS.items():
        ruta = salidas / nombre
        if ruta.exists():
            figuras.append({"titulo": titulo, "descripcion": desc, "mime": TIPO_MIME[ruta.suffix],
                            "b64": base64.b64encode(ruta.read_bytes()).decode("ascii")})
    inicio = None
    if resumen.get("inicio"):
        inicio = f"{datetime.fromisoformat(resumen['inicio']):%d/%m/%Y %H:%M}"
    casos = []
    for a in (resumen.get("anomalias") or {}).get("casos", []):
        if resumen.get("inicio"):
            momento = f"{datetime.fromisoformat(resumen['inicio']) + timedelta(seconds=a['t']):%H:%M:%S}"
        else:
            m, sg = divmod(int(a["t"]), 60)
            momento = f"{m:02d}:{sg:02d}"
        casos.append({**a, "momento": momento})
    generado = resumen.get("fecha_analisis") or datetime.now().isoformat(timespec="seconds")
    html = Environment(autoescape=True).from_string(PLANTILLA).render(
        nombre=Path(resumen["carpeta"]).resolve().name,
        inicio=inicio,
        duracion=_duracion(resumen["duracion_s"]),
        kpis=kpis(resumen),
        frase=resumen["frase_carril"],
        conclusiones=resumen.get("conclusiones") or conclusiones(resumen),
        figuras=figuras,
        por_carril=resumen["por_carril"],
        mas_usado=resumen["carril_mas_usado"]["carril"],
        por_clase=list(resumen["por_clase"].items()),
        total=max(resumen["total_vehiculos"], 1),
        linea=resumen.get("linea_conteo"),
        por_intervalo=resumen.get("por_intervalo", []),
        pico=resumen["pico"]["intervalo"],
        nombre_intervalo=NOMBRE_INTERVALO[resumen["intervalo"]],
        fuente=resumen.get("fuente"),
        carpeta=str(Path(resumen["carpeta"]).resolve()),
        limpieza=resumen["limpieza"],
        parametros=resumen.get("parametros") or {"intervalo": resumen["intervalo"], "por_deteccion": False},
        carriles_origen=resumen["carriles_origen"],
        anomalias=casos,
        generado=f"{datetime.fromisoformat(generado):%d/%m/%Y %H:%M}",
    )
    ruta = salidas / "reporte.html"
    ruta.write_text(html, encoding="utf-8")
    return ruta


if __name__ == "__main__":
    carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    salidas = carpeta / "salidas" if (carpeta / "salidas").is_dir() else carpeta
    res = json.loads((salidas / "resumen.json").read_text(encoding="utf-8"))
    print(generar_reporte(salidas, res))
