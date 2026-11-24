"""
Genera la presentación base (sin animaciones ni videos) con python-pptx.

Convención de nombres que luego lee animar.ps1 (PowerPoint por COM):
  A<orden>_<efecto>_<disparador>   forma o grupo animado
  V<orden>_<video>_<disparador>    hueco donde va videos/<video>.mp4
    efecto: fade | wipe | zoom | fly | appear
    disparador: click | with | after
  Las formas con el mismo <orden> forman un paso: la primera usa su disparador
  y las demás van "con la anterior".

Salida: base.pptx  (luego: powershell -File animar.ps1)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = {  # mismas medidas que anims.py
    "dos_problemas": (12.13, 3.70), "seis_gdl": (6.30, 4.30), "ego_motion": (12.13, 4.25),
    "profundidad": (7.40, 5.25), "calibracion": (12.13, 3.80), "deriva": (8.20, 5.25),
    "textura": (5.95, 4.15), "ransac": (5.95, 4.15),
}
BG, CARD, INNER, BORDER = "0E1621", "172433", "1F3044", "2C4560"
TEAL, AMBER, TXT, MUTED, CORAL = "2DE2CF", "E879F9", "F2F5F8", "A3B3C4", "FF5D73"
FONT, MONO = "Calibri", "Consolas"
TOTAL = 13

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def rgb(h): return RGBColor.from_string(h)


# ------------------------------------------------------------------ helpers
def box(sh, x, y, w, h, fill=CARD, line=BORDER, lw=1.25, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, name=None):
    s = sh.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = rgb(line); s.line.width = Pt(lw)
    s.shadow.inherit = False
    if name: s.name = name
    s.text_frame.text = ""
    return s


def text(sh, x, y, w, h, paras, size=18, color=TXT, bold=False, italic=False, align="l", anchor="t",
         font=FONT, name=None, spacing=1.0, margin=0.0):
    """paras: str | lista de párrafos; cada párrafo es str o lista de (texto, {opciones})."""
    tb = sh.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin); tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str): paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        para.line_spacing = spacing
        runs = [(p, {})] if isinstance(p, str) else p
        for t, o in runs:
            r = para.add_run(); r.text = t
            f = r.font; f.name = o.get("font", font); f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold); f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
        if isinstance(p, list) and p and "space_before" in p[0][1]:
            para.space_before = Pt(p[0][1]["space_before"])
    if name: tb.name = name
    return tb


def arrow(sh, x1, y1, x2, y2, color=TEAL, w=2.0, head=True):
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    if head:
        ln = c.line._get_or_add_ln()
        te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle"); te.set("w", "med"); te.set("len", "med")
    return c


def group(slide, name):
    g = slide.shapes.add_group_shape(); g.name = name
    return g


def new_slide(notes):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
    s.notes_slide.notes_text_frame.text = notes.strip()
    return s


def header(s, title, kicker=None, idea=None):
    text(s.shapes, 0.6, 0.32, 12.1, 0.7, title, size=32, bold=True)
    if kicker or idea:
        runs = []
        if kicker: runs.append((kicker + "  ·  ", {"color": TEAL, "bold": True, "italic": False}))
        if idea: runs.append((idea, {"color": MUTED, "italic": True}))
        text(s.shapes, 0.6, 0.98, 12.1, 0.45, [runs], size=18)


def footer(s, n, arg=None):
    text(s.shapes, 11.6, 7.0, 1.13, 0.32, f"{n} / {TOTAL}", size=11, color=MUTED, align="r")
    if arg:
        text(s.shapes, 0.6, 7.0, 1.1, 0.32, "Argumentos", size=11, color=MUTED)
        for k in range(1, 8):
            x = 1.62 + (k - 1) * 0.34
            on, past = k == arg, k < arg
            c = box(s.shapes, x, 7.03, 0.26, 0.26, fill=TEAL if on else (INNER if past else BG),
                    line=TEAL if (on or past) else BORDER, lw=1, shape=MSO_SHAPE.OVAL)
            tf = c.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(k); r.font.size = Pt(9.5); r.font.bold = on; r.font.name = FONT
            r.font.color.rgb = rgb(BG if on else (TEAL if past else MUTED))


def video(s, key, x, y, order, trigger="with"):
    w, h = SIZES[key]
    return box(s.shapes, x, y, w, h, fill=CARD, line=None, shape=MSO_SHAPE.RECTANGLE,
               name=f"V{order:02d}_{key}_{trigger}")


def card(sh, x, y, w, h, title=None, body=None, accent=TEAL, size=17, title_size=18, fill=CARD, bar=True,
         body_color=TXT):
    box(sh, x, y, w, h, fill=fill, line=BORDER)
    if bar:
        box(sh, x, y + 0.14, 0.07, h - 0.28, fill=accent, line=None, shape=MSO_SHAPE.RECTANGLE)
    paras = []
    if title: paras.append([(title, {"bold": True, "color": accent, "size": title_size})])
    if body:
        for i, b in enumerate(body if isinstance(body, list) else [body]):
            runs = b if isinstance(b, list) else [(b, {})]
            runs = [(t, {**{"size": size, "color": body_color, "space_before": 4 if (paras or i) else 0}, **o})
                    for t, o in runs]
            paras.append(runs)
    text(sh, x + 0.25, y + 0.1, w - 0.4, h - 0.2, paras, anchor="m", spacing=1.05)


def mini_person(sh, x, y, s, color="3A4E66"):
    """persona con los pies en (x, y), altura ≈ 1.25·s pulgadas"""
    box(sh, x - 0.17 * s, y - 0.80 * s, 0.34 * s, 0.80 * s, fill=color, line=None, radius=0.45)
    box(sh, x - 0.16 * s, y - 1.22 * s, 0.32 * s, 0.32 * s, fill=color, line=None, shape=MSO_SHAPE.OVAL)


def id_box(sh, x, y, w, h, label="ID 7", color=AMBER):
    box(sh, x, y, w, h, fill=None, line=color, lw=2.25, shape=MSO_SHAPE.RECTANGLE)
    t = box(sh, x - 0.012, y - 0.30, 0.62, 0.30, fill=color, line=None, shape=MSO_SHAPE.RECTANGLE)
    text(sh, x - 0.012, y - 0.30, 0.62, 0.30, label, size=12, bold=True, color=BG, align="c", anchor="m")


# ================================================================ diapositivas
# ---- 1. Portada
s = new_slide("""
(Portada, unos 20 segundos.)
Buenos días. Mi parte de la Etapa 02 es la actividad 1: explicar por qué es verdadera una afirmación que se repite mucho en visión artificial: que la odometría visual es bastante más difícil que el tracking 2D de las cámaras de seguridad.
Lo voy a defender con siete argumentos y, al final, los resumo en una tabla.
(Las dos ilustraciones aparecen solas: a la izquierda el mundo 2D, una caja con un ID; a la derecha el mundo 3D, los ejes de una cámara que se mueve.)
""")
box(s.shapes, 0, 0, 0.12, 7.5, fill=TEAL, line=None, shape=MSO_SHAPE.RECTANGLE)
text(s.shapes, 0.75, 1.35, 7.2, 0.4, "ETAPA 02  ·  ACTIVIDAD 1", size=16, bold=True, color=TEAL)
text(s.shapes, 0.75, 1.85, 7.4, 2.6, "¿Por qué la odometría visual es más difícil que el tracking 2D?",
     size=44, bold=True, spacing=0.95)
text(s.shapes, 0.75, 4.45, 7.2, 0.5, "Siete argumentos para una afirmación verdadera", size=22, color=MUTED, italic=True)
text(s.shapes, 0.75, 6.35, 7.6, 0.4, "Visión artificial  ·  [Nombre del expositor]  ·  [Universidad]  ·  noviembre 2026",
     size=14, color=MUTED)
g = group(s, "A01_fade_after")
box(g.shapes, 8.55, 1.25, 3.6, 2.7, fill=CARD, line=AMBER, lw=1.5)
box(g.shapes, 8.8, 1.5, 3.1, 2.0, fill=INNER, line=BORDER, shape=MSO_SHAPE.RECTANGLE)
box(g.shapes, 8.8, 2.95, 3.1, 0.55, fill="1A2A3B", line=None, shape=MSO_SHAPE.RECTANGLE)
mini_person(g.shapes, 10.0, 3.3, 1.1)
id_box(g.shapes, 9.68, 1.88, 0.66, 1.52)
text(g.shapes, 8.8, 3.55, 3.1, 0.35, "2D · una caja y un ID", size=14, color=AMBER, align="c")
g = group(s, "A02_fade_after")
box(g.shapes, 9.5, 4.1, 3.3, 2.75, fill=CARD, line=TEAL, lw=1.5)
ox, oy = 10.55, 5.95
arrow(g.shapes, ox, oy, ox + 1.35, oy + 0.35, TEAL, 2.5)
arrow(g.shapes, ox, oy, ox, oy - 1.15, TEAL, 2.5)
arrow(g.shapes, ox, oy, ox + 0.85, oy - 0.75, TEAL, 2.5)
text(g.shapes, ox + 1.35, oy + 0.18, 0.4, 0.35, "x", size=15, color=TEAL)
text(g.shapes, ox - 0.38, oy - 1.25, 0.4, 0.35, "z", size=15, color=TEAL)
text(g.shapes, ox + 0.85, oy - 1.0, 0.4, 0.35, "y", size=15, color=TEAL)
box(g.shapes, ox - 0.07, oy - 0.07, 0.14, 0.14, fill=AMBER, line=None, shape=MSO_SHAPE.OVAL)
text(g.shapes, 9.6, 4.2, 3.1, 0.35, "3D · 6 grados de libertad", size=14, color=TEAL, align="c")
footer(s, 1)

# ---- 2. La afirmación
s = new_slide("""
Esta es la afirmación que nos toca analizar. La leo completa. [Leer la cita.]
[CLIC] Mi veredicto: es VERDADERA.
[CLIC] La idea central, en una frase: el tracking 2D sigue cosas DENTRO de una imagen; la odometría visual tiene que recuperar cómo se movió la cámara en el mundo 3D, y solo tiene imágenes 2D para hacerlo.
[CLIC] Una aclaración antes de seguir: decir "más difícil" no quiere decir que el tracking 2D sea fácil. También tiene oclusiones, cruces entre personas y cambios de ID. La diferencia es de naturaleza: el 2D es un problema de asociación en la imagen; la odometría visual es un problema de geometría y de estimación continua del movimiento en 3D.
""")
header(s, "La afirmación a explicar")
box(s.shapes, 0.6, 1.5, 12.13, 2.45, fill=CARD, line=BORDER)
text(s.shapes, 0.85, 1.38, 1.0, 1.2, "“", size=96, bold=True, color=TEAL)
text(s.shapes, 1.75, 1.7, 10.6, 2.1,
     "La odometría visual (y el Pose Tracking 3D) es considerablemente más difícil de desarrollar e "
     "implementar con éxito que el tracking de objetos en videos de seguridad (2D).",
     size=25, italic=True, anchor="m", spacing=1.05)
g = group(s, "A01_zoom_click")
st = box(g.shapes, 9.15, 3.45, 3.35, 0.95, fill=BG, line=TEAL, lw=3.5, radius=0.12)
text(g.shapes, 9.15, 3.45, 3.35, 0.95, "VERDADERO", size=38, bold=True, color=TEAL, align="c", anchor="m")
g.rotation = -5
g = group(s, "A02_fade_click")
card(g.shapes, 0.6, 4.6, 5.85, 1.35, "Tracking 2D", "sigue cosas dentro de una imagen", accent=AMBER, size=20)
card(g.shapes, 6.88, 4.6, 5.85, 1.35, "Odometría visual / Pose 3D",
     "recupera el movimiento 3D de la cámara a partir de imágenes 2D", accent=TEAL, size=20)
text(s.shapes, 0.6, 6.2, 12.13, 0.6,
     [[("Ojo: ", {"bold": True, "color": AMBER}),
       ("«más difícil» no significa que el tracking 2D sea trivial: también tiene oclusiones, cruces y cambios de ID.",
        {"italic": True, "color": MUTED})]], size=17, name="A03_fade_click")
footer(s, 2)

# ---- 3. Dos problemas distintos
s = new_slide("""
Un repaso rápido de los dos problemas, porque toda la comparación depende de esto. El video arranca solo.
A la izquierda, una cámara de seguridad: está fija. Lo único que cambia son las personas. El sistema detecta a cada una (por ejemplo con YOLO) y le asigna un ID que mantiene entre frames (ByteTrack, BoT-SORT). La salida es una caja y un número, en píxeles.
A la derecha, la cámara va montada en algo que se mueve: un robot, un dron, un celular. Aquí la pregunta ya no es dónde está la persona, sino dónde estoy yo. La salida es mi posición y mi orientación, en metros y grados; el minimapa dibuja la trayectoria.
[CLIC] El tracking 2D responde: ¿dónde está ahora lo que vi antes?
[CLIC] La odometría visual responde: ¿cómo me moví yo en el mundo real?
""")
header(s, "Dos problemas distintos", "Repaso", "¿dónde está lo que vi?  frente a  ¿cómo me moví yo?")
video(s, "dos_problemas", 0.6, 1.55, 0, "with")
g = group(s, "A01_fade_click")
card(g.shapes, 0.6, 5.42, 5.95, 1.45, "¿Dónde está ahora lo que vi antes?",
     "Detectar (YOLO) y asociar entre frames (ByteTrack, BoT-SORT).", accent=AMBER, size=16)
g = group(s, "A02_fade_click")
card(g.shapes, 6.78, 5.42, 5.95, 1.45, "¿Cómo me moví yo en el mundo real?",
     "Estimar cuánto y hacia dónde se movió la cámara: posición y orientación.", accent=TEAL, size=16)
footer(s, 3)

# ---- 4. Argumento 1
s = new_slide("""
Primer argumento: qué se estima.
A la izquierda, el tracking 2D. Lo que busca está en la imagen: la caja tiene posición y tamaño en píxeles, cuatro números, más un ID. Se puede medir directamente, y un humano lo puede etiquetar con el mouse.
[CLIC] A la derecha, lo que estima la odometría visual: la pose de la cámara. Son seis grados de libertad: tres de traslación (x, y, z) y tres de rotación (roll, pitch, yaw). En el video, la cámara se mueve en cada uno por separado, y al final en todos a la vez, que es lo que pasa en la realidad.
Lo importante no es solo que sean más números: es que ninguno de ellos aparece en la imagen. Nadie ve "yaw = 12 grados" en un frame. Hay que deducirlo con geometría a partir de cómo se desplazan los puntos.
[CLIC] Por eso: en 2D se mide; en 3D se deduce. Es un problema inverso.
""")
header(s, "Se estima algo que no se ve", "Argumento 1", "de una caja en la imagen a 6 grados de libertad")
box(s.shapes, 0.6, 1.55, 5.6, 5.25, fill=CARD, line=BORDER)
text(s.shapes, 0.85, 1.65, 5.2, 0.45, "Tracking 2D", size=19, bold=True, color=AMBER)
box(s.shapes, 0.85, 2.15, 5.1, 2.65, fill=INNER, line=BORDER, shape=MSO_SHAPE.RECTANGLE)
box(s.shapes, 0.85, 4.05, 5.1, 0.75, fill="1A2A3B", line=None, shape=MSO_SHAPE.RECTANGLE)
box(s.shapes, 1.25, 2.45, 0.9, 1.6, fill=CARD, line=BORDER, shape=MSO_SHAPE.RECTANGLE)
mini_person(s.shapes, 3.6, 4.5, 1.5)
id_box(s.shapes, 3.15, 2.45, 0.9, 2.12)
for i, (k, v) in enumerate((("x", "412"), ("y", "230"), ("w", "64"), ("h", "150"))):
    bx = 0.85 + i * 1.29
    box(s.shapes, bx, 4.98, 1.17, 0.6, fill=INNER, line=AMBER, lw=1)
    text(s.shapes, bx, 4.98, 1.17, 0.6, [[(k + " ", {"color": MUTED}), (v, {"bold": True})]], size=17,
         align="c", anchor="m")
text(s.shapes, 0.85, 5.62, 5.2, 0.35, "píxeles, + 1 ID", size=13, color=MUTED, align="r")
g = group(s, "A02_fade_click")
box(g.shapes, 0.85, 6.0, 5.1, 0.65, fill=BG, line=AMBER, lw=1.25)
text(g.shapes, 0.85, 6.0, 5.1, 0.65, [[("2D = medir", {"color": AMBER, "bold": True}),
                                        ("   ·   ", {"color": MUTED}),
                                        ("3D = deducir (problema inverso)", {"color": TEAL, "bold": True})]],
     size=17, align="c", anchor="m")
g = group(s, "A01_fade_click")
box(g.shapes, 6.4, 1.55, 6.33, 5.25, fill=CARD, line=BORDER)
text(g.shapes, 6.65, 1.65, 5.9, 0.45, "Odometría visual / pose 3D", size=19, bold=True, color=TEAL)
text(g.shapes, 6.65, 6.33, 5.9, 0.4, "ninguno de estos números aparece en la imagen", size=14, italic=True,
     color=MUTED)
video(s, "seis_gdl", 6.42, 2.05, 1, "with")
footer(s, 4, arg=1)

# ---- 5. Argumento 2
s = new_slide("""
Segundo argumento: la cámara de seguridad está quieta, la de la odometría visual no.
Miren las flechas: son el flujo óptico, cuánto se desplaza cada punto entre un frame y el siguiente.
A la izquierda, con la cámara fija, los puntos del fondo no se mueven. Lo único con flechas es la persona: lo que se mueve ES lo que interesa. Por eso funciona algo tan simple como restar el fondo.
A la derecha, la cámara avanza y TODA la imagen se mueve: los puntos salen desde el centro, que es hacia donde voy. Ese movimiento de toda la imagen es el ego-motion, el movimiento propio de la cámara.
Ahora fíjense cuando cruza el peatón (flechas rojas): su movimiento no encaja con el resto. El sistema tiene que decidir qué flechas son mías y cuáles son de otros. Si se equivoca y usa las del peatón, estima mal su propio movimiento.
[CLIC] Fondo fijo: lo que se mueve es el objeto.
[CLIC] Cámara móvil: toda la imagen cambia.
[CLIC] Y los objetos independientes contaminan la estimación.
""")
header(s, "Cámara fija vs. cámara en movimiento", "Argumento 2", "hay que separar el movimiento propio del ajeno")
video(s, "ego_motion", 0.6, 1.55, 0, "with")
cw = (12.13 - 0.4) / 3
for i, (acc, t1, t2) in enumerate((
        (AMBER, "Fondo fijo", "lo que se mueve es el objeto"),
        (TEAL, "Cámara móvil", "toda la imagen cambia (ego-motion)"),
        (CORAL, "Objetos independientes", "un peatón puede arruinar la estimación"))):
    g = group(s, f"A{i + 1:02d}_fade_click")
    card(g.shapes, 0.6 + i * (cw + 0.2), 5.95, cw, 0.95, t1, t2, accent=acc, size=15, title_size=16)
footer(s, 5, arg=2)

# ---- 6. Argumento 3
s = new_slide("""
Tercer argumento: al pasar del mundo 3D a una imagen 2D, se pierde la profundidad.
Arriba: un objeto pequeño y cerca, y uno grande y lejos, producen exactamente la misma imagen. Si ves un auto pequeño en la foto, no sabes si es un auto de juguete cerca o uno real lejos.
Abajo, la consecuencia para la odometría visual con una sola cámara: estas tres trayectorias (×0,5, ×1 y ×2) generan las mismas imágenes, porque todo el mundo está escalado igual. Los rayos hacia el punto ámbar son paralelos. La cámara sabe hacia dónde se movió, pero no cuántos metros.
[CLIC] La proyección borra la profundidad.
[CLIC] Con una cámara (monocular) hay ambigüedad de escala, y además la escala se va desviando con el tiempo (deriva de escala).
[CLIC] Para resolverlo se agrega hardware: una segunda cámara (estéreo), un sensor de profundidad o una IMU. Más costo y más calibración.
[CLIC] En pose tracking 3D de personas pasa lo mismo: varias posturas 3D producen la misma silueta 2D.
""")
header(s, "Al pasar de 3D a 2D se pierde la profundidad", "Argumento 3", "se sabe la dirección, no los metros")
video(s, "profundidad", 0.6, 1.55, 0, "with")
items = [
    (TEAL, "La proyección borra la profundidad", None),
    (TEAL, "Monocular: escala desconocida", "y además se desvía con el tiempo"),
    (AMBER, "Solución: más sensores", "estéreo, profundidad o IMU → más costo"),
    (TEAL, "Pose humana 3D", "varias posturas → la misma silueta 2D"),
]
for i, (acc, t1, t2) in enumerate(items):
    g = group(s, f"A{i + 1:02d}_fade_click")
    card(g.shapes, 8.25, 1.55 + i * 1.34, 4.48, 1.2, t1, t2, accent=acc, size=15, title_size=17)
footer(s, 6, arg=3)

# ---- 7. Argumento 4
s = new_slide("""
Cuarto argumento: la odometría visual necesita conocer muy bien su cámara.
Izquierda: la lente deforma la imagen; las rectas salen curvas (distorsión radial). Si no se corrige, cada medición sale torcida.
Derecha: un recorrido de una manzana con cuatro giros de 90 grados. La línea punteada es el recorrido real. Veamos qué pasa si la distancia focal que uso está mal por unos pocos por ciento: cada giro de 90 se mide como 87 u 86 grados, y al volver al inicio el error ya son varios metros. Un error chico en la calibración se propaga a TODA la trayectoria.
[CLIC] Lo que hay que conocer: intrínsecos (focal, centro óptico), distorsión y, si hay varios sensores, su posición relativa. Y luego geometría multivista: matriz esencial, PnP, triangulación.
[CLIC] El tracking 2D trabaja directo en píxeles: no necesita nada de esto para funcionar.
""")
header(s, "Hace falta calibración y geometría precisa", "Argumento 4", "un error pequeño se propaga a toda la trayectoria")
video(s, "calibracion", 0.6, 1.55, 0, "with")
cw = (12.13 - 0.6) / 4
cards4 = [(TEAL, "Intrínsecos", "distancia focal, centro óptico"),
          (TEAL, "Distorsión", "y posición entre sensores"),
          (TEAL, "Geometría multivista", "matriz esencial, PnP, triangulación")]
g = group(s, "A01_fade_click")
for i, (acc, t1, t2) in enumerate(cards4):
    card(g.shapes, 0.6 + i * (cw + 0.2), 5.55, cw, 1.3, t1, t2, accent=acc, size=15, title_size=17)
g = group(s, "A02_fade_click")
card(g.shapes, 0.6 + 3 * (cw + 0.2), 5.55, cw, 1.3, "Tracking 2D", "trabaja en píxeles: no lo necesita",
     accent=AMBER, size=15, title_size=17)
footer(s, 7, arg=4)

# ---- 8. Argumento 5
s = new_slide("""
Quinto argumento, y para mí el más importante: los errores se acumulan.
La odometría visual estima el movimiento entre un frame y el siguiente, y suma. Cada paso tiene un error pequeño, pero ese error se queda y se arrastra. Es como caminar con los ojos cerrados contando pasos: al principio vas bien, pero al dar la vuelta a la manzana ya no sabes dónde estás.
Miren la curva de abajo: el error de la odometría visual (turquesa) solo crece. El tracking 2D (ámbar) también se equivoca, cambia el ID 7 por el 12, pero se recupera y el error vuelve a cero: es LOCAL.
Al final aparece la solución clásica: el cierre de ciclo. El sistema reconoce que volvió a un lugar conocido y reparte el error a lo largo del camino. Funciona, pero hay que detectar el lugar y volver a optimizar todo.
[CLIC] Un número para recordar: un solo grado de error al girar, después de 100 metros, son 1,75 m de desvío lateral (100 × sen 1°). Y el rumbo se sigue equivocando, así que crece cada vez más rápido.
[CLIC] Corregirlo cuesta: optimización local, bundle adjustment, cierre de ciclos.
[CLIC] En 2D, si se pierde un ID, se reasocia y no contamina el resto del video.
""")
header(s, "Los errores se acumulan: la deriva", "Argumento 5", "cada paso es relativo al anterior")
video(s, "deriva", 0.6, 1.55, 0, "with")
g = group(s, "A01_fade_click")
box(g.shapes, 9.05, 1.55, 3.68, 2.15, fill=CARD, line=TEAL, lw=1.5)
text(g.shapes, 9.25, 1.62, 3.3, 1.0, [[("1°", {"size": 50, "bold": True, "color": TEAL}),
                                       ("  de error al girar", {"size": 17, "color": MUTED})]], anchor="m")
text(g.shapes, 9.25, 2.62, 3.3, 1.0, [[("tras 100 m ", {"color": MUTED}), ("≈ 1,75 m", {"bold": True, "color": TXT}),
                                       (" de desvío", {"color": MUTED})],
                                      [("100 · sen 1°", {"size": 13, "color": MUTED, "italic": True})]],
     size=19, anchor="m")
g = group(s, "A02_fade_click")
card(g.shapes, 9.05, 3.9, 3.68, 1.35, "Corregirla cuesta",
     "optimización local, bundle adjustment, cierre de ciclos", accent=TEAL, size=15, title_size=17)
g = group(s, "A03_fade_click")
card(g.shapes, 9.05, 5.45, 3.68, 1.35, "En 2D el fallo es local",
     "se pierde un ID, se reasocia y listo", accent=AMBER, size=15, title_size=17)
footer(s, 8, arg=5)

# ---- 9. Argumento 6
s = new_slide("""
Sexto argumento: el entorno.
Izquierda: la odometría visual vive de los puntos que puede seguir entre imágenes. Esto es un detector real (FAST, de OpenCV) corriendo sobre dos paredes. La de ladrillos da más de 150 puntos; la pared lisa, cero. Ahora la cámara se mueve rápido y la imagen se desenfoca: los puntos desaparecen. Luego baja la luz: lo mismo. Sin puntos no hay movimiento que estimar.
[CLIC] Entonces: poca textura, poca luz, desenfoque y objetos que se mueven dejan a la odometría visual sin datos fiables.
[CLIC] Derecha: e incluso con puntos, muchos emparejamientos están mal. RANSAC lo resuelve probando: toma dos puntos al azar, traza el modelo y cuenta cuántos puntos están de acuerdo. Repite y se queda con el que más apoyo tiene. Al final, en turquesa, la recta de RANSAC; en ámbar, lo que pasa si simplemente promedio todo: los errores la arrastran.
En odometría visual, los puntos son emparejamientos entre imágenes y la «recta» es el movimiento de la cámara.
[CLIC] Hay que filtrar muchos outliers en cada frame, en tiempo real.
""")
header(s, "El entorno lo pone difícil", "Argumento 6", "sin textura, sin luz o con prisa, no hay qué seguir")
video(s, "textura", 0.6, 1.55, 0, "with")
g = group(s, "A01_fade_click")
card(g.shapes, 0.6, 5.85, 5.95, 1.0, "Pocos puntos fiables",
     "poca textura, poca luz, desenfoque, objetos dinámicos", accent=AMBER, size=15, title_size=17)
video(s, "ransac", 6.78, 1.55, 2, "click")
g = group(s, "A03_fade_click")
card(g.shapes, 6.78, 5.85, 5.95, 1.0, "Muchos emparejamientos erróneos",
     "hay que filtrarlos (RANSAC) en cada frame", accent=CORAL, size=15, title_size=17)
footer(s, 9, arg=6)

# ---- 10. Argumento 7
s = new_slide("""
Séptimo argumento: la madurez de las herramientas.
A la izquierda, tracking 2D. Con Ultralytics, tres líneas: cargar un YOLO preentrenado y llamar a track con ByteTrack. Hay modelos entrenados con datasets enormes (COCO, MOT17) y el enfoque "detectar y asociar" funciona bien en la mayoría de los videos de seguridad.
[CLIC] A la derecha, la cadena de la odometría visual. [Los pasos van apareciendo.] Calibrar, sincronizar sensores, extraer y emparejar puntos, filtrar con RANSAC, estimar la pose, recuperar la escala, optimizar, cerrar ciclos y, para saber si funciona, compararla con una verdad de terreno (GPS de precisión o captura de movimiento), que es cara y difícil de conseguir.
Ojo, existen librerías serias, como ORB-SLAM3 o VINS-Fusion. Pero no se usan solo con instalarlas: cada sensor, entorno y velocidad exige calibrar y ajustar parámetros.
[CLIC] Y una última diferencia: el costo del error. Si el tracking 2D cambia un ID, un conteo sale mal. Si la odometría visual se equivoca, el robot o el dron choca.
""")
header(s, "Herramientas: listas para usar vs. a la medida", "Argumento 7",
       "detectar y asociar ya está resuelto; la odometría visual se ajusta a cada caso")
box(s.shapes, 0.6, 1.55, 5.3, 4.15, fill=CARD, line=BORDER)
text(s.shapes, 0.85, 1.65, 4.9, 0.45, "Tracking 2D", size=19, bold=True, color=AMBER)
box(s.shapes, 0.85, 2.2, 4.8, 1.95, fill=BG, line=BORDER, radius=0.04)
code = [[("from", {"color": TEAL}), (" ultralytics ", {}), ("import", {"color": TEAL}), (" YOLO", {})],
        [("model = YOLO(", {}), ('"yolo11n.pt"', {"color": AMBER}), (")", {})],
        [("model.track(", {}), ('"video.mp4"', {"color": AMBER}), (",", {})],
        [("            tracker=", {}), ('"bytetrack.yaml"', {"color": AMBER}), (")", {})]]
text(s.shapes, 1.0, 2.3, 4.6, 1.8, code, size=13.5, font=MONO, anchor="m", spacing=1.15)
text(s.shapes, 0.85, 4.3, 4.8, 1.3, [[("4 líneas", {"bold": True, "color": AMBER}),
                                      (" · modelos preentrenados", {})],
                                     [("datasets enormes (COCO, MOT17)", {"space_before": 4})]],
     size=16, color=TXT)
box(s.shapes, 6.1, 1.55, 6.63, 4.15, fill=CARD, line=BORDER)
text(s.shapes, 6.35, 1.65, 6.2, 0.45, "Odometría visual", size=19, bold=True, color=TEAL)
steps = ["Calibrar la cámara", "Sincronizar sensores", "Extraer y emparejar puntos", "Filtrar con RANSAC",
         "Estimar la pose", "Recuperar la escala", "Optimizar (bundle adj.)", "Cerrar ciclos",
         "Evaluar con verdad de terreno"]
cw3, ch3 = 1.98, 0.88
for i, st_ in enumerate(steps):
    r, c = divmod(i, 3)
    x, y = 6.35 + c * (cw3 + 0.12), 2.2 + r * (ch3 + 0.12)
    g = group(s, f"A{i + 1:02d}_fade_{'click' if i == 0 else 'after'}")
    box(g.shapes, x, y, cw3, ch3, fill=INNER, line=TEAL if i < 8 else AMBER, lw=1)
    box(g.shapes, x + 0.1, y + 0.27, 0.34, 0.34, fill=TEAL if i < 8 else AMBER, line=None, shape=MSO_SHAPE.OVAL)
    text(g.shapes, x + 0.1, y + 0.27, 0.34, 0.34, str(i + 1), size=12, bold=True, color=BG, align="c", anchor="m")
    text(g.shapes, x + 0.52, y + 0.04, cw3 - 0.58, ch3 - 0.08, st_, size=14, anchor="m", spacing=0.95)
g = group(s, "A09_fade_after")  # misma orden que el último paso
text(g.shapes, 6.35, 5.2, 6.2, 0.4, "Hay librerías (ORB-SLAM3, VINS-Fusion), pero no son «instalar y listo».",
     size=13.5, italic=True, color=MUTED)
g = group(s, "A10_wipe_click")
box(g.shapes, 0.6, 5.88, 12.13, 0.98, fill=CARD, line=BORDER)
text(g.shapes, 0.85, 5.88, 2.0, 0.98, "¿Y si falla?", size=19, bold=True, anchor="m")
text(g.shapes, 2.75, 5.88, 4.7, 0.98, [[("2D: ", {"bold": True, "color": AMBER}),
                                        ("un ID cambiado → un conteo mal hecho", {})]], size=17, anchor="m")
text(g.shapes, 7.45, 5.88, 5.2, 0.98, [[("OV: ", {"bold": True, "color": CORAL}),
                                        ("una pose mal estimada → el robot choca", {})]], size=17, anchor="m")
footer(s, 10, arg=7)

# ---- 11. Tabla resumen
s = new_slide("""
Resumo los siete argumentos en una tabla. Cada clic agrega una fila, una por argumento.
[CLIC] Qué estima: caja e ID en píxeles, frente a una pose de 6 grados de libertad.
[CLIC] Cámara: fija frente a en movimiento, con el ego-motion que hay que separar.
[CLIC] Profundidad: el 2D no la necesita; la odometría visual tiene que recuperarla, y con una cámara no conoce la escala.
[CLIC] Calibración: no hace falta frente a imprescindible.
[CLIC] Error: local frente a acumulativo; esta es la fila más importante.
[CLIC] Entorno: el 2D tolera bastante; la odometría visual es sensible a textura, luz y velocidad.
[CLIC] Herramientas: listas para usar frente a ajuste por sensor y escenario.
""")
header(s, "Comparación resumen", "Resumen", "los siete argumentos en una tabla")
cols = [(0.6, 2.75), (3.45, 4.55), (8.1, 4.63)]
box(s.shapes, cols[1][0], 1.55, cols[1][1], 0.55, fill=AMBER, line=None, radius=0.12)
text(s.shapes, cols[1][0], 1.55, cols[1][1], 0.55, "Tracking 2D (seguridad)", size=17, bold=True, color=BG,
     align="c", anchor="m")
box(s.shapes, cols[2][0], 1.55, cols[2][1], 0.55, fill=TEAL, line=None, radius=0.12)
text(s.shapes, cols[2][0], 1.55, cols[2][1], 0.55, "Odometría visual / Pose 3D", size=17, bold=True, color=BG,
     align="c", anchor="m")
rows = [("Qué estima", "Caja + ID en píxeles", "Pose de 6 GdL en el espacio"),
        ("Cámara", "Fija", "En movimiento (ego-motion)"),
        ("Profundidad", "No la necesita", "Hay que recuperarla (escala)"),
        ("Calibración", "No hace falta", "Imprescindible"),
        ("Error", "Local: se recupera", "Acumulativo: deriva"),
        ("Entorno", "Tolera bastante", "Sensible a textura, luz y velocidad"),
        ("Herramientas", "YOLO + ByteTrack, listas", "Ajuste por sensor y escenario")]
for i, (a, b, c) in enumerate(rows):
    y = 2.2 + i * 0.66
    g = group(s, f"A{i + 1:02d}_wipe_click")
    for j, (val, (x, w)) in enumerate(zip((a, b, c), cols)):
        box(g.shapes, x, y, w, 0.58, fill=CARD if j else INNER, line=BORDER, lw=1, radius=0.1)
        col = MUTED if j == 0 else TXT
        text(g.shapes, x + 0.2, y, w - 0.3, 0.58, [[(f"{i + 1}  ", {"color": TEAL, "bold": True}), (a, {})]]
             if j == 0 else val, size=16, color=col, bold=(j == 2 and i == 4), anchor="m")
footer(s, 11)

# ---- 12. Conclusión
s = new_slide("""
Para cerrar.
La afirmación es verdadera. La odometría visual tiene que estimar movimiento en 3D, con una cámara que se mueve, sin medir la profundidad directamente, con una calibración exigente y con errores que se acumulan.
[CLIC] Estas son las cinco palabras clave. [Aparecen una a una.]
[CLIC] El matiz: más difícil no significa que el tracking 2D sea trivial. Tiene sus propios retos (oclusiones, cruces, cambios de ID), pero trabaja en el plano de la imagen y con herramientas maduras.
[CLIC] Y el puente con la actividad 4 de mis compañeros: en cuanto la cámara de seguridad se mueve (una PTZ o un dron), el tracking 2D solo ya no basta y hay que compensar el movimiento de la cámara; es decir, hace falta odometría visual.
""")
header(s, "Conclusión", "Cierre", "la afirmación es verdadera")
box(s.shapes, 0.6, 1.55, 12.13, 1.45, fill=CARD, line=TEAL, lw=1.5)
text(s.shapes, 0.9, 1.6, 11.6, 1.35,
     [[("Es ", {}), ("verdadero", {"bold": True, "color": TEAL}),
       (": la odometría visual estima movimiento 3D con una cámara que se mueve, sin profundidad directa, "
        "con calibración exigente y con errores que se acumulan.", {})]], size=21, anchor="m", spacing=1.05)
kw = ["6 GdL", "cámara móvil", "sin profundidad", "calibración", "deriva"]
kx = 0.6
for i, k in enumerate(kw):
    w = 0.42 + 0.135 * len(k)
    g = group(s, f"A{i + 1:02d}_zoom_{'click' if i == 0 else 'after'}")
    box(g.shapes, kx, 3.2, w, 0.55, fill=INNER, line=TEAL, lw=1.25, radius=0.5)
    text(g.shapes, kx, 3.2, w, 0.55, k, size=17, bold=True, color=TEAL, align="c", anchor="m")
    kx += w + 0.18
g = group(s, "A06_fade_click")
card(g.shapes, 0.6, 3.98, 5.95, 1.5, "Matiz",
     "«Más difícil» no significa que el tracking 2D sea trivial: también tiene oclusiones y cambios de ID.",
     accent=AMBER, size=15.5)
g = group(s, "A07_fade_click")
card(g.shapes, 6.78, 3.98, 5.95, 1.5, "Puente a la actividad 4",
     "Si la cámara se mueve (PTZ, dron), el tracking 2D solo no basta: hace falta odometría visual.",
     accent=TEAL, size=15.5)
refs = [
    "Scaramuzza, D. y Fraundorfer, F. (2011). Visual Odometry, Part I: The First 30 Years and Fundamentals. IEEE Robotics & Automation Magazine, 18(4).",
    "Fraundorfer, F. y Scaramuzza, D. (2012). Visual Odometry, Part II: Matching, Robustness, Optimization, and Applications. IEEE RAM, 19(2).",
    "Hartley, R. y Zisserman, A. (2004). Multiple View Geometry in Computer Vision (2.ª ed.). Cambridge University Press.",
    "Zhang, Y. et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV.",
    "Aharon, N., Orfaig, R. y Bobrovsky, B.-Z. (2022). BoT-SORT: Robust Associations Multi-Pedestrian Tracking. arXiv:2206.14651.",
    "Campos, C. et al. (2021). ORB-SLAM3. IEEE Transactions on Robotics, 37(6).  ·  OpenCV: calibración de cámara y flujo óptico.",
]
text(s.shapes, 0.6, 5.62, 2.0, 0.3, "Referencias", size=12, bold=True, color=MUTED)
text(s.shapes, 0.6, 5.88, 12.13, 1.1, refs, size=9.5, color=MUTED, spacing=1.0)
footer(s, 12)

# ---- 13. Preguntas
s = new_slide("""
Respuestas preparadas:
• ¿La odometría visual siempre acumula error? Sí: cada estimación es relativa a la anterior. Se reduce con optimización (bundle adjustment) y cierre de ciclos, pero no desaparece del todo.
• ¿Cómo se resuelve la escala en monocular? Con estéreo, un sensor de profundidad, una IMU (odometría visual-inercial) o conociendo el tamaño real de algún objeto.
• ¿El tracking 2D nunca falla? Falla con oclusiones, cruces y cambios de apariencia; la diferencia es que el error es local y se recupera.
• ¿Y si la cámara de seguridad se mueve (PTZ o dron)? El tracking 2D deja de bastar: hay que compensar el movimiento de la cámara, y eso es odometría visual (actividad 4).
• ¿Qué es RANSAC? Un método que prueba modelos con subconjuntos aleatorios y se queda con el que más datos explica, descartando los emparejamientos erróneos (outliers).
• ¿Por qué 1° da 1,75 m? Porque tras 100 m el desvío lateral es 100 · sen(1°) ≈ 1,745 m; y como el rumbo sigue acumulando error, en la práctica crece más rápido.
• ¿Las redes neuronales no lo resuelven? Hay odometría visual aprendida, pero generalizar a cámaras y entornos nuevos sigue siendo difícil; los sistemas en producción siguen siendo, sobre todo, geométricos o híbridos.
""")
box(s.shapes, 0, 0, 0.12, 7.5, fill=TEAL, line=None, shape=MSO_SHAPE.RECTANGLE)
text(s.shapes, 0.75, 2.4, 11.8, 1.3, "¿Preguntas?", size=60, bold=True)
text(s.shapes, 0.75, 3.75, 11.8, 0.6, "Gracias", size=24, color=MUTED, italic=True)
g = group(s, "A01_fade_after")
for i, (c, t) in enumerate(((AMBER, "Tracking 2D: medir en la imagen"), (TEAL, "Odometría visual: deducir el movimiento en 3D"))):
    box(g.shapes, 0.75, 4.75 + i * 0.6, 0.18, 0.18, fill=c, line=None, shape=MSO_SHAPE.OVAL)
    text(g.shapes, 1.1, 4.62 + i * 0.6, 10, 0.45, t, size=18, color=MUTED)
footer(s, 13)

out = os.path.join(HERE, "base.pptx")
prs.save(out)
print("Guardado:", out, len(prs.slides), "diapositivas")
