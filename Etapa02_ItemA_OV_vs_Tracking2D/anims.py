"""
Animaciones (mp4) para la Etapa 02, Actividad 1:
"¿Por qué la odometría visual es más difícil que el tracking 2D?"

Cada video se renderiza con matplotlib al tamaño exacto (en pulgadas) del hueco
que ocupa en la diapositiva, así el texto del video tiene el mismo tamaño en
puntos que el texto de PowerPoint.

Uso:
    python anims.py                 # renderiza todos
    python anims.py deriva          # solo uno
    python anims.py deriva --frame 9.5   # guarda un PNG de prueba en t = 9,5 s
"""
import os, sys, shutil, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
import cv2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
os.makedirs(OUT, exist_ok=True)

BG, CARD, INNER, BORDER = "#0E1621", "#172433", "#1F3044", "#2C4560"
TEAL, AMBER, TXT, MUTED, CORAL = "#2DE2CF", "#E879F9", "#F2F5F8", "#A3B3C4", "#FF5D73"
plt.rcParams.update({"font.family": "Calibri", "axes.unicode_minus": False})
FPS, DPI = 25, 160

# Tamaño (pulgadas) de cada hueco en la diapositiva. build_deck.py usa los mismos.
SIZES = {
    "dos_problemas": (12.13, 3.70),
    "seis_gdl":      (6.30, 4.30),
    "ego_motion":    (12.13, 4.25),
    "profundidad":   (7.40, 5.25),
    "calibracion":   (12.13, 3.80),
    "deriva":        (8.20, 5.25),
    "textura":       (5.95, 4.15),
    "ransac":        (5.95, 4.15),
}


def ease(u):
    u = float(np.clip(u, 0, 1))
    return u * u * (3 - 2 * u)


def num(v, d=1):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def panel(fig, rect, xlim=(0, 16), ylim=(0, 9), fc=INNER, border=True):
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_facecolor(fc)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BORDER if border else fc); s.set_linewidth(1.2)
    return ax


def render(name, draw, seconds, test_t=None):
    size = SIZES[name]
    fig = plt.figure(figsize=size, dpi=DPI)
    fig.patch.set_facecolor(CARD)

    def grab(t):
        fig.clf(); fig.patch.set_facecolor(CARD)
        draw(fig, t)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, w = img.shape[:2]
        return np.ascontiguousarray(img[: h - h % 2, : w - w % 2])

    if test_t is not None:
        p = os.path.join(OUT, f"_test_{name}.png")
        plt.imsave(p, grab(test_t)); print("PNG:", p); return

    n = int(round(seconds * FPS))
    first = grab(0); h, w = first.shape[:2]
    mp4 = os.path.join(OUT, f"{name}.mp4")
    ff = shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                             "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    proc.stdin.write(first.tobytes())
    img = first
    for i in range(1, n):
        img = grab(i / FPS)
        proc.stdin.write(img.tobytes())
    proc.stdin.close(); proc.wait()
    plt.imsave(os.path.join(OUT, f"{name}_final.png"), img)
    plt.close(fig)
    print(f"Listo: {mp4}  {w}x{h}  {n} frames")


# ---------------------------------------------------------------- figuras
def person(ax, x, y, s, col=MUTED, z=3):
    """Persona estilizada con los pies en (x, y) y altura ~ 3*s."""
    ax.add_patch(FancyBboxPatch((x - 0.45 * s, y), 0.9 * s, 1.9 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.4 * s}",
                                fc=col, ec="none", alpha=0.85, zorder=z))
    ax.add_patch(Circle((x, y + 2.45 * s), 0.42 * s, fc=col, ec="none", alpha=0.85, zorder=z))


# ======================================================== 1. dos problemas
def draw_dos_problemas(fig, t):
    fig.text(0.012, 0.93, "CÁMARA FIJA", color=AMBER, fontsize=15, fontweight="bold", va="center")
    fig.text(0.118, 0.93, "· tracking 2D en video de seguridad", color=MUTED, fontsize=13, va="center")
    fig.text(0.512, 0.93, "CÁMARA EN MOVIMIENTO", color=TEAL, fontsize=15, fontweight="bold", va="center")
    fig.text(0.708, 0.93, "· odometría visual", color=MUTED, fontsize=13, va="center")

    # ---- izquierda: cámara de seguridad
    ax = panel(fig, [0.008, 0.03, 0.488, 0.84])
    ax.add_patch(Rectangle((0, 0), 16, 3.2, fc="#1A2A3B", ec="none"))           # suelo
    ax.plot([0, 16], [3.2, 3.2], color=BORDER, lw=1.5)
    for x0 in (1.2, 9.6):                                                          # puertas
        ax.add_patch(Rectangle((x0, 3.2), 1.8, 3.6, fc=CARD, ec=BORDER, lw=1.2))
    ax.add_patch(Rectangle((4.3, 4.6), 3.6, 1.8, fc=CARD, ec=BORDER, lw=1.2))     # ventana
    ax.add_patch(Rectangle((12.6, 3.2), 2.6, 0.8, fc=CARD, ec=BORDER, lw=1.2))    # banca
    blink = (t * 1.6) % 2 < 1.3
    ax.add_patch(Circle((0.7, 8.3), 0.2, fc=CORAL if blink else INNER, ec="none"))
    ax.text(1.1, 8.3, f"REC  00:00:{int(t):02d}", color=MUTED, fontsize=11, va="center")

    people = [
        (1, 1.0 + 0.95 * t, 2.2, 0.62),
        (2, 15.0 - 0.75 * t, 2.6, 0.55),
        (3, 8.0 + 2.6 * np.sin(0.45 * t), 0.7, 0.78),
    ]
    for pid, x, y, s in people:
        person(ax, x, y, s)
        bx, by, bw, bh = x - 0.75 * s, y - 0.12, 1.5 * s, 3.1 * s
        ax.add_patch(Rectangle((bx, by), bw, bh, fc="none", ec=AMBER, lw=1.8, zorder=5))
        ax.add_patch(Rectangle((bx, by + bh), 1.25, 0.5, fc=AMBER, ec="none", zorder=5))
        ax.text(bx + 0.62, by + bh + 0.25, f"ID {pid}", color=BG, fontsize=10, fontweight="bold",
                ha="center", va="center", zorder=6)
    ax.text(15.7, 0.25, "salida: caja + ID, en píxeles", color=AMBER, fontsize=12, ha="right", va="bottom")

    # ---- derecha: pasillo en primera persona
    ax = panel(fig, [0.504, 0.03, 0.488, 0.84])
    ax.add_patch(Rectangle((0, 0), 16, 9, fc="#14202D", ec="none"))
    yaw = 0.10 * np.sin(0.55 * t)
    speed, cx, cy, f = 1.5, 8.0, 4.7, 4.2
    d = speed * t

    def proj(x, y, z):
        xc = x * np.cos(yaw) - z * np.sin(yaw)
        zc = x * np.sin(yaw) + z * np.cos(yaw)
        if zc < 0.35:
            return None
        return cx + f * xc / zc, cy + f * y / zc

    W, HF, HC = 1.8, -1.3, 1.3
    # bordes longitudinales
    for (x, y) in ((-W, HF), (W, HF), (-W, HC), (W, HC)):
        pts = [proj(x, y, z) for z in np.linspace(0.4, 80, 60)]
        pts = np.array([p for p in pts if p])
        ax.plot(pts[:, 0], pts[:, 1], color=BORDER, lw=1.6)
    S = 2.0
    k0 = int(d // S)
    for k in range(k0, k0 + 32):
        z = k * S - d
        if z < 0.4:
            continue
        a, b = proj(-W, HF, z), proj(W, HF, z)
        al = float(np.clip(1.2 - z / 40, 0.12, 1))
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=BORDER, lw=1.1, alpha=al)
        if k % 3 == 0:                      # puertas en ambas paredes
            for side in (-W, W):
                q = [proj(side, HF, z), proj(side, HF, z + 1.1), proj(side, 0.7, z + 1.1), proj(side, 0.7, z)]
                if all(q):
                    ax.add_patch(Polygon(q, closed=True, fc=CARD, ec=TEAL, lw=1.0, alpha=0.35 + 0.5 * al))
        if k % 2 == 0:                      # luces de techo
            q = [proj(-0.4, HC, z), proj(0.4, HC, z), proj(0.4, HC, z + 0.5), proj(-0.4, HC, z + 0.5)]
            if all(q):
                ax.add_patch(Polygon(q, closed=True, fc=TXT, ec="none", alpha=0.25 + 0.45 * al))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)

    # minimapa: trayectoria estimada vista desde arriba
    ts = np.linspace(0, t, max(2, int(t * 25)))
    yaws = 0.10 * np.sin(0.55 * ts)
    xs = np.cumsum(-np.sin(yaws) * speed / 25) * 6  # exagerado ×6 para que se vea
    zs = ts * speed
    mx = fig.add_axes([0.855, 0.06, 0.13, 0.36]); mx.set_facecolor(CARD)
    mx.set_xticks([]); mx.set_yticks([])
    for s in mx.spines.values(): s.set_color(TEAL); s.set_linewidth(1)
    mx.set_xlim(-6, 6); mx.set_ylim(-1.5, 20)
    mx.plot(xs, zs, color=TEAL, lw=2)
    hd = yaws[-1]
    tri = np.array([[0, 1.6], [-0.7, -0.5], [0.7, -0.5]])
    R = np.array([[np.cos(hd), -np.sin(hd)], [np.sin(hd), np.cos(hd)]])
    mx.add_patch(Polygon(tri @ R.T * [1, 1] + [xs[-1], zs[-1]], fc=TEAL, ec="none"))
    mx.text(0, 18.6, "mi trayectoria", color=TEAL, fontsize=9.5, ha="center", va="center")
    ax.text(0.3, 0.25, f"salida: x = {num(xs[-1] / 6, 2)} m   z = {num(zs[-1], 1)} m   "
                       f"giro = {num(np.degrees(hd), 1)}°", color=TEAL, fontsize=12, va="bottom")


# ======================================================== 2. seis GdL
def rotm(roll, pitch, yaw):
    """Mundo: X derecha, Y adelante, Z arriba. roll sobre Y, pitch sobre X, yaw sobre Z."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


SEG = 1.8
LABELS6 = [("x", "m"), ("y", "m"), ("z", "m"), ("roll", "°"), ("pitch", "°"), ("yaw", "°")]


def pose6(t):
    v = np.zeros(6); active = None
    amp = [0.8, 0.8, 0.6, 35, 30, 40]
    if t < 6 * SEG:
        k = int(t // SEG); u = (t - k * SEG) / SEG
        v[k] = amp[k] * np.sin(np.pi * u); active = k
    else:
        u = t - 6 * SEG
        g = ease(u / 0.8) * (1 - ease((u - 4.6) / 0.8))
        fr = [0.9, 0.7, 1.1, 1.3, 0.8, 1.0]
        ph = [0, 1, 2, 0.5, 1.5, 2.5]
        v = np.array([amp[i] * 0.7 * np.sin(fr[i] * u * 2 + ph[i]) for i in range(6)]) * g
        active = "all"
    return v, active


def draw_seis_gdl(fig, t):
    v, active = pose6(t)
    ax = fig.add_axes([-0.06, -0.06, 0.78, 1.10], projection="3d")
    ax.set_facecolor(CARD)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0)); a.line.set_color((0, 0, 0, 0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.0, 1.9); ax.set_zlim(-1.3, 1.2)
    ax.set_box_aspect((2.6, 2.9, 2.5))
    ax.view_init(elev=20, azim=-62)
    # suelo
    for g in np.linspace(-1.3, 1.3, 7):
        ax.plot([g, g], [-1.0, 1.9], [-1.25, -1.25], color=BORDER, lw=0.8)
    for g in np.linspace(-1.0, 1.9, 7):
        ax.plot([-1.3, 1.3], [g, g], [-1.25, -1.25], color=BORDER, lw=0.8)
    # ejes del mundo (fijos)
    for d, lab in ((np.array([1, 0, 0]), "X"), (np.array([0, 1, 0]), "Y"), (np.array([0, 0, 1]), "Z")):
        p = d * 0.5 + np.array([-1.25, -0.95, -1.25])
        ax.plot([-1.25, p[0]], [-0.95, p[1]], [-1.25, p[2]], color=MUTED, lw=1.2)
        ax.text(*(p + d * 0.12), lab, color=MUTED, fontsize=9)

    R = rotm(*np.radians(v[3:])); T = v[:3]
    w, h, dz = 0.55, 0.40, 0.95
    corners = np.array([[-w, dz, -h], [w, dz, -h], [w, dz, h], [-w, dz, h]])
    C = corners @ R.T + T
    O = T
    for c in C:
        ax.plot([O[0], c[0]], [O[1], c[1]], [O[2], c[2]], color=TEAL, lw=1.8)
    loop = np.vstack([C, C[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=TEAL, lw=2.4)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection([C], facecolors=TEAL, alpha=0.18, edgecolors="none"))
    # "arriba" de la cámara: triangulito
    up = np.array([[-0.12, dz, h + 0.02], [0.12, dz, h + 0.02], [0, dz, h + 0.17]]) @ R.T + T
    ax.add_collection3d(Poly3DCollection([up], facecolors=TEAL, alpha=0.9, edgecolors="none"))
    # ejes propios de la cámara
    for d in np.eye(3):
        e = (d * 0.45) @ R.T + T
        ax.plot([O[0], e[0]], [O[1], e[1]], [O[2], e[2]], color=AMBER, lw=1.3, alpha=0.8)
    # sombra en el suelo
    ax.scatter([T[0]], [T[1]], [-1.25], color=TEAL, s=30, alpha=0.35)

    # ---- lectura de los 6 números
    x0 = 0.70
    fig.text(x0, 0.90, "Pose de la cámara", color=TXT, fontsize=14, fontweight="bold")
    fig.text(x0, 0.83, "= 6 números", color=TEAL, fontsize=13)
    for i, (lab, unit) in enumerate(LABELS6):
        y = 0.70 - i * 0.092 - (0.04 if i >= 3 else 0)
        on = active == "all" or active == i
        col = TEAL if on else MUTED
        if on:
            fig.patches.append(FancyBboxPatch((x0 - 0.01, y - 0.035), 0.29, 0.07, transform=fig.transFigure,
                                              boxstyle="round,pad=0,rounding_size=0.012",
                                              fc=INNER, ec=TEAL if active == i else BORDER, lw=1.2))
        val = num(v[i], 2) if unit == "m" else num(v[i], 0)
        fig.text(x0 + 0.01, y, lab, color=col, fontsize=13, fontweight="bold" if on else "normal", va="center")
        fig.text(x0 + 0.265, y, f"{val} {unit}", color=col, fontsize=13, ha="right", va="center")
    fig.text(x0, 0.735, "traslación", color=MUTED, fontsize=10, style="italic")
    fig.text(x0, 0.415, "rotación", color=MUTED, fontsize=10, style="italic")


# ======================================================== 3. ego-motion
rng3 = np.random.default_rng(3)
_pts = []
for _ in range(110):                      # fachadas
    side = rng3.choice([-1, 1])
    _pts.append((side * 3.6, rng3.uniform(-1.5, 3.5), rng3.uniform(2.5, 24)))
for _ in range(45):                       # suelo
    _pts.append((rng3.uniform(-3.4, 3.4), -1.5, rng3.uniform(2.5, 24)))
PTS3 = np.array(_pts)
PED_LOCAL = np.array([[0, -1.5 + h, 0] for h in (0.15, 0.6, 1.0, 1.4)] +
                     [[0.25, -0.5, 0], [-0.25, -0.5, 0], [0.2, -1.2, 0], [-0.2, -1.2, 0]])


def draw_ego_motion(fig, t):
    f, cx, cy, speed, dt = 6.0, 8.0, 5.0, 2.2, 0.3
    fig.text(0.012, 0.935, "CÁMARA FIJA", color=AMBER, fontsize=15, fontweight="bold", va="center")
    fig.text(0.118, 0.935, "· el fondo no se mueve", color=MUTED, fontsize=13, va="center")
    fig.text(0.512, 0.935, "CÁMARA EN MOVIMIENTO", color=TEAL, fontsize=15, fontweight="bold", va="center")
    fig.text(0.708, 0.935, "· toda la imagen se mueve", color=MUTED, fontsize=13, va="center")

    def proj(P):
        z = np.maximum(P[:, 2], 0.3)
        return np.c_[cx + f * P[:, 0] / z, cy + f * P[:, 1] / z]

    ped_x = lambda tt: -2.6 + 0.7 * tt
    for side in (0, 1):
        moving = side == 1
        ax = panel(fig, [0.008 + side * 0.496, 0.03, 0.488, 0.84], fc="#14202D")
        # bordes de la calle
        for x, y in ((-3.6, -1.5), (3.6, -1.5)):
            q = proj(np.array([[x, y, 0.5], [x, y, 60]]))
            ax.plot(q[:, 0], q[:, 1], color=BORDER, lw=1.4)
        off = speed * t if moving else 0.0
        P = PTS3.copy(); P[:, 2] = (P[:, 2] - off - 2.5) % 21.5 + 2.5
        Pp = P.copy()
        if moving:
            Pp[:, 2] = P[:, 2] + speed * dt
        a, b = proj(Pp), proj(P)
        ok = (P[:, 2] > 2.8) & (Pp[:, 2] < 24)
        if moving:
            ax.quiver(a[ok, 0], a[ok, 1], (b - a)[ok, 0], (b - a)[ok, 1], color=TEAL, angles="xy",
                      scale_units="xy", scale=1, width=0.0032, headwidth=4, headlength=4.5, alpha=0.9)
        ax.scatter(b[ok, 0], b[ok, 1], s=7, color=TEAL if moving else MUTED, alpha=0.9, zorder=3)

        # peatón (en la derecha aparece a los 4,5 s)
        tp = t if not moving else t - 4.5
        if tp >= 0:
            z_ped = 4.5
            Pd = PED_LOCAL + [ped_x(tp), 0, z_ped]
            Pd0 = PED_LOCAL + [ped_x(tp - dt), 0, z_ped + (speed * dt if moving else 0)]
            q, q0 = proj(Pd), proj(Pd0)
            base = proj(np.array([[ped_x(tp), -1.5, z_ped]]))[0]
            s = f / z_ped * 0.62
            person(ax, base[0], base[1], s, col="#3A4E66", z=4)
            col = AMBER if not moving else CORAL
            ax.quiver(q0[:, 0], q0[:, 1], (q - q0)[:, 0] * 2.2, (q - q0)[:, 1] * 2.2, color=col, angles="xy",
                      scale_units="xy", scale=1, width=0.0045, headwidth=4, headlength=4.5, zorder=6)
            if moving and tp > 0.8:
                ax.text(base[0], base[1] + 3.9 * s + 0.4, "¿me muevo yo o se mueve él?", color=CORAL,
                        fontsize=12.5, fontweight="bold", ha="center", va="bottom", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=CORAL, lw=1))
        if moving:
            ax.plot([cx], [cy], marker="+", color=TXT, ms=10, mew=1.5, alpha=0.6)
            ax.text(cx + 0.3, cy - 0.15, "hacia donde voy", color=TXT, fontsize=10.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#14202D", ec="none"), zorder=8)
        msg = ("solo se mueve lo que interesa" if not moving else
               "hay que separar MI movimiento (ego-motion) del de los demás")
        ax.text(0.3, 0.25, msg, color=AMBER if not moving else TEAL, fontsize=12.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="#14202D", ec="none"))
        ax.set_xlim(0, 16); ax.set_ylim(0, 9)


# ======================================================== 4. profundidad
def draw_profundidad(fig, t):
    ax = panel(fig, [0.0, 0.0, 1.0, 1.0], xlim=(0, 16), ylim=(0, 11.35), fc=CARD, border=False)
    # --- parte A: el mismo tamaño en la imagen
    ax.text(0.3, 10.9, "1 · Pequeño y cerca, o grande y lejos", color=TXT, fontsize=14, fontweight="bold", va="center")
    O = np.array([1.3, 8.1]); xi = 2.6
    u = ease(t / 6.0) if t < 6.0 else 1.0
    dist = 2.8 + 8.4 * u             # 2,8 → 11,2 (unidades de dibujo)
    k = 0.19
    hh = k * dist                     # mitad de la altura del objeto
    # cámara
    ax.add_patch(Polygon([[O[0] - 0.9, O[1] - 0.45], [O[0] - 0.9, O[1] + 0.45], [O[0], O[1]]],
                         fc=TEAL, ec="none"))
    ax.add_patch(Rectangle((O[0] - 1.25, O[1] - 0.3), 0.4, 0.6, fc=TEAL, ec="none"))
    # plano de imagen
    ax.plot([xi, xi], [O[1] - 1.5, O[1] + 1.5], color=TXT, lw=2.5)
    ax.text(xi, O[1] - 1.85, "imagen", color=MUTED, fontsize=11, ha="center")
    pi = k * (xi - O[0])
    ax.plot([xi, xi], [O[1] - pi, O[1] + pi], color=TEAL, lw=6, solid_capstyle="butt")
    # fantasmas de posiciones anteriores
    for gd in (2.8, 5.6, 8.4):
        if gd < dist - 0.2:
            gh = k * gd
            ax.add_patch(Rectangle((O[0] + gd - 0.12 * gh, O[1] - gh), 0.24 * gh, 2 * gh, fc="none",
                                   ec=MUTED, lw=1, ls=(0, (3, 3)), alpha=0.5))
    X = O[0] + dist
    for sgn in (1, -1):
        ax.plot([O[0], X], [O[1], O[1] + sgn * hh], color=AMBER, lw=1.2, alpha=0.8)
    ax.add_patch(FancyBboxPatch((X - 0.12 * hh, O[1] - hh), 0.24 * hh, 2 * hh,
                                boxstyle=f"round,pad=0,rounding_size={0.05 * hh}", fc=INNER, ec=AMBER, lw=2))
    scale = dist / 2.8
    ax.text(15.7, 10.9, f"tamaño ×{num(scale)}   distancia ×{num(scale)}", color=AMBER, fontsize=13,
            ha="right", va="center")
    if t > 2.0:
        ax.text(xi + 0.25, O[1] + 1.75, "en la imagen mide lo mismo", color=TEAL, fontsize=12, va="center")

    # --- parte B: ambigüedad de escala
    if t < 6.6:
        return
    tb = t - 6.6
    ax.plot([0.3, 15.7], [5.55, 5.55], color=BORDER, lw=1)
    ax.text(0.3, 5.05, "2 · Con una sola cámara, ¿cuántos metros me moví?", color=TXT, fontsize=14,
            fontweight="bold", va="center")
    base = np.array([0.9, 1.8])
    s_ = np.linspace(0, 1, 60)
    path = np.c_[s_ * 3.3, 0.55 * np.sin(s_ * np.pi * 1.2)]
    lm = np.array([2.9, 0.95])                     # un punto del mundo (en escala ×1)
    scales = [(0.5, (0, (1, 2))), (1.0, "-"), (2.0, (0, (5, 3)))]
    for i, (sc, ls) in enumerate(scales):
        a = ease((tb - i * 1.0) / 1.0)
        if a <= 0:
            continue
        n = max(2, int(len(s_) * a))
        P = base + path[:n] * sc
        ax.plot(P[:, 0], P[:, 1], color=TEAL, lw=2.4, ls=ls, alpha=0.55 + 0.15 * i)
        end = P[-1]
        ax.add_patch(Circle(end, 0.16, fc=TEAL, ec="none"))
        L = base + lm * sc
        ax.add_patch(Polygon([L + [0, 0.32], L + [-0.28, -0.18], L + [0.28, -0.18]], fc=AMBER, ec="none"))
        if a >= 1:
            ax.plot([end[0], L[0]], [end[1], L[1]], color=AMBER, lw=1, ls=(0, (2, 2)), alpha=0.9)
            ax.text(end[0] + 0.15, end[1] - 0.55, f"×{num(sc)}", color=TEAL, fontsize=12, va="center")
    if tb > 3.4:
        ax.text(15.7, 3.05, "Las tres trayectorias\nproducen exactamente\nlas mismas imágenes.", color=TXT,
                fontsize=13, ha="right", va="center", linespacing=1.3)
        ax.text(15.7, 1.25, "Se sabe la dirección,\nno los metros.", color=TEAL, fontsize=13,
                fontweight="bold", ha="right", va="center", linespacing=1.3)


# ======================================================== 5. calibración
def route(turn_scale):
    """Recorrido de una manzana: 4 tramos rectos + 4 giros de 90° a la izquierda (metros)."""
    segs = [(40, 0), (12, np.pi / 2), (24, 0), (12, np.pi / 2), (40, 0), (12, np.pi / 2), (24, 0), (12, np.pi / 2)]
    ds = 0.25; x = y = th = 0.0; P = [(0, 0)]
    for L, ang in segs:
        n = int(L / ds)
        for _ in range(n):
            th += ang / n * turn_scale
            x += ds * np.cos(th); y += ds * np.sin(th); P.append((x, y))
    return np.array(P)


TRUE_ROUTE = route(1.0)


def draw_calibracion(fig, t):
    # --- izquierda: distorsión
    k1 = -0.26 * ease(t / 4.0)
    fig.text(0.012, 0.93, "Distorsión de la lente", color=TXT, fontsize=14, fontweight="bold", va="center")
    ax = panel(fig, [0.008, 0.04, 0.33, 0.80], xlim=(-1.25, 1.25), ylim=(-0.92, 0.92), fc="#14202D")
    g = np.linspace(-1.0, 1.0, 11)
    sline = np.linspace(-1.0, 1.0, 80)
    for c in g:
        for P in (np.c_[np.full_like(sline, c), sline * 0.72], np.c_[sline, np.full_like(sline, c * 0.72)]):
            r2 = (P ** 2).sum(1, keepdims=True)
            D = P * (1 + k1 * r2)
            ax.plot(P[:, 0], P[:, 1], color=BORDER, lw=0.8, alpha=0.6)
            ax.plot(D[:, 0], D[:, 1], color=AMBER, lw=1.3)
    ax.text(-1.18, -0.86, f"k₁ = {num(k1, 2)}", color=AMBER, fontsize=12.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="#14202D", ec="none"))
    ax.text(1.18, -0.86, "rectas que salen curvas", color=MUTED, fontsize=11, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="#14202D", ec="none"))

    # --- derecha: error de focal → trayectoria que no cierra
    e = 0.05 * ease(t / 7.5)
    fig.text(0.355, 0.93, "Error en la distancia focal", color=TXT, fontsize=14, fontweight="bold", va="center")
    ax = panel(fig, [0.35, 0.04, 0.40, 0.80], xlim=(-20, 60), ylim=(-7, 50), fc="#14202D")
    ax.set_aspect("auto")
    ax.plot(TRUE_ROUTE[:, 0], TRUE_ROUTE[:, 1], color=TXT, lw=1.6, ls=(0, (4, 3)), alpha=0.7)
    est = route(1 / (1 + e))
    ax.plot(est[:, 0], est[:, 1], color=TEAL, lw=2.6)
    ax.add_patch(Circle((0, 0), 1.1, fc=TXT, ec="none"))
    ax.add_patch(Circle(est[-1], 1.1, fc=TEAL, ec="none"))
    gap = np.hypot(*est[-1])
    if gap > 1.5:
        ax.plot([0, est[-1, 0]], [0, est[-1, 1]], color=CORAL, lw=1.5)
    ax.plot([-17, -13], [46.5, 46.5], color=TXT, lw=1.6, ls=(0, (4, 3)), alpha=0.7)
    ax.text(-12, 46.5, "real", color=TXT, fontsize=11, va="center")
    ax.plot([-3, 1], [46.5, 46.5], color=TEAL, lw=2.6)
    ax.text(2, 46.5, "estimada", color=TEAL, fontsize=11, va="center")

    # --- lectura y "deslizador"
    x0 = 0.77
    fig.text(x0, 0.80, "error en la focal", color=MUTED, fontsize=12)
    fig.text(x0, 0.69, f"+{num(e * 100, 1)} %", color=TEAL, fontsize=26, fontweight="bold", va="center")
    sx = fig.add_axes([x0, 0.555, 0.21, 0.05]); sx.axis("off"); sx.set_xlim(0, 5); sx.set_ylim(-1, 1)
    sx.plot([0, 5], [0, 0], color=BORDER, lw=5, solid_capstyle="round")
    sx.plot([0, e * 100], [0, 0], color=TEAL, lw=5, solid_capstyle="round")
    sx.plot([e * 100], [0], "o", color=TXT, ms=10)
    fig.text(x0, 0.43, "cada giro de 90° se mide como", color=MUTED, fontsize=12)
    fig.text(x0, 0.33, f"{num(90 / (1 + e), 1)}°", color=TXT, fontsize=22, fontweight="bold", va="center")
    fig.text(x0, 0.20, "al volver al inicio:", color=MUTED, fontsize=12)
    fig.text(x0, 0.10, f"{num(gap, 1)} m de desvío", color=CORAL if gap > 1.5 else TXT, fontsize=19,
             fontweight="bold", va="center")


# ======================================================== 6. deriva
def loop_path():
    # rectángulo redondeado de 60 × 36 m, recorrido antihorario, paso 0,5 m
    segs = [(54, 0), (6 * np.pi / 2, np.pi / 2), (30, 0), (6 * np.pi / 2, np.pi / 2),
            (54, 0), (6 * np.pi / 2, np.pi / 2), (30, 0), (6 * np.pi / 2, np.pi / 2)]
    ds = 0.5; x = y = th = 0.0; P = [(0, 0)]; turns = []
    for L, ang in segs:
        n = int(round(L / ds))
        for _ in range(n):
            turns.append(ang / n)
            th += ang / n; x += ds * np.cos(th); y += ds * np.sin(th); P.append((x, y))
    return np.array(P), np.array(turns), ds


DRIFT_TRUE, DRIFT_TURNS, DRIFT_DS = loop_path()


def drift_est(seed=7):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0012, 0.0045, len(DRIFT_TURNS))          # error de giro por paso (rad)
    sc = 1 + rng.normal(0.0, 0.03, len(DRIFT_TURNS))               # error de avance por paso
    x = y = th = 0.0; P = [(0, 0)]
    for dth, s in zip(DRIFT_TURNS + noise, sc):
        th += dth; x += DRIFT_DS * s * np.cos(th); y += DRIFT_DS * s * np.sin(th); P.append((x, y))
    return np.array(P)


DRIFT_EST = drift_est()
DRIFT_ERR = np.hypot(*(DRIFT_EST - DRIFT_TRUE).T)
_N = len(DRIFT_TRUE)
DRIFT_CORR = DRIFT_EST - np.linspace(0, 1, _N)[:, None] * (DRIFT_EST[-1] - DRIFT_TRUE[-1])
T_TRACE, T_LC0, T_LC1 = 10.5, 11.6, 13.2


def draw_deriva(fig, t):
    u = min(t / T_TRACE, 1.0)
    i = max(2, int(u * (_N - 1)))
    lc = ease((t - T_LC0) / (T_LC1 - T_LC0))

    fig.text(0.015, 0.95, "Odometría visual: caminar por una manzana y volver al inicio", color=TXT,
             fontsize=13.5, fontweight="bold", va="center")
    ax = panel(fig, [0.012, 0.03, 0.56, 0.86], xlim=(-12, 72), ylim=(-12, 50), fc="#14202D")
    ax.plot(DRIFT_TRUE[:i, 0], DRIFT_TRUE[:i, 1], color=TXT, lw=1.6, ls=(0, (4, 3)), alpha=0.75)
    E = DRIFT_EST[:i] * (1 - lc) + DRIFT_CORR[:i] * lc
    ax.plot(E[:, 0], E[:, 1], color=TEAL, lw=2.6)
    ax.add_patch(Circle(DRIFT_TRUE[i - 1], 1.0, fc=TXT, ec="none"))
    ax.add_patch(Circle(E[-1], 1.2, fc=TEAL, ec="none"))
    ax.add_patch(Rectangle((-1.5, -1.5), 3, 3, fc="none", ec=TXT, lw=1.2))
    ax.text(0, -3.2, "inicio", color=MUTED, fontsize=10.5, ha="center", va="top")
    if lc <= 0:
        err = DRIFT_ERR[i - 1]
        if err > 1:
            ax.plot([DRIFT_TRUE[i - 1, 0], E[-1, 0]], [DRIFT_TRUE[i - 1, 1], E[-1, 1]], color=CORAL, lw=1.6)
        ax.text(70, 47, f"error acumulado: {num(err, 1)} m", color=CORAL if err > 3 else TXT, fontsize=14,
                fontweight="bold", ha="right", va="top")
    else:
        ax.text(70, 47, "cierre de ciclo: «¡ya estuve aquí!»", color=TEAL, fontsize=14, fontweight="bold",
                ha="right", va="top")
        if lc >= 1:
            ax.text(70, 42.5, "el error se reparte y se corrige", color=MUTED, fontsize=12, ha="right", va="top")
    ax.plot([], [], color=TXT, ls=(0, (4, 3)), label="real")
    ax.plot([], [], color=TEAL, lw=2.6, label="estimada")
    lg = ax.legend(loc="center", frameon=False, fontsize=11.5, labelcolor=TXT, bbox_to_anchor=(0.5, 0.5))

    # --- columna derecha: tracking 2D vs OV
    tl = fig.add_axes([0.61, 0.62, 0.375, 0.17])
    tl.set_xlim(0, 1); tl.set_ylim(0, 1); tl.axis("off")
    fig.text(0.61, 0.86, "Tracking 2D: la misma persona", color=AMBER, fontsize=13, fontweight="bold", va="center")
    sw0, sw1 = 0.40, 0.52
    segs = [(0, sw0, AMBER, "ID 7"), (sw0, sw1, CORAL, "ID 12"), (sw1, 1, AMBER, "ID 7")]
    for a, b, col, lab in segs:
        if u <= a:
            continue
        bb = min(b, u)
        tl.add_patch(Rectangle((a, 0.25), bb - a, 0.5, fc=col, ec=CARD, lw=1.5))
        if bb - a > 0.06:
            tl.text((a + bb) / 2, 0.5, lab, color=BG, fontsize=11, fontweight="bold", ha="center", va="center")
    if u > sw1 + 0.04:
        tl.text((sw0 + sw1) / 2, 0.02, "se equivoca\ny se recupera", color=CORAL, fontsize=10, ha="center",
                va="top", linespacing=1.1)

    ex = fig.add_axes([0.62, 0.10, 0.365, 0.36]); ex.set_facecolor("#14202D")
    for s in ex.spines.values(): s.set_color(BORDER)
    ex.tick_params(colors=MUTED, labelsize=9.5, length=2)
    ex.set_xlim(0, 1); ex.set_ylim(-0.5, DRIFT_ERR.max() * 1.15)
    ex.set_xticks([]); ex.set_xlabel("tiempo", color=MUTED, fontsize=11)
    ex.set_ylabel("error (m)", color=MUTED, fontsize=11)
    fig.text(0.61, 0.52, "¿Cómo crece el error?", color=TXT, fontsize=13, fontweight="bold", va="center")
    xs = np.linspace(0, 1, _N)
    ex.plot(xs[:i], DRIFT_ERR[:i], color=TEAL, lw=2.2)
    e2 = np.where((xs > sw0) & (xs < sw1), DRIFT_ERR.max() * 0.35, 0.0)
    ex.plot(xs[:i], e2[:i], color=AMBER, lw=2.2)
    if u > 0.75:
        ex.text(0.04, DRIFT_ERR.max() * 0.98, "OV: se acumula", color=TEAL, fontsize=11.5, va="top")
        ex.text(0.04, DRIFT_ERR.max() * 0.80, "2D: error local", color=AMBER, fontsize=11.5, va="top")


# ======================================================== 7. textura
def make_walls(W=320, H=230, seed=4):
    rng = np.random.default_rng(seed)
    tex = np.full((H, W), 120, np.uint8)
    bh, bw = 26, 58
    for r in range(0, H // bh + 1):
        off = (r % 2) * bw // 2
        for c in range(-1, W // bw + 2):
            x0, y0 = c * bw + off, r * bh
            g = int(rng.integers(95, 175))
            cv2.rectangle(tex, (x0 + 2, y0 + 2), (x0 + bw - 2, y0 + bh - 2), g, -1)
    tex = cv2.add(tex, rng.integers(0, 25, (H, W)).astype(np.uint8))
    for _ in range(14):
        cv2.circle(tex, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(3, 9)),
                   int(rng.integers(40, 90)), -1)
    yy, xx = np.mgrid[0:H, 0:W]
    flat = (175 + 20 * xx / W - 10 * yy / H).astype(np.float32)
    flat += rng.normal(0, 1.2, (H, W))
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    return tex, flat


WALL_TEX, WALL_FLAT = make_walls()
FAST = cv2.FastFeatureDetector_create(threshold=28, nonmaxSuppression=True)


def degrade(img, blur, light):
    out = img.copy()
    if blur > 1:
        k = np.zeros((1, blur), np.float32); k[0, :] = 1 / blur
        out = cv2.filter2D(out, -1, k)
    if light < 0.999:
        rng = np.random.default_rng(int(light * 1000))
        f = out.astype(np.float32) * light + rng.normal(0, 3, out.shape)
        out = np.clip(f, 0, 255).astype(np.uint8)
    return out


def textura_state(t):
    if t < 3.5:
        return 1, 1.0, "condiciones normales"
    if t < 8.0:
        return int(1 + 17 * ease((t - 3.5) / 4.0)), 1.0, "desenfoque por movimiento rápido"
    return 1, 1.0 - 0.82 * ease((t - 8.0) / 3.5), "poca luz"


def draw_textura(fig, t):
    blur, light, label = textura_state(t)
    fig.text(0.02, 0.94, "¿Cuántos puntos puedo seguir?", color=TXT, fontsize=14, fontweight="bold", va="center")
    for j, (img, name) in enumerate(((WALL_TEX, "pared con textura"), (WALL_FLAT, "pared lisa"))):
        im = degrade(img, blur, light)
        kps = FAST.detect(im, None)
        ax = fig.add_axes([0.02 + j * 0.49, 0.29, 0.47, 0.56])
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(BORDER)
        if kps:
            P = np.array([k.pt for k in kps])
            ax.scatter(P[:, 0], P[:, 1], s=14, facecolors="none", edgecolors=TEAL, linewidths=1.0)
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
        fig.text(0.02 + j * 0.49, 0.235, name, color=MUTED, fontsize=11.5, va="center")
        n = len(kps)
        fig.text(0.49 + j * 0.49, 0.235, f"{n} puntos", color=TEAL if n > 40 else CORAL, fontsize=15,
                 fontweight="bold", ha="right", va="center")
    fig.text(0.02, 0.09, label, color=AMBER if label != "condiciones normales" else MUTED, fontsize=14,
             fontweight="bold" if label != "condiciones normales" else "normal", va="center")


# ======================================================== 8. RANSAC
rng8 = np.random.default_rng(11)
RX_IN = np.sort(rng8.uniform(0.5, 9.5, 26))
RY_IN = 0.55 * RX_IN + 1.2 + rng8.normal(0, 0.18, RX_IN.size)
RX_OUT = rng8.uniform(0.5, 9.5, 12)
RY_OUT = rng8.uniform(4.6, 7.4, 12)
RX = np.r_[RX_IN, RX_OUT]; RY = np.r_[RY_IN, RY_OUT]
SAMPLES = [(28, 31), (3, 33), (5, 30), (2, 20), (36, 9), (6, 23)]
TOL = 0.45


def fit_line(i, j):
    m = (RY[j] - RY[i]) / (RX[j] - RX[i]); b = RY[i] - m * RX[i]
    return m, b


def inliers(m, b):
    return np.abs(RY - (m * RX + b)) / np.sqrt(1 + m * m) < TOL


IT_DUR = 1.15


def draw_ransac(fig, t):
    fig.text(0.02, 0.94, "RANSAC: quedarse con lo consistente", color=TXT, fontsize=14, fontweight="bold",
             va="center")
    ax = panel(fig, [0.02, 0.17, 0.96, 0.70], xlim=(0, 10), ylim=(0, 8.2), fc="#14202D")
    xx = np.array([0, 10])
    k = int(t // IT_DUR); n_it = len(SAMPLES)
    best = None; best_n = -1
    for it in range(min(k, n_it)):
        m, b = fit_line(*SAMPLES[it]); c = inliers(m, b).sum()
        if c > best_n: best, best_n = (m, b), c
    final = k >= n_it + 1
    if not final:
        ax.scatter(RX, RY, s=26, color=MUTED, zorder=3)
        if best is not None:
            ax.plot(xx, best[0] * xx + best[1], color=TEAL, lw=2.2, alpha=0.9, zorder=2)
        if k < n_it:
            i, j = SAMPLES[k]; m, b = fit_line(i, j); msk = inliers(m, b)
            ph = (t - k * IT_DUR) / IT_DUR
            if ph > 0.25:
                ax.fill_between(xx, m * xx + b - TOL * np.sqrt(1 + m * m), m * xx + b + TOL * np.sqrt(1 + m * m),
                                color=AMBER, alpha=0.12, lw=0)
                ax.plot(xx, m * xx + b, color=AMBER, lw=1.5, ls=(0, (4, 3)))
            if ph > 0.5:
                ax.scatter(RX[msk], RY[msk], s=30, color=AMBER, zorder=4)
                ax.text(9.8, 0.35, f"prueba {k + 1}: {msk.sum()} puntos de acuerdo", color=AMBER, fontsize=12.5,
                        ha="right", va="bottom", zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="#14202D", ec="none"))
            ax.scatter(RX[[i, j]], RY[[i, j]], s=140, facecolors="none", edgecolors=TXT, linewidths=2, zorder=5)
        if best is not None:
            ax.text(0.2, 7.9, f"mejor hasta ahora: {best_n}", color=TEAL, fontsize=12, va="top")
        fig.text(0.02, 0.085, "1) elige 2 puntos al azar   2) traza la recta   3) cuenta cuántos coinciden",
                 color=MUTED, fontsize=11.5, va="center")
    else:
        m, b = best; msk = inliers(m, b)
        mls, bls = np.polyfit(RX, RY, 1)
        ax.plot(xx, mls * xx + bls, color=AMBER, lw=2, ls=(0, (5, 3)))
        ax.plot(xx, m * xx + b, color=TEAL, lw=2.6)
        ax.scatter(RX[msk], RY[msk], s=30, color=TEAL, zorder=4)
        ax.scatter(RX[~msk], RY[~msk], s=50, color=CORAL, marker="x", linewidths=2, zorder=4)
        ax.text(9.8, 0.35, "RANSAC", color=TEAL, fontsize=13, fontweight="bold", ha="right", va="bottom")
        ax.text(0.2, 7.9, "promedio de todos (engañado por los errores)", color=AMBER, fontsize=12, va="top")
        fig.text(0.02, 0.085, "En OV los puntos son emparejamientos y la «recta» es el movimiento de la cámara.",
                 color=MUTED, fontsize=11.5, va="center")


ANIMS = {
    "dos_problemas": (draw_dos_problemas, 12.0),
    "seis_gdl": (draw_seis_gdl, 6 * SEG + 5.6),
    "ego_motion": (draw_ego_motion, 12.0),
    "profundidad": (draw_profundidad, 12.5),
    "calibracion": (draw_calibracion, 10.5),
    "deriva": (draw_deriva, 15.0),
    "textura": (draw_textura, 13.0),
    "ransac": (draw_ransac, (len(SAMPLES) + 1) * IT_DUR + 4.0),
}

if __name__ == "__main__":
    args = sys.argv[1:]
    test = None
    if "--frame" in args:
        i = args.index("--frame"); test = float(args[i + 1]); del args[i:i + 2]
    names = args or list(ANIMS)
    for nm in names:
        fn, dur = ANIMS[nm]
        render(nm, fn, dur, test)
