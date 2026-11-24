"""
Animación: robot con ruedas en suelo firme vs. suelo resbaladizo.

Arriba: las ruedas no patinan -> lo que cuentan los encoders = lo que avanzó.
Abajo: en la zona de barro las ruedas giran pero el robot avanza menos.
       Fantasma ámbar = donde los encoders creen que está.
       Marca turquesa = lo que estima la cámara (odometría visual).

Salida: salida/robot_resbala.mp4 y salida/robot_resbala_final.png
Uso:  python robot_anim.py
"""
import os, subprocess, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salida")
os.makedirs(OUT, exist_ok=True)

CARD, INNER, BORDER, BG = "#172433", "#1F3044", "#2C4560", "#0E1621"
TEAL, AMBER, TXT, MUTED, MUD = "#3DD6C4", "#FFB627", "#F2F5F8", "#A3B3C4", "#5A4526"
plt.rcParams.update({"font.family": "Calibri"})

FPS, T_MOVE, T_HOLD = 25, 7.5, 2.5
W_TOTAL = 6.0                 # metros que cuentan las ruedas al final
X0, SC = 0.9, 1.55            # origen y escala (unidades de dibujo por metro)
R = 0.27                      # radio de rueda (dibujo)
ZONE = (1.5, 2.7)             # zona resbaladiza en metros reales
SLIP_EFF = 0.35               # en la zona, el robot avanza 35 % de lo que giran las ruedas
G1, G2 = 3.95, 0.75           # altura del suelo de cada carril

def real_from_wheel(s):
    """Avance real en el carril resbaladizo dado lo que giraron las ruedas."""
    a, b = ZONE
    s_in = a
    s_out = a + (b - a) / SLIP_EFF
    if s <= s_in: return s
    if s <= s_out: return a + (s - s_in) * SLIP_EFF
    return b + (s - s_out)

def fmt(v): return f"{v:.1f} m".replace(".", ",")

def ease(u): return u * u * (3 - 2 * u)

def robot(ax, x, g, ang, ghost=False):
    col = AMBER if ghost else TEAL
    ls = (0, (4, 3)) if ghost else "-"
    fill = "none" if ghost else INNER
    z = 3 if ghost else 4
    ax.add_patch(FancyBboxPatch((x - 0.62, g + 0.36), 1.24, 0.52, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fill, ec=col, lw=1.8, ls=ls, zorder=z))
    # cámara al frente
    ax.add_patch(Rectangle((x + 0.18, g + 0.88), 0.36, 0.22, fc=fill if not ghost else "none", ec=col, lw=1.5, ls=ls, zorder=z))
    if not ghost: ax.add_patch(Circle((x + 0.58, g + 0.99), 0.06, fc=TEAL, ec="none", zorder=z + 1))
    for wx in (x - 0.36, x + 0.36):
        ax.add_patch(Circle((wx, g + R), R, fc=BG if not ghost else "none", ec=col if ghost else MUTED, lw=1.6, alpha=0.7 if ghost else 1, zorder=z + 1))
        if not ghost:
            for k in range(3):
                a = ang + k * np.pi / 3
                ax.plot([wx - R * 0.85 * np.cos(a), wx + R * 0.85 * np.cos(a)],
                        [g + R - R * 0.85 * np.sin(a), g + R + R * 0.85 * np.sin(a)], color=MUTED, lw=1.4, zorder=z + 2)
            ax.add_patch(Circle((wx, g + R), 0.05, fc=TEAL, ec="none", zorder=z + 3))

def ground(ax, g, slippery):
    ax.plot([0.3, 12.5], [g, g], color=MUTED, lw=1.5, zorder=1)
    for m in range(0, 8):
        x = X0 + m * SC
        if x > 12.4: break
        ax.plot([x, x], [g - 0.08, g], color=MUTED, lw=1, zorder=1)
        ax.text(x, g - 0.14, f"{m} m", color=MUTED, fontsize=9, ha="center", va="top")
    if slippery:
        a, b = X0 + ZONE[0] * SC, X0 + ZONE[1] * SC
        ax.add_patch(Rectangle((a, g - 0.06), b - a, 0.12, fc=MUD, ec="none", zorder=2))
        xs = np.linspace(a, b, 120)
        for k in range(3):
            ax.plot(xs, g + 0.1 + 0.11 * k + 0.035 * np.sin(xs * 9 + k), color=AMBER, lw=1, alpha=0.35, zorder=1)
        ax.text((a + b) / 2, g - 0.42, "barro / arena", color=AMBER, fontsize=10, ha="center", va="top", alpha=0.9)

rng = np.random.default_rng(3)
particles = []  # [x, y, vx, vy, vida]

def frame(t, fig, ax):
    ax.clear(); ax.set_xlim(0, 12.8); ax.set_ylim(0, 6.6); ax.axis("off")
    u = ease(min(t / T_MOVE, 1.0))
    sw = W_TOTAL * u
    ang = -sw * SC / R
    # ---- carril 1: suelo firme
    ax.text(0.35, 6.3, "SUELO FIRME", color=TEAL, fontsize=13, fontweight="bold", va="center")
    ground(ax, G1, False)
    robot(ax, X0 + sw * SC, G1, ang)
    ax.text(12.45, 6.3, f"ruedas: {fmt(sw)}    real: {fmt(sw)}", color=TXT, fontsize=13, ha="right", va="center")
    ax.text(12.45, 5.85, "coinciden", color=TEAL, fontsize=12, ha="right", va="center", style="italic")
    ax.plot([0.3, 12.5], [3.3, 3.3], color=BORDER, lw=1)
    # ---- carril 2: suelo resbaladizo
    sr = real_from_wheel(sw)
    ax.text(0.35, 2.95, "SUELO RESBALADIZO", color=AMBER, fontsize=13, fontweight="bold", va="center")
    ground(ax, G2, True)
    diff = sw - sr
    if diff > 0.05:
        robot(ax, X0 + sw * SC, G2, ang, ghost=True)
        ax.text(X0 + sw * SC, G2 + 1.32, "encoders", color=AMBER, fontsize=10, ha="center", alpha=min(1, diff * 3))
    xr = X0 + sr * SC
    robot(ax, xr, G2, ang)
    # barro que salta cuando la rueda patina
    slipping = ZONE[0] < sr < ZONE[1] and t < T_MOVE
    if slipping:
        for _ in range(3):
            particles.append([xr - 0.36 - R * 0.7, G2 + 0.08, -rng.uniform(0.8, 2.2), rng.uniform(1.0, 2.6), 1.0])
    dt = 1 / FPS
    for p in particles:
        p[0] += p[2] * dt; p[1] += p[3] * dt; p[3] -= 6 * dt; p[4] -= 1.6 * dt
    particles[:] = [p for p in particles if p[4] > 0 and p[1] > G2 - 0.05]
    if particles:
        P = np.array(particles)
        ax.scatter(P[:, 0], P[:, 1], s=9, color=AMBER, alpha=np.clip(P[:, 4], 0, 1) * 0.8, zorder=6, lw=0)
    ax.text(xr, G2 + 1.32, "cámara", color=TEAL, fontsize=10, ha="center", fontweight="bold")
    ax.add_patch(Polygon([[xr - 0.1, G2 + 1.24], [xr + 0.1, G2 + 1.24], [xr, G2 + 1.13]], fc=TEAL, ec="none"))
    ax.text(12.45, 2.95, f"ruedas: {fmt(sw)}", color=AMBER, fontsize=13, ha="right", va="center")
    ax.text(12.45, 2.5, f"real: {fmt(sr)}    cámara: {fmt(sr)}", color=TXT, fontsize=13, ha="right", va="center")
    if slipping:
        ax.text(0.35, 2.5, "¡las ruedas patinan!", color=AMBER, fontsize=12, ha="left", va="center", style="italic")
    if t >= T_MOVE:  # flecha final del error
        y, xg = G2 + 0.6, X0 + sw * SC
        ax.annotate("", xy=(xg - 0.68, y), xytext=(xr + 0.68, y), arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.6))
        ax.text((xr + xg) / 2, y + 0.1, f"error: {diff:.1f} m".replace(".", ","),
                color=AMBER, fontsize=12, ha="center", va="bottom", fontweight="bold")
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

fig = plt.figure(figsize=(6.4, 3.3), dpi=200); fig.patch.set_facecolor(CARD)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(CARD)
n = int((T_MOVE + T_HOLD) * FPS)
first = frame(0, fig, ax); h, w = first.shape[:2]
ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
mp4 = os.path.join(OUT, "robot_resbala.mp4")
proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                         "-preset", "slow", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
particles.clear()
for i in range(n):
    img = frame(i / FPS, fig, ax)
    proc.stdin.write(img.tobytes())
proc.stdin.close(); proc.wait()
plt.imsave(os.path.join(OUT, "robot_resbala_final.png"), img)
print("Listo:", mp4, f"{w}x{h}", n, "frames")
