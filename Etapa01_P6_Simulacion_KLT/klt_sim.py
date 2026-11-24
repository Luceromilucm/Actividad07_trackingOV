"""
Simulación de odometría visual monocular con tracking KLT.

1. Renderiza un pasillo 3D texturizado (ray casting) visto por una cámara
   que sigue una trayectoria conocida (ground truth).
2. Sigue puntos entre frames con KLT (Lucas-Kanade piramidal de OpenCV).
3. Estima el movimiento con matriz esencial + recoverPose y lo acumula.
4. Compara contra el ground truth y contra odometría de ruedas con deslizamiento.

Salidas en ./salida/ : GIF de la demo y figuras para las diapositivas.
Uso:  python klt_sim.py
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salida")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# Paleta de las diapositivas
BG, CARD, BORDER = "#0E1621", "#172433", "#2C4560"
TEAL, AMBER, TXT, MUTED = "#3DD6C4", "#FFB627", "#F2F5F8", "#A3B3C4"
def bgr(h): h = h.lstrip("#"); return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

# ------------------------------------------------------------------ escena
W_IMG, H_IMG, F = 640, 400, 380.0
K = np.array([[F, 0, W_IMG / 2], [0, F, H_IMG / 2], [0, 0, 1]])
HALF_W, FLOOR_Y, CEIL_Y, END_Z = 2.0, 1.2, -1.6, 40.0   # y hacia abajo
PX_PER_M, TEX = 110, 1024                               # textura: 1024 px ~ 9.3 m

def make_texture(seed, tint):
    r = np.random.default_rng(seed)
    t = r.normal(0.45, 0.12, (TEX, TEX)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), 3)
    for _ in range(260):                     # manchas, bloques y líneas: textura con esquinas
        x, y = r.integers(0, TEX, 2); s = int(r.integers(6, 40)); v = float(r.uniform(0.05, 0.95))
        k = r.integers(0, 3)
        if k == 0: cv2.rectangle(t, (int(x), int(y)), (int(x + s), int(y + s * r.uniform(.5, 2))), v, -1)
        elif k == 1: cv2.circle(t, (int(x), int(y)), s // 2, v, -1)
        else: cv2.line(t, (int(x), int(y)), (int(x + r.integers(-80, 80)), int(y + r.integers(-80, 80))), v, 3)
    t = cv2.GaussianBlur(t, (0, 0), 1.0)
    t = np.clip(t, 0, 1)
    rgb = np.stack([t * c for c in tint], -1)
    return (rgb * 255).astype(np.uint8)

TEXTURES = {
    "floor": make_texture(1, (0.55, 0.75, 0.85)),
    "ceil": make_texture(2, (0.50, 0.55, 0.65)),
    "left": make_texture(3, (0.80, 0.85, 0.95)),
    "right": make_texture(4, (0.85, 0.82, 0.75)),
    "end": make_texture(5, (0.9, 0.9, 0.9)),
}

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

SS = 2  # supermuestreo para evitar aliasing (ayuda a KLT)
uu, vv = np.meshgrid((np.arange(W_IMG * SS) + 0.5) / SS, (np.arange(H_IMG * SS) + 0.5) / SS)
RAYS_C = np.stack([(uu - K[0, 2]) / F, (vv - K[1, 2]) / F, np.ones_like(uu)], -1)

def render(C, R_wc):
    d = RAYS_C @ R_wc.T
    big = np.full(d.shape[:2], np.inf)
    hit = np.full(d.shape[:2], -1)
    planes = [("left", 0, -HALF_W), ("right", 0, HALF_W), ("floor", 1, FLOOR_Y), ("ceil", 1, CEIL_Y), ("end", 2, END_Z)]
    ts = []
    for i, (_, ax, val) in enumerate(planes):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (val - C[ax]) / d[..., ax]
        t[~np.isfinite(t) | (t <= 1e-6)] = np.inf
        m = t < big; big[m] = t[m]; hit[m] = i; ts.append(t)
    P = C + d * big[..., None]
    img = np.zeros(d.shape[:2] + (3,), np.float32)
    for i, (name, ax, _) in enumerate(planes):
        m = hit == i
        if not m.any(): continue
        a, b = [(2, 1), (2, 1), (0, 2), (0, 2), (0, 1)][i]
        mx = np.mod(P[..., a] * PX_PER_M, TEX).astype(np.float32)
        my = np.mod(P[..., b] * PX_PER_M, TEX).astype(np.float32)
        s = cv2.remap(TEXTURES[name], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).astype(np.float32)
        img[m] = s[m]
    fog = np.exp(-np.clip(big, 0, 60) / 18.0)[..., None]  # profundidad: se oscurece a lo lejos
    img = img * (0.25 + 0.75 * fog)
    img = cv2.resize(img, (W_IMG, H_IMG), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 255).astype(np.uint8)

# ------------------------------------------------------- trayectoria real
N = 110
s = np.linspace(0, 1, N)
gt_z = 1.0 + 13.0 * s
gt_x = 0.9 * np.sin(2 * np.pi * s * 1.0)
dx, dz = np.gradient(gt_x), np.gradient(gt_z)
gt_yaw = np.arctan2(dx, dz)
gt_y = 0.03 * np.sin(6 * np.pi * s)  # pequeño balanceo vertical
GT_C = np.stack([gt_x, gt_y, gt_z], 1)
GT_R = [rot_y(a) for a in gt_yaw]

# ------------------------------------------------------- KLT + odometría
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FEAT = dict(maxCorners=350, qualityLevel=0.01, minDistance=9, blockSize=7)

frames, grays = [], []
print("Renderizando escena...")
for i in range(N):
    f = render(GT_C[i], GT_R[i]); frames.append(f); grays.append(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY))

est_C, est_R = [GT_C[0].copy()], [GT_R[0].copy()]
tracks = []           # lista de listas de puntos (para dibujar estelas)
pts = cv2.goodFeaturesToTrack(grays[0], **FEAT).reshape(-1, 2)
tracks = [[p] for p in pts]
viz_tracks, n_tracked = [], []
for k in range(1, N):
    p0 = np.float32([t[-1] for t in tracks]).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[k - 1], grays[k], p0, None, **LK)
    p0b, stb, _ = cv2.calcOpticalFlowPyrLK(grays[k], grays[k - 1], p1, None, **LK)
    fb = np.linalg.norm(p0 - p0b, axis=2).ravel()
    good = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < 0.8)
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    E, mask = cv2.findEssentialMat(a, b, K, cv2.RANSAC, 0.999, 0.8)
    _, R, t, mpose = cv2.recoverPose(E, a, b, K, mask=mask)
    scale = np.linalg.norm(GT_C[k] - GT_C[k - 1])   # monocular: la escala viene de fuera
    Rw = est_R[-1]
    est_C.append(est_C[-1] + scale * (Rw @ (-R.T @ t).ravel()))
    est_R.append(Rw @ R.T)
    tracks = [tr + [p] for tr, p, g in zip(tracks, p1.reshape(-1, 2), good) if g]
    n_tracked.append(len(tracks))
    if len(tracks) < 180:  # reponer puntos donde no hay
        m = np.full(grays[k].shape, 255, np.uint8)
        for tr in tracks: cv2.circle(m, tuple(int(v) for v in tr[-1]), 12, 0, -1)
        new = cv2.goodFeaturesToTrack(grays[k], mask=m, **FEAT)
        if new is not None: tracks += [[p] for p in new.reshape(-1, 2)]
    viz_tracks.append([np.array(tr[-12:]) for tr in tracks if len(tr) > 1])

EST_C = np.array(est_C)
dist = np.sum(np.linalg.norm(np.diff(GT_C, axis=0), axis=1))
err_final = np.linalg.norm(EST_C[-1] - GT_C[-1])
print(f"Recorrido {dist:.1f} m | error final VO {err_final:.2f} m ({100 * err_final / dist:.1f} %) | puntos/frame ~{np.mean(n_tracked):.0f}")

# --------------------------------------- odometría de ruedas con deslizamiento
step = np.linalg.norm(np.diff(GT_C[:, [0, 2]], axis=0), axis=1)
dyaw = np.diff(gt_yaw)
mid_z = 0.5 * (GT_C[1:, 2] + GT_C[:-1, 2])
SAND = (5.0, 9.5)
in_sand = (mid_z > SAND[0]) & (mid_z < SAND[1])
slip = np.where(in_sand, rng.uniform(0.35, 0.6, len(step)), rng.uniform(0.0, 0.04, len(step)))
enc_step = step * (1 + slip)                               # las ruedas giran de más
enc_dyaw = dyaw + np.where(in_sand, rng.normal(0.012, 0.01, len(step)), rng.normal(0, 0.002, len(step)))
enc = [GT_C[0, [0, 2]]]; yaw = gt_yaw[0]
for d_, dy in zip(enc_step, enc_dyaw):
    yaw += dy; enc.append(enc[-1] + d_ * np.array([np.sin(yaw), np.cos(yaw)]))
ENC = np.array(enc)
err_enc = np.linalg.norm(ENC[-1] - GT_C[-1, [0, 2]])
print(f"error final encoders {err_enc:.2f} m ({100 * err_enc / dist:.1f} %)")

# ------------------------------------------------------------------ figuras
plt.rcParams.update({"font.family": "Calibri", "font.size": 13, "text.color": TXT,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

def top_view(ax, upto=None, show_enc=False, legend=True, horiz=False):
    """Vista superior. horiz=True dibuja z en el eje horizontal (formato apaisado)."""
    n = len(GT_C) if upto is None else upto
    P = (lambda x, z: (z, x)) if horiz else (lambda x, z: (x, z))
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.plot(*P(GT_C[:, 0], GT_C[:, 2]), "--", color=MUTED, lw=1.6, label="Trayectoria real", alpha=0.9 if upto is None else 0.35)
    if upto is not None: ax.plot(*P(GT_C[:n, 0], GT_C[:n, 2]), "--", color=MUTED, lw=1.6)
    if show_enc:
        (ax.axvspan if horiz else ax.axhspan)(*SAND, color=AMBER, alpha=0.10, lw=0)
        ax.plot(*P(ENC[:, 0], ENC[:, 1]), color=AMBER, lw=2.4, label="Encoders de rueda")
    ax.plot(*P(EST_C[:n, 0], EST_C[:n, 2]), color=TEAL, lw=2.6, label="Odometría visual (KLT)")
    ax.plot(*P(EST_C[n - 1, 0], EST_C[n - 1, 2]), "o", color=TEAL, ms=8)
    ax.set_xlabel("z (m)" if horiz else "x (m)"); ax.set_ylabel("x (m)" if horiz else "z (m)")
    ax.set_aspect("equal"); ax.grid(color=BORDER, lw=0.6)
    if legend:
        lg = ax.legend(loc="upper left", frameon=True, fontsize=11)
        lg.get_frame().set_facecolor(BG); lg.get_frame().set_edgecolor(BORDER)

# 1) encoders vs VO (apaisado)
fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=220); fig.patch.set_facecolor(CARD)
top_view(ax, show_enc=True, legend=False, horiz=True)
ax.set_xlim(0, max(ENC[:, 1].max(), GT_C[:, 2].max()) + 0.6); ax.set_ylim(-2.2, max(2.2, ENC[:, 0].max() + 0.4))
ax.text(np.mean(SAND), ax.get_ylim()[1] - 0.25, "zona con deslizamiento", color=AMBER, fontsize=11, ha="center", va="top")
ax.text(ENC[-1, 1] - 0.15, ENC[-1, 0], f"error {err_enc:.1f} m", color=AMBER, fontsize=11, ha="right", va="center")
ax.text(EST_C[-1, 2] + 0.15, EST_C[-1, 0] - 0.55, f"error {err_final:.2f} m", color=TEAL, fontsize=11, ha="right", va="top")
lg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=11)
fig.savefig(f"{OUT}/encoders_vs_vo.png", facecolor=CARD, bbox_inches="tight", pad_inches=0.12); plt.close(fig)

# 2) frame con puntos KLT (tipos de tracking: puntos)
def draw_tracks(img, tr_list, thick=1):
    o = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    o = cv2.addWeighted(o, 0.75, np.zeros_like(o), 0.25, 0)
    for tr in tr_list:
        cv2.polylines(o, [tr.astype(np.int32).reshape(-1, 1, 2)], False, bgr(TEAL), thick, cv2.LINE_AA)
        cv2.circle(o, tuple(int(v) for v in tr[-1]), 2 + thick, bgr(AMBER), -1, cv2.LINE_AA)
    return cv2.cvtColor(o, cv2.COLOR_BGR2RGB)
KF = 45
Image.fromarray(draw_tracks(frames[KF], viz_tracks[KF - 1])).save(f"{OUT}/tipo_puntos.png")

# 3) tracking de objetos: dos objetos moviéndose, detección por color + asociación por cercanía
obj_frames = []
objs = [dict(p=np.array([90., 250.]), v=np.array([4.2, -0.6]), r=26, c=(240, 120, 90)),
        dict(p=np.array([520., 300.]), v=np.array([-3.5, -1.2]), r=20, c=(120, 200, 255))]
ids, hist, nxt = {}, {}, 1
base = frames[0]
for k in range(40):
    im = base.copy()
    for o in objs:
        q = (o["p"] + k * o["v"]).astype(int)
        cv2.circle(im, tuple(q), o["r"], o["c"], -1, cv2.LINE_AA)
    # detección: saturación alta
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    m = (hsv[..., 1] > 120).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    dets = [(cents[j], stats[j]) for j in range(1, n) if stats[j, 4] > 300]
    new_ids = {}
    for c, st in dets:  # asociación: ID del track más cercano del frame anterior
        best = min(ids.items(), key=lambda kv: np.linalg.norm(kv[1] - c), default=None)
        if best and np.linalg.norm(best[1] - c) < 30 and best[0] not in new_ids: tid = best[0]
        else: tid = nxt; nxt += 1
        new_ids[tid] = c; hist.setdefault(tid, []).append((c, st))
    ids = new_ids
o = cv2.cvtColor(im, cv2.COLOR_RGB2BGR); o = cv2.addWeighted(o, 0.75, np.zeros_like(o), 0.25, 0)
for tid, hl in hist.items():
    pts_ = np.array([h[0] for h in hl]).astype(np.int32)
    cv2.polylines(o, [pts_.reshape(-1, 1, 2)], False, bgr(TEAL), 2, cv2.LINE_AA)
    x, y, w, h, _ = hl[-1][1]
    cv2.rectangle(o, (x - 6, y - 6), (x + w + 6, y + h + 6), bgr(AMBER), 2, cv2.LINE_AA)
    cv2.rectangle(o, (x - 7, y - 30), (x + 52, y - 6), bgr(AMBER), -1)
    cv2.putText(o, f"ID {tid}", (x - 2, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr(BG), 2, cv2.LINE_AA)
Image.fromarray(cv2.cvtColor(o, cv2.COLOR_BGR2RGB)).save(f"{OUT}/tipo_objetos.png")

# 4) pose de la cámara: vista superior con "frustums"
fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200); fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
ax.plot(EST_C[:, 2], EST_C[:, 0], color=MUTED, lw=1, alpha=0.6)
for i in range(0, N, 11):
    Cc, Rr = EST_C[i], est_R[i]
    fw, rt = Rr @ np.array([0, 0, 1.0]), Rr @ np.array([1.0, 0, 0])
    tip = Cc + 0.9 * fw
    a_, b_ = tip + 0.45 * rt, tip - 0.45 * rt
    ax.fill([Cc[2], a_[2], b_[2]], [Cc[0], a_[0], b_[0]], color=TEAL, alpha=0.25, lw=0)
    ax.plot([a_[2], Cc[2], b_[2]], [a_[0], Cc[0], b_[0]], color=TEAL, lw=1.6)
    ax.plot(Cc[2], Cc[0], "o", color=AMBER, ms=4)
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(0.2, 15.2); ax.set_ylim(-1.5, 1.5)
fig.tight_layout(pad=0.2); fig.savefig(f"{OUT}/tipo_pose.png", facecolor=CARD); plt.close(fig)

# 5) GIF: vista de cámara con estelas KLT + trayectoria que se va dibujando
print("Generando GIF...")
gif = []
xs = np.concatenate([GT_C[:, 0], EST_C[:, 0]]); zs = np.concatenate([GT_C[:, 2], EST_C[:, 2]])
for k in range(1, N, 1):
    left = draw_tracks(frames[k], viz_tracks[k - 1], 1)
    fig, ax = plt.subplots(figsize=(3.6, 4.0), dpi=100); fig.patch.set_facecolor(CARD)
    top_view(ax, upto=k + 1, legend=False)
    ax.set_xlim(xs.min() - 1.2, xs.max() + 1.2); ax.set_ylim(0, zs.max() + 0.6)
    ax.set_title(f"frame {k:3d}   puntos: {len(viz_tracks[k-1])}", color=MUTED, fontsize=11)
    fig.tight_layout(pad=0.4); fig.canvas.draw()
    right = np.asarray(fig.canvas.buffer_rgba())[..., :3]; plt.close(fig)
    right = cv2.resize(right, (int(right.shape[1] * H_IMG / right.shape[0]), H_IMG), interpolation=cv2.INTER_AREA)
    gif.append(Image.fromarray(np.hstack([left, np.full((H_IMG, 8, 3), bgr(BG)[::-1], np.uint8), right])))
gif[0].save(f"{OUT}/demo_klt.gif", save_all=True, append_images=gif[1:], duration=90, loop=0, optimize=True)
gif[-1].save(f"{OUT}/demo_klt_final.png")
np.savez(f"{OUT}/resultados.npz", gt=GT_C, est=EST_C, enc=ENC, err_vo=err_final, err_enc=err_enc, dist=dist)
print("Listo:", OUT)
