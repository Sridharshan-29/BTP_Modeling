import os
import numpy as np
import pandas as pd
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lattice_geometry import lattice_layout

# ---------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------
WINNERS_CSV = os.path.join(HERE, "design_sweep_winners.csv")
BLOCK_SIZE_MM = 20.0
OPERATING_J = 0.2
CHECK_J = 0.135
LAYER_COLORS = ["#b22222", "#2f6fa5", "#2e8b57", "#c28f1e"]


def role_title(role):
    if role == "optimum":
        return "OPTIMUM"
    if role.startswith("pareto_alt_"):
        return "PARETO ALTERNATIVE " + role.split("_")[-1]
    return role.upper()


def load_winners(path=WINNERS_CSV):
    df = pd.read_csv(path)
    return [r for _, r in df.iterrows()]


def jtag(j):
    return f"{j:g}".replace(".", "p")


# ---------------------------------------------------------------------
# 2. Geometry: filaments of width d and layer height h (oval cross-section), as the printer lays them.
#    Every layer is ONE continuous square-wave serpentine; the next layer is rotated 90 degrees.
# ---------------------------------------------------------------------
def layer_segments(L, positions, F, d, h):
    rr = d / 2.0
    zc = h / 2.0 + L * h
    lo, hi = -F / 2 + rr, F / 2 - rr
    segs, joints = [], []
    n = len(positions)
    for k in range(n):
        if L % 2 == 0:
            segs.append(("y", positions[k], zc, lo, hi))
        else:
            segs.append(("x", positions[k], zc, lo, hi))
        if k < n - 1:
            end = hi if k % 2 == 0 else lo
            if L % 2 == 0:
                segs.append(("x", end, zc, positions[k], positions[k + 1]))
                joints.append((positions[k], end, zc))
                joints.append((positions[k + 1], end, zc))
            else:
                segs.append(("y", end, zc, positions[k], positions[k + 1]))
                joints.append((end, positions[k], zc))
                joints.append((end, positions[k + 1], zc))
    return segs, joints


def build(row):
    d, s, h = float(row["d_mm"]), float(row["s_mm"]), float(row["h_mm"])
    n_layers = int(row["N_layers"])
    n, positions, F = lattice_layout(d, s, block_size=BLOCK_SIZE_MM)
    segs, joints = [], []
    for L in range(n_layers):
        sg, jt = layer_segments(L, positions, F, d, h)
        segs += [(L, *x) for x in sg]
        joints += [(L, *x) for x in jt]
    return segs, joints, n, F, d / 2.0, h / 2.0, n_layers * h


def info_lines(row):
    d, s, h = float(row["d_mm"]), float(row["s_mm"]), float(row["h_mm"])
    return [
        f"{role_title(row['role'])}   [{row['label']}]",
        f"Nozzle (filament width) d = {d:.2f} mm, layer height h = {h:.2f} mm",
        f"Line spacing s = {s:.2f} mm   (s/d = {s/d:.2f}, open gap = {s-d:.2f} mm)",
        f"N = {int(row['N_layers'])} layers, thickness {float(row['thickness_mm']):.2f} mm",
        f"{int(row['n_filaments'])} lines per layer, footprint {float(row['footprint_mm']):.1f} x {float(row['footprint_mm']):.1f} mm",
        f"Total porosity {float(row['eps_total']):.1%}, theoretical {float(row['areal_capacity_mAh_cm2']):.1f} mAh/cm2",
        f"Delivered at {CHECK_J} mA/cm2: {float(row[f'deliv_{jtag(CHECK_J)}_mAh_cm2']):.2f} mAh/cm2",
        f"Delivered at {OPERATING_J} mA/cm2: {float(row[f'deliv_{jtag(OPERATING_J)}_mAh_cm2']):.2f} mAh/cm2 "
        f"(supports cathode <= {float(row['max_cathode_at_op_mAh_cm2']):.1f})",
        f"Usable {float(row['usable_Ah_L_at_op']):.1f} Ah/L at {OPERATING_J} mA/cm2, max current {float(row['j_max_mA_cm2']):.2f} mA/cm2",
    ]


# ---------------------------------------------------------------------
# 3. Matplotlib overview (works anywhere, no pyvista needed)
# ---------------------------------------------------------------------
def seg_surface(seg, rr, rz, n_theta=24):
    L, axis, fixed, zc, a0, a1 = seg
    th = np.linspace(0, 2 * np.pi, n_theta)
    T, TH = np.meshgrid(np.array([a0, a1]), th)
    c = fixed + rr * np.cos(TH)
    z = zc + rz * np.sin(TH)
    return (T, c, z) if axis == "x" else (c, T, z)


def sphere_surface(j, rr, rz, n=12):
    L, x, y, z = j
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    return (x + rr * np.outer(np.cos(u), np.sin(v)),
            y + rr * np.outer(np.sin(u), np.sin(v)),
            z + rz * np.outer(np.ones_like(u), np.cos(v)))


def overview_png(designs, fname=os.path.join(HERE, "lattice_three_designs.png")):
    fig = plt.figure(figsize=(6.2 * len(designs), 11))
    for k, row in enumerate(designs):
        segs, joints, n, F, rr, rz, total_h = build(row)
        ax = fig.add_subplot(3, len(designs), k + 1, projection="3d")
        for sg in segs:
            X, Y, Z = seg_surface(sg, rr, rz)
            ax.plot_surface(X, Y, Z, color=LAYER_COLORS[sg[0] % 4], linewidth=0, antialiased=True)
        for j in joints:
            X, Y, Z = sphere_surface(j, rr, rz)
            ax.plot_surface(X, Y, Z, color=LAYER_COLORS[j[0] % 4], linewidth=0)
        zs = max(total_h * 3.0, 3.0)
        ax.set_box_aspect((F, F, zs))
        ax.set_xlim(-F / 2, F / 2)
        ax.set_ylim(-F / 2, F / 2)
        ax.set_zlim(0, zs)
        ax.view_init(elev=35, azim=-55)
        ax.set_xlabel("x [mm]", fontsize=7)
        ax.set_ylabel("y [mm]", fontsize=7)
        ax.set_zlabel("z [mm]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{role_title(row['role'])}: {row['label']}", fontsize=10)

        ax2 = fig.add_subplot(3, len(designs), len(designs) + k + 1)
        for sg in segs:
            L, axis, fixed, zc, a0, a1 = sg
            if axis == "x":
                xy, w, hgt = (a0, fixed - rr), a1 - a0, 2 * rr
            else:
                xy, w, hgt = (fixed - rr, a0), 2 * rr, a1 - a0
            ax2.add_patch(Rectangle(xy, w, hgt, facecolor=LAYER_COLORS[L % 4], edgecolor="k",
                                    linewidth=0.2, alpha=0.5))
        ax2.set_xlim(-F / 2 - 0.5, F / 2 + 0.5)
        ax2.set_ylim(-F / 2 - 0.5, F / 2 + 0.5)
        ax2.set_aspect("equal")
        ax2.set_title("Top view (all layers overlaid)", fontsize=9)
        ax2.set_xlabel("x [mm]", fontsize=8)
        ax2.set_ylabel("y [mm]", fontsize=8)
        ax2.tick_params(labelsize=7)

        ax3 = fig.add_subplot(3, len(designs), 2 * len(designs) + k + 1)
        ax3.axis("off")
        ax3.text(0.0, 1.0, "\n".join(info_lines(row)), va="top", ha="left", fontsize=8.5, family="monospace")
    fig.suptitle("Optimised DIW hard-carbon anode: each layer one continuous square-bend serpentine, next layer rotated 90 deg (colour = layer)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=170)
    plt.close(fig)
    print("Saved:", fname)


# ---------------------------------------------------------------------
# 4. Pyvista: rotatable HTML, still PNG, auto-rotating GIF per design
# ---------------------------------------------------------------------
def pyvista_views(row):
    import pyvista as pv
    segs, joints, n, F, rr, rz, total_h = build(row)
    text = "\n".join(info_lines(row))
    name = row["role"]

    def scene(size):
        p = pv.Plotter(off_screen=True, window_size=size)
        for L, axis, fixed, zc, a0, a1 in segs:
            mid = (a0 + a1) / 2.0
            center = (mid, fixed, zc) if axis == "x" else (fixed, mid, zc)
            direction = (1, 0, 0) if axis == "x" else (0, 1, 0)
            m = pv.Cylinder(center=center, direction=direction, radius=rr, height=a1 - a0,
                            resolution=40, capping=True)
            m.points[:, 2] = zc + (m.points[:, 2] - zc) * (rz / rr)
            p.add_mesh(m, color=LAYER_COLORS[L % 4], smooth_shading=True)
        for L, x, y, z in joints:
            m = pv.Sphere(radius=rr, center=(x, y, z), theta_resolution=24, phi_resolution=24)
            m.points[:, 2] = z + (m.points[:, 2] - z) * (rz / rr)
            p.add_mesh(m, color=LAYER_COLORS[L % 4], smooth_shading=True)
        p.add_text(text, position="upper_left", font_size=8, color="black")
        p.set_background("white")
        p.camera_position = "iso"
        return p

    p = scene((1300, 1000))
    p.show_axes()
    p.screenshot(os.path.join(HERE, f"lattice_{name}.png"))
    p.close()
    print(f"Saved: lattice_{name}.png")

    p = scene((1100, 900))
    p.show_axes()
    p.export_html(os.path.join(HERE, f"lattice_{name}.html"))
    p.close()
    print(f"Saved: lattice_{name}.html  (open in any browser, drag to rotate)")


# ---------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    designs = load_winners()
    for r in designs:
        print(f"{r['role']}: {r['label']}  d={r['d_mm']} h={r['h_mm']} s={r['s_mm']:.2f} N={int(r['N_layers'])}")
    overview_png(designs)
    try:
        import pyvista
    except ImportError:
        print("pyvista not installed: skipping HTML/GIF (pip install pyvista imageio trame trame-vtk trame-vuetify)")
    else:
        for r in designs:
            try:
                pyvista_views(r)
            except Exception as e:
                print(f"pyvista output failed for {r['role']}: {e}")
