from functools import lru_cache
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------
# 1. Layout / porosity (pure numpy -- no pyvista needed)
# ---------------------------------------------------------------------
def lattice_layout(w, s, block_size=20.0):
    n = max(2, int(round(block_size / s)) + 1)
    half_span = (n - 1) * s / 2.0
    positions = np.linspace(-half_span, half_span, n)
    footprint = (n - 1) * s + w
    return n, positions, footprint

def macro_porosity_finite_block(w, s, n):
    F = (n - 1) * s + w
    return 1.0 - n * w / F

def macro_porosity_periodic(w, s):
    return 1.0 - w / s

# ---------------------------------------------------------------------
# 2. Voxel model of one periodic woodpile cell  (s x s in plane, N layers)
# ---------------------------------------------------------------------
def _best_pitch_resolution(w, s, lo=12, hi=20):
    best, err = lo, 9e9
    for n in range(lo, hi + 1):
        e = abs(round(w / s * n) / n - w / s)
        if e < err - 1e-12:
            best, err = n, e
    return best

def woodpile_voxel_mask(w, s, h, n_layers, n_pitch=None, nz_per_layer=None):
    n_pitch = n_pitch or _best_pitch_resolution(w, s)
    nw = int(round(w / s * n_pitch))
    nw = max(1, min(nw, n_pitch))
    nzl = nz_per_layer or max(4, int(round(n_pitch * h / s)))
    nx = ny = n_pitch
    nz = nzl * n_layers
    mask = np.zeros((nx, ny, nz), dtype=bool)
    rod = np.zeros(n_pitch, dtype=bool)
    rod[:nw] = True
    for L in range(n_layers):
        sl = slice(L * nzl, (L + 1) * nzl)
        if L % 2 == 0:
            mask[:, rod, sl] = True
        else:
            mask[rod, :, sl] = True
    dx = s / n_pitch
    dz = h / nzl
    return mask, dx, dz

def voxel_porosity(mask):
    return 1.0 - mask.mean()

# ---------------------------------------------------------------------
# 3. Effective through-thickness transport (finite-volume homogenisation)
# ---------------------------------------------------------------------
def effective_conductivity_z(K, dx, dz):
    nx, ny, nz = K.shape
    dy = dx
    N = nx * ny * nz
    idx = np.arange(N).reshape(nx, ny, nz)
    rows, cols, vals = [], [], []
    diag = np.zeros(N)

    def pair(Ka, Kb, ia, ib, area, dist):
        G = area * 2.0 * Ka * Kb / (dist * (Ka + Kb))
        ia, ib, G = ia.ravel(), ib.ravel(), G.ravel()
        rows.extend([ia, ib]); cols.extend([ib, ia]); vals.extend([-G, -G])
        np.add.at(diag, ia, G); np.add.at(diag, ib, G)

    pair(K, np.roll(K, -1, axis=0), idx, np.roll(idx, -1, axis=0), dy * dz, dx)
    pair(K, np.roll(K, -1, axis=1), idx, np.roll(idx, -1, axis=1), dx * dz, dy)
    pair(K[:, :, :-1], K[:, :, 1:], idx[:, :, :-1], idx[:, :, 1:], dx * dy, dz)

    b = np.zeros(N)
    top = idx[:, :, -1].ravel(); bot = idx[:, :, 0].ravel()
    Gtop = K[:, :, -1].ravel() * dx * dy / (dz / 2.0)
    Gbot = K[:, :, 0].ravel() * dx * dy / (dz / 2.0)
    diag[top] += Gtop; b[top] += Gtop * 1.0
    diag[bot] += Gbot

    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    M = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N)).tocsc()
    phi = spla.spsolve(M, b)
    flux = np.sum(Gtop * (1.0 - phi[top]))
    thickness = nz * dz
    area = nx * dx * ny * dy
    return flux * thickness / area


@lru_cache(maxsize=512)
def _transport_cached(w_r, s_r, h_r, n_layers, eps_micro_r, b_micro_r):
    w, s, h = w_r, s_r, h_r
    mask, dx, dz = woodpile_voxel_mask(w, s, h, n_layers)
    D_filament = eps_micro_r ** b_micro_r
    K_ion = np.where(mask, D_filament, 1.0)
    f_ion = effective_conductivity_z(K_ion, dx, dz)
    VOID = 1e-9
    K_el = np.where(mask, 1.0, VOID)
    g_el = effective_conductivity_z(K_el, dx, dz)
    return f_ion, g_el, voxel_porosity(mask), mask.shape


def effective_transport_z(w, s, h, n_layers, eps_micro=0.51, b_micro=1.5):
    f_ion, g_el, por, shape = _transport_cached(round(w, 5), round(s, 5), round(h, 5),
                                                int(min(n_layers, 4)), round(eps_micro, 5), round(b_micro, 5))
    eps_macro_periodic = macro_porosity_periodic(w, s)
    eps_total_periodic = eps_macro_periodic + (1 - eps_macro_periodic) * eps_micro
    b_eff = float(np.log(f_ion) / np.log(eps_total_periodic))
    return {"f_ion": float(f_ion), "g_el": float(g_el),
            "porosity_voxel": float(por), "grid": shape,
            "eps_macro_periodic": float(eps_macro_periodic),
            "eps_total_periodic": float(eps_total_periodic),
            "b_eff": b_eff}


def bruggeman_equivalent_exponent(f_ion, eps_total):
    return float(np.log(f_ion) / np.log(eps_total))


# ---------------------------------------------------------------------
# 4. Ink recipe -> volume fractions
# ---------------------------------------------------------------------
def ink_hc_volume_share(mass_ratio=(88.0, 6.0, 6.0), densities=(1.95, 1.9, 1.6)):
    vols = [m / r for m, r in zip(mass_ratio, densities)]
    return vols[0] / sum(vols)


def combine_with_micro_porosity(eps_macro, eps_micro=0.51, hc_share=None, R_n=3.48e-6):
    if hc_share is None:
        hc_share = ink_hc_volume_share()
    eps_total = eps_macro + (1 - eps_macro) * eps_micro
    eps_am = (1 - eps_macro) * (1 - eps_micro) * hc_share
    eps_filler = (1 - eps_macro) * (1 - eps_micro) * (1 - hc_share)
    assert abs(eps_total + eps_am + eps_filler - 1.0) < 1e-9
    return {"eps_macro": eps_macro, "eps_total": eps_total, "eps_am_total": eps_am,
            "eps_filler_total": eps_filler, "a_n_total": 3 * eps_am / R_n,
            "hc_share": hc_share,
            "assumptions_used": {"eps_micro": eps_micro, "hc_share": hc_share, "R_n": R_n}}


# ---------------------------------------------------------------------
# 5. 3-D mesh + rendering (pyvista imported lazily so the rest runs without it)
# ---------------------------------------------------------------------
def build_woodpile_lattice(d, s, n_layers, block_size=20.0, layer_height=0.8):
    import pyvista as pv
    n, positions, F = lattice_layout(d, s, block_size)
    h = layer_height
    radius = d / 2.0
    rods = []
    for L in range(n_layers):
        zc = L * h + h / 2.0
        direction = (1, 0, 0) if L % 2 == 0 else (0, 1, 0)
        for off in positions:
            center = (0.0, off, zc) if L % 2 == 0 else (off, 0.0, zc)
            rods.append(pv.Cylinder(center=center, direction=direction,
                                    radius=radius, height=F, resolution=24))
    return rods, n, F


def render_lattice(rods, fname="lattice_render.png", title=None):
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=(1200, 1000))
    for r in rods:
        p.add_mesh(r, color="firebrick", smooth_shading=False)
    if title:
        p.add_text(title, font_size=10, color="black")
    p.set_background("white"); p.camera_position = "iso"
    p.screenshot(fname)
    print("saved", fname)


def render_lattice_interactive(rods):
    import pyvista as pv
    p = pv.Plotter(window_size=(1200, 1000))
    for r in rods:
        p.add_mesh(r, color="firebrick")
    p.set_background("white"); p.camera_position = "iso"
    p.show()


# ---------------------------------------------------------------------
# 6. Self-tests
# ---------------------------------------------------------------------
def _validate_homogenisation():
    nx = ny = 8
    K = np.full((nx, ny, 12), 0.37)
    assert abs(effective_conductivity_z(K, 1.0, 1.0) - 0.37) < 1e-9, "uniform"
    K = np.ones((nx, ny, 12)); K[:, :, :6] = 1.0; K[:, :, 6:] = 0.1
    exp = 2.0 / (1 / 1.0 + 1 / 0.1)
    assert abs(effective_conductivity_z(K, 1.0, 1.0) - exp) < 1e-9, "series"
    K = np.ones((nx, ny, 12)); K[:4, :, :] = 0.1
    exp = 0.5 * 1.0 + 0.5 * 0.1
    assert abs(effective_conductivity_z(K, 1.0, 1.0) - exp) < 1e-9, "parallel"
    print("homogenisation solver: uniform / series / parallel exact cases PASS")


if __name__ == "__main__":
    _validate_homogenisation()
    eps_micro = 0.51
    print(f"\nHC volume share of dry solids from your recipe: {ink_hc_volume_share():.3f}")
    print(f"monolithic (no channels) baseline: f_ion = eps_micro^1.5 = {eps_micro**1.5:.3f}\n")
    print("eps_macro(voxel) below is the PERIODIC-cell porosity f_ion/b_eff were actually solved on --")
    print("it is NOT a cross-check against eps_macro(finite); those are genuinely different numbers,")
    print("more so the fewer filaments (n) fit across the block. b_eff is applied to eps_total(finite).\n")
    print(" d(mm) s/d  n h(mm) N | eps_macro(finite) eps_macro(voxel=periodic) | eps_total(finite) | f_ion  Bruggeman-1.5  b_eff | g_el")
    for d in (1.6, 3.3):
        for ratio in (1.3, 1.6, 2.0):
            for h, N in ((0.8, 2), (0.8, 4), (1.0, 6)):
                s = d * ratio
                n, pos, F = lattice_layout(d, s)
                em = macro_porosity_finite_block(d, s, n)
                tr = effective_transport_z(d, s, h, N, eps_micro)
                pb = combine_with_micro_porosity(em, eps_micro)
                brug = pb["eps_total"] ** 1.5
                b_eff = tr["b_eff"]
                print(f" {d:4.1f} {ratio:4.1f} {n:3d} {h:4.1f} {N} |   {em:6.3f}          {tr['porosity_voxel']:6.3f}            "
                      f"|  {pb['eps_total']:6.3f}  | {tr['f_ion']:6.3f}   {brug:6.3f}      {b_eff:5.2f} | {tr['g_el']:6.3f}")
