import os
import time
import numpy as np
import pandas as pd
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import half_cell as hc
from lattice_geometry import lattice_layout, combine_with_micro_porosity, effective_transport_z
from half_cell import run_discharge, theoretical_capacity_Ah

# ---------------------------------------------------------------------
# 1. Printer limits (Eazao Potter X-1) and design grid
# ---------------------------------------------------------------------
NOZZLE_VALUES_MM = [1.6, 2.0, 2.5, 3.3]
LAYER_HEIGHT_VALUES_MM = [0.4, 0.6, 0.8, 1.0]
S_OVER_D_RATIOS = [1.5, 1.8, 2.2, 2.6, 3.0]
N_LAYERS_VALUES = [1, 2, 3, 4]
THICKNESS_MIN_MM = 0.4
THICKNESS_MAX_MM = 1.8
BLOCK_SIZE_MM = 20.0
MIN_FILAMENTS = 7
ELLIPSE_FILL = np.pi / 4.0

DESIGN_GRID = [
    {"d_mm": d, "s_mm": d * ratio, "ratio": ratio, "N": N, "h_mm": h,
     "label": f"d{d}_h{h}_r{ratio}_N{N}"}
    for d in NOZZLE_VALUES_MM
    for h in LAYER_HEIGHT_VALUES_MM
    for ratio in S_OVER_D_RATIOS
    for N in N_LAYERS_VALUES if THICKNESS_MIN_MM <= N * h <= THICKNESS_MAX_MM
]

# ---------------------------------------------------------------------
# 2. Application: solar-charged buffer / remote sensor node. Operating point and cathode
# ---------------------------------------------------------------------
CATHODE_MAH_CM2 = 3.5
MAX_N_OVER_P = 2.0
MIN_DELIVERED_OVER_CATHODE = 1.1
OPERATING_J = 0.2
CHECK_J = 0.135
PLATING_SAFE_CUTOFF_V = 0.02
CURRENT_GRID_MA_CM2 = [0.135, 0.2, 0.3, 0.5, 0.75, 1.0]
R_CONTACT_BASE = 2e-3
R_CONTACT_BAND = [0.0, 2e-3, 4.32e-3]
V_CATHODE_ASSUMED = 3.4
MAX_PARETO_SHOWN = 3

# ---------------------------------------------------------------------
# 3. Geometry and single-design evaluation
# ---------------------------------------------------------------------
def jtag(j):
    return f"{j:g}".replace(".", "p")


def geometry_of(design):
    d, s, h, N = design["d_mm"], design["s_mm"], design["h_mm"], design["N"]
    n, positions, F = lattice_layout(d, s, block_size=BLOCK_SIZE_MM)
    area_frac = (n * F * d + (n - 1) * (s - d) * d) / F ** 2
    solid_vol_frac = min(ELLIPSE_FILL * area_frac, 0.95)
    eps_macro = 1.0 - solid_vol_frac
    pb = combine_with_micro_porosity(eps_macro)
    tr = effective_transport_z(d, s, h, N)
    area_m2 = (F * 1e-3) ** 2
    thickness_m = N * h * 1e-3
    eps_am = pb["eps_am_total"]
    Qth_Ah = theoretical_capacity_Ah(thickness_m, area_m2, eps_am)
    areal = Qth_Ah * 1e3 / (area_m2 * 1e4)
    sigma_param = 256.0 * tr["g_el"] / max(eps_am, 1e-6) ** 1.5
    row = {
        "label": design["label"], "d_mm": d, "s_mm": s, "ratio": design["ratio"], "h_mm": h,
        "N_layers": N, "thickness_mm": N * h, "footprint_mm": F, "n_filaments": n,
        "eps_macro": eps_macro, "eps_total": pb["eps_total"], "eps_am": eps_am,
        "b_eff": tr["b_eff"], "g_el": tr["g_el"], "Qth_Ah": Qth_Ah,
        "areal_capacity_mAh_cm2": areal,
    }
    common = dict(porosity=pb["eps_total"], thickness=thickness_m, area=area_m2,
                  eps_am=eps_am, b_electrolyte=tr["b_eff"], sigma_param=sigma_param)
    return row, common, area_m2, areal


def run_at(common, area_m2, areal, j_mA_cm2, r_contact):
    current_A = j_mA_cm2 * 1e-3 * area_m2 * 1e4
    max_hours = 1.3 * areal / j_mA_cm2 + 1.0
    r = run_discharge(**common, current_A=current_A, max_hours=max_hours, r_contact=r_contact)
    if r["status"] != "ok":
        return np.nan, np.nan, r["status"]
    return r["Q_Ah"] * 1e3 / (area_m2 * 1e4), r["V_avg"], "ok"


def max_feasible_current(js, delivered, threshold):
    ok = [(j, q) for j, q in zip(js, delivered) if not np.isnan(q)]
    if not ok or ok[0][1] < threshold:
        return np.nan
    for (j0, q0), (j1, q1) in zip(ok[:-1], ok[1:]):
        if q1 < threshold:
            return j0 + (j1 - j0) * (q0 - threshold) / (q0 - q1)
    return ok[-1][0]


def evaluate_design(design):
    hc.V_CUTOFF = PLATING_SAFE_CUTOFF_V
    row, common, area_m2, areal = geometry_of(design)
    row["eligible"], row["skip_reason"] = True, ""
    if row["n_filaments"] < MIN_FILAMENTS:
        row["eligible"], row["skip_reason"] = False, f"fewer than {MIN_FILAMENTS} filaments"
    elif areal < MIN_DELIVERED_OVER_CATHODE * CATHODE_MAH_CM2:
        row["eligible"], row["skip_reason"] = False, "theoretical capacity below 1.1x cathode"
    elif areal > MAX_N_OVER_P * CATHODE_MAH_CM2:
        row["eligible"], row["skip_reason"] = False, f"N/P above {MAX_N_OVER_P} (oversized anode)"
    for j in CURRENT_GRID_MA_CM2:
        row[f"deliv_{jtag(j)}_mAh_cm2"] = np.nan
        row[f"Vavg_{jtag(j)}_V"] = np.nan
    if not row["eligible"]:
        return row
    for j in CURRENT_GRID_MA_CM2:
        q, v, status = run_at(common, area_m2, areal, j, R_CONTACT_BASE)
        row[f"deliv_{jtag(j)}_mAh_cm2"] = q
        row[f"Vavg_{jtag(j)}_V"] = v
        if status != "ok":
            print(f"   [WARN] {design['label']} at {j} mA/cm2: {status}")
    return row


# ---------------------------------------------------------------------
# 4. Objectives: (1) usable volumetric capacity at the operating current, (2) highest current with >= 1.1x cathode
# ---------------------------------------------------------------------
def score_designs(df):
    df = df.copy()
    thick_cm = df["thickness_mm"] * 0.1
    q_op = df[f"deliv_{jtag(OPERATING_J)}_mAh_cm2"]
    q_ck = df[f"deliv_{jtag(CHECK_J)}_mAh_cm2"]
    df["usable_Ah_L_at_op"] = q_op / thick_cm
    df["usable_Ah_L_at_check"] = q_ck / thick_cm
    df["retention_at_op_pct"] = 100 * q_op / df["areal_capacity_mAh_cm2"]
    df["max_cathode_at_op_mAh_cm2"] = q_op / MIN_DELIVERED_OVER_CATHODE
    df["max_cathode_at_check_mAh_cm2"] = q_ck / MIN_DELIVERED_OVER_CATHODE
    df["N_over_P"] = df["areal_capacity_mAh_cm2"] / CATHODE_MAH_CM2
    thr = MIN_DELIVERED_OVER_CATHODE * CATHODE_MAH_CM2
    jmax = []
    for _, r in df.iterrows():
        q = [r[f"deliv_{jtag(j)}_mAh_cm2"] for j in CURRENT_GRID_MA_CM2]
        jmax.append(max_feasible_current(CURRENT_GRID_MA_CM2, q, thr))
    df["j_max_mA_cm2"] = jmax
    df["feasible"] = (q_op >= thr) & df["eligible"] & (df["N_over_P"] <= MAX_N_OVER_P)
    df["fullcell_Wh_L_est"] = df["usable_Ah_L_at_op"] * (V_CATHODE_ASSUMED - df[f"Vavg_{jtag(OPERATING_J)}_V"])
    return df


def pareto_front(df):
    pool = df[df["feasible"]]
    keep = []
    for i, r in pool.iterrows():
        dominated = ((pool["usable_Ah_L_at_op"] >= r["usable_Ah_L_at_op"]) &
                     (pool["j_max_mA_cm2"] >= r["j_max_mA_cm2"]) &
                     ((pool["usable_Ah_L_at_op"] > r["usable_Ah_L_at_op"]) |
                      (pool["j_max_mA_cm2"] > r["j_max_mA_cm2"]))).any()
        if not dominated:
            keep.append(i)
    return pool.loc[keep].sort_values(["usable_Ah_L_at_op", "j_max_mA_cm2"], ascending=False)


# ---------------------------------------------------------------------
# 5. Contact-resistance band
# ---------------------------------------------------------------------
def contact_band(winners):
    hc.V_CUTOFF = PLATING_SAFE_CUTOFF_V
    by_label = {d["label"]: d for d in DESIGN_GRID}
    rows = []
    for role, w in winners.items():
        _, common, area_m2, areal = geometry_of(by_label[w["label"]])
        for j in (CHECK_J, OPERATING_J):
            for r_c in R_CONTACT_BAND:
                q, v, status = run_at(common, area_m2, areal, j, r_c)
                rows.append({"role": role, "label": w["label"], "j_mA_cm2": j, "R_contact_Ohm_m2": r_c,
                             "delivered_mAh_cm2": q,
                             "retention_pct": 100 * q / areal if status == "ok" else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 6. Run everything
# ---------------------------------------------------------------------
if __name__ == "__main__":
    hc.V_CUTOFF = PLATING_SAFE_CUTOFF_V
    print("Application: solar-charged buffer / remote sensor node, low-power slow discharge, thick high-areal-capacity cell")
    print(f"Cathode {CATHODE_MAH_CM2} mAh/cm2 (N/P <= {MAX_N_OVER_P}) | operating {OPERATING_J} mA/cm2 (check {CHECK_J}) | "
          f"cutoff {PLATING_SAFE_CUTOFF_V*1e3:.0f} mV | R_contact {R_CONTACT_BASE} Ohm m2")
    print(f"Printer: nozzle {NOZZLE_VALUES_MM} mm, layer height {LAYER_HEIGHT_VALUES_MM} mm, "
          f"thickness {THICKNESS_MIN_MM}-{THICKNESS_MAX_MM} mm")
    print(f"{len(DESIGN_GRID)} grid points\n")
    rows = []
    t_start = time.time()
    for k, design in enumerate(DESIGN_GRID, 1):
        t0 = time.time()
        row = evaluate_design(design)
        rows.append(row)
        note = row["skip_reason"] if not row["eligible"] else f"delivered@{OPERATING_J}={row[f'deliv_{jtag(OPERATING_J)}_mAh_cm2']:.2f}"
        print(f"[{k}/{len(DESIGN_GRID)}] {design['label']}  areal={row['areal_capacity_mAh_cm2']:.2f}  {note}  "
              f"({time.time()-t0:.0f}s, total {(time.time()-t_start)/60:.1f} min)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "design_sweep_results_partial.csv"), index=False)

    scored = score_designs(pd.DataFrame(rows))
    scored.to_csv(os.path.join(HERE, "design_sweep_results.csv"), index=False)
    print("\nSaved: design_sweep_results.csv\n")

    cols = ["label", "n_filaments", "N_layers", "thickness_mm", "eps_total", "areal_capacity_mAh_cm2",
            f"deliv_{jtag(CHECK_J)}_mAh_cm2", f"deliv_{jtag(OPERATING_J)}_mAh_cm2", "retention_at_op_pct",
            "max_cathode_at_op_mAh_cm2", "usable_Ah_L_at_op", "j_max_mA_cm2", "feasible"]
    print(scored[scored["eligible"]][cols].round(2).to_string(index=False))
    print("\nSkipped:")
    print(scored[~scored["eligible"]][["label", "skip_reason"]].to_string(index=False))

    front = pareto_front(scored)
    if len(front) == 0:
        print(f"\nNo design delivers {MIN_DELIVERED_OVER_CATHODE}x the {CATHODE_MAH_CM2} mAh/cm2 cathode at {OPERATING_J} mA/cm2.")
        print("Largest cathode each design supports is in column max_cathode_at_op_mAh_cm2.")
        raise SystemExit

    front = front.head(MAX_PARETO_SHOWN)
    winners = {("optimum" if i == 0 else f"pareto_alt_{i}"): w for i, (_, w) in enumerate(front.iterrows())}
    print("\n=== OPTIMUM AND PARETO ALTERNATIVES (objective 1: usable Ah/L at "
          f"{OPERATING_J} mA/cm2; objective 2: highest current with >= {MIN_DELIVERED_OVER_CATHODE}x cathode) ===")
    for role, w in winners.items():
        print(f"\n{role}: {w['label']}  (d={w['d_mm']} mm, h={w['h_mm']} mm, s={w['s_mm']:.2f} mm, N={int(w['N_layers'])}, "
              f"thickness={w['thickness_mm']:.2f} mm, {int(w['n_filaments'])} lines, eps_total={w['eps_total']:.3f})")
        print(f"  theoretical areal capacity {w['areal_capacity_mAh_cm2']:.2f} mAh/cm2, N/P vs {CATHODE_MAH_CM2} cathode = {w['N_over_P']:.1f}")
        for j in (CHECK_J, OPERATING_J):
            q = w[f"deliv_{jtag(j)}_mAh_cm2"]
            print(f"  {j} mA/cm2: delivered {q:.2f} mAh/cm2 ({100*q/w['areal_capacity_mAh_cm2']:.0f}%), "
                  f"supports a cathode up to {q/MIN_DELIVERED_OVER_CATHODE:.2f} mAh/cm2")
        print(f"  usable volumetric capacity {w['usable_Ah_L_at_op']:.1f} Ah/L at {OPERATING_J} mA/cm2; "
              f"highest current with >= {MIN_DELIVERED_OVER_CATHODE}x cathode: {w['j_max_mA_cm2']:.2f} mA/cm2")
        print(f"  full-cell energy estimate (cathode {V_CATHODE_ASSUMED} V assumed): {w['fullcell_Wh_L_est']:.0f} Wh/L")

    wdf = pd.DataFrame([{"role": role, **w.to_dict()} for role, w in winners.items()])
    wdf.to_csv(os.path.join(HERE, "design_sweep_winners.csv"), index=False)
    print("\nSaved: design_sweep_winners.csv")

    band = contact_band(winners)
    band.to_csv(os.path.join(HERE, "design_sweep_rcontact_band.csv"), index=False)
    print("\n=== Contact-resistance band (retention % of theoretical) ===")
    print(band.pivot_table(index=["role", "label", "j_mA_cm2"], columns="R_contact_Ohm_m2",
                           values="retention_pct").round(0).to_string())
    print("\nSaved: design_sweep_rcontact_band.csv")

    ok = scored[scored["eligible"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for _, r in ok.iterrows():
        ax.plot(CURRENT_GRID_MA_CM2, [r[f"deliv_{jtag(j)}_mAh_cm2"] for j in CURRENT_GRID_MA_CM2], color="0.82", lw=0.8)
    for (role, w), col in zip(winners.items(), ["tab:red", "tab:blue", "tab:green"]):
        ax.plot(CURRENT_GRID_MA_CM2, [w[f"deliv_{jtag(j)}_mAh_cm2"] for j in CURRENT_GRID_MA_CM2], "o-", color=col,
                label=f"{role}: {w['label']}")
    ax.axhline(MIN_DELIVERED_OVER_CATHODE * CATHODE_MAH_CM2, color="k", ls="--",
               label=f"{MIN_DELIVERED_OVER_CATHODE}x cathode ({CATHODE_MAH_CM2} mAh/cm2)")
    ax.axvline(OPERATING_J, color="0.5", ls=":")
    ax.set_xlabel("Current density [mA/cm2]")
    ax.set_ylabel(f"Delivered above {PLATING_SAFE_CUTOFF_V*1e3:.0f} mV [mAh/cm2]")
    ax.set_title("Delivered areal capacity vs current")
    ax.legend(fontsize=7)
    ax = axes[1]
    sc = ax.scatter(ok["usable_Ah_L_at_op"], ok["j_max_mA_cm2"].fillna(0), c=ok["areal_capacity_mAh_cm2"], cmap="viridis", s=55)
    for _, r in ok.iterrows():
        ax.annotate(r["label"], (r["usable_Ah_L_at_op"], 0 if np.isnan(r["j_max_mA_cm2"]) else r["j_max_mA_cm2"]),
                    fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="Theoretical areal capacity [mAh/cm2]")
    ax.set_xlabel(f"Usable volumetric capacity at {OPERATING_J} mA/cm2 [Ah/L]")
    ax.set_ylabel(f"Highest current with >= {MIN_DELIVERED_OVER_CATHODE}x cathode [mA/cm2]")
    ax.set_title("Two objectives (top right is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "design_sweep_tradeoff.png"), dpi=200)
    print("Saved: design_sweep_tradeoff.png")
