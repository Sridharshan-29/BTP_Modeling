import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybamm
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# 0. Digitised curves & Calibrated 4-Knot Profiles (Train RMSE = 12.10 mV)
# ---------------------------------------------------------------------
XLSX_PATH = os.path.join(HERE, "HC_digitized_parameters_blue.xlsx")
ocp_df = pd.read_excel(XLSX_PATH, sheet_name="U_HC_OCP")
d1n_df = pd.read_excel(XLSX_PATH, sheet_name="D1n_diffusivity")
kn_df = pd.read_excel(XLSX_PATH, sheet_name="kn_rate_constant")
d2_df = pd.read_excel(XLSX_PATH, sheet_name="D2_electrolyte_diffusivity")
kappa_df = pd.read_excel(XLSX_PATH, sheet_name="kappa_electrolyte_conductivity")

OCP_C = ocp_df["c_mol_m3"].to_numpy()
OCP_U = np.minimum.accumulate(ocp_df["U_V"].to_numpy())
D1N_C = d1n_df["c_mol_m3"].to_numpy()
KN_C = kn_df["c_mol_m3"].to_numpy()

_D1N_D_raw = d1n_df["D1n_m2_s"].to_numpy()
_KN_K_raw = kn_df["kn_m2p5_mol1p5_s"].to_numpy()

c_knots = np.array([0.0, 3600.0, 8000.0, 13821.0])
s_D_opt = np.array([0.9347, 0.0930, 0.0209, 1.7649])
s_k_opt = np.array([0.9631, 0.4273, 0.0941, 0.0770])

interp_s_D = np.interp(D1N_C, c_knots, s_D_opt)
interp_s_k = np.interp(KN_C, c_knots, s_k_opt)

D1N_D = _D1N_D_raw * interp_s_D
KN_K = _KN_K_raw * interp_s_k

_d2_sorted = d2_df.sort_values("c2_mol_m3")
D2_C = _d2_sorted["c2_mol_m3"].to_numpy(); D2_D = _d2_sorted["D2_m2_s"].to_numpy()
_kap_sorted = kappa_df.sort_values("c2_mol_m3")
KAPPA_C = _kap_sorted["c2_mol_m3"].to_numpy(); KAPPA_K = _kap_sorted["kappa_S_m"].to_numpy()

# ---------------------------------------------------------------------
# 1. Physical Parameters
# ---------------------------------------------------------------------
F = 96485.0
rho_HC = 1950.0
R_n = 3.48e-6
c_max = 13821.0

alpha_ct = 0.5
t_plus = 0.45
c_e_init = 1000.0
delta_sep = 220e-6
BRUGG = 1.5

delta_coin = 64e-6
A_coin = 2.545e-4
eps_coin = 0.51
eps_filler_coin = 0.001

SIGMA_PARAM_PAPER = 256.0
V_CUTOFF = 0.0003
C0_FRACTION = 1e-5

R_CONTACT_HALF_CELL = 2e-3

def theoretical_capacity_Ah(thickness, area, eps_am):
    return c_max * F * thickness * area * eps_am / 3600.0

# ---------------------------------------------------------------------
# 2. Interpolants
# ---------------------------------------------------------------------
def _smooth_extrapolated_interp(x, xp, fp, name):
    c_clamped = pybamm.minimum(pybamm.maximum(x, float(xp.min())), float(xp.max()))
    base = pybamm.Interpolant(xp, fp, c_clamped, name=name, interpolator="linear")
    slope_hi = float((fp[-1] - fp[-2]) / (xp[-1] - xp[-2]))
    slope_lo = float((fp[1] - fp[0]) / (xp[1] - xp[0]))
    over = pybamm.maximum(x - float(xp.max()), 0)
    under = pybamm.maximum(float(xp.min()) - x, 0)
    return base + slope_hi * over - slope_lo * under

def hc_ocp(sto):
    c_s = sto * c_max
    return _smooth_extrapolated_interp(c_s, OCP_C, OCP_U, "U_HC(c_s)")

def hc_diffusivity(sto, T):
    c_s = sto * c_max
    return _smooth_extrapolated_interp(c_s, D1N_C, D1N_D, "D1n(c_s)")

def hc_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    k_n =  _smooth_extrapolated_interp(c_s_surf, KN_C, KN_K, "kn(c_s)")
    return (F * k_n * (c_e ** alpha_ct) * (c_s_surf ** alpha_ct)
            * ((c_s_max - c_s_surf) ** (1 - alpha_ct)))

def _make_electrolyte_functions():
    def electrolyte_diffusivity(c_e, T):
        return _smooth_extrapolated_interp(c_e, D2_C, D2_D, "D2(c_e)")
    def electrolyte_conductivity(c_e, T):
        return _smooth_extrapolated_interp(c_e, KAPPA_C, KAPPA_K, "kappa(c_e)")
    return electrolyte_diffusivity, electrolyte_conductivity

# ---------------------------------------------------------------------
# 3. Parameter set
# ---------------------------------------------------------------------
def build_parameter_values(porosity, thickness, area, eps_am,
                           particle_radius=R_n, b_electrolyte=BRUGG,
                           sigma_param=SIGMA_PARAM_PAPER, b_solid=1.5,
                           r_contact=R_CONTACT_HALF_CELL):
    pv = pybamm.ParameterValues("Xu2019")
    D_e, kap_e = _make_electrolyte_functions()
    side = float(np.sqrt(area))
    Q_nom = theoretical_capacity_Ah(thickness, area, eps_am)
    overrides = {
        "Electrode height [m]": side, "Electrode width [m]": side,
        "Positive electrode thickness [m]": thickness,
        "Separator thickness [m]": delta_sep,
        "Positive particle radius [m]": particle_radius,

        "Positive electrode porosity [1]": porosity,
        "Positive electrode porosity": porosity,
        "Positive electrode active material volume fraction [1]": eps_am,
        "Positive electrode active material volume fraction": eps_am,
        "Separator porosity [1]": 0.70, "Separator porosity": 0.70,

        "Positive electrode Bruggeman coefficient (electrolyte)": b_electrolyte,
        "Separator Bruggeman coefficient (electrolyte)": BRUGG,
        "Positive electrode Bruggeman coefficient (electrode)": b_solid,
        "Positive electrode conductivity [S.m-1]": sigma_param,

        "Positive particle diffusivity [m2.s-1]": hc_diffusivity,
        "Positive electrode diffusivity [m2.s-1]": hc_diffusivity,

        "Positive electrode exchange-current density [A.m-2]": hc_exchange_current_density,
        "Positive electrode OCP [V]": hc_ocp,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,

        "Maximum concentration in positive electrode [mol.m-3]": c_max,
        "Initial concentration in positive electrode [mol.m-3]": 1e-5 * c_max,

        "Initial concentration in electrolyte [mol.m-3]": c_e_init,
        "Cation transference number": t_plus,
        "Electrolyte diffusivity [m2.s-1]": D_e,
        "Electrolyte conductivity [S.m-1]": kap_e,

        "Contact resistance [Ohm]": r_contact / area,
        "Positive electrode charge transfer coefficient": alpha_ct,

        "Nominal cell capacity [A.h]": Q_nom,
        "Lower voltage cut-off [V]": V_CUTOFF,
        "Upper voltage cut-off [V]": 2.0,
    }
    pv.update(overrides, check_already_exists=False)
    return pv, overrides, [k for k in overrides if k not in pv.keys()]

def make_model(use_contact_resistance=False):
    opts = {"working electrode": "positive"}
    if use_contact_resistance:
        opts["contact resistance"] = "true"
    try:
        return pybamm.lithium_ion.DFN(opts), use_contact_resistance
    except Exception:
        return pybamm.lithium_ion.DFN({"working electrode": "positive"}), False

# ---------------------------------------------------------------------
# 4. Simulation Execution Helpers
# ---------------------------------------------------------------------
_trapz = getattr(np, "trapezoid", None) or np.trapz

def run_discharge(*, porosity, thickness, area, eps_am, current_A, max_hours,
                  particle_radius=R_n, b_electrolyte=BRUGG, sigma_param=SIGMA_PARAM_PAPER,
                  b_solid=1.5, r_contact=R_CONTACT_HALF_CELL, audit=False):
    model, cr_on = make_model(r_contact > 0)
    pv, overrides, new_keys = build_parameter_values(
        porosity, thickness, area, eps_am, particle_radius, b_electrolyte,
        sigma_param, b_solid, r_contact)

    step = f"Discharge at {current_A:.8f} A for {max_hours:.3f} hours or until {V_CUTOFF} V"
    sim = pybamm.Simulation(model, parameter_values=pv, experiment=pybamm.Experiment([step]))

    try:
        sol = sim.solve(solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6))
    except Exception:
        try:
            sol = sim.solve(solver=pybamm.CasadiSolver(mode="safe"))
        except Exception as e:
            return {"status": "solver_error", "error": str(e)}

    try:
        t_h = np.asarray(sol["Time [h]"].entries)
        V = np.asarray(sol["Voltage [V]"].entries)
        Q = np.asarray(sol["Discharge capacity [A.h]"].entries)
    except Exception:
        return {"status": "extract_error"}

    return {"status": "ok", "Q_Ah": float(Q[-1]), "V_avg": float(_trapz(V, t_h) / (t_h[-1] - t_h[0])),
            "t_h": float(t_h[-1] - t_h[0]), "V0": float(V[0]), "V_end": float(V[-1]),
            "hit_cutoff": bool(V[-1] <= V_CUTOFF + 0.003), "Q_arr": Q, "V_arr": V,
            "contact_R_active": cr_on}

def coin_cell_run(divisor=None, current_A=None, audit=False):
    eps_am = 1 - eps_coin - eps_filler_coin
    Qth = theoretical_capacity_Ah(delta_coin, A_coin, eps_am)
    i_A = Qth / divisor if current_A is None else current_A
    hours = 1.15 * Qth / i_A + 1.0
    r = run_discharge(porosity=eps_coin, thickness=delta_coin, area=A_coin, eps_am=eps_am,
                      current_A=i_A, max_hours=hours, r_contact=R_CONTACT_HALF_CELL, audit=audit)
    m_g = rho_HC * delta_coin * A_coin * eps_am * 1e3
    return r, m_g, Qth, i_A

def monotone_curve(q, v):
    order = np.argsort(q, kind="stable")
    q, v = np.asarray(q)[order], np.asarray(v)[order]
    keep = np.concatenate(([True], np.diff(q) > 1e-9))
    return q[keep], v[keep]

def pchip(q, v):
    q, v = monotone_curve(q, v)
    return PchipInterpolator(q, v, extrapolate=False)

def rmse_pchip(q_model, v_model, q_ref, v_ref):
    f_model, f_ref = pchip(q_model, v_model), pchip(q_ref, v_ref)
    lo = max(np.min(q_model), np.min(q_ref))
    hi = min(np.max(q_model), np.max(q_ref))
    qc = np.linspace(lo, hi, 1000)
    err = f_model(qc) - f_ref(qc)
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(err))

def plot_comparison(q_model, v_model, q_ref, v_ref, model_label, ref_label, title, fname, rmse):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(q_ref, v_ref, "k-", linewidth=1.6, label=ref_label)
    ax.plot(q_model, v_model, "C0", linewidth=1.4, label=model_label)
    ax.set_xlabel("Specific capacity [mAh/g]"); ax.set_ylabel("Voltage vs Na+/Na [V]")
    ax.set_title(f"{title}\nRMSE = {rmse*1000:.2f} mV", fontsize=10)
    ax.legend(fontsize=8); ax.set_xlim(0, 230)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)

def print_table(q_model, v_model, q_ref, v_ref, ref_name):
    fm, fr = pchip(q_model, v_model), pchip(q_ref, v_ref)
    print(f"  Q(mAh/g)   model   {ref_name}")
    for q in (5, 15, 30, 50, 80, 100, 120, 150, 180):
        print(f"  {q:7.0f}   {float(fm(q)):7.3f}  {float(fr(q)):7.3f}")
