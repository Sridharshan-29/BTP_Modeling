import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.interpolate import interp1d
import pybamm
import Half_Cell_Base as hc

HERE = os.path.dirname(os.path.abspath(__file__))

CRATES_XLSX_PATH = os.path.join(HERE, "CRATES.xlsx")

TRAIN_CONFIG = {
    "C30": {"sheet": "C30", "current_A": 0.100e-3, "weight": 1.0},
    "C17": {"sheet": "C17", "current_A": 0.1765e-3, "weight": 1.0},
}
EMF_CURRENT_A = 0.00295e-3

C_KNOTS = np.array([0.0, 3600.0, 8000.0, float(hc.c_max)])

def load_training_data():
    exp_data = {}
    xls = pd.ExcelFile(CRATES_XLSX_PATH)
    for key, cfg in TRAIN_CONFIG.items():
        if cfg["sheet"] in xls.sheet_names:
            df = pd.read_excel(CRATES_XLSX_PATH, sheet_name=cfg["sheet"], usecols=["Q_mAh_g", "U_V"]).dropna()
            q_exp, u_exp = hc.monotone_curve(df["Q_mAh_g"].to_numpy(), df["U_V"].to_numpy())
            exp_data[key] = (q_exp, u_exp)
            print(f"Loaded Training Sheet '{cfg['sheet']}': {len(q_exp)} points (Max Q = {q_exp.max():.2f} mAh/g)")

    q_emf = hc.ocp_df["Q_mAh_g"].to_numpy()
    u_emf = hc.OCP_U
    exp_data["EMF"] = hc.monotone_curve(q_emf, u_emf)
    print(f"Loaded Reference EMF curve directly from ocp_df: {len(q_emf)} points")
    return exp_data

TRAIN_DATA = load_training_data()

def create_knot_functions(s_D_vec, s_k_vec):
    interp_s_D = interp1d(C_KNOTS, s_D_vec, kind='linear', fill_value='extrapolate')
    interp_s_k = interp1d(C_KNOTS, s_k_vec, kind='linear', fill_value='extrapolate')

    d1n_mod = hc.D1N_D * np.clip(interp_s_D(hc.D1N_C), 0.01, 10.0)
    kn_mod = hc.KN_K * np.clip(interp_s_k(hc.KN_C), 0.01, 10.0)

    def hc_diffusivity_opt(sto, T):
        return hc._smooth_extrapolated_interp(sto * hc.c_max, hc.D1N_C, d1n_mod, "D1n_opt(c_s)")

    def hc_exchange_current_density_opt(c_e, c_s_surf, c_s_max, T):
        k_n = hc._smooth_extrapolated_interp(c_s_surf, hc.KN_C, kn_mod, "kn_opt(c_s)")
        return (hc.F * k_n * (c_e ** hc.alpha_ct) * (c_s_surf ** hc.alpha_ct)
                * ((c_s_max - c_s_surf) ** (1 - hc.alpha_ct)))

    return hc_diffusivity_opt, hc_exchange_current_density_opt

def run_discharge_safe(current_A, D_func, K_func):
    eps_am = 1 - hc.eps_coin - hc.eps_filler_coin
    Qth = hc.theoretical_capacity_Ah(hc.delta_coin, hc.A_coin, eps_am)
    hours = 1.25 * Qth / current_A + 1.0

    model, cr_on = hc.make_model(hc.R_CONTACT_PAPER > 0)
    pv, overrides, new_keys = hc.build_parameter_values(
        porosity=hc.eps_coin,
        thickness=hc.delta_coin,
        area=hc.A_coin,
        eps_am=eps_am,
        particle_radius=hc.R_n,
        b_electrolyte=hc.BRUGG,
        sigma_param=hc.SIGMA_PARAM_PAPER,
        b_solid=1.5,
        r_contact=hc.R_CONTACT_PAPER
    )

    pv.update({
        "Positive particle diffusivity [m2.s-1]": D_func,
        "Positive electrode diffusivity [m2.s-1]": D_func,
        "Positive electrode exchange-current density [A.m-2]": K_func,
    }, check_already_exists=False)

    step = f"Discharge at {current_A:.8f} A for {hours:.3f} hours or until {hc.V_CUTOFF} V"
    sim = pybamm.Simulation(model, parameter_values=pv, experiment=pybamm.Experiment([step]))

    try:
        sol = sim.solve(solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6))
    except Exception:
        try:
            sol = sim.solve(solver=pybamm.CasadiSolver(mode="safe"))
        except Exception:
            return None

    try:
        t_h = np.asarray(sol["Time [h]"].entries)
        V = np.asarray(sol["Voltage [V]"].entries)
        Q = np.asarray(sol["Discharge capacity [A.h]"].entries)
        m_g = hc.rho_HC * hc.delta_coin * hc.A_coin * eps_am * 1e3
        q_mod = Q * 1e3 / m_g
        return hc.monotone_curve(q_mod, V)
    except Exception:
        return None

ITERATION_COUNTER = 0

def objective_4knot(x):
    global ITERATION_COUNTER
    ITERATION_COUNTER += 1

    s_D_vec = x[0:4]
    s_k_vec = x[4:8]

    D_func, K_func = create_knot_functions(s_D_vec, s_k_vec)

    sim_c30 = run_discharge_safe(TRAIN_CONFIG["C30"]["current_A"], D_func, K_func)
    if sim_c30 is None: return 1e6
    rmse_c30, _ = hc.rmse_pchip(sim_c30[0], sim_c30[1], TRAIN_DATA["C30"][0], TRAIN_DATA["C30"][1])

    sim_c17 = run_discharge_safe(TRAIN_CONFIG["C17"]["current_A"], D_func, K_func)
    if sim_c17 is None: return 1e6
    rmse_c17, _ = hc.rmse_pchip(sim_c17[0], sim_c17[1], TRAIN_DATA["C17"][0], TRAIN_DATA["C17"][1])

    sim_emf = run_discharge_safe(EMF_CURRENT_A, D_func, K_func)
    if sim_emf is None: return 1e6
    rmse_emf, _ = hc.rmse_pchip(sim_emf[0], sim_emf[1], TRAIN_DATA["EMF"][0], TRAIN_DATA["EMF"][1])

    total_loss = (rmse_c30 + rmse_c17 + 0.5 * rmse_emf) / 2.5

    if ITERATION_COUNTER % 5 == 0:
        print(f"Iter {ITERATION_COUNTER:3d} | Train RMSE: {total_loss*1000:.2f} mV | s_D={np.round(s_D_vec, 3)} | s_k={np.round(s_k_vec, 3)}")

    return total_loss

def run_optimization():
    print("\n" + "=" * 75)
    print("      4-KNOT PHYSICAL JOINT OPTIMIZER (C30 + C17 + EMF)           ")
    print("      Knots c = [0, 3600, 8000, 13821] mol/m3                     ")
    print("=" * 75)

    x0 = np.array([0.95, 0.50, 0.05, 1.50,   1.50, 0.25, 0.10, 0.05])
    bounds = [(0.01, 5.0)] * 8

    print("\nStarting Nelder-Mead Optimization Routine...")
    res = opt.minimize(
        objective_4knot,
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 300, 'disp': True, 'adaptive': True}
    )

    print("\n" + "=" * 75)
    print("OPTIMIZATION STATUS & DIAGNOSTICS:")
    print(f"  Success Flag   : {res.success}")
    print(f"  Message        : {res.message}")
    print(f"  Iterations     : {res.nit}")
    print(f"  Func Evals     : {res.nfev}")
    print("-" * 75)
    print(f"  Final Train RMSE : {res.fun*1000:.2f} mV")
    print(f"  Optimized s_D    : {np.round(res.x[0:4], 4)}")
    print(f"  Optimized s_k    : {np.round(res.x[4:8], 4)}")
    print("=" * 75)

    with open(os.path.join(HERE, "optimization_result.txt"), "w") as f:
        f.write("4-knot optimisation result\n")
        f.write(f"knots c [mol/m3] : {np.round(C_KNOTS, 1).tolist()}\n")
        f.write(f"s_D              : {np.round(res.x[0:4], 4).tolist()}\n")
        f.write(f"s_k              : {np.round(res.x[4:8], 4).tolist()}\n")
        f.write(f"final train RMSE : {res.fun*1000:.2f} mV\n")
        f.write(f"iterations       : {res.nit}\n")
        f.write(f"function evals   : {res.nfev}\n")
        f.write(f"success flag     : {res.success}\n")
        f.write(f"message          : {res.message}\n")
    print(f"Saved: {os.path.join(HERE, 'optimization_result.txt')}")

    if res.fun < 0.030:
        s_D_opt = res.x[0:4]
        s_k_opt = res.x[4:8]
        interp_s_D = np.interp(hc.D1N_C, C_KNOTS, s_D_opt)
        interp_s_k = np.interp(hc.KN_C, C_KNOTS, s_k_opt)

        hc.D1N_D = hc.D1N_D * interp_s_D
        hc.KN_K = hc.KN_K * interp_s_k
        print("\nUpdated half_cell.py D1N_D and KN_K with calibrated 4-knot vectors!")

if __name__ == "__main__":
    run_optimization()
