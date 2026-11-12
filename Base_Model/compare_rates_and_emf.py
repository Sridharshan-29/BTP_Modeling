import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Half_Cell_Base as hc

HERE = os.path.dirname(os.path.abspath(__file__))

CRATES_XLSX_PATH = os.path.join(HERE, "CRATES.xlsx")

DATASET_CONFIG = {
    "EMF": {
        "name": "EMF (C/1000)",
        "current_A": 0.00295e-3,
        "sheet": None,
        "color_mod": "#444444",
        "fname": "plot_model_vs_EMF_experiment_Base.png"
    },
    "C30": {
        "name": "C/30",
        "current_A": 0.10000e-3,
        "sheet": "C30",
        "color_mod": "#d62728",
        "fname": "plot_model_vs_C30_experiment_Base.png"
    },
    "C17": {
        "name": "C/17",
        "current_A": 0.17650e-3,
        "sheet": "C17",
        "color_mod": "#1f77b4",
        "fname": "plot_model_vs_C17_experiment_Base.png"
    },
    "C22": {
        "name": "C/22",
        "current_A": 0.1364e-3,
        "sheet": "C22",
        "color_mod": "#026d10",
        "fname": "plot_model_vs_C22_experiment_Base.png"
    }
}

def load_experimental_data(key):
    cfg = DATASET_CONFIG[key]
    if key == "EMF":
        q_exp = hc.ocp_df["Q_mAh_g"].to_numpy()
        u_exp = hc.OCP_U
        return hc.monotone_curve(q_exp, u_exp)
    else:
        d = pd.read_excel(CRATES_XLSX_PATH, sheet_name=cfg["sheet"], usecols=["Q_mAh_g", "U_V"]).dropna()
        return hc.monotone_curve(d["Q_mAh_g"].to_numpy(), d["U_V"].to_numpy())

def generate_individual_plot(key, q_mod, v_mod, exp_q, exp_u, rmse_mV, bias_mV):
    cfg = DATASET_CONFIG[key]
    current_mA = cfg["current_A"] * 1e3

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)

    ax.plot(exp_q, exp_u, color="#000000", linestyle="-", linewidth=2.0,
            label=f"paper experiment, {cfg['name']} (digitised)")

    ax.plot(q_mod, v_mod, color=cfg["color_mod"], linestyle="--", linewidth=1.8,
            label=f"PyBaMM model, {cfg['name']} ({current_mA:.4f} mA)")

    ax.set_xlabel('transferred charge / mAh g$^{-1}$', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('V$_{HC}$ / V vs. Na$^{+}$/Na RE', fontsize=12, fontweight='bold', labelpad=8)

    ax.set_xlim(0, 250)
    ax.set_ylim(0, 2.0)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

    ax.tick_params(direction='in', top=True, right=True, labelsize=10.5, length=6, width=1.2)
    ax.tick_params(which='minor', direction='in', top=True, right=True, length=3, width=0.8)
    ax.minorticks_on()

    ax.legend(loc='upper right', fontsize=9.5, frameon=True, edgecolor='gray', facecolor='white', framealpha=0.95)

    fig.suptitle(f"Model vs. Experimental {cfg['name']} ({current_mA:.4f} mA)",
                 fontsize=14, fontweight='bold', y=0.98)
    ax.set_title(f"RMSE = {rmse_mV:.2f} mV  |  Mean Error = {bias_mV:+.1f} mV  |",
                 fontsize=10.5, fontweight='semibold', color='#333333', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(HERE, cfg["fname"]), dpi=300)
    plt.close(fig)
    print(f"  Saved plot: {os.path.join(HERE, cfg['fname'])}")

def run_all_evaluations():
    print("=" * 75)
    print("   AUTOMATED MULTI-RATE & EMF MODEL EVALUATION (EMF, C/30, C/17, C/22)   ")
    print("=" * 75)

    results = {}
    for key, cfg in DATASET_CONFIG.items():
        print(f"\n--> Evaluating {cfg['name']} ({cfg['current_A']*1e3:.4f} mA)...")
        r, m_g, Qth, i_A = hc.coin_cell_run(current_A=cfg["current_A"], audit=False)
        if r["status"] != "ok":
            print(f"    [ERROR] Simulation failed for {cfg['name']}")
            continue

        q_mod, v_mod = hc.monotone_curve(r["Q_arr"] * 1e3 / m_g, r["V_arr"])
        exp_q, exp_u = load_experimental_data(key)

        rmse, bias = hc.rmse_pchip(q_mod, v_mod, exp_q, exp_u)
        rmse_mV, bias_mV = rmse * 1000.0, bias * 1000.0

        print(f"  Delivered Capacity : {q_mod[-1]:.1f} mAh/g (Exp: {exp_q.max():.1f} mAh/g)")
        print(f"  RMSE vs Experiment : {rmse_mV:.2f} mV (Mean Error: {bias_mV:+.1f} mV)")

        hc.print_table(q_mod, v_mod, exp_q, exp_u, f"exp {cfg['name']}")
        generate_individual_plot(key, q_mod, v_mod, exp_q, exp_u, rmse_mV, bias_mV)

        results[key] = (rmse_mV, bias_mV, q_mod[-1], exp_q.max())

    print("\n" + "=" * 75)
    print("SUMMARY TABLE Across EMF, C/30, C/17, C/22:")
    print(f"{'Dataset':<12} | {'RMSE [mV]':<10} | {'Mean Bias [mV]':<15} | {'Model Cap':<12} | {'Exp Cap':<12}")
    print("-" * 75)
    for key, res in results.items():
        print(f"{DATASET_CONFIG[key]['name']:<12} | {res[0]:<10.2f} | {res[1]:<+15.1f} | {res[2]:<12.1f} | {res[3]:<12.1f}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_all_evaluations()
