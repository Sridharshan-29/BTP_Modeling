import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import half_cell as hc

HERE = os.path.dirname(os.path.abspath(__file__))

CRATES_XLSX_PATH = os.path.join(HERE, "CRATES.xlsx")
VALIDATION_SHEET = "C22"
I_C22_A = 0.1364e-3
PLOT_FNAME = os.path.join(HERE, "plot_validation_C22_out_of_sample.png")

def load_validation_data():
    d = pd.read_excel(CRATES_XLSX_PATH, sheet_name=VALIDATION_SHEET, usecols=["Q_mAh_g", "U_V"]).dropna()
    return hc.monotone_curve(d["Q_mAh_g"].to_numpy(), d["U_V"].to_numpy())

def generate_validation_plot(q_mod, v_mod, exp_q, exp_u, rmse_mV, bias_mV):
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)

    ax.plot(exp_q, exp_u, color="#000000", linestyle="-", linewidth=2.0,
            label="paper experiment, C/22 (held-out validation)")

    ax.plot(q_mod, v_mod, color="#2ca02c", linestyle="--", linewidth=1.8,
            label=f"PyBaMM model, C/22 ({I_C22_A*1e3:.4f} mA)")

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

    fig.suptitle(f"Out-of-Sample Model Validation: C/22 Rate ({I_C22_A*1e3:.4f} mA)",
                 fontsize=14, fontweight='bold', y=0.98)
    ax.set_title(f"Validation RMSE = {rmse_mV:.2f} mV  |  Mean Error = {bias_mV:+.1f} mV  | ",
                 fontsize=10.5, fontweight='semibold', color='#333333', pad=10)

    plt.tight_layout()
    plt.savefig(PLOT_FNAME, dpi=300)
    plt.close(fig)
    print(f"Saved validation plot: '{PLOT_FNAME}'")

def main():
    print("=" * 75)
    print(f"      OUT-OF-SAMPLE VALIDATION ON HELD-OUT SHEET: '{VALIDATION_SHEET}'      ")
    print("=" * 75)

    r, m_g, Qth, _ = hc.coin_cell_run(current_A=I_C22_A, audit=True)
    if r["status"] != "ok":
        print("Validation run failed:", r)
        return

    q_mod, v_mod = hc.monotone_curve(r["Q_arr"] * 1e3 / m_g, r["V_arr"])
    exp_q, exp_u = load_validation_data()

    rmse, bias = hc.rmse_pchip(q_mod, v_mod, exp_q, exp_u)
    rmse_mV, bias_mV = rmse * 1000.0, bias * 1000.0

    print(f"\n[Validation Result - {VALIDATION_SHEET} (Out-of-Sample Test)]")
    print(f"  Applied Current          : {I_C22_A*1e3:.4f} mA")
    print(f"  Model Delivered Capacity : {q_mod[-1]:.1f} mAh/g")
    print(f"  Exp Delivered Capacity   : {exp_q.max():.1f} mAh/g")
    print(f"  Validation RMSE          : {rmse_mV:.2f} mV")
    print(f"  Mean Error (Bias)        : {bias_mV:+.1f} mV\n")

    hc.print_table(q_mod, v_mod, exp_q, exp_u, f"exp {VALIDATION_SHEET}")
    generate_validation_plot(q_mod, v_mod, exp_q, exp_u, rmse_mV, bias_mV)

if __name__ == "__main__":
    main()
