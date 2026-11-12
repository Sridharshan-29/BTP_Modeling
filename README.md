# Na || Hard Carbon Half-Cell Model and Printed Anode Design

PyBaMM half-cell model of a hard carbon (HC) anode in sodium-ion cells, calibrated against Chayambuka et al. (2022) data and used to choose direct-ink-written (DIW) lattice designs.

Run each script from its own folder (or from anywhere). Results are saved in the same folder as the script.

| Folder | What it does | Run |
|---|---|---|
| `1_Base_Model` | Base model with no calibration, compared with EMF, C/30, C/22 and C/17 | `compare_rates_and_emf.py` |
| `2_Optimization` | Fits concentration-dependent diffusivity and rate-constant multipliers | `optimize_hc_parameters.py` |
| `3_Optimized_Model` | Calibrated model, compared with EMF, C/30, C/17, and validated on C/22 | `compare_rates_and_emf.py`, `validate_out_of_sample.py` |
| `4_Design_Optimization` | Lattice geometry to model inputs, design sweep, and viewer for the selected designs | `design_sweep.py`, then `view_lattice.py` |

Install the packages with `pip install -r requirements.txt` (`pyvista` is optional and only used for the 3D views).
