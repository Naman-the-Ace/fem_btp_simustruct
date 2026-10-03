import sys, numpy as np
sys.path.insert(0, '.')
from src.fem_solver import analytical_stress_field
from src.inference import run_ai_inference

for load_val in [1e4, 1e5, 1e6, 5e6, 1e7]:
    r_fem = analytical_stress_field(1.0, 0.5, [{'rx':0.05,'ry':0.05,'cx':0.5,'cy':0.5}], 'structural_steel_a36', load_val, 'tension', n_grid=80)
    r_ai = run_ai_inference({'plate_width':1.0,'plate_height':0.5,'holes':[{'rx':0.05,'ry':0.05,'cx':0.5,'cy':0.5}]},
                            {'E':200e9,'nu':0.26,'sigma_y':250e6}, {'type':'tension','magnitude':load_val,'angle':0}, 'structural_steel_a36', n_grid=80)
    
    err = np.abs(np.array(r_ai['stress_vm']) - np.array(r_fem['stress_vm'])) / (np.abs(np.array(r_fem['stress_vm'])) + 1e-8) * 100
    print(f"Load {load_val/1e6:5.2f} MPa | Mean Error: {np.mean(err):5.1f}% | FEM max: {r_fem['sigma_max_mpa']:6.2f} | AI max: {r_ai['sigma_max_mpa']:6.2f}")
