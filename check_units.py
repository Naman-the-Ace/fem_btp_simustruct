"""Unit consistency check — run from fem_btp root."""
import sys, numpy as np
sys.path.insert(0, '.')
from src.fem_solver import analytical_stress_field
from src.inference import run_ai_inference

r_fem = analytical_stress_field(1.0, 0.5, [{"rx":0.05,"ry":0.05,"cx":0.5,"cy":0.5}],
    "structural_steel_a36", 1e5, "tension", n_grid=80)
r_ai = run_ai_inference(
    {"plate_width":1.0,"plate_height":0.5,"holes":[{"rx":0.05,"ry":0.05,"cx":0.5,"cy":0.5}]},
    {"E":200e9,"nu":0.26,"sigma_y":250e6},
    {"type":"tension","magnitude":1e5,"angle":0},
    "structural_steel_a36", n_grid=80)

ai_sv  = np.array(r_ai["stress_vm"])
fem_sv = np.array(r_fem["stress_vm"])
err = np.abs(ai_sv - fem_sv) / (np.abs(fem_sv) + 1e-8) * 100

print("=== UNIT CONSISTENCY CHECK ===")
print("FEM max stress:", round(float(fem_sv.max()), 4), "MPa (expect ~0.25)")
print("AI  max stress:", round(float(ai_sv.max()), 4), "MPa")
print("FEM sigma_max_mpa:", round(r_fem["sigma_max_mpa"], 4), "SCF:", round(r_fem["scf"], 3))
print("AI  sigma_max_mpa:", round(r_ai["sigma_max_mpa"], 4),  "SCF:", round(r_ai["scf"], 3))
print("Mean Error:", round(float(np.mean(err)), 1), "% (was 98%)")
print("Safety Factor FEM:", round(r_fem["safety_factor"], 2))
print("Safety Factor AI:",  round(r_ai["safety_factor"], 2))
