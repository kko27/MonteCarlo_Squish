# Python code that generates the svmultiphysics inputs for the Monte-Carlo
# study of the active-layer orientation

# The active (domain 1) layer is the crescent from 00_assign_domains.py: the outer
# shell (cell radius > RADIUS_THRESHOLD) inside a cone of half angle
# PERCENTAGE_COVERAGE * 180 deg about the cone axis.  The cap is axisymmetric about
# that axis, so a uniformly random orientation of the layer is a cone axis drawn
# uniformly on the unit sphere:  cos(theta) ~ U[-1, 1],  phi ~ U[0, 2 pi).
# Fibers are not rotated; only DOMAIN_ID changes from sample to sample.

# For each draw, create a run directory inside Samples
# Label Samples as s0001, s0002, ..., s1000
# Each sample holds solver.xml and its own mesh-complete.domains.vtu
# Save the sample parameters as a txt file with fields: sample, axis_x, axis_y, axis_z, theta_deg, phi_deg
# This txt file is saved as samples.txt inside Samples

# The runs themselves are launched by ../driver.py (see ../submit_sweep.sbatch)

import os
import re
import sys

import numpy as np
import pyvista as pv

# Paths (resolved relative to this script so it can be run from anywhere)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
BASELINE_DIR = os.path.join(ROOT_DIR, 'Baseline')
SAMPLES_DIR  = os.path.join(ROOT_DIR, 'Samples')

# Sample Space
N_SAMPLES = 1000
SEED      = 20261010          # fixed so the sample set can be regenerated exactly

# Skin layer (same for every sample; see 00_assign_domains.py)
RADIUS_THRESHOLD    = 300.0   # um
PERCENTAGE_COVERAGE = 0.6     # fraction of 360 deg spanned by the domain 1 cone

# Refuse to overwrite inputs of samples that have already been run: the results
# would no longer match the orientations in samples.txt
if os.path.isdir(SAMPLES_DIR):
    started = [d for d in os.listdir(SAMPLES_DIR)
               if os.path.exists(os.path.join(SAMPLES_DIR, d, '.claim'))]
    if started:
        sys.exit(f'{len(started)} samples in {SAMPLES_DIR} have already been run '
                 f'(e.g. {sorted(started)[0]}). Move or delete Samples/ before regenerating.')

# Read in the baseline files once
with open(os.path.join(BASELINE_DIR, 'solver_tanH.xml')) as f:
    baseline_xml = f.read()

mesh = pv.read(os.path.join(BASELINE_DIR, 'mesh_scaled', 'mesh-complete.mesh.vtu'))
cell_centers = mesh.cell_centers().points
radii = np.linalg.norm(cell_centers, axis=1)
outer = radii > RADIUS_THRESHOLD
unit_centers = cell_centers / np.maximum(radii, 1e-12)[:, None]
half_angle = np.deg2rad(PERCENTAGE_COVERAGE * 360.0 / 2.0)


def crescent_domain_id(axis):
    """Outer shell cells within half_angle of axis are domain 1; everything else is domain 2."""
    polar = np.arccos(np.clip(unit_centers @ axis, -1.0, 1.0))
    return np.where(outer & (polar <= half_angle), 1, 2).astype(np.int32)


def make_solver_xml():
    xml = baseline_xml
    # Point the mesh, fibers and data files back to Baseline
    xml = xml.replace('mesh_scaled/', '../../Baseline/mesh_scaled/')
    # ... except the domain file, which is written per sample
    xml = re.sub(r'<Domain_file_path>\s*[^<]*</Domain_file_path>',
                 '<Domain_file_path> mesh-complete.domains.vtu </Domain_file_path>', xml)
    xml = re.sub(r'<Temporal_values_file_path>\s*(\w+\.dat)\s*</Temporal_values_file_path>',
                 r'<Temporal_values_file_path> ../../Baseline/\1 </Temporal_values_file_path>', xml)
    return xml


# Uniform cone axes on the unit sphere
rng = np.random.default_rng(SEED)
cos_theta = rng.uniform(-1.0, 1.0, N_SAMPLES)
phi = rng.uniform(0.0, 2.0 * np.pi, N_SAMPLES)
sin_theta = np.sqrt(1.0 - cos_theta**2)
axes = np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

solver_xml = make_solver_xml()
os.makedirs(SAMPLES_DIR, exist_ok=True)

with open(os.path.join(SAMPLES_DIR, 'samples.txt'), 'w') as f_samples:
    f_samples.write('sample axis_x axis_y axis_z theta_deg phi_deg n_domain1_cells\n')

    for i, axis in enumerate(axes):
        name = f's{i + 1:04d}'

        # Create a save directory for each sample
        sample_dir = os.path.join(SAMPLES_DIR, name)
        os.makedirs(sample_dir, exist_ok=True)

        with open(os.path.join(sample_dir, 'solver.xml'), 'w') as f:
            f.write(solver_xml)

        domain_id = crescent_domain_id(axis)
        mesh.cell_data['DOMAIN_ID'] = domain_id
        mesh.save(os.path.join(sample_dir, 'mesh-complete.domains.vtu'))

        theta_deg = np.degrees(np.arccos(axis[2]))
        phi_deg = np.degrees(phi[i])
        f_samples.write(f'{name} {axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f} '
                        f'{theta_deg:.4f} {phi_deg:.4f} {int((domain_id == 1).sum())}\n')

print(f'Generated {N_SAMPLES} samples (seed {SEED}, coverage {PERCENTAGE_COVERAGE}, '
      f'radius threshold {RADIUS_THRESHOLD}) in {SAMPLES_DIR}')
