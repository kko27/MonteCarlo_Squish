import os
import numpy as np
import pyvista as pv

# read in the mesh-complete.mesh.vtu data from mesh_scaled
# assign two domains to the mesh based on the radius of the mesh points
# If the radius is within 300 (user-specified) assign domain 2, else assign domain 1
RADIUS_THRESHOLD = 300.0

# Flag that if true, creates mesh files with domains assigned with a crescent shape
CRESCENT_SHAPE = True
# Coverage of skin layer: fraction of 360 degrees spanned by the domain 1 cone
# (e.g. 0.7 -> 252 degrees, i.e. +/- 126 degrees about the cone axis)
PERCENTAGE_COVERAGE = 0.6
# The cone axis starts at +z and rotates clockwise about the x-axis (viewed from +x
# looking toward the origin) from 0 up to MAX_ROTATION_DEG in N_ROTATIONS equal steps
MAX_ROTATION_DEG = 180.0
N_ROTATIONS = 4
CRESCENT_DIR = "crescent_domains"

mesh = pv.read("mesh_scaled/mesh-complete.mesh.vtu")

# Calculate the radius of each cell (distance of its centroid from the origin)
cell_centers = mesh.cell_centers().points
radii = np.linalg.norm(cell_centers, axis=1)

# Save the data as a new field called DOMAIN_ID which is Cell Data
# This data can be appended to the mesh
domain_id = np.where(radii <= RADIUS_THRESHOLD, 2, 1).astype(np.int32)
mesh.cell_data["DOMAIN_ID"] = domain_id

mesh.save("mesh_scaled/mesh-complete.domains.vtu")

if CRESCENT_SHAPE:
    os.makedirs(CRESCENT_DIR, exist_ok=True)

    half_angle = np.deg2rad(PERCENTAGE_COVERAGE * 360.0 / 2.0)
    outer = radii > RADIUS_THRESHOLD
    unit_centers = cell_centers / np.maximum(radii, 1e-12)[:, None]

    for rot_deg in np.linspace(0.0, MAX_ROTATION_DEG, N_ROTATIONS + 1):
        # Clockwise about x (viewed from +x): +z tips toward +y
        rot = np.deg2rad(rot_deg)
        axis = np.array([0.0, np.sin(rot), np.cos(rot)])

        # Polar angle of each cell centroid measured from the cone axis
        polar = np.arccos(np.clip(unit_centers @ axis, -1.0, 1.0))

        # Outer shell cells inside the cone stay domain 1; everything else is domain 2
        crescent_id = np.where(outer & (polar <= half_angle), 1, 2).astype(np.int32)
        mesh.cell_data["DOMAIN_ID"] = crescent_id

        out_path = os.path.join(CRESCENT_DIR, f"mesh-complete.domains_rot{int(round(rot_deg)):03d}.vtu")
        mesh.save(out_path)
        print(f"Saved {out_path}")
