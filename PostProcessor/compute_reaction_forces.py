"""Net platen reaction force for the squish (ustruct) simulation.

For a body in quasi-static equilibrium the force transmitted across a surface S is

    F = integral over S of  sigma_cauchy . n  dA          (deformed configuration)

so the platen reaction is the traction  sigma . n  integrated over the *contact
face only* (wall_top / wall_bottom), with normals and areas taken in the deformed
mesh.  The BCs here are RigidPlanePenalty (not Dirichlet), so there are no nodal
reactions to read off -- post-processing the stress field is the right route.

result_*.vtu points are the reference mesh, so we warp_by_vector('Displacement')
before integrating.  Cauchy_stress is a 6-component Voigt vector.

Units: model is um / mg / s  =>  stress in Pa, force in pN  (1 Pa.um^2 = 1e-12 N).

Usage (from the repository root):
    python3 PostProcessor/compute_reaction_forces.py                 # every DONE sample without a CSV
    python3 PostProcessor/compute_reaction_forces.py Samples/s0001   # just these samples
    ... --delete-vtu        remove result_*.vtu once the CSV is written

Writes:  Samples/sNNNN/<n>-procs/net_reaction_force.csv   (time, F vector + magnitude, for each face)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
MESH_SURF = os.path.join(ROOT_DIR, "Baseline", "mesh_scaled", "mesh-surfaces")
FACES = ["sphere_top", "sphere_bottom"]
CSV_NAME = "net_reaction_force.csv"


def voigt_to_mat(v):
    """svMultiPhysics symmetric-tensor output order: [xx, yy, zz, xy, yz, xz]."""
    xx, yy, zz, xy, yz, xz = v.T
    m = np.empty((v.shape[0], 3, 3))
    m[:, 0, 0], m[:, 1, 1], m[:, 2, 2] = xx, yy, zz
    m[:, 0, 1] = m[:, 1, 0] = xy
    m[:, 0, 2] = m[:, 2, 0] = xz
    m[:, 1, 2] = m[:, 2, 1] = yz
    return m


def face_node_mask(face_vtp, ref_points):
    """Boolean mask over the volume mesh nodes that lie on a boundary face."""
    f = pv.read(face_vtp)
    d, idx = cKDTree(ref_points).query(f.points)
    assert d.max() < 1e-3, f"{face_vtp}: face/volume points do not match ({d.max():.3g})"
    mask = np.zeros(len(ref_points), dtype=bool)
    mask[idx] = True
    return mask


def reaction_force(vol, node_mask):
    """Sum  sigma . n * dA  over the surface cells whose nodes are all in node_mask (deformed)."""
    warped = vol.warp_by_vector("Displacement")
    surf = warped.extract_surface(pass_pointid=True, algorithm="dataset_surface")
    orig = np.asarray(surf.point_data["vtkOriginalPointIds"])
    tris = surf.faces.reshape(-1, 4)
    assert (tris[:, 0] == 3).all(), "expected a triangulated boundary"
    keep = np.flatnonzero(node_mask[orig[tris[:, 1:]]].all(axis=1))
    if keep.size == 0:
        return np.zeros(3)
    sub = surf.extract_cells(keep).extract_surface(algorithm="dataset_surface")
    sub = sub.compute_normals(cell_normals=True, point_normals=False,
                              auto_orient_normals=True).point_data_to_cell_data()
    s = voigt_to_mat(np.asarray(sub.cell_data["Cauchy_stress"]))
    n = np.asarray(sub.cell_data["Normals"])
    a = sub.compute_cell_sizes(area=True).cell_data["Area"]
    t = np.einsum("kij,kj->ki", s, n)                 # traction per cell
    return (t * a[:, None]).sum(axis=0)               # net force vector


def results_dirs(sample):
    """svMultiPhysics writes to "<nprocs>-procs"; normally there is exactly one."""
    return sorted(d for d in glob.glob(os.path.join(sample, "*-procs")) if os.path.isdir(d))


def is_done(sample):
    status_file = os.path.join(sample, "status")
    if not os.path.exists(status_file):
        return False
    with open(status_file) as f:
        return f.readline().strip() == "DONE"


def process_run(dir_path):
    files = sorted(glob.glob(os.path.join(dir_path, "result_*.vtu")),
                   key=lambda p: int(re.search(r"result_(\d+)", p).group(1)))
    if not files:
        raise RuntimeError(f"no result_*.vtu in {dir_path}")

    ref = pv.read(files[0]).points.copy()
    masks = {face: face_node_mask(os.path.join(MESH_SURF, f"{face}.vtp"), ref) for face in FACES}

    times = []
    forces = {face: [] for face in FACES}          # list of [Fx, Fy, Fz] per step
    for fn in files:
        vol = pv.read(fn)
        times.append(float(vol.field_data["TimeValue"][0]))
        for face in FACES:
            forces[face].append(reaction_force(vol, masks[face]))

    times = np.asarray(times)
    order = np.argsort(times)

    header = ["time_s"]
    cols = [times[order]]
    for face in FACES:
        arr = np.asarray(forces[face])[order]
        header += [f"{face}_Fx_pN", f"{face}_Fy_pN", f"{face}_Fz_pN", f"{face}_Fmag_pN"]
        cols += [arr[:, 0], arr[:, 1], arr[:, 2], np.linalg.norm(arr, axis=1)]

    # Write to a temporary name first so a half-written CSV is never mistaken for a result
    out_csv = os.path.join(dir_path, CSV_NAME)
    tmp_csv = out_csv + ".tmp"
    np.savetxt(tmp_csv, np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="")
    os.replace(tmp_csv, out_csv)

    top = np.asarray(forces["sphere_top"])[order][:, 2]
    bot = np.asarray(forces["sphere_bottom"])[order][:, 2]
    rel = np.abs(top + bot) / np.maximum(np.abs(top), 1e-30)
    print(f"wrote {out_csv} ({len(files)} steps); top/bottom F_z balance: "
          f"mean {rel.mean()*100:.1f}%  max {rel.max()*100:.1f}%")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", nargs="*", help="sample directories (default: every DONE sample)")
    parser.add_argument("--delete-vtu", action="store_true",
                        help="delete result_*.vtu once net_reaction_force.csv is written")
    args = parser.parse_args()

    samples = args.samples or sorted(d for d in glob.glob(os.path.join(SAMPLES_DIR, "s[0-9]*"))
                                     if os.path.isdir(d))
    n_fail = 0
    for sample in samples:
        if not is_done(sample):
            continue
        for dir_path in results_dirs(sample):
            out_csv = os.path.join(dir_path, CSV_NAME)
            vtus = glob.glob(os.path.join(dir_path, "result_*.vtu"))
            if not os.path.exists(out_csv):
                print(f"=== {os.path.basename(os.path.normpath(sample))}", flush=True)
                try:
                    process_run(dir_path)
                except Exception as e:          # keep going; the VTUs are left for a retry
                    print(f"FAILED {dir_path}: {e!r}", flush=True)
                    n_fail += 1
                    continue
            if args.delete_vtu and vtus:
                for fn in vtus:
                    os.remove(fn)
                print(f"deleted {len(vtus)} result_*.vtu in {dir_path}", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
