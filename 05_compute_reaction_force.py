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
Outputs are written in pN; plot is |F_z| in uN.

Writes:  net_reaction_force.csv   (time, F vector + magnitude, for each face)
         net_reaction_force.png   (axial reaction vs time, both faces)
"""

import glob
import os
import re

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt


def voigt_to_mat(v):
    """svMultiPhysics symmetric-tensor output order: [xx, yy, zz, xy, yz, xz]."""
    xx, yy, zz, xy, yz, xz = v.T
    m = np.empty((v.shape[0], 3, 3))
    m[:, 0, 0], m[:, 1, 1], m[:, 2, 2] = xx, yy, zz
    m[:, 0, 1] = m[:, 1, 0] = xy
    m[:, 0, 2] = m[:, 2, 0] = xz
    m[:, 1, 2] = m[:, 2, 1] = yz
    return m


def face_node_set(face_vtp, ref_points):
    """Indices into the volume mesh of the nodes that lie on a boundary face."""
    f = pv.read(face_vtp)
    d, idx = cKDTree(ref_points).query(f.points)
    assert d.max() < 1e-3, f"{face_vtp}: face/volume points do not match ({d.max():.3g})"
    return set(idx.tolist())


def reaction_force(vol, node_set):
    """Sum  sigma . n * dA  over the surface cells belonging to node_set (deformed)."""
    warped = vol.warp_by_vector("Displacement")
    surf = warped.extract_surface(pass_pointid=True, algorithm="dataset_surface")
    orig = surf.point_data["vtkOriginalPointIds"]
    keep = [c for c in range(surf.n_cells)
            if all(orig[p] in node_set for p in surf.get_cell(c).point_ids)]
    if not keep:
        return np.zeros(3)
    sub = surf.extract_cells(keep).extract_surface(algorithm="dataset_surface")
    sub = sub.compute_normals(cell_normals=True, point_normals=False,
                              auto_orient_normals=True).point_data_to_cell_data()
    s = voigt_to_mat(np.asarray(sub.cell_data["Cauchy_stress"]))
    n = np.asarray(sub.cell_data["Normals"])
    a = sub.compute_cell_sizes(area=True).cell_data["Area"]
    t = np.einsum("kij,kj->ki", s, n)                 # traction per cell
    return (t * a[:, None]).sum(axis=0)               # net force vector


def main():
    # fold_list = ['../Sphere_FrankStarling_SensStudy/rot000', 
    #              '../Sphere_FrankStarling_SensStudy/rot045',
    #              '../Sphere_FrankStarling_SensStudy/rot090',
    #              '../Sphere_FrankStarling_SensStudy/rot135',
    #              '../Sphere_FrankStarling_SensStudy/rot180',
    #              ]

    fold_list = ['../Compression_ActiveStress_Sensitivity/Tmax_100', 
                '../Compression_ActiveStress_Sensitivity/Tmax_150',
                '../Compression_ActiveStress_Sensitivity/Tmax_200',
                '../Compression_ActiveStress_Sensitivity/Tmax_250',
                '../Compression_ActiveStress_Sensitivity/Tmax_300',
                ]

    # fold_list = ['../FrankStarlingSquish_70/']

    
    for fold in fold_list:
        print(f"Processing {fold}...")
        DIR_PATH = f"{fold}/24-procs"
        # MESH_SURF = "../Sphere_FrankStarling_SensStudy/mesh_scaled/mesh-surfaces"
        MESH_SURF = "../Compression_ActiveStress_Sensitivity/mesh_scaled/mesh-surfaces"
        FACES = ["sphere_top", "sphere_bottom"]          # actuator + fixed reaction; used to cross-check
        OUT_CSV = os.path.join(DIR_PATH, "net_reaction_force.csv")
        OUT_PNG = os.path.join(DIR_PATH, "net_reaction_force.png")

        files = sorted(glob.glob(os.path.join(DIR_PATH, "result_*.vtu")),
                    key=lambda p: int(re.search(r"result_(\d+)", p).group(1)))
        if not files:
            raise SystemExit(f"no result_*.vtu in {DIR_PATH}")

        ref = pv.read(files[0]).points.copy()
        nodes = {face: face_node_set(os.path.join(MESH_SURF, f"{face}.vtp"), ref)
                for face in FACES}

        times = []
        forces = {face: [] for face in FACES}          # list of [Fx, Fy, Fz] per step
        for fn in files:
            vol = pv.read(fn)
            t = float(vol.field_data["TimeValue"][0])
            times.append(t)
            line = [f"t={t:6.2f}s"]
            for face in FACES:
                f_vec = reaction_force(vol, nodes[face])
                forces[face].append(f_vec)
                line.append(f"{face}: Fz={f_vec[2]: .3e} pN")
            print("  ".join(line))

        times = np.asarray(times)
        order = np.argsort(times)
        times = times[order]

        # --- CSV -------------------------------------------------------------------
        header = ["time_s"]
        cols = [times]
        for face in FACES:
            arr = np.asarray(forces[face])[order]
            header += [f"{face}_Fx_pN", f"{face}_Fy_pN", f"{face}_Fz_pN", f"{face}_Fmag_pN"]
            cols += [arr[:, 0], arr[:, 1], arr[:, 2], np.linalg.norm(arr, axis=1)]
        np.savetxt(OUT_CSV, np.column_stack(cols), delimiter=",",
                header=",".join(header), comments="")
        print(f"\nwrote {OUT_CSV}")

        # --- plot ---------------------------------------------------------------
        plt.figure()
        for face in FACES:
            fz = np.asarray(forces[face])[order][:, 2]
            plt.plot(times, np.abs(fz) / 1e6, "-o", ms=3, label=f"{face}  |$F_z$|")
        plt.xlabel("Time (s)")
        plt.ylabel(r"Axial reaction $|F_z|$  (µN)")
        plt.title("Squish reaction force")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=150)
        plt.close()
        print(f"wrote {OUT_PNG}")

        # --- balance check ----------------------------------------------------
        top = np.asarray(forces["sphere_top"])[order][:, 2]
        bot = np.asarray(forces["sphere_bottom"])[order][:, 2]
        rel = np.abs(top + bot) / np.maximum(np.abs(top), 1e-30)
        print(f"\ntop/bottom F_z balance: mean {rel.mean()*100:.1f}%  max {rel.max()*100:.1f}%  "
            f"(0% = perfect equilibrium)")


if __name__ == "__main__":
    main()
