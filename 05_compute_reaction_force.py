"""Axial platen force for the squish (ustruct) simulation, from cross-sections.

The part of the sphere above a horizontal cut is loaded in z only by the top
platen contact and by the stress transmitted across the cut: the rest of its
surface is traction-free, pin_axis_top constrains x/y only, there are no body
forces, and inertia is negligible (~1e-5 uN).  Equilibrium therefore gives

    F_platen = - integral over the cut of  sigma_zz  dA        (deformed configuration)

for any horizontal cut through the sphere.  Integrating sigma . n over the
sphere_top face instead is not usable: that face is the whole upper hemisphere,
not the contact patch, and the nodal-averaged stress on the free surface is far
from traction-free once the fibres are active (spurious dips and sign flips at
twitch onset).

Cuts are taken at 50%, 25% and 75% of the deformed sphere's height, per frame.
Fixed heights (as in Sphere_Parameter_Tune) don't work here: only the top platen
moves, so the body is not centred on z = 0 and a cut at z = +150 um ends up within
~65 um of the contact patch late in the run, where the mesh can't resolve the
stress (it drops ~30%).  The mid-height cut is the reported force; the other two
are a check, since in equilibrium all three must agree.

result_*.vtu points are the reference mesh, so we warp_by_vector('Displacement')
before slicing.  Cauchy_stress is a 6-component Voigt vector [xx, yy, zz, xy, yz, xz].

Units: model is um / mg / s  =>  stress in Pa, force in pN  (1 Pa.um^2 = 1e-12 N).
Force is compressive-positive.

Writes:  section_force.csv   (time, force at each cut)
         section_force.png   (force at each cut vs time)
"""

import glob
import os
import re

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt


OUT_NAME = "section_force.csv"
CUTS_FRAC = [0.50, 0.25, 0.75]  # fraction of deformed height (bottom -> top); first one is the reported force
SPREAD_WARN = 0.10              # warn when the cuts disagree by more than 10% of the peak force (~5% is typical)


def section_force(warped, frac):
    """Compressive axial force (pN) transmitted across the horizontal plane at
    fraction `frac` of the deformed body's height."""
    zmin, zmax = warped.bounds[4], warped.bounds[5]
    z = zmin + frac * (zmax - zmin)
    s = warped.slice(normal="z", origin=(0.0, 0.0, z)).triangulate()
    s = s.compute_cell_sizes(area=True).point_data_to_cell_data()
    szz = np.asarray(s.cell_data["Cauchy_stress"])[:, 2]
    return -float(np.sum(szz * s.cell_data["Area"]))


def process_run(DIR_PATH):
    OUT_CSV = os.path.join(DIR_PATH, OUT_NAME)
    OUT_PNG = os.path.join(DIR_PATH, "section_force.png")

    files = sorted(glob.glob(os.path.join(DIR_PATH, "result_*.vtu")),
                   key=lambda p: int(re.search(r"result_(\d+)", p).group(1)))
    if not files:
        raise SystemExit(f"no result_*.vtu in {DIR_PATH}")

    times = []
    forces = []                                     # one row per step, one column per cut
    for fn in files:
        vol = pv.read(fn)
        t = float(vol.field_data["TimeValue"][0])
        warped = vol.warp_by_vector("Displacement")
        f = [section_force(warped, a) for a in CUTS_FRAC]
        times.append(t)
        forces.append(f)
        print(f"t={t:6.2f}s  " + "  ".join(f"{100 * a:.0f}%: {fz: .3e} pN" for a, fz in zip(CUTS_FRAC, f)))

    times = np.asarray(times)
    order = np.argsort(times)
    times = times[order]
    forces = np.asarray(forces)[order]

    # --- equilibrium check: all cuts carry the same force ----------------------
    spread = np.ptp(forces, axis=1).max() / np.abs(forces[:, 0]).max()
    print(f"max spread between cuts: {100 * spread:.1f}% of peak force")
    if spread > SPREAD_WARN:
        print(f"WARNING: cuts disagree by more than {100 * SPREAD_WARN:.0f}%, check {DIR_PATH}")

    # --- CSV -------------------------------------------------------------------
    header = ["time_s"] + [f"F_cut{100 * a:.0f}pct_pN" for a in CUTS_FRAC]
    np.savetxt(OUT_CSV, np.column_stack([times, forces]), delimiter=",",
               header=",".join(header), comments="")
    print(f"\nwrote {OUT_CSV}")

    # --- plot ------------------------------------------------------------------
    plt.figure()
    for k, a in enumerate(CUTS_FRAC):
        plt.plot(times, forces[:, k] / 1e6, "-o", ms=3, label=f"cut at {100 * a:.0f}% height")
    plt.xlabel("Time (s)")
    plt.ylabel("Axial platen force (µN)")
    plt.title("Squish platen force")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"wrote {OUT_PNG}")


def main():
    fold_list = ['../Sphere_FrankStarling_SensStudy/rot000',
                 '../Sphere_FrankStarling_SensStudy/rot045',
                 '../Sphere_FrankStarling_SensStudy/rot090',
                 '../Sphere_FrankStarling_SensStudy/rot135',
                 '../Sphere_FrankStarling_SensStudy/rot180',
                 ]

    # fold_list = ['../Compression_ActiveStress_Sensitivity/Tmax_100',
    #              '../Compression_ActiveStress_Sensitivity/Tmax_150',
    #              '../Compression_ActiveStress_Sensitivity/Tmax_200',
    #              '../Compression_ActiveStress_Sensitivity/Tmax_250',
    #              '../Compression_ActiveStress_Sensitivity/Tmax_300',
    #              ]

    # fold_list = ['../FrankStarlingSquish_70/']

    for fold in fold_list:
        print(f"Processing {fold}...")
        process_run(f"{fold}/24-procs")


if __name__ == "__main__":
    main()
