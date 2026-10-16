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

Cuts are taken at 50%, 25% and 75% of the deformed sphere's height, per frame
(the body is not centred on z = 0 once the top platen has moved, so fixed
heights can end up next to the contact patch).  The mid-height cut is the
reported force; the other two are a check, since in equilibrium all three
must agree.

result_*.vtu points are the reference mesh, so we warp_by_vector('Displacement')
before slicing.  Cauchy_stress is a 6-component Voigt vector [xx, yy, zz, xy, yz, xz].

Units: model is um / mg / s  =>  stress in Pa, force in pN  (1 Pa.um^2 = 1e-12 N).
Force is compressive-positive.

Usage (from the repository root):
    python3 PostProcessor/compute_reaction_forces.py                 # every DONE sample without a CSV
    python3 PostProcessor/compute_reaction_forces.py Samples/s0001   # just these samples
    ... --delete-vtu        remove result_*.vtu once the CSV is written

Writes:  Samples/sNNNN/<n>-procs/section_force.csv   (time, force at each cut)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pyvista as pv


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
CSV_NAME = "section_force.csv"
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

    times = []
    forces = []                                     # one row per step, one column per cut
    for fn in files:
        vol = pv.read(fn)
        times.append(float(vol.field_data["TimeValue"][0]))
        warped = vol.warp_by_vector("Displacement")
        forces.append([section_force(warped, a) for a in CUTS_FRAC])

    times = np.asarray(times)
    order = np.argsort(times)
    forces = np.asarray(forces)[order]

    header = ["time_s"] + [f"F_cut{100 * a:.0f}pct_pN" for a in CUTS_FRAC]

    # Write to a temporary name first so a half-written CSV is never mistaken for a result
    out_csv = os.path.join(dir_path, CSV_NAME)
    tmp_csv = out_csv + ".tmp"
    np.savetxt(tmp_csv, np.column_stack([times[order], forces]), delimiter=",",
               header=",".join(header), comments="")
    os.replace(tmp_csv, out_csv)

    # Equilibrium check: all cuts carry the same force
    spread = np.ptp(forces, axis=1).max() / np.maximum(np.abs(forces[:, 0]).max(), 1e-30)
    print(f"wrote {out_csv} ({len(files)} steps); max spread between cuts: "
          f"{100 * spread:.1f}% of peak force")
    if spread > SPREAD_WARN:
        print(f"WARNING: cuts disagree by more than {100 * SPREAD_WARN:.0f}%, check {dir_path}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", nargs="*", help="sample directories (default: every DONE sample)")
    parser.add_argument("--delete-vtu", action="store_true",
                        help=f"delete result_*.vtu once {CSV_NAME} is written")
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
