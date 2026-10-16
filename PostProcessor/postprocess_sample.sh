#!/bin/bash
# Per-sample post-processing called by `driver.py run --postprocess ...` after each
# DONE simulation:  writes section_force.csv, then deletes the result_*.vtu.
#
#   PostProcessor/postprocess_sample.sh Samples/s0001
#
# Runs in its own shell, so loading the python modules here does not touch the
# solver environment of the sweep worker.

set -euo pipefail

type module >/dev/null 2>&1 || source /etc/profile

module --force purge
module load devel
module load math
module load python/3.12.1
module load viz
module load py-numpy/2.2.6_py312
module load py-scipy/1.16.0_py312
module load py-matplotlib/3.8.3_py312

# pyvista lives in the venv
source /scratch/users/kbko/python-envs/test/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

cd "$(dirname "$0")/.."
exec python3 PostProcessor/compute_reaction_forces.py --delete-vtu "$@"
