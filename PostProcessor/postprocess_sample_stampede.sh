#!/bin/bash
# Stampede3 version of postprocess_sample.sh, called by `driver.py run --postprocess ...`
# after each DONE simulation:  writes section_force.csv, then deletes the result_*.vtu.
#
#   PostProcessor/postprocess_sample_stampede.sh Samples/s0001
#
# Runs in its own shell, so loading the python module here does not touch the
# solver environment of the sweep worker.

set -euo pipefail

type module >/dev/null 2>&1 || source /etc/profile

# numpy, scipy, matplotlib and pyvista all live in the venv
module load python/3.12.11
source /work2/11985/kbko/stampede3/python-envs/test/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

cd "$(dirname "$0")/.."
exec python3 PostProcessor/compute_reaction_forces.py --delete-vtu "$@"
