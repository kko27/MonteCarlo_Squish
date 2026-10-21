#!/bin/bash
# Stampede3 version of submit_postprocess.sbatch.
# Post-processing for the sweep: platen reaction forces for every DONE sample that
# does not have them yet (deleting its VTUs), then the Monte-Carlo statistics.
# Submit from the repository root once the sweep (or part of it) has finished:
#
#   sbatch submit_postprocess_stampede.sh                        # whole trace
#   sbatch submit_postprocess_stampede.sh --tmin 5 --tmax 6.25   # statistics on a time window
#
# Resubmitting is safe; samples that already have section_force.csv are skipped.
# Outputs: Samples/sNNNN/<n>-procs/section_force.csv, Samples/force_statistics.csv,
#          Samples/sample_metrics.csv, Samples/mc_*.png
#
# Note: skx bills the whole node for the whole run time, even for this serial job.

#SBATCH -J MCPost                   # Job name
#SBATCH -o postprocess_%j.out       # Name of stdout output file
#SBATCH -e postprocess_%j.err       # Name of stderr error file
#SBATCH -p skx                      # Queue (partition) name
#SBATCH -N 1                        # Total # of nodes
#SBATCH -n 1                        # Total # of tasks
#SBATCH -t 12:00:00                 # Run time (hh:mm:ss)
#SBATCH --mail-user=kbko@stanford.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# numpy, scipy, matplotlib and pyvista all live in the venv
module load python/3.12.11
source /work2/11985/kbko/stampede3/python-envs/test/bin/activate

export MPLBACKEND=Agg          # no display on compute nodes
export OMP_NUM_THREADS=1       # serial job

cd "$SLURM_SUBMIT_DIR"

# Fail fast if the venv is missing a package or was built with another python
python3 -c "import sys, numpy, scipy, matplotlib, pyvista; \
print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__, \
'| matplotlib', matplotlib.__version__, '| pyvista', pyvista.__version__)"

python3 PostProcessor/compute_reaction_forces.py --delete-vtu || echo "some samples failed; see above"
python3 PostProcessor/compute_statistics.py "$@"
