#!/bin/bash
# Post-processing for the sweep: platen reaction forces for every DONE sample that
# does not have them yet (deleting its VTUs), then the Monte-Carlo statistics.
# Submit from the repository root once the sweep (or part of it) has finished:
#
#   sbatch submit_postprocess.sbatch                        # whole trace
#   sbatch submit_postprocess.sbatch --tmin 5 --tmax 6.25   # statistics on a time window
#
# Resubmitting is safe; samples that already have section_force.csv are skipped.
# Outputs: Samples/sNNNN/24-procs/section_force.csv, Samples/force_statistics.csv,
#          Samples/sample_metrics.csv, Samples/mc_*.png

#SBATCH --job-name=MCPost
#SBATCH --partition=amarsden
#SBATCH --output=postprocess_%j.out
#SBATCH --error=postprocess_%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000
#SBATCH --mail-user=kbko@stanford.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module --force purge
module load devel
module load math
module load python/3.12.1
module load viz 
module load py-numpy/2.2.6_py312
module load py-scipy/1.16.0_py312
module load py-matplotlib/3.8.3_py312

# matplotlib and pyvista live in the venv
source /scratch/users/kbko/python-envs/test/bin/activate

export MPLBACKEND=Agg          # no display on compute nodes
export OMP_NUM_THREADS=1       # match the single CPU requested

cd "$SLURM_SUBMIT_DIR"

# Fail fast if the module numpy/scipy don't play well with the venv packages
python3 -c "import numpy, scipy, matplotlib, pyvista; \
print('numpy', numpy.__version__, '| scipy', scipy.__version__, \
'| matplotlib', matplotlib.__version__, '| pyvista', pyvista.__version__)"

python3 PostProcessor/compute_reaction_forces.py --delete-vtu || echo "some samples failed; see above"
python3 PostProcessor/compute_statistics.py "$@"
