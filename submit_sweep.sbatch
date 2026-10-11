#!/bin/bash
# One sweep worker: runs pending samples in Samples/ one after another until none
# are left or the wall time is nearly used up.  Submit from the repository root
# (which should live on $SCRATCH).  Each DONE run is post-processed right away
# (net_reaction_force.csv) and its result_*.vtu are deleted to save space.
#
#   sbatch --array=1-8 submit_sweep.sbatch     # 8 workers in parallel
#   python3 driver.py status                   # check progress
#
# Resubmitting later is safe; workers only pick up samples nobody has claimed.

#SBATCH --job-name=MCOrient
#SBATCH --partition=amarsden
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=8000
#SBATCH --ntasks-per-node=24
#SBATCH --mail-user=kbko@stanford.edu
#SBATCH --mail-type=END,FAIL

module --force purge
module load devel
module load math
module load imkl/2019
module load cuda/10.2.89
module load openmpi/4.0.3
module load hdf5/1.10.6
module load gcc/12.4.0
module load freetype/2.8.1
module load libxml2/2.9.4
module load libxkbcommon/0.9.1
module load libfabric/1.9.1
module load ucx/1.8.1
module load system

module load szip/2.1.1
module load cmake/3.24.2
module load libpng/1.2.57
module load fontconfig/2.12.4
module load llvm/4.0.0
module load x11/7.7
module load zlib/1.2.11
export LD_LIBRARY_PATH=/home/groups/amarsden/lib-perigee/VTK-7.1.1-shared/lib:$LD_LIBRARY_PATH
PATH=$PATH:$HOME/.local/bin:$HOME/bin
export PATH
export VTK_DIR=/home/groups/amarsden/lib-perigee/VTK-7.1.1-shared

SVMP_EXE=/home/users/kbko/ActiveStressSquish/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics

cd "$SLURM_SUBMIT_DIR"

# driver.py only needs the standard library, so the system python3 is enough.
# Per-run limits: --time-limit (default 7200 s) kills a run that takes too long;
# runs are also killed on the first "linear system solution has not converged".
exec python3 driver.py run --exe "$SVMP_EXE" --launcher srun \
    --postprocess PostProcessor/postprocess_sample.sh
