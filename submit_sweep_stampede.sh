#!/bin/bash
# Stampede3 version of submit_sweep.sbatch.
# One sweep worker: runs pending samples in Samples/ one after another (each on a
# full 48-core SKX node) until none are left or the wall time is nearly used up.
# Submit from the repository root (which lives on $WORK).  Each DONE run
# is post-processed right away (section_force.csv) and its result_*.vtu are deleted.
#
#   for i in $(seq 1 4); do sbatch submit_sweep_stampede.sh; done   # 4 workers in parallel
#   python3 driver.py status                                         # check progress
#
# Resubmitting later is safe; workers only pick up samples nobody has claimed.
# Results go to Samples/sNNNN/48-procs/ (Sherlock runs used 24-procs/).

#SBATCH -J MCOrient                 # Job name
#SBATCH -o sweep_%j.out             # Name of stdout output file
#SBATCH -e sweep_%j.err             # Name of stderr error file
#SBATCH -p skx                      # Queue (partition) name
#SBATCH -N 1                        # Total # of nodes
#SBATCH -n 48                       # Total # of mpi tasks
#SBATCH -t 48:00:00                 # Run time (hh:mm:ss), skx maximum
#SBATCH --mail-user=kbko@stanford.edu
#SBATCH --mail-type=END,FAIL

module load hypre/2.30.0-i64 hdf5/1.14.6 blis/1.1 boost/1.86.0 vtk/9.3.0

SVMP_EXE=/work2/11985/kbko/stampede3/svmpWithoutTrilinos/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics

cd "$SLURM_SUBMIT_DIR"

# driver.py only needs the standard library, so the system python3 is enough.
# Per-run limits: --time-limit (default 7200 s) kills a run that takes too long;
# runs are also killed on the first "linear system solution has not converged".
# Use ibrun instead of mpirun or mpiexec.
exec python3 driver.py run --exe "$SVMP_EXE" --launcher ibrun \
    --postprocess PostProcessor/postprocess_sample_stampede.sh
