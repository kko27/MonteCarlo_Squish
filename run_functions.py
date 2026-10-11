"""Helper functions for the Monte-Carlo orientation sweep driver.

Bookkeeping lives inside each sample directory (Samples/sNNNN/):
    .claim/       created atomically (os.mkdir) by the worker that runs the sample;
                  holds an `owner` file with the Slurm job id and host
    status        one of RUNNING, DONE, DIVERGED, TIMEOUT, FAILED
    sim.log       stdout/stderr of the solver
    post.log      stdout/stderr of the per-sample post-processing (reaction force)

A sample with no .claim directory is pending.  Several Slurm jobs can therefore
work through the same Samples/ directory without running a sample twice.

Written for the system python3 on Sherlock (3.6), so only the standard library is
used and nothing newer than 3.6 syntax.
"""

import os
import re
import shutil
import signal
import socket
import subprocess
import time

ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(ROOT_DIR, 'Samples')
SWEEP_LOG   = os.path.join(SAMPLES_DIR, 'sweep_log.csv')

# Warning svMultiPhysics writes to histor.dat when GMRES fails.  Once it shows up
# the residuals are already NaN, and each further Newton iteration burns ~5 min,
# so the run is killed on the first occurrence.
DIVERGENCE_MSG = 'The linear system solution has not converged'

PENDING  = 'PENDING'
RUNNING  = 'RUNNING'
DONE     = 'DONE'
DIVERGED = 'DIVERGED'
TIMEOUT  = 'TIMEOUT'
FAILED   = 'FAILED'
FINISHED = (DONE, DIVERGED, TIMEOUT, FAILED)


class Terminated(Exception):
    """Raised from the SIGTERM handler (scancel or Slurm hitting wall time)."""


def raise_terminated(signum, frame):
    raise Terminated()


# ----------------------------------------------------------------------------
# Sample bookkeeping
# ----------------------------------------------------------------------------
def list_samples(samples_dir=SAMPLES_DIR):
    """Sample directories (s001, s002, ...) in run order."""
    names = [d for d in os.listdir(samples_dir)
             if re.fullmatch(r's\d+', d) and os.path.isdir(os.path.join(samples_dir, d))]
    return [os.path.join(samples_dir, d) for d in sorted(names, key=lambda d: int(d[1:]))]


def claim_dir(sample_dir):
    return os.path.join(sample_dir, '.claim')


def read_status(sample_dir):
    path = os.path.join(sample_dir, 'status')
    if not os.path.exists(path):
        return RUNNING if os.path.isdir(claim_dir(sample_dir)) else PENDING
    with open(path) as f:
        return f.readline().strip() or PENDING


def write_status(sample_dir, status, **info):
    """First line is the status; any extra info is written as key=value lines."""
    with open(os.path.join(sample_dir, 'status'), 'w') as f:
        f.write(status + '\n')
        for key, value in info.items():
            f.write('{}={}\n'.format(key, value))


def read_owner(sample_dir):
    """Slurm job id that claimed this sample, or None."""
    try:
        with open(os.path.join(claim_dir(sample_dir), 'owner')) as f:
            return f.readline().split()[0]
    except (OSError, IndexError):
        return None


def try_claim(sample_dir, job_id):
    """Atomically claim a sample.  Returns False if another worker already has it."""
    try:
        os.mkdir(claim_dir(sample_dir))
    except FileExistsError:
        return False
    with open(os.path.join(claim_dir(sample_dir), 'owner'), 'w') as f:
        f.write('{} {} {}\n'.format(job_id, socket.gethostname(), time.strftime('%Y-%m-%d %H:%M:%S')))
    write_status(sample_dir, RUNNING, job=job_id)
    return True


def release_claim(sample_dir):
    """Return a sample to the pending pool (used when a run is cut short by wall time)."""
    status_path = os.path.join(sample_dir, 'status')
    if os.path.exists(status_path):
        os.remove(status_path)
    shutil.rmtree(claim_dir(sample_dir), ignore_errors=True)


def append_log(job_id, sample_dir, status, elapsed):
    """One line per finished attempt in Samples/sweep_log.csv (O_APPEND, so jobs can share it)."""
    new_file = not os.path.exists(SWEEP_LOG)
    with open(SWEEP_LOG, 'a') as f:
        if new_file:
            f.write('timestamp,job_id,sample,status,elapsed_s\n')
        f.write('{},{},{},{},{:.0f}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'), job_id,
                                              os.path.basename(sample_dir), status, elapsed))


# ----------------------------------------------------------------------------
# Slurm
# ----------------------------------------------------------------------------
def parse_slurm_time(text):
    """'[D-]HH:MM:SS', 'MM:SS' or 'SS' -> seconds.  None if unlimited/unparseable."""
    text = text.strip()
    m = re.fullmatch(r'(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?', text)
    if not m:
        return None
    days = int(m.group(1) or 0)
    parts = [int(p) for p in m.group(2, 3, 4) if p is not None]
    while len(parts) < 3:          # 'MM:SS' -> [0, MM, SS]
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def slurm_time_left(job_id):
    """Seconds of wall time left for this job, or None outside Slurm / if unknown."""
    if job_id is None:
        return None
    try:
        out = subprocess.check_output(['squeue', '-h', '-j', str(job_id), '-o', '%L'],
                                      universal_newlines=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return parse_slurm_time(out) if out.strip() else None


def slurm_job_alive(job_id):
    """True if the job is still pending/running (used to spot stale claims)."""
    try:
        out = subprocess.check_output(['squeue', '-h', '-j', str(job_id), '-o', '%T'],
                                      universal_newlines=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False        # squeue errors on job ids it no longer knows about
    return bool(out.strip())


# ----------------------------------------------------------------------------
# Running one simulation
# ----------------------------------------------------------------------------
def histor_has_diverged(histor_path):
    try:
        with open(histor_path, errors='replace') as f:
            return DIVERGENCE_MSG in f.read()
    except OSError:
        return False        # not written yet


def kill_process_group(proc, grace=60):
    """SIGTERM the solver (srun forwards it to the MPI ranks), then SIGKILL if needed."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def run_simulation(sample_dir, cmd, results_dir, time_limit, poll_interval=30):
    """Run the solver in sample_dir and watch it.

    Returns (status, elapsed_seconds) where status is
        DONE      solver exited with code 0 and no divergence warning
        DIVERGED  DIVERGENCE_MSG appeared in <results_dir>/histor.dat (run killed)
        TIMEOUT   run exceeded time_limit seconds (run killed)
        FAILED    solver exited with a non-zero code
    """
    # Start from a clean results directory so a histor.dat left by an earlier,
    # interrupted attempt can't trigger a false divergence.
    results_path = os.path.join(sample_dir, results_dir)
    shutil.rmtree(results_path, ignore_errors=True)
    histor_path = os.path.join(results_path, 'histor.dat')

    start = time.time()
    with open(os.path.join(sample_dir, 'sim.log'), 'w') as log:
        # New session -> the solver gets its own process group that we can kill as a whole.
        proc = subprocess.Popen(cmd, cwd=sample_dir, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            status = None
            while proc.poll() is None:
                time.sleep(poll_interval)
                if histor_has_diverged(histor_path):
                    status = DIVERGED
                elif time.time() - start > time_limit:
                    status = TIMEOUT
                if status is not None:
                    kill_process_group(proc)
                    break
        finally:
            # Also reached on SIGTERM / Ctrl-C: never leave the solver running.
            kill_process_group(proc)
    elapsed = time.time() - start

    if status is None:
        if histor_has_diverged(histor_path):
            status = DIVERGED
        elif proc.returncode != 0:
            status = FAILED
        else:
            status = DONE
    return status, elapsed


def run_postprocess(sample_dir, cmd):
    """Run the post-processing command on a DONE sample (sample_dir is appended to cmd).

    Returns True if it exited with code 0.  Output goes to <sample_dir>/post.log.
    """
    with open(os.path.join(sample_dir, 'post.log'), 'w') as log:
        proc = subprocess.Popen(cmd + [sample_dir], cwd=ROOT_DIR, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait()
        finally:
            kill_process_group(proc, grace=10)
    return proc.returncode == 0
