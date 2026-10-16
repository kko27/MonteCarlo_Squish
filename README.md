# Monte-Carlo Simulations of Spheroid Orientation 

This repository contains the python driver code that will generate monte-carlo simulations to quantify the effect of random orientation of the spheroid squished between 2 parallel plates. The `00_assign_domains.py` script assigns a contiguous skin layer (`RADIUS_THRESHOLD`) that is of a certain specified `PERCENTAGE_COVERAGE`. To simulate the randomness of the active domain layer (i.e., layer 1), this continguous layer should rotate in any orientation. This is sampled from a uniform randonmess: any orientation of the active layer is equally likely. The ultimate readout of this experiment is to find the uncertainty bounds of the random placement of the active layer with respect to the parallel plates. This is because the upper plate is measuring a bulk force (using the `05_compute_reaction_force.py` and `viz_forces.py` scripts) and where the active myocyte layer is changes the result. The force traces will then be averaged/analyzed to represent the uncertainty of individual measurements. This will allow us to provide some confidence interval for real experimental uncertainty.

## Sampling

The active layer is the crescent of `00_assign_domains.py`: cells with centroid radius > `RADIUS_THRESHOLD` (300 µm) inside a cone of half angle `PERCENTAGE_COVERAGE` × 180° (0.6 → ±108°) about the cone axis. The cap is axisymmetric about that axis, so a uniformly random orientation is a cone axis drawn uniformly on the unit sphere: $\cos\theta \sim U[-1, 1]$, $\phi \sim U[0, 2\pi)$. 1000 axes are drawn with a fixed seed (`SEED` in the generator), so the set can be regenerated exactly. Only `DOMAIN_ID` changes between samples; the fiber fields, material parameters, activation, and plate motion are the baseline ones for every sample.

## File Architecture

- `Baseline/`: `solver_tanH.xml`, the activation and plate-step waveforms, and `mesh_scaled/` (shared by every sample)
- `Generator/generate_orientations.py`: builds `Samples/s0001 ... s1000` (each with `solver.xml` and its own `mesh-complete.domains.vtu`) and `Samples/samples.txt` (sample, axis_x, axis_y, axis_z, theta_deg, phi_deg, n_domain1_cells)
- `driver.py` / `run_functions.py`: sweep worker. Runs pending samples one after another, records the outcome of each in `Samples/sNNNN/status` and `Samples/sweep_log.csv`, and post-processes every DONE sample
- `submit_sweep.sbatch`: Sherlock job script that runs one worker
- `PostProcessor/postprocess_sample.sh`: called by the worker after each DONE run; loads the python modules/venv and runs `compute_reaction_forces.py --delete-vtu` on that sample
- `PostProcessor/compute_reaction_forces.py`: platen force from horizontal cross-sections of the deformed sphere (`section_force.csv`), same calculation as `05_compute_reaction_force.py`; with `--delete-vtu` removes `result_*.vtu` once the CSV is written
- `PostProcessor/compute_statistics.py`: Monte-Carlo statistics of the platen |F_z| over all samples
- `submit_postprocess.sbatch`: catches up on any DONE sample without a CSV, then runs the statistics

`00_assign_domains.py`, `05_compute_reaction_force.py` and `viz_forces.py` are the original single-run scripts and are not used by the sweep.

## Running on Sherlock

Put the repository on `$SCRATCH`. Each run writes ~520 MB of VTUs while it runs; they are deleted as soon as the reaction force is extracted, so the finished sweep needs only ~2 GB.

```bash
python3 Generator/generate_orientations.py   # once (needs numpy + pyvista, ~30 s)
sbatch --array=1-8 submit_sweep.sbatch       # 8 parallel 24 h workers
python3 driver.py status                     # progress (-v lists every sample)
```

Each worker claims the next unclaimed sample (an atomic `.claim` directory in the sample folder), so any number of workers can share `Samples/`, and resubmitting later only picks up what is left. A run is stopped and the worker moves on to the next sample when:

1. `histor.dat` contains `WARNING: The linear system solution has not converged` → `DIVERGED`
2. it has run for more than 2 hours (`--time-limit`) → `TIMEOUT`

A worker won't start a new run with less than 40 minutes of wall time left, and keeps 20 minutes (`--margin`) for post-processing. A run that is cut off by the job's wall time (or by `scancel`) goes back to `PENDING` rather than being marked as failed. A solver exiting with a non-zero code is marked `FAILED`. If a sample's post-processing fails (see `Samples/sNNNN/post.log`) it stays `DONE` with its VTUs, and `submit_postprocess.sbatch` picks it up.

```bash
python3 driver.py reset --stale                 # free samples whose job died (e.g. node failure)
python3 driver.py reset --status TIMEOUT FAILED # queue those samples again
```

Once the sweep (or part of it) is done:

```bash
sbatch submit_postprocess.sbatch                         # whole trace
sbatch submit_postprocess.sbatch --tmin 5 --tmax 6.25    # statistics on a time window
```

Outputs in `Samples/`:

- `force_statistics.csv`: per time point, n, mean, std and 2.5/50/97.5 percentiles of the platen |F_z| (µN)
- `sample_metrics.csv`: per sample, orientation plus peak and mean |F_z| in the window
- `mc_force_band.png`: mean force trace with the 2.5–97.5 percentile band
- `mc_peak_hist.png`: distribution of the peak force
- `mc_peak_vs_theta.png`: peak force vs cone-axis polar angle (shows how much of the spread θ alone explains)
- `mc_convergence.png`: running mean and 95% CI of the peak vs number of samples (checks that 1000 samples is enough)


