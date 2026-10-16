# For Each Sample inside Samples, open the 24-procs directory and read in the
# section_force.csv text file. Open the F_cut50pct_pN column (platen force), save it
# to an array or list for further processing
#
# After reading in all the sample data, find the mean of the F_cut50pct_pN
# for every time point (from all samples). Also find the 1SD (standard deviation)
# for each time point.

import glob
import os

import numpy as np
from matplotlib import pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
CSV_PATTERN = os.path.join(SAMPLES_DIR, "s*", "24-procs", "section_force.csv")

COLUMN = "F_cut50pct_pN"
PN_TO_UN = 1e-6                 # pN -> µN

COLOR = "#2a78d6"               # mean line (same as compute_statistics.py)
BAND_COLOR = "#9ec5f4"          # ±1 SD band


def read_csv(path):
    """Return (time_s, column) from one section_force.csv."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, header.index("time_s")], data[:, header.index(COLUMN)]


# Read every finished sample
csv_files = sorted(glob.glob(CSV_PATTERN))
if not csv_files:
    raise SystemExit(f"No section_force.csv found under {SAMPLES_DIR}")

time_s = None
forces = []                     # one |Fz| trace (µN) per sample
names = []
for path in csv_files:
    name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    t, fz = read_csv(path)
    if time_s is None:
        time_s = t
    elif t.shape != time_s.shape or not np.allclose(t, time_s):
        print(f"Skipping {name}: time points do not match {names[0]}")
        continue
    forces.append(np.abs(fz) * PN_TO_UN)
    names.append(name)

forces = np.array(forces)       # shape (n_samples, n_times)
n_samples = forces.shape[0]
print(f"Read {n_samples} samples, {len(time_s)} time points each")

# Mean and 1 SD across samples at every time point
mean = forces.mean(axis=0)
std = forces.std(axis=0, ddof=1) if n_samples > 1 else np.zeros_like(mean)

np.savetxt(os.path.join(SAMPLES_DIR, "top_Fz_mean_std.csv"),
           np.column_stack([time_s, np.full_like(time_s, n_samples), mean, std]),
           delimiter=",", header="time_s,n,mean_abs_Fz_uN,std_abs_Fz_uN", comments="")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(time_s, mean - 2*std, mean + 2*std, color=BAND_COLOR, linewidth=0,
                label="±2 SD")
ax.plot(time_s, mean, color=COLOR, linewidth=1, label="Mean")
ax.set_xlabel("Time (s)", fontsize=14)
ax.set_ylabel("Platen |F$_z$| (µN)", fontsize=14)
ax.set_title(f"Platen force, n = {n_samples} samples", fontsize=14)
ax.legend(fontsize=12, frameon=False)
ax.grid(True, color="#e0e0e0", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(SAMPLES_DIR, "top_Fz_mean_std.png"), dpi=450)
plt.close(fig)
