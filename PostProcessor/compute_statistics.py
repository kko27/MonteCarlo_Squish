"""Monte-Carlo statistics of the platen force over the random active-layer orientations.

Reads Samples/samples.txt and every Samples/sNNNN/<n>-procs/net_reaction_force.csv,
and summarises |F_z| on the top plate, the bottom plate, and their sum
(as in viz_forces.py).  Forces are reported in uN.

Usage (from the repository root):
    python3 PostProcessor/compute_statistics.py                  # whole trace
    python3 PostProcessor/compute_statistics.py --tmin 5 --tmax 6.25

Writes:
    Samples/force_statistics.csv   per time point: n, mean, std, 2.5/50/97.5 percentiles
    Samples/sample_metrics.csv     per sample: orientation + peak / mean |F_z| in [tmin, tmax]
    Samples/mc_force_band.png      mean trace with 95% band (one panel per measure)
    Samples/mc_peak_hist.png       distribution of the peak |F_z|
    Samples/mc_peak_vs_theta.png   peak |F_z| vs cone-axis polar angle
    Samples/mc_convergence.png     running mean and 95% CI of the peak vs number of samples
"""

import argparse
import glob
import os

import numpy as np
from matplotlib import pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "Samples")
CSV_NAME = "net_reaction_force.csv"

MEASURES = ["top", "bottom", "total"]
COLOR = "#2a78d6"            # line / markers
BAND_COLOR = "#9ec5f4"       # 95% band
PCTS = [2.5, 50.0, 97.5]


def read_samples():
    """samples.txt -> dict name -> row (dict of floats)."""
    with open(os.path.join(SAMPLES_DIR, "samples.txt")) as f:
        keys = f.readline().split()
        rows = [line.split() for line in f if line.strip()]
    return {r[0]: dict(zip(keys[1:], map(float, r[1:]))) for r in rows}


def load_forces(name):
    """|F_z| in uN for top, bottom, total, plus time; None if the CSV is missing."""
    paths = glob.glob(os.path.join(SAMPLES_DIR, name, "*-procs", CSV_NAME))
    if not paths:
        return None
    data = np.loadtxt(paths[0], delimiter=",", skiprows=1, ndmin=2)
    top = np.abs(data[:, 3]) / 1e6
    bot = np.abs(data[:, 7]) / 1e6
    return data[:, 0], {"top": top, "bottom": bot, "total": top + bot}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tmin", type=float, default=-np.inf, help="start of the analysis window (s)")
    parser.add_argument("--tmax", type=float, default=np.inf, help="end of the analysis window (s)")
    args = parser.parse_args()

    samples = read_samples()
    names, traces, time = [], {m: [] for m in MEASURES}, None
    for name in sorted(samples):
        loaded = load_forces(name)
        if loaded is None:
            continue
        t, f = loaded
        if time is None:
            time = t
        elif len(t) != len(time) or not np.allclose(t, time):
            print(f"skipping {name}: time points differ from the first sample ({len(t)} vs {len(time)})")
            continue
        names.append(name)
        for m in MEASURES:
            traces[m].append(f[m])
    if not names:
        raise SystemExit(f"no {CSV_NAME} found under {SAMPLES_DIR}")
    traces = {m: np.asarray(v) for m, v in traces.items()}      # (n_samples, n_times)
    n = len(names)
    print(f"{n} samples with reaction forces")

    window = (time >= args.tmin) & (time <= args.tmax)
    if not window.any():
        raise SystemExit(f"no time points in [{args.tmin}, {args.tmax}]")

    # --- per time point statistics ----------------------------------------------
    header, cols = ["time_s", "n"], [time, np.full(len(time), n)]
    for m in MEASURES:
        x = traces[m]
        p = np.percentile(x, PCTS, axis=0)
        header += [f"{m}_mean_uN", f"{m}_std_uN"] + [f"{m}_p{q:g}_uN" for q in PCTS]
        cols += [x.mean(axis=0), x.std(axis=0, ddof=1) if n > 1 else np.zeros(len(time))] + list(p)
    out = os.path.join(SAMPLES_DIR, "force_statistics.csv")
    np.savetxt(out, np.column_stack(cols), delimiter=",", fmt="%.6g", header=",".join(header), comments="")
    print(f"wrote {out}")

    # --- per sample metrics --------------------------------------------------------
    peak = {m: traces[m][:, window].max(axis=1) for m in MEASURES}
    mean = {m: traces[m][:, window].mean(axis=1) for m in MEASURES}
    theta = np.array([samples[s]["theta_deg"] for s in names])
    out = os.path.join(SAMPLES_DIR, "sample_metrics.csv")
    with open(out, "w") as f:
        f.write("sample,theta_deg,phi_deg,axis_z," +
                ",".join(f"{m}_peak_uN,{m}_mean_uN" for m in MEASURES) + "\n")
        for i, s in enumerate(names):
            row = samples[s]
            f.write(f"{s},{row['theta_deg']:.4f},{row['phi_deg']:.4f},{row['axis_z']:.6f}," +
                    ",".join(f"{peak[m][i]:.6g},{mean[m][i]:.6g}" for m in MEASURES) + "\n")
    print(f"wrote {out}")

    print(f"\npeak |F_z| in t = [{time[window][0]:.2f}, {time[window][-1]:.2f}] s  (uN)")
    for m in MEASURES:
        lo, med, hi = np.percentile(peak[m], PCTS)
        sem = peak[m].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        print(f"  {m:6s} mean {peak[m].mean():.4g} +/- {1.96 * sem:.2g} (95% CI of mean)   "
              f"median {med:.4g}   95% range [{lo:.4g}, {hi:.4g}]   CV {peak[m].std() / peak[m].mean() * 100:.1f}%")

    # --- plots -------------------------------------------------------------------------
    tw = time[window]
    fig, axes = plt.subplots(len(MEASURES), 1, figsize=(8, 9), sharex=True)
    for ax, m in zip(axes, MEASURES):
        x = traces[m][:, window]
        lo, med, hi = np.percentile(x, PCTS, axis=0)
        ax.fill_between(tw, lo, hi, color=BAND_COLOR, linewidth=0, label="2.5–97.5 percentile")
        ax.plot(tw, x.mean(axis=0), color=COLOR, linewidth=2, label="mean")
        ax.set_ylabel(f"{m} |$F_z$| (µN)", fontsize=12)
        ax.grid(True, color="#e0e0e0", linewidth=0.5)
    axes[0].legend(frameon=False)
    axes[0].set_title(f"Platen force over {n} random active-layer orientations")
    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(SAMPLES_DIR, "mc_force_band.png"), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(MEASURES), figsize=(12, 3.8))
    for ax, m in zip(axes, MEASURES):
        ax.hist(peak[m], bins=min(40, max(5, n // 10)), color=COLOR, edgecolor="white", linewidth=1)
        for q in np.percentile(peak[m], [2.5, 97.5]):
            ax.axvline(q, color="#52514e", linestyle="--", linewidth=1)
        ax.set_xlabel(f"peak {m} |$F_z$| (µN)")
    axes[0].set_ylabel("samples")
    fig.tight_layout()
    fig.savefig(os.path.join(SAMPLES_DIR, "mc_peak_hist.png"), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(MEASURES), figsize=(12, 3.8), sharex=True)
    for ax, m in zip(axes, MEASURES):
        ax.scatter(theta, peak[m], s=10, color=COLOR, alpha=0.6, linewidths=0)
        ax.set_xlabel("cone-axis polar angle θ (deg)")
        ax.set_ylabel(f"peak {m} |$F_z$| (µN)")
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.grid(True, color="#e0e0e0", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(SAMPLES_DIR, "mc_peak_vs_theta.png"), dpi=300)
    plt.close(fig)

    k = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(12, 3.8), sharex=True)
    for ax, m in zip(axes, MEASURES):
        run_mean = np.cumsum(peak[m]) / k
        run_sq = np.cumsum(peak[m] ** 2) / k
        run_sd = np.sqrt(np.maximum(run_sq - run_mean ** 2, 0) * k / np.maximum(k - 1, 1))
        half = 1.96 * run_sd / np.sqrt(k)
        ax.fill_between(k, run_mean - half, run_mean + half, color=BAND_COLOR, linewidth=0)
        ax.plot(k, run_mean, color=COLOR, linewidth=2)
        ax.set_xlabel("number of samples")
        ax.set_ylabel(f"running mean peak {m} (µN)")
        ax.grid(True, color="#e0e0e0", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(SAMPLES_DIR, "mc_convergence.png"), dpi=300)
    plt.close(fig)
    print(f"wrote plots to {SAMPLES_DIR}/mc_*.png")


if __name__ == "__main__":
    main()
