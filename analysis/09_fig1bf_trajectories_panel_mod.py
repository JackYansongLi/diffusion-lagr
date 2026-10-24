# -*- coding: utf-8 -*-
"""
09_fig1bf_trajectories_panel_mod.py -- part A3-7: reproduction of the trajectory panels of Fig. 1b-f of the paper

Panel layout of Fig. 1 (checked against the rendered page 2 of the PDF):
  b/c/d : one velocity component of DM-1c against time (0..200 tau_eta), with the windows at tau/tau_eta = 100 / 5 / 1 marked by a red segment
  e     : the three velocity components against time, DNS above and DM-3c below (source data t,vx,vy,vz of Fig. 1e)
  f     : 3D trajectory coloured by |a|, DNS next to DM-3c (source data x,y,z,acc)

Data sources:
  DM-1c : Data/tmp/dm1c_358400x2000x1_f32.npy (u1c, scalar min/max)
  DM-3c : Data/tmp/dm3c_358400x2000x3_f32.npy (u3c, per-component min/max)
  DNS   : Data/Lagr_u3c_diffusion.h5 (the full u3c set)
Positions are integrated from the velocities (the source files hold real positions, only velocities are local): x(t+1) = x(t) + v(t)*dts, dts = 2.25e-3 code units

Usage: python 09_fig1bf_trajectories_panel_mod.py [--traj-idx 137] [--n-plot 2000]
Output: results/figs/A9_fig1bcd_traj.png, A9_fig1ef_multicomp.png
"""
import argparse
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figs")
os.makedirs(FIGS, exist_ok=True)

DTS = 2.25e-3          # storage interval in code units (= 0.1 τη, τη=2.1e-2)
ETA = 4.2e-3           # Kolmogorov length in code units (Table 1)


def denorm_npy(path, mn, mx, idx, k):
    a = np.load(path, mmap_mode="r")
    u = np.asarray(a[idx:idx + k], dtype=np.float32)
    mn = np.asarray(mn, dtype=np.float32).reshape(1, 1, -1)
    mx = np.asarray(mx, dtype=np.float32).reshape(1, 1, -1)
    return (u + 1.0) * ((mx - mn) / 2.0) + mn


def denorm_h5(path, idx, k):
    with h5py.File(path, "r") as f:
        mn = f["min"][()]
        mx = f["max"][()]
        u = f["train"][idx:idx + k].astype(np.float32)
    mn = np.asarray(mn, dtype=np.float32).reshape(1, 1, -1)
    mx = np.asarray(mx, dtype=np.float32).reshape(1, 1, -1)
    return (u + 1.0) * ((mx - mn) / 2.0) + mn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-idx", type=int, default=137)
    ap.add_argument("--n-plot", type=int, default=2000)
    args = ap.parse_args()
    i0, K = args.traj_idx, args.n_plot
    t = np.arange(K) * 0.1                       # τ/τη

    # ---------- Fig. 1b/c/d: a single velocity component of DM-1c ----------
    with h5py.File(os.path.join(DATA, "Lagr_u1c_diffusion.h5"), "r") as f:
        mn1 = float(f["min"][()])
        mx1 = float(f["max"][()])
    d1c = denorm_npy(os.path.join(DATA, "tmp", "dm1c_358400x2000x1_f32.npy"),
                     [mn1], [mx1], i0, 3)[:, :K, 0]           # three trajectories
    sig = d1c.std(axis=1, keepdims=True)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    for ax, v, lag, lab in zip(axes, d1c, [1000, 50, 10], ["100", "5", "1"]):
        ax.plot(t, v / sig[0], color="k", lw=0.8)
        # window marker: a red segment of length = lag in the middle of the trace
        c0 = K // 2
        ax.plot(t[c0:c0 + lag], (v / sig[0])[c0:c0 + lag], color="tab:red", lw=2.5)
        ax.set_ylabel("$V_i/\\sigma(V_i)$")
        ax.text(0.02, 0.82, "$\\tau/\\tau_\\eta$ = %s" % lab, transform=ax.transAxes,
                fontsize=10, color="tab:red")
        ax.set_ylim(-5, 5)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("$\\tau/\\tau_\\eta$")
    axes[0].set_title("(a) Fig.1b-d reproduction: DM-1c velocity segments (windows $\\tau/\\tau_\\eta$=100/5/1)")
    fig.tight_layout()
    p = os.path.join(FIGS, "A9_fig1bcd_traj_mod.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved", p)

    # ---------- Fig. 1e + 1f: DNS against DM-3c, three components ----------
    with h5py.File(os.path.join(DATA, "Lagr_u3c_diffusion.h5"), "r") as f:
        mn3 = f["min"][()]
        mx3 = f["max"][()]
    vdm = denorm_npy(os.path.join(DATA, "tmp", "dm3c_358400x2000x3_f32.npy"),
                     mn3, mx3, i0, 1)[0, :K, :]
    vdns = denorm_h5(os.path.join(DATA, "Lagr_u3c_diffusion.h5"), i0, 1)[0, :K, :]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for ax, v, name in zip(axes, [vdns, vdm], ["DNS", "DM-3c"]):
        for j, (c, comp) in enumerate(zip(["tab:blue", "tab:orange", "tab:green"], "xyz")):
            ax.plot(t, v[:, j] / v[:, j].std(), color=c, lw=0.7, label="$V_%s$" % comp)
        ax.set_ylabel("$V_i/\\sigma(V_i)$")
        ax.set_ylim(-5, 5)
        ax.grid(alpha=0.25)
        ax.text(0.01, 0.80, name, transform=ax.transAxes, fontsize=11, weight="bold")
        ax.legend(ncol=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("$\\tau/\\tau_\\eta$")
    axes[0].set_title("(b) Fig.1e reproduction: three-velocity-component series")
    fig.tight_layout()
    p = os.path.join(FIGS, "A9_fig1ef_multicomp_mod.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved", p)

    # 3D trajectory (integrated from the velocities) coloured by |a|; both panels share one colour range, otherwise they cannot be compared
    trajs = []
    for v, name in [(vdns, "DNS"), (vdm, "DM-3c")]:
        pos = np.cumsum(v, axis=0) * DTS / ETA             # -> eta units
        acc = np.zeros(len(v))
        acc[1:] = np.linalg.norm(np.diff(v, axis=0), axis=1) / DTS
        trajs.append((pos, acc, name))
    vmax = max(float(np.quantile(a, 0.999)) for _, a, _ in trajs)
    fig = plt.figure(figsize=(10.5, 4.6))
    for k, (pos, acc, name) in enumerate(trajs):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=acc, s=1.5, cmap="jet",
                   vmin=0, vmax=vmax)
        ax.set_title("%s (color: |a|, 0..%.1f)" % (name, vmax), fontsize=10)
        ax.set_xlabel("x/η", fontsize=8)
        ax.set_ylabel("y/η", fontsize=8)
        ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="|a| (code units)")
    fig.suptitle("Fig.1f reproduction: 3D trajectories (positions integrated from v)")
    fig.tight_layout()
    p = os.path.join(FIGS, "A9_fig1f_3dtraj.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
