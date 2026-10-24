"""36_tetrad_r_reconstruction_accuracy.py -- how accurate is r(t) reconstructed from the 6-channel velocity tensor?

Benchmark: the raw file also stores POSITIONS, so the exact r(t) = |X1(t)-X2(t)| (minimum image) is known.
Compares two quadratures for the relative displacement (both use dts = 2.25e-3):
  left    : dr(t) = sum_{m<=t} (V1-V2)(m) dts
  trapz   : dr(t) = sum_{m} 0.5*[(V1-V2)(m-1)+(V1-V2)(m)] dts
Reports (a) pointwise errors and (b) the pair-dispersion statistics at checkpoints:
  <r^2>, <r^4>, flatness F = <r^4>/<r^2>^2  -- positions vs reconstruction.
Uses 120 pairs sampled uniformly across the index range (i.e. across the 32x32 pencil columns).
"""
import os

import numpy as np
import h5py

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "Data", "full_traj_tracers.h5")
L = 2.0 * np.pi
DTS = 2.25e-3
CHK = [10, 50, 100, 200, 400, 800, 1200, 1999]
NP = 120


def mi(d):
    return d - L * np.round(d / L)


with h5py.File(RAW, "r") as f:
    d = f["traj3d"]
    ks = (np.linspace(0, 81919, NP)).astype(int)
    ks = ks[:-1]                       # keep 119 < 81920
    rows = np.vstack([d[4 * k:4 * k + 4, :, 0:6].astype("float64") for k in ks])
print("loaded", rows.shape, flush=True)

n = len(ks)
r_pos = np.zeros((n, 2000)); r_l = np.zeros((n, 2000)); r_t = np.zeros((n, 2000))
for j in range(n):
    q = rows[4 * j:4 * j + 4]
    X1, V1 = q[0, :, 0:3], q[0, :, 3:6]
    X2, V2 = q[1, :, 0:3], q[1, :, 3:6]
    r0 = mi(X1[0] - X2[0])
    r_pos[j] = np.linalg.norm(mi(X1 - X2), axis=1)
    dV = V1 - V2                     # PITFALL: do NOT minimum-image a velocity difference (|dV| can exceed pi)
    dr_l = np.cumsum(dV * DTS, axis=0)
    r_l[j] = np.linalg.norm(mi(r0[None, :] + dr_l), axis=1)      # fold the DISPLACEMENT only
    dV_t = 0.5 * (dV[1:] + dV[:-1]) * DTS
    dr_t = np.cumsum(dV_t, axis=0)
    r_t[j, 1:] = np.linalg.norm(mi(r0[None, :] + dr_t), axis=1)
    r_t[j, 0] = r_pos[j, 0]

print("== (a) pointwise error vs exact r(t) ==", flush=True)
for j in [0, 1, 2]:
    e_l = np.abs(r_l[j] - r_pos[j]) / np.maximum(r_pos[j], 1e-9)
    e_t = np.abs(r_t[j] - r_pos[j]) / np.maximum(r_pos[j], 1e-9)
    print(f"  pair {j}: rel.err mean left={e_l.mean():.2%} trapz={e_t.mean():.2%} | "
          f"abs.err mean left={np.abs(r_l[j]-r_pos[j]).mean():.3e} trapz={np.abs(r_t[j]-r_pos[j]).mean():.3e}", flush=True)

print("== (b) pair-dispersion statistics: exact vs reconstructed ==", flush=True)
print("    t |   <r^2>_pos  <r^2>_left  <r^2>_trapz |   F_pos   F_left  F_trapz", flush=True)
for t in CHK:
    a, bl, bt = r_pos[:, t], r_l[:, t], r_t[:, t]
    m2p, m2l, m2t = (a ** 2).mean(), (bl ** 2).mean(), (bt ** 2).mean()
    Fp = (a ** 4).mean() / m2p ** 2
    Fl = (bl ** 4).mean() / m2l ** 2
    Ft = (bt ** 4).mean() / m2t ** 2
    print(f"  {t:5d} | {m2p:9.5f} {m2l:10.5f} {m2t:11.5f} | {Fp:7.3f} {Fl:8.3f} {Ft:8.3f}", flush=True)
