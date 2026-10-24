# -*- coding: utf-8 -*-
"""64_order_evidence_data.py --- C02 data stage: how strongly consecutive batches are correlated

Rescues the measurement that was previously run from a scratch file
(~/.hermes/cache/scratch/order_probe.py, 2026-10-18) and whose results are recorded in
RESULTS_ANALYSIS.md section 7.19, so that the numbers behind the discussion document are
reproducible from the project script set.

Question: the training dataset is stored as successive 32x32 spatial pencil columns, and
training reads it in storage order (shuffle defaults to False). How much slower does the
batch-mean series fluctuate, and how correlated is it, compared with the four-window read
of a four-GPU run?

Calibers (identical to section 7.19, so the numbers are directly comparable):
  data      : Data/Lagr_u1c_diffusion.h5, first 131,072 trajectories, every 5th time step
  our read  : 256 consecutive trajectories per batch  (our single-GPU run)
  official  : four windows of 64 consecutive trajectories, quarters of the read region
              apart, averaged (the four-GPU distributed read)
  shuffled  : random permutation of the 131,072 trajectories read in batches of 256
              (this gives the pure-noise floor of the batch-mean scatter)
  rho1      : lag-1 autocorrelation of the batch-mean series, mean removed

The input and output paths follow LTS_ROOT (the directory that holds results/ and Data/), e.g.
  LTS_ROOT=/path/to/PostP python3 64_order_evidence_data.py

Product: results/order_evidence.npz
"""
import os

import h5py
import numpy as np

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
N = 131072


def rho1(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    return float((x[:-1] * x[1:]).sum() / (x * x).sum())


path = os.path.join(DATA, "Lagr_u1c_diffusion.h5")
with h5py.File(path, "r") as f:
    X = np.asarray(f["train"][0:N, ::5, 0], dtype=np.float64)   # (N, K) memory-light read
print("read %s -> X%s  (global mean %.6f)" % (path, X.shape, X.mean()), flush=True)

K = X.shape[1]
wb = X.reshape(N // 256, 256, K).mean(axis=(1, 2))              # our read
sb = X.reshape(N // 64, 64, K).mean(axis=(1, 2))                # 64-window means
Kq = len(sb) // 4
ob = (sb[0:Kq] + sb[Kq:2 * Kq] + sb[2 * Kq:3 * Kq] + sb[3 * Kq:4 * Kq]) / 4.0   # official style
rng = np.random.default_rng(0)
Xr = X[rng.permutation(N)]
wn_series = Xr.reshape(N // 256, 256, K).mean(axis=(1, 2))       # shuffled floor

res = {
    "wb": wb, "ob": ob, "wn_series": wn_series,
    "std_wb": np.float64(wb.std(ddof=1)),
    "std_ob": np.float64(ob.std(ddof=1)),
    "std_wn": np.float64(wn_series.std(ddof=1)),
    "rho1_wb": np.float64(rho1(wb)),
    "rho1_ob": np.float64(rho1(ob)),
    "rho1_wn": np.float64(rho1(wn_series)),
    "global_mean": np.float64(X.mean()),
    "n_read": np.int64(N),
    "stride": np.int64(5),
}
print("[1] batch-mean scatter (std):  ours %.3e | official style %.3e | shuffled floor %.3e"
      % (res["std_wb"], res["std_ob"], res["std_wn"]), flush=True)
print("    ratio ours/official = %.2f x" % (res["std_wb"] / res["std_ob"]), flush=True)
print("[2] lag-1 autocorrelation rho1: ours %.3f | official style %.3f | shuffled %.3f"
      % (res["rho1_wb"], res["rho1_ob"], res["rho1_wn"]), flush=True)
st_w = np.sqrt(max(wb.var(ddof=1) - res["std_wn"] ** 2, 0.0))
st_o = np.sqrt(max(ob.var(ddof=1) - res["std_wn"] ** 2 / 4.0, 0.0))
res["structural_wb"] = np.float64(st_w)
res["structural_ob"] = np.float64(st_o)
# RESULTS_ANALYSIS.md section 7.19 quotes the structural component relative to the global
# mean of the *full* training set (0.043136, section 7.18.3). Print both denominators so the
# two calibers cannot be confused again: the window mean is not the full-set mean.
MEAN_FULL = 0.043136
res["mean_full"] = np.float64(MEAN_FULL)
res["mean_window"] = np.float64(X.mean())
print("[3] structural component: ours %.3e (%.0f%% of its std, %.1f%% of the full-set mean "
      "%.6f / %.1f%% of the window mean %.6f) | official style %.3e (%.1f%% / %.1f%%)"
      % (st_w, 100 * st_w / wb.std(ddof=1), 100 * st_w / MEAN_FULL, MEAN_FULL, 100 * st_w / X.mean(),
         X.mean(), st_o, 100 * st_o / MEAN_FULL, 100 * st_o / X.mean()), flush=True)
print("    note: the window mean differs from the full-set mean by %+.1f%% -- the first "
      "block of the stored order is not representative of the whole set."
      % (100 * (X.mean() / MEAN_FULL - 1.0)), flush=True)

np.savez(os.path.join(RESULTS, "order_evidence.npz"), **res)
print("saved", os.path.join(RESULTS, "order_evidence.npz"), flush=True)
