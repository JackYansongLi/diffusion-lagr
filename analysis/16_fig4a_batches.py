"""Error-bar statistics for Fig. 4a: the dataset is cut into N independent trajectory blocks and S(p),
p = 2, 4, 6 is computed per lag,
with the min/max across blocks taken as the lower and upper edge of the bar, as in the caption of
Fig. 4 of the paper.

Caption of the paper: "The error bars represent the minimum and maximum values obtained for each
measure
by dividing the entire dataset used to compute the statistics into ten different independent
batches of smaller size.'

Output: results/fig4a_batches_<ds>.npz
  lags(1024), S2b/S4b/S6b (nb,1024) the per-block mean, S2_lo/hi and the like (1024,) the block min/max,
  n_traj, n_batch, batch_size, dataset
Usage: python 16_fig4a_batches.py --dataset dns|dm3c|dm1c [--batches 10] [--workers 10]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

LAGS = np.arange(1, 1025)          # stored step; tau/tau_eta = 0.1*lag
BLOCK = 2048                       # trajectories per worker block (about 49 MB each, f32 x 2000 x 3)


def _denorm_params(name):
    """The min/max used to de-normalise: read from the HDF5 attributes when possible, otherwise from
    structfuncs_<name>_full.npz
        (whose min/max keys hold the same constants; see 02_structure_functions.py)."""
    try:
        import h5py
    except ImportError:
        z = np.load(os.path.join(RESULTS, "structfuncs_%s_full.npz" % name))
        return z["min"], z["max"]
    fn = "Lagr_u3c_diffusion.h5" if name in ("dns", "dm3c") else "Lagr_u1c_diffusion.h5"
    with h5py.File(os.path.join(DATA, fn), "r") as f:
        return f["min"][()], f["max"][()]


def get_dataset(name):
    if name == "dns":
        import h5py
        p = os.path.join(DATA, "Lagr_u3c_diffusion.h5")
        with h5py.File(p, "r") as f:
            mn, mx = f["min"][()], f["max"][()]
            n = f["train"].shape[0]
        return dict(kind="h5", path=p, min=mn, max=mx, n_traj=n, C=3)
    if name == "dm3c":
        mn, mx = _denorm_params("dm3c")
        return dict(kind="npy", path=os.path.join(DATA, "tmp", "dm3c_358400x2000x3_f32.npy"),
                    min=mn, max=mx, n_traj=358400, C=3)
    if name == "dm1c":
        mn, mx = _denorm_params("dm1c")
        return dict(kind="npy", path=os.path.join(DATA, "tmp", "dm1c_358400x2000x1_f32.npy"),
                    min=float(np.asarray(mn).ravel()[0]), max=float(np.asarray(mx).ravel()[0]),
                    n_traj=358400, C=1)
    if name == "dm1c_repro":
        # Our own DM-1c sampling (self-trained DM-1c, ema_0.9999_250000.pt, 28 runs x 12800).
        # Same u1c physical domain as the official dm1c -> reuse its de-normalisation scalars.
        mn, mx = _denorm_params("dm1c")
        return dict(kind="npy", path=os.path.join(DATA, "tmp", "dm1c_repro_358400x2000x1_f32.npy"),
                    min=float(np.asarray(mn).ravel()[0]), max=float(np.asarray(mx).ravel()[0]),
                    n_traj=358400, C=1)
    raise ValueError(name)


def worker(task):
    """(batch_id, ds, i0, i1) -> (batch_id, S(nl,3,C), cnt(nl,C)), where S holds the moment sums of
    p = 2, 4, 6 and keeps the component axis
        so that one pass serves both batching conventions."""
    bid, ds, i0, i1 = task
    mn = np.asarray(ds["min"], dtype=np.float32).reshape(1, 1, -1)
    mx = np.asarray(ds["max"], dtype=np.float32).reshape(1, 1, -1)
    if ds["kind"] == "h5":
        import h5py
        with h5py.File(ds["path"], "r") as f:
            u = f["train"][i0:i1].astype(np.float32)
    else:
        a = np.load(ds["path"], mmap_mode="r")
        u = np.asarray(a[i0:i1], dtype=np.float32)
    scale = (mx - mn) / 2.0
    v = (u + 1.0) * scale + mn                      # back to physical units
    del u, scale
    B, K, C = v.shape
    S = np.zeros((len(LAGS), 3, C), dtype=np.float64)          # the component axis is kept so both conventions come out of one pass
    cnt = np.zeros((len(LAGS), C), dtype=np.float64)
    for li, tau in enumerate(LAGS):
        d = v[:, tau:, :] - v[:, :K - tau, :]
        m2 = d * d                                             # δ²
        del d
        S[li, 0] += np.sum(m2, axis=(0, 1), dtype=np.float64)  # p=2, per component
        m4 = m2 * m2                                           # δ⁴
        S[li, 1] += np.sum(m4, axis=(0, 1), dtype=np.float64)  # p=4
        m6 = m4 * m2                                           # δ⁶
        S[li, 2] += np.sum(m6, axis=(0, 1), dtype=np.float64)  # p=6
        del m2, m4, m6
        cnt[li] += B * (K - tau)
    del v
    return bid, S, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dns", "dm3c", "dm1c", "dm1c_repro"])
    ap.add_argument("--batches", type=int, default=10, help="number of independent blocks (the paper uses 10)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--n-traj", type=int, default=0, help="use only the first N trajectories (0 = all; for smoke tests)")
    ap.add_argument("--out-tag", default="", help="output-name suffix, so a smoke test does not overwrite the real npz")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    if args.n_traj:
        ds["n_traj"] = args.n_traj
    n = ds["n_traj"]
    bs = n // args.batches
    print(f"[16_fig4a_batches] dataset={args.dataset} n_traj={n} batches={args.batches} "
          f"batch_size={bs} workers={args.workers} lags={LAGS[0]}..{LAGS[-1]}", flush=True)

    # the work items are the trajectory blocks of every batch
    tasks = []
    for b in range(args.batches):
        j0 = b * bs
        j1 = (b + 1) * bs if b < args.batches - 1 else n
        for i0 in range(j0, j1, args.block):
            tasks.append((b, ds, i0, min(i0 + args.block, j1)))
    print(f"  tasks={len(tasks)} blocks", flush=True)

    C = ds["C"]
    Ssum = np.zeros((args.batches, len(LAGS), 3, C))
    cnt = np.zeros((args.batches, len(LAGS), C))
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for bid, S, c in pool.imap_unordered(worker, tasks, chunksize=1):
            Ssum[bid] += S
            cnt[bid] += c
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} blocks ({time.time()-t0:.0f}s)", flush=True)

    # convention A (the earlier one): 10 trajectory blocks with the three components merged
    Sp = Ssum.sum(axis=3) / cnt.sum(axis=2)[:, :, None]        # (nb, nl, 3)
    lo = Sp.min(axis=0)
    hi = Sp.max(axis=0)
    # convention B (the paper's, captions of Figs. 4 and 5): components kept apart, i.e. nb x C
    # component blocks (30 for the DNS and DM-3c, 10 for DM-1c)
    # transpose before reshaping: reshaping (nb, nl, 3, C) directly reorders by memory layout and
    # yields garbage
    Vc = (Ssum / cnt[:, :, None, :]).transpose(0, 3, 1, 2).reshape(args.batches * C, len(LAGS), 3)
    lo30 = Vc.min(axis=0)
    hi30 = Vc.max(axis=0)
    out = os.path.join(RESULTS, f"fig4a_batches_{args.dataset}{args.out_tag}.npz")
    np.savez_compressed(out, lags=LAGS,
                        S2b=Sp[:, :, 0], S4b=Sp[:, :, 1], S6b=Sp[:, :, 2],
                        S2_lo=lo[:, 0], S2_hi=hi[:, 0],
                        S4_lo=lo[:, 1], S4_hi=hi[:, 1],
                        S6_lo=lo[:, 2], S6_hi=hi[:, 2],
                        S2_lo_cB=lo30[:, 0], S2_hi_cB=hi30[:, 0],
                        S4_lo_cB=lo30[:, 1], S4_hi_cB=hi30[:, 1],
                        S6_lo_cB=lo30[:, 2], S6_hi_cB=hi30[:, 2],
                        n_traj=n, n_batch=args.batches, n_batch_cB=args.batches * C,
                        batch_size=bs, dataset=args.dataset)
    print(f"  saved {out} ({time.time()-t0:.0f}s)", flush=True)
    for i, p in enumerate((2, 4, 6)):
        rA = (hi[:, i] - lo[:, i]) / Sp[:, :, i].mean(axis=0)
        rB = (hi30[:, i] - lo30[:, i]) / Vc[:, :, i].mean(axis=0)
        print(f"  p={p}: relative bar width, median, convention A (10 merged blocks) {np.median(rA):.4f} | "
              f"convention B (paper, {args.batches * C} component blocks) {np.median(rB):.4f} | ratio {np.median(rB) / np.median(rA):.2f}",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
