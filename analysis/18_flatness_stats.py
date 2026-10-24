"""18_flatness_stats.py -- moment statistics needed for the generalized flatness of Fig. 4b,
F_tau^(p) = S_tau^(p) / [S_tau^(2)]^(p/2).

Why this script exists: Fig. 4b needs p = 4, 6, 8, while the existing structfuncs_<ds>_full.npz stops
at p = 6,
so one more full pass has to produce the **p = 8** moment (p = 2, 4, 6 are recomputed in the same
pass as a cross-check).

One pass yields both:
  - the full-resolution mean of S2/S4/S6/S8 (1024 lags, used for the flatness curves);
  - the S(p) of every trajectory block and component, shape (nb, nl, 4, C), so that the bars can later
    be reduced to a min/max envelope under any convention:
      convention A (the earlier one) = 10 trajectory blocks with the components merged;
      convention B (the paper's, captions of Figs. 4 and 5) = components kept apart, i.e. 30 component
      batches for the DNS and DM-3c and 10 for DM-1c.

The data sources and de-normalisation constants are those of 16_fig4a_batches.py.
Output: results/flatness_<ds>_full.npz
Usage: python3 18_flatness_stats.py --dataset dns|dm1c|dm3c
"""
import argparse
import importlib
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_b16 = importlib.import_module("16_fig4a_batches")
LAGS, get_dataset, RESULTS, DATA = _b16.LAGS, _b16.get_dataset, _b16.RESULTS, _b16.DATA

ORDERS = (2, 4, 6, 8)
BLOCK = 512


def worker(task):
    """(batch_id, ds, i0, i1) -> (batch_id, S(nl,4,C), cnt(nl,C)), where S holds the moment sums of
    p = 2, 4, 6, 8 per component."""
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
    v = (u + 1.0) * scale + mn
    del u, scale
    B, K, C = v.shape
    S = np.zeros((len(LAGS), 4, C), dtype=np.float64)
    cnt = np.zeros((len(LAGS), C), dtype=np.float64)
    for li, tau in enumerate(LAGS):
        d = v[:, tau:, :] - v[:, :K - tau, :]
        m2 = d * d                                     # δ²
        del d
        S[li, 0] += np.sum(m2, axis=(0, 1), dtype=np.float64)
        m4 = m2 * m2                                   # δ⁴
        S[li, 1] += np.sum(m4, axis=(0, 1), dtype=np.float64)
        m6 = m4 * m2                                   # δ⁶
        S[li, 2] += np.sum(m6, axis=(0, 1), dtype=np.float64)
        m8 = m4 * m4                                   # δ⁸
        S[li, 3] += np.sum(m8, axis=(0, 1), dtype=np.float64)
        del m2, m4, m6, m8
        cnt[li] += B * (K - tau)
    del v
    return bid, S, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dns", "dm3c", "dm1c", "dm1c_repro"])
    ap.add_argument("--batches", type=int, default=10, help="number of trajectory blocks (the paper uses 10)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--n-traj", type=int, default=0, help="use only the first N trajectories (0 = all; for smoke tests)")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    if args.n_traj:
        ds["n_traj"] = args.n_traj
    n, C = ds["n_traj"], ds["C"]
    bs = n // args.batches
    print(f"[18_flatness_stats] dataset={args.dataset} n_traj={n} C={C} batches={args.batches} "
          f"batch_size={bs} workers={args.workers} orders={ORDERS}", flush=True)

    tasks = []
    for b in range(args.batches):
        j0 = b * bs
        j1 = (b + 1) * bs if b < args.batches - 1 else n
        for i0 in range(j0, j1, args.block):
            tasks.append((b, ds, i0, min(i0 + args.block, j1)))
    print(f"  tasks={len(tasks)} blocks", flush=True)

    Ssum = np.zeros((args.batches, len(LAGS), 4, C))
    cnt = np.zeros((args.batches, len(LAGS), C))
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for bid, S, c in pool.imap_unordered(worker, tasks, chunksize=1):
            Ssum[bid] += S
            cnt[bid] += c
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} blocks ({time.time() - t0:.0f}s)", flush=True)

    Vbc = Ssum / cnt[:, :, None, :]                     # S(p) of every block and component: (nb, nl, 4, C)
    mean = Ssum.sum(axis=0).sum(axis=2) / cnt.sum(axis=0).sum(axis=1)[:, None]   # full-resolution mean, (nl, 4)
    out = os.path.join(RESULTS, f"flatness_{args.dataset}{args.out_tag}_full.npz")
    np.savez_compressed(out, lags=LAGS, orders=np.array(ORDERS), mean=mean,
                        Sbc=Vbc, n_batch=args.batches, C=C, n_traj=n, batch_size=bs, dataset=args.dataset)
    print(f"  saved {out} ({time.time() - t0:.0f}s)", flush=True)
    for i, p in enumerate(ORDERS):
        s = mean[:, i]
        msg = f"  p={p}: S median {np.median(s):.4g} max {s.max():.4g}"
        if p > 2:                                       # rough comparison of the flatness with the paper
            f_ours = s / mean[:, 0] ** (p / 2.0)
            msg += f" | F median {np.median(f_ours):.4g}"
        print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
