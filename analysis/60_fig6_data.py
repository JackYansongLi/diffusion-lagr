#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""60_fig6_data.py -- data for Fig. 6: the training-loss curve and, for the inset, the fourth-order
flatness of the DNS.

Conventions (see RESULTS_ANALYSIS.md section 7.17)
-----------------------------------
Main panel (training protocol)
- quantity: <Lsimple_n> against iteration for DM-1c in the paper (`MOESM9: Figure6.out`, 33450 points,
  stride 10, x in [0, 334500], y in [0.0341, 1.002])
- our source: the table log `runs/dm1c_0929_2316/run.out` (`| loss | x |` and `| step | n |` in pairs,
  blocks separated by ---)
  => 26911 (step, loss) pairs, step in [0, 266000], median stride 10 (the 0 comes from a resumed
  duplicate, the 100 from a thinned stretch)
- axes: log-log (the main panel of the paper ticks x at 10^0..10^6 and y at 10^-1..10^0; the source data
  start at x = 0 and are invisible on a log axis, as in the paper)

inset（fourth-order flatness）
- quantity: F_tau^(4) = S_tau^(4)/[S_tau^(2)]^2 (equation 5 of the paper; S_p averaged over the
  components, lags 1..1024, so tau/tau_eta = 0.1*lag in [0.1, 102.4])
- source: the existing `flatness_dns_full.npz` (written by `18_flatness_stats.py`; n_traj = 327680, 10
  blocks, C = 3)
  => reused as is, **not recomputed** (the DNS curve was checked in sections 7.11 and 3.12: it differs
  from Figure6_inset_DNS of the paper by +2.26/+0.14/+0.97/-0.64%)
- bars: the convention already used for Fig. 4b (min/max envelope over the F of every block and
  component, `19_fig4b_flatness_mod.py --err-def B`)
- reference: `Figure6_inset_DNS.out` (1024 points, columns = tau, mean, err-, err+)
- the A/B/C curves were to be added once remote sampling finished (A: 10k, B: 30k, C: 250k samples; C
  could come from the existing 28-run pool)
"""
import argparse
import os
import re

import numpy as np

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RUN_OUT = os.environ.get("LTS_RUN_OUT") or os.path.join(
    ROOT, "Data", "Reproduce", "runs", "dm1c_0929_2316", "run.out")
KV = re.compile(r"^\|\s*([A-Za-z_0-9]+)\s*\|\s*([-\d.eE+]+)\s*\|")


def parse_run_out(path):
    """Extract (step, loss) from the table log: loss and step are paired inside a block."""
    rec, cur = [], {}
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            m = KV.match(s)
            if not m:
                if s.startswith("---") and "loss" in cur and "step" in cur:
                    rec.append((cur["step"], cur["loss"]))
                    cur = {}
                continue
            k, v = m.group(1), m.group(2)
            try:
                v = float(v)
            except ValueError:
                continue
            if k in ("loss", "step"):
                cur[k] = v
    return np.array(rec, dtype=float)


def flatness_envelope(npz_path, order=4):
    """F = S_p / (S_2)^(p/2), with the mean curve averaged over components and a min/max envelope over every
    block and component."""
    z = np.load(npz_path, allow_pickle=True)
    orders = z["orders"]
    mean = z["mean"]                      # (nl, 4), averaged over the components
    Sbc = z["Sbc"]                        # (nb, nl, 4, C)
    i2 = int(np.where(orders == 2)[0][0])
    ip = int(np.where(orders == order)[0][0])
    F_mean = mean[:, ip] / mean[:, i2] ** (order / 2.0)
    F_bc = Sbc[:, :, ip, :] / Sbc[:, :, i2, :] ** (order / 2.0)   # (nb, nl, C)
    # transpose before reshaping: reshape(-1, nl) directly would mix the lag axis in
    #    (measured: written that way the min was corrupted from 42.33 to 2.90 and the envelope went wrong)
    F_flat = F_bc.transpose(0, 2, 1).reshape(-1, F_bc.shape[1])   # (nb*C, nl)
    return z["lags"], F_mean, F_flat.min(axis=0), F_flat.max(axis=0), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-out", default=RUN_OUT)
    ap.add_argument("--dns-npz", default=os.path.join(RESULTS, "flatness_dns_full.npz"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig6_dm1c_repro.npz"))
    ap.add_argument("--log", default=os.path.join(RESULTS, "log_60_fig6.txt"))
    ap.add_argument("--order", type=int, default=4)
    args = ap.parse_args()
    log = open(args.log, "w", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    # ---- main panel: training loss ----
    rec = parse_run_out(args.run_out)
    step, loss = rec[:, 0], rec[:, 1]
    say("[60] main panel loss: %d points | step [%.5g, %.5g] | loss [%.5g, %.5g] | median stride %.4g"
        % (len(rec), step.min(), step.max(), loss.min(), loss.max(), float(np.median(np.diff(step[:2000])))))
    pos = step > 0
    say("      points usable on a log axis (step > 0): %d; points with loss <= 0: %d" % (int(pos.sum()), int((loss <= 0).sum())))

    # ---- inset: fourth-order flatness of the DNS ----
    lags, F_dns, lo, hi, zd = flatness_envelope(args.dns_npz, args.order)
    tau = lags * 0.1
    say("[60] inset DNS: n_traj=%s blocks=%s C=%s | F%ds range [%.4g, %.4g] | tau/tau_eta in [%.2f, %.2f]"
        % (zd["n_traj"], zd["n_batch"], zd["C"], args.order, F_dns.min(), F_dns.max(), tau[0], tau[-1]))
    for t in (0.1, 1.0, 10.0, 100.0):
        j = int(np.argmin(np.abs(tau - t)))
        say("      tau/tau_eta=%-6.1f F=%.4g  [%.4g, %.4g]  (envelope %.2f%%/+%.2f%%)"
            % (t, F_dns[j], lo[j], hi[j], 100 * (lo[j] - F_dns[j]) / F_dns[j], 100 * (hi[j] - F_dns[j]) / F_dns[j]))

    np.savez_compressed(args.out, loss_step=step, loss=loss,
                        tau=tau, F_dns=F_dns, F_dns_lo=lo, F_dns_hi=hi,
                        order=args.order, dns_n_traj=zd["n_traj"], dns_n_batch=zd["n_batch"],
                        dns_C=zd["C"], run_out=args.run_out, dns_npz=args.dns_npz)
    say("[60] → %s" % os.path.basename(args.out))
    log.close()


if __name__ == "__main__":
    main()
