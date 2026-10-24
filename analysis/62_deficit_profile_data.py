# -*- coding: utf-8 -*-
"""62_deficit_profile_data.py --- C01 data stage: three-source F4(tau) with 10-batch envelopes

Merges the three same-schema structure-function archives into the minimal dataset needed
for one figure:
  flatness_dns_full.npz        (DNS,               n_traj = 327680)
  flatness_dm1c_full.npz       (official DM-1c,    n_traj = 358400)
  flatness_dm1c_repro_full.npz (our retrained DM-1c, n_traj = 358400)

Calibers are copied verbatim from 19_fig4b_flatness_mod.py / 50_fig4b_cmp.py:
  F4 = S4 / S2^2, with S_p read from the stored 'mean' array (orders = [2,4,6,8]);
  envelope = min/max over the 10 stored batches (caliber A: components averaged first,
  which is identical for the C=1 sources used here);
  tau/tau_eta = 0.1 * lag (the project-wide lag mapping).

The output path follows LTS_ROOT (the directory that holds results/ and Data/), e.g.
  LTS_ROOT=/path/to/PostP python3 62_deficit_profile_data.py

Product: results/deficit_profile.npz
"""
import os

import numpy as np

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
IX = {2: 0, 4: 1, 6: 2, 8: 3}
SOURCES = [("dns", "dns"), ("dm1c", "official"), ("dm1c_repro", "ours")]

out = {}
for ds, tag in SOURCES:
    z = np.load(os.path.join(RESULTS, "flatness_%s_full.npz" % ds))
    lags, mean, Sbc = z["lags"], z["mean"], z["Sbc"]        # (nl,), (nl,4), (nb,nl,4,C)
    nb, nl, _, C = Sbc.shape
    S = Sbc.mean(axis=3)                                    # caliber A
    f4 = mean[:, IX[4]] / mean[:, IX[2]] ** 2.0
    f4b = S[:, :, IX[4]] / S[:, :, IX[2]] ** 2.0
    out["tau"] = 0.1 * lags
    out["F4_%s" % tag] = f4
    out["lo_%s" % tag] = f4b.min(axis=0)
    out["hi_%s" % tag] = f4b.max(axis=0)
    out["ntraj_%s" % tag] = np.int64(z["n_traj"])
    print("%-9s n_traj=%d nb=%d C=%d | F4(0.1tn)=%.2f F4(1tn)=%.2f F4(2tn)=%.2f | env@0.1tn=%.2f..%.2f"
          % (tag, z["n_traj"], nb, C, f4[0], f4[9], f4[19], f4b[:, 0].min(), f4b[:, 0].max()),
          flush=True)

for num in ("ours", "dns"):
    r = out["F4_%s" % num] / out["F4_official"]
    out["ratio_%s" % num] = r
    print("ratio %-5s vs official:" % num, flush=True)
    for L in (1, 2, 3, 5, 10, 20):
        print("   lag %2d (tau/teta=%4.1f): %.4f  (%+.2f%%)" % (L, 0.1 * L, r[L - 1], 100 * (r[L - 1] - 1.0)),
              flush=True)

np.savez(os.path.join(RESULTS, "deficit_profile.npz"), **out)
print("saved", os.path.join(RESULTS, "deficit_profile.npz"), flush=True)
