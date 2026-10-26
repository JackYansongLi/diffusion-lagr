# Reproduction and validation workbench

**Status: bounded feasibility/validation deliverable, not full paper reproduction.**
Target: arXiv:2307.08529v2 (28 April 2024), main F1-F6 plus supplementary S1-S2.
Upstream: `leopc999/diffusion-lagr@77da6cdc67775aa45af6c4148068f00a75e54c41`.
This branch preserves the upstream training/sampling implementation and places new work in `repro/`.

## What actually ran

- NumPy numerical tests against analytic references and the two upstream schedule functions.
- Statistics on the bundled **256 physical DNS trajectories**, with all three components kept in the same bootstrap cluster.
- Analytic noise-schedule recalculation. Paper linear endpoints and upstream linear endpoints are explicitly separate variants.
- Sampler planning only. No checkpoint inference or training has run.

The one-component demo contains three component records for each of the same physical trajectories. It is not an independent second dataset. Figures with a `demo_` prefix are pipeline diagnostics, not reproductions of paper figures.

## Run without installing the model environment

From the repository root, activate an existing environment with NumPy. For the plotting command also use Matplotlib. HDF5 requires h5py or an existing `h5dump` executable. The latter fallback is size-capped to 64 MiB and is intended only for the bundled demo.

```sh
python -m unittest discover -s repro/tests -v
python -m repro.validate_demo
MPLCONFIGDIR=repro/.cache/matplotlib python -m repro.plot_audit
MPLCONFIGDIR=repro/.cache/matplotlib python -m repro.explore_estimators
python -m repro.report_tables
cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

The measured Mac run used Python 3.12.12, NumPy 2.3.5, Matplotlib 3.10.7 and HDF5 CLI 2.2.0. SciPy 1.16.3 was available but not required by these kernels. Exact input and code hashes, environment, seeds, output hashes and commands are in `results/*manifest.json`; sufficient per-trajectory statistics are retained for independent checks.

## Checkpoint sampling plan (does not execute by default)

```sh
python -m repro.sample_sharded \
  --config repro/configs/dm1c.json --checkpoint repro/checkpoints/dm1c.pt \
  --output repro/runs/dm1c-seed230708529 --num-samples 16 --batch-size 4 --history
```

A future, explicitly budgeted GPU run adds `--execute`. It requires the selected PyTorch environment and a verified checkpoint; none is installed or fetched automatically. The new sampler is **not runtime-tested against PyTorch yet**. It writes every batch independently, avoids all-gather and the full-sample RAM buffer, selects only V416/V216/V48/V0, and uses global batch IDs for non-overlapping seeds. For parallel processes choose distinct `--rank` values with one common `--world-size` and output directory. Device allocation is explicit. Existing rank outputs are refused, not overwritten. Fixed batch size and rank-independent batch ownership preserve each batch's seed; bitwise cross-device equivalence is not promised.

`--history` stores four states, compared with 801 in the upstream history routine. Final values are clipped as upstream; noisy intermediate states are not clipped. The selected history indices follow the paper's remaining-step fractions. Configs record the full model arguments. Checkpoints load with `weights_only=True`; the code does not fall back to arbitrary pickle execution.

## Statistical definitions and caveats

Input axes are trajectory, time, component. Even structure functions average uncentered powers of velocity increments. Generalized flatness divides the pth moment by S2^(p/2). Mixed flatness averages xy/xz/yz numerator pairs and divides by the pooled S2 squared. Paper-type ranges use ten trajectory partitions per component; they are min/max envelopes, **not** confidence intervals. Additional percentile bootstrap intervals resample whole physical trajectories, retaining temporal and component dependence. They assume exchangeability of observed traces and exclude shared-DNS, model-training and systematic errors.

Local exponents use second-order differences of log moments against log lag. Publication comparisons require the dense integer grid 1..1024; sparse demo diagnostics do not implement Figure 5 comparisons. Histograms retain out-of-range mass and do not renormalize away the tails. Acceleration supports interior central, forward, and NumPy-gradient stencils with explicit dt. The per-sample versus physical-time unit issue must be reported before comparing Table II.

`explore_estimators` separately measures all integer lags 1..1024 on the 256-trace demo, finite-subset high-order sensitivity, and 2000 trajectory-cluster bootstrap resamples for Table II quantities. The dense-grid plot is still a demo estimator study, not a DNS/DM paper comparison. The Figure 3b audit plots **1-alpha_bar on a logarithmic ordinate**; its CSV retains both alpha_bar and noise variance, including the zero-noise A endpoint omitted from the log curve.

A separate read-only audit of commit `47b024b` passed all 19 tests and independently matched numerical summaries at rtol=1e-14 (sufficient-statistic relative difference <=4.74e-16). It identified and prompted the Figure 3b ordinate correction. It did not test neural inference. See `manifests/independent_review.json` for scope.

## Data and rights

`manifests/assets.json` records public archive checksums and byte sizes supplied by the independent paper audit. The original demo HDF5 files already belong to upstream. No new full data or checkpoints are committed. Public research data are attributed to the authors' CC-BY-4.0 record https://www.openaccessrepository.it/records/z9akz-5km92 (DOI 10.15161/oar.it/143615). No large download is needed for the commands above.

See `manifests/figures.json` for every panel's actual status and dependencies. Future author-sample reanalysis must be called **author-sample reanalysis**, separate from new checkpoint inference or retraining.
