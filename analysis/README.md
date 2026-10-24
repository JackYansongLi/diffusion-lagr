# analysis -- scripts behind the discussion document

Scripts that produce every number and every curve of

> *Reproducing and extending Li et al. (2024) on a 12-channel tetrad model* (discussion draft,
> `Results/Docs/202610_diffusion_single_tetrad/`)

They were written inside the single-particle post-processing tree (`PostP/scripts/`) and are
collected here so that the document's results can be re-derived outside that tree, with three
changes only: (i) English comments, (ii) a single path root, and (iii) the helpers of
`48_fig1a_fig2_cmp_mod.py` are taken from `14_fig1a_pdf_mod.py`, the split successor of the
frozen merged script it used to import, so that nothing here imports a file that is not here.

## What is here, and what is not

Included: the data-processing and plotting scripts for the figures and tables of the document,
plus two helpers (`tetrad_lib.py`, `_vec_regen.py`).

**Not** included, by design:

* the `.npz` arrays the scripts read and write -- they are products, not sources; the data
  stage scripts below regenerate them from the archives;
* the retrained DM-1c sample archive (`samples_358400x2000x1_DM-1c_reproduce.npz`, ~11.5 GB);
* the raw trajectory HDF5 files (`Lagr_u1c_diffusion.h5` 15.7 GB, `Lagr_u3c_diffusion.h5`
  15.7 GB, `Lagr_tetrad_u12c.h5` 31.5 GB, `full_traj_tracers.h5` 94.4 GB);
* the paper's source data beyond the public supplement files (see below);
* the compiled PDF and the vector figure files.

## Pointing the scripts at a data tree

Every script resolves its inputs through one root, taken from `LTS_ROOT`:

```bash
export LTS_ROOT=/path/to/PostP      # the directory that holds results/ and Data/
python 19_fig4b_flatness_mod.py --dataset dns
```

The default, when `LTS_ROOT` is unset, is the parent of this `analysis/` directory. The tree
the scripts expect under that root is

```
$LTS_ROOT/results/                              # .npz caches and figures/ (created on demand)
$LTS_ROOT/Data/Li_et_al_2024_source_data/       # public supplements, see below
$LTS_ROOT/Data/Lagr_u1c_diffusion.h5            # raw DNS, not distributed
$LTS_ROOT/Data/Lagr_u3c_diffusion.h5            # raw DNS, not distributed
$LTS_ROOT/Data/Reproduce/samples_*.npz          # retrained samples, not distributed
```

The paper's source data are the supplementary files of Li et al. (2024),
*Nature Machine Intelligence* 6, 1440 (article `s42256-024-00810-0`):

| file | used for |
| --- | --- |
| `42256_2024_810_MOESM4_ESM.txt` | Fig. 1a of the paper (velocity-increment PDFs) |
| `42256_2024_810_MOESM5_ESM.txt` | Fig. 2 of the paper (acceleration PDF, $C_\tau$ inset) |
| `42256_2024_810_MOESM6_ESM.txt` | Figs. 3c--3e of the paper (PDFs, $S^{(2)}_\tau$, $F^{(4)}_\tau$) |
| `42256_2024_810_MOESM7_ESM.txt` | Figs. 4a--4c (structure functions, flatness) |
| `42256_2024_810_MOESM8_ESM.txt` | Fig. 5a (local exponent) |
| `42256_2024_810_MOESM9_ESM.txt` | Fig. 6 (training loss) |

## Scripts

### Statistics and tables

| script | produces |
| --- | --- |
| `02_structure_functions.py` | streaming $S^{(p)}_\tau$, $F^{(p)}_\tau$, $\zeta(p,\tau)$ for all datasets (statistics cache) |
| `07_fig1_fig2_table2.py` | single-time second-order correlations (Table 2 of the paper) |
| `71_fig2_accpdf_diff_data.py` | PDF differences quoted in the text: mass ratios, per-bin deviation, TV distance |
| `72_fig3_diff_and_table2.py` | Table 2 reproduction, $C_\tau$ inset, per-stage differences |
| `73_fig4b_mean_diff.py` | mean flatness of the retrained model against the paper's, by lag band |
| `36_tetrad_r_reconstruction_accuracy.py` | trapezoidal reconstruction error of the tetrad geometry |

### Data stages (write the `.npz` the plotting scripts read)

| script | writes | read by |
| --- | --- | --- |
| `16_fig4a_batches.py` | `fig4a_batches_<ds>.npz` -- per-block $S^{(2)},S^{(4)},S^{(6)}$ and their min/max | 17, 49, 54, 55, 56 |
| `18_flatness_stats.py` | `flatness_<ds>_full.npz` -- $S^{(p)}$, $p=2,4,6,8$, per block and component | 19, 50, 51, 60, 73 |
| `22_fig4c_mixed_batches.py` | `fig4c_mixed_<ds>.npz` -- mixed flatness per block, both bar conventions | 23, 24 |
| `24_fig4c_calibers.py` | `fig4c_calibers_<ds>.npz` -- eleven bar conventions side by side | 23, 26, 27 |
| `25_fig4c_split_test.py` | `fig4c_split_<ds>.npz` -- ordered against shuffled splitting | 26, 27 |
| `52_fig3c_pdf_data.py` | `fig3c_pdf_dm1c_repro.npz` -- increment PDFs, fine and paper grid | 53 |
| `54_fig3d_s2_data.py` | `fig3d_s2_dm1c_repro.npz` -- $S^{(2)}_\tau$ per stage with a 10-block envelope | 55, 72 |
| `56_fig3e_f4_data.py` | `fig3e_f4_dm1c_repro.npz`, `fig3e_f4_dns.npz` -- per-unit $S^{(2)},S^{(4)}$ | 57, 72 |
| `58_fig3f_traj_data.py` | `fig3f_traj_dm1c_repro.npz` -- one trajectory through the four stages | 59 |
| `60_fig6_data.py` | `fig6_dm1c_repro.npz` -- training loss plus the DNS inset curve | 61 |
| `47_dm1c_repro_unpack.py` | `Data/tmp/dm1c_repro_*.npy` -- stage slices of the retrained archive | 54, 56 |

The heavier stages read the raw HDF5 files and take from minutes to hours; the plotting scripts
never touch them. `--n-traj`/`--limit-traj`/`--limit-blocks` cut a stage down for a smoke test, and
the `--out-tag`/`--out` options keep such a run from overwriting the real cache:

```bash
export LTS_ROOT=/path/to/PostP
python 16_fig4a_batches.py --dataset dm1c --batches 2 --n-traj 2000 --out-tag _smoke
python 52_fig3c_pdf_data.py --limit-traj 2000 --out /tmp/fig3c_smoke.npz
```

Two stages (`24_fig4c_calibers.py`, `25_fig4c_split_test.py`) have no output flag and always write
to `$LTS_ROOT/results`; back those two files up before a smoke run.

### Single-particle figures (Fig. 1--5 of the paper)

| script | produces |
| --- | --- |
| `09_fig1bf_trajectories_panel_mod.py` | trajectory-level panels, labels a--b |
| `70_fig1f_3dtraj_style_mod.py` | 3D trajectory coloured by $|a_i|$, label c |
| `14_fig1a_pdf_mod.py` | velocity-increment PDFs at four lags |
| `15_fig2_accpdf_mod.py` | acceleration PDF and autocorrelation inset |
| `17_fig4a_structfuncs_mod.py` | $S^{(p)}_\tau$, $p = 2, 4, 6$ |
| `19_fig4b_flatness_mod.py` | $F^{(p)}_\tau$, $p = 4, 6, 8$ |
| `23_fig4c_mixed_mod.py` | mixed flatness with global-denominator bars |
| `26_fig4c_variants.py` | four error-bar conventions side by side |
| `27_fig4c_width_vs_N.py` | bar width against trajectory count |
| `29_fig5a_dns_deviation_localize.py` | localisation of the large-lag deviation (initial version) |
| `29_fig5a_dns_deviation_localize_mod.py` | same, revision that draws the figure used in the document |
| `06_fig5a_local_slope_mod.py` | local exponent $\zeta(4,\tau)$ |
| `53_fig3c_pdf_plot_mod.py` | increment PDFs at $\tau_\eta$, panel labels a--e |
| `55_fig3d_s2_plot_panel_mod.py` | $S^{(2)}_\tau$ against lag, labels a--e |
| `57_fig3e_f4_plot_panel_mod.py` | $F^{(4)}_\tau$ against lag, labels a--e |
| `59_fig3f_traj_plot_mod.py` | one trajectory across the four denoising stages |
| `61_fig6_plot_mod.py` | training loss against the published run |

### Model-to-model comparisons

`48_fig1a_fig2_cmp_mod.py`, `49_fig4a_cmp.py`, `50_fig4b_cmp.py`, `51_fig5a_cmp.py` draw the
comparisons that the reproduction panels absorb (paper samples against the retrained model),
i.e. the two-curve versions of Figs. 1a, 2, 4a, 4b and 5a.

### Retraining diagnostics

| script | produces |
| --- | --- |
| `62_deficit_profile_data.py` | three-source $F^{(4)}(\tau)$ with 10-batch envelopes (data stage) |
| `63_deficit_profile_plot_mod_pub.py` | dissipation-range deficit figure (reads the above) |
| `64_order_evidence_data.py` | evidence on the ordering of the training data (data stage) |
| `65_order_evidence_plot.py` | the corresponding figure |

### Tetrad benchmark (Biferale et al. 2005 comparison)

| script | produces |
| --- | --- |
| `43_bif05_fig1_eigenvalues_data.py` / `_plot.py` | eigenvalue trajectories against the 2005 reference |
| `44_bif05_volume_data.py` / `_plot.py` | tetrad volume distribution |
| `45_bif05_shape_indices_data.py` / `_plot.py` | shape indices $I_i$ |
| `46_bif05_shape_pdf_data_mod.py` / `46_bif05_shape_pdf_plot_mod.py` | shape-index PDFs (the `_mod` pair is the one drawn) |

### Helpers

* `tetrad_lib.py` -- shared helpers for the tetrad scripts (geometry, units, I/O, figure style).
  Run the self-test of the parent project after editing it.
* `_vec_regen.py -- <script.py> [args]` -- re-runs a plotting script and additionally writes a
  vector PDF twin of every PNG it saves (the document embeds `.eps`/`.pdf`). Transparent
  backgrounds are pre-composited over white, and figures listed in `VEC_NO_FLATTEN`
  (comma separated basenames) are exempt.

## Conventions worth knowing before reading the scripts

* `*_mod*` files are revisions of a frozen initial version; the initial version is kept
  unchanged on purpose, so both may appear.
* Lags are mapped to $\tau/\tau_\eta = 0.1 \times$ (stored lag index) throughout.
* The label strings inside `results/fig4c_calibers_*.npz` (the `names` array, e.g. the convention-C
  entry) are still Chinese: they are data identifiers used for lookups in `23_`, `26_` and `27_`,
  not comments. Do not translate them; regenerate the npz with renamed labels if they must change.
* The DM panels follow the figure convention of the paper: 10 batches, released samples; the
  error-bar convention finally used for the mixed flatness is described in the document,
  section 2.2, and is the one implemented in `23_fig4c_mixed_mod.py`.
* `analysis/` is intentionally flat: no subdirectories. Keep it that way when adding scripts.

## Requirements

Python 3.10+ with `numpy`, `matplotlib` and `h5py` (see `requirements.txt`). Every script is
standalone apart from `tetrad_lib.py`; run them from this directory or with its path on
`PYTHONPATH`.
