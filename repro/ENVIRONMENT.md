# Environment and resource audit

Measured host: Apple M1 Pro, 8 CPU cores (6 performance, 2 efficiency), 14-core GPU,
16 GiB unified memory, macOS arm64, approximately 729 GiB available storage at audit.
No CUDA GPU is available. Existing TeX and Poppler executables can compile/render the report.
No credentials, host identifiers, private configuration, or full environment-variable dump is recorded.

GitHub identity was verified independently by the connected service and `gh api user`:
JackYansongLi (id 53420718). The initial sandboxed CLI auth check was misleading because
DNS/network was blocked; the approved network read succeeded. Fork parent and origin
were verified before clone. The fork is new and no existing user work was overwritten.

The author's environment is Ubuntu 20.04.4, Python 3.7.16, PyTorch 1.13.1,
OpenRTE 4.0.2 and CUDA 11.5 on A100. setup.py does not pin the stack or declare
NumPy, mpi4py and h5py completely. A model environment lock is therefore **pending**
selection/validation on the eventual compute target; requirements-analysis.txt only
pins the environment used for this bounded analysis. Do not claim the model stack
has been reproduced on the Mac.

The original loader requires MPI and an MPI-enabled h5py driver. Its deterministic
flag sets shuffle=True when deterministic=True; this appears reversed relative to
its documentation. Default training currently uses shuffle=False. We preserve this
baseline, record it as a methodological uncertainty, and do not silently change it.
The original sampler derives seeds from an integer CUDA_VISIBLE_DEVICES; this is
fragile for comma-separated device lists and repeats seeds across nodes. The new
opt-in sampler uses explicit global batch seeds and avoids MPI for independent ranks.

Storage estimates (decimal GB; not downloads):
- DNS3c 15.729 GB. DNS1c is an additional 15.729 GB, avoid duplicating for analysis.
- Published DM1c and DM3c samples total 11.469 GB. DNS3c + both = 27.197 GB.
- Extracting both NPZ members to memory-mapped NPY adds about 11.469 GB.
- Allow at least 50 GB for a useful published-data analysis workspace, plus outputs.
- 86 * 327680 scalar trajectories at length2000 float32 require 225.444 GB;
  22 * 327680 require 57.672 GB, before replication/checkpoints/history. The reference
  interpretation of 'DNS trajectory count' must be confirmed because component pooling
  can multiply these budgets by three. Statistics should stream from shards.

Compute estimates: the paper states about25h (DM1c) and40h (DM3c), each on4 A100s,
i.e. approximately100 and160 A100 GPU-hours. README gives rounded1 and2 days.
Paper's4million-trajectory sampling example takes200h on4 A100s (800 GPU-hours),
but this cannot be transferred to the Mac or every model without a real benchmark.
Checkpoint-based inference avoids training but remains costly at rare-tail sample counts.
Published author-sample reanalysis avoids model inference altogether.

No sustained training, model inference, software installation, purchase, external
contact or large-data download was performed. User's one-hour deadline restricts
this deliverable to bounded measured validation and explicit remaining work.
