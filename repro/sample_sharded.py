"""Opt-in bounded-memory checkpoint sampling; original scripts are unchanged.

Run without --execute to print the plan without importing torch or opening a
checkpoint. Independent rank processes own disjoint batches; no all-gather.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from .provenance import manifest, sha256, write_json
from .sampling_plan import batch_plan, history_steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=230708529)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--history", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    plan = list(batch_plan(args.num_samples, args.batch_size, args.rank, args.world_size, args.seed))
    if not plan:
        raise SystemExit("This rank has no batches; reduce world-size")
    states = history_steps(config["diffusion_steps"]) if args.history else [0]
    samples_for_rank = sum(item["count"] for item in plan)
    print(json.dumps({"classification": "sampling_plan_not_executed" if not args.execute else "checkpoint_sampling",
                      "rank": args.rank, "batches": len(plan), "samples_for_rank": samples_for_rank,
                      "saved_states": states,
                      "raw_output_bytes_for_rank": samples_for_rank * config["in_channels"] * config["image_size"] * 4 * len(states),
                      "seed_scheme": "base seed + global batch ID; unchanged for fixed batch size and any world size"}, indent=2))
    if not args.execute:
        return
    if not args.checkpoint.is_file():
        raise SystemExit("Checkpoint is missing; no automatic download")
    rank_manifest = args.output / f"rank{args.rank:04d}.json"
    if rank_manifest.exists() or any(args.output.glob(f"rank{args.rank:04d}-*.npz")):
        raise SystemExit("Existing rank output detected; choose a fresh directory")
    import torch as th
    from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
    defaults = model_and_diffusion_defaults()
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown model settings: {sorted(unknown)}")
    defaults.update(config)
    if defaults["class_cond"] or defaults["use_fp16"]:
        raise ValueError("This audited entrypoint currently supports unconditional float32 models")
    th.use_deterministic_algorithms(True)
    th.backends.cudnn.benchmark = False
    model, diffusion = create_model_and_diffusion(**defaults)
    # Never fall back to unbounded pickle execution for an untrusted checkpoint.
    model.load_state_dict(th.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    record = manifest("python -m repro.sample_sharded " + " ".join(sys.argv[1:]),
                      inputs=[args.checkpoint, args.config], seed=args.seed,
                      classification="checkpoint_sampling_not_yet_paper_validated")
    record.update({"status": "running", "effective_model_config": defaults,
                   "rank": args.rank, "world_size": args.world_size, "device": args.device,
                   "torch_version": th.__version__, "saved_states": states,
                   "normalization": "[-1,1]; denormalize with matching training dataset bounds",
                   "batch_size": args.batch_size, "requested_samples_all_ranks": args.num_samples,
                   "seed_scheme": "base seed + global batch ID", "shards": []})
    write_json(rank_manifest, record)
    for item in plan:
        started = time.monotonic()
        th.manual_seed(item["seed"])
        if th.cuda.is_available():
            th.cuda.manual_seed_all(item["seed"])
        saved = {}
        for step, sample in enumerate(diffusion.p_sample_loop_progressive(
                model, (item["count"], config["in_channels"], config["image_size"]),
                device=th.device(args.device), clip_denoised=True)):
            remaining = diffusion.num_timesteps - 1 - step
            if remaining in states:
                values = sample["sample"]
                if remaining == 0:
                    values = values.clamp(-1, 1)
                saved[f"V_{remaining}"] = values.detach().permute(0, 2, 1).cpu().numpy().copy()
        if set(saved) != {f"V_{n}" for n in states}:
            raise RuntimeError("incomplete requested denoising history")
        target = args.output / f"rank{args.rank:04d}-batch{item['batch_id']:07d}.npz"
        with target.open("xb") as handle:
            np.savez(handle, **saved)
        record["shards"].append({**item, "name": target.name, "sha256": sha256(target),
                                  "seconds": time.monotonic() - started})
        write_json(rank_manifest, record)
    record["status"] = "complete"
    write_json(rank_manifest, record)


if __name__ == "__main__":
    main()
