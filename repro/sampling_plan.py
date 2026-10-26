"""Rank-unique batch ownership without CUDA device-string assumptions."""


def batch_plan(num_samples, batch_size, rank=0, world_size=1, seed=230708529):
    if num_samples < 1 or batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid sample, batch, rank or world size")
    if seed < 0 or seed + num_samples >= 2**63:
        raise ValueError("seed outside supported range")
    batches = (num_samples + batch_size - 1) // batch_size
    for batch_id in range(rank, batches, world_size):
        start = batch_id * batch_size
        yield {"batch_id": batch_id, "start": start,
               "count": min(batch_size, num_samples - start), "seed": seed + batch_id}


def history_steps(steps=800):
    # V_n is the state *after* the step at PyTorch time index n.
    return [round(.52 * steps), round(.27 * steps), round(.06 * steps), 0]
