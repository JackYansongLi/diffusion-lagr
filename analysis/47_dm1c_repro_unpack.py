#!/usr/bin/env python
"""47_dm1c_repro_unpack.py -- extract the final-sample stage of OUR OWN DM-1c sampling
into a flat float32 npy in Data/tmp, in the same layout as the official DM-1c unpack.

Windows side -- run through scripts/run.bat (see HANDOFF):
    run.bat scripts/47_dm1c_repro_unpack.py
or from WSL:
    export LTS_ROOT=/path/to/PostP && python 47_dm1c_repro_unpack.py

Input : $LTS_ROOT/Data/Reproduce/samples_358400x2000x1_DM-1c_reproduce.npz
        member 'arr'  (S, N, L, C) float32   -- S = denoising stages, per-run concatenation
        member 'stages' (S,) int64           -- diffusion step of each stage (0 = n=0 = final sample)
Output: $LTS_ROOT/Data/tmp/dm1c_repro_<N>x<L>x<C>_f32.npy  (verbatim slice, still normalised)
        results\\log_47_dm1c_repro_unpack.txt (via the .bat wrapper)

The slice written is the stage whose value equals --stage-value (default 0).
Values stay in the normalised [-1, 1] space; de-normalisation happens at read time
(u1c global scalars min=-11.24481512 / max=10.31416093), exactly like the official file.
"""
import argparse
import io
import os
import time
import zipfile

import numpy as np

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
TMP_DEFAULT = os.path.join(ROOT, "Data", "tmp")
NPZ_DEFAULT = os.path.join(ROOT, "Data", "Reproduce",
                           "samples_358400x2000x1_DM-1c_reproduce.npz")


def read_npy_header(fp):
    """(shape, fortran_order, dtype) from an open .npy stream; version safe."""
    ver = np.lib.format.read_magic(fp)
    if ver[0] == 1:
        return np.lib.format.read_array_header_1_0(fp)
    return np.lib.format.read_array_header_2_0(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ_DEFAULT)
    ap.add_argument("--outdir", default=TMP_DEFAULT)
    ap.add_argument("--stage-value", type=int, default=0,
                    help="stage value to extract; 0 = n=0 = final sample")
    ap.add_argument("--verify-rows", type=int, default=64,
                    help="rows of the output re-read and compared against a direct npz read")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    a = ap.parse_args()

    t0 = time.time()
    with zipfile.ZipFile(a.npz) as zf:
        names = zf.namelist()
        arr_name = None
        for nm in names:
            if not nm.endswith(".npy"):
                continue
            with zf.open(nm) as fp:
                shape, fortran, dtype = read_npy_header(fp)
            if len(shape) == 4:
                arr_name, arr_shape, arr_dtype = nm, shape, dtype
            if nm == "stages.npy":
                stages = np.load(io.BytesIO(zf.read(nm)))
        if arr_name is None:
            raise SystemExit(f"no 4-D .npy member found in {a.npz}: {names}")
        S, N, L, C = arr_shape
        print(f"npz        : {a.npz}")
        print(f"size       : {os.path.getsize(a.npz)/1e9:.3f} GB")
        print(f"member     : {arr_name} shape={arr_shape} dtype={arr_dtype} order={'F' if fortran else 'C'}")
        print(f"stages     : {stages}")
        if fortran:
            raise SystemExit("member is Fortran ordered -- slice copy would not be contiguous")
        hits = np.flatnonzero(stages == a.stage_value)
        if hits.size == 0:
            raise SystemExit(f"stage value {a.stage_value} not present in {stages}")
        sidx = int(hits[0])
        print(f"extracting : stage value {a.stage_value} -> index {sidx} of {S}")

        out_name = f"dm1c_repro_{N}x{L}x{C}_f32.npy"
        dst = os.path.join(a.outdir, out_name)
        if os.path.exists(dst) and not a.force:
            print(f"[skip] {out_name} already exists ({os.path.getsize(dst)/1e9:.3f} GB); use --force to redo")
            return
        os.makedirs(a.outdir, exist_ok=True)

        per_stage = int(N) * int(L) * int(C)
        itemsize = np.dtype(arr_dtype).itemsize
        skip_bytes = sidx * per_stage * itemsize
        copy_bytes = per_stage * itemsize
        CH = 1 << 24  # 16 MiB

        with zf.open(arr_name) as fp:
            read_npy_header(fp)          # position now at the first data byte
            out_dtype = np.dtype(arr_dtype).newbyteorder("<")
            with open(dst, "wb") as out:
                hdr = {"descr": np.lib.format.dtype_to_descr(out_dtype),
                       "fortran_order": False,
                       "shape": (int(N), int(L), int(C))}
                np.lib.format.write_array_header_1_0(out, hdr)
                done = 0
                while done < skip_bytes:
                    b = fp.read(min(CH, skip_bytes - done))
                    if not b:
                        raise SystemExit("npz member ended while skipping earlier stages")
                    done += len(b)
                done = 0
                while done < copy_bytes:
                    b = fp.read(min(CH, copy_bytes - done))
                    if not b:
                        raise SystemExit("npz member ended while copying the target stage")
                    out.write(b)
                    done += len(b)
        dt = time.time() - t0
        print(f"[done] {out_name}: {os.path.getsize(dst)/1e9:.3f} GB written in {dt:.0f}s")

    # --- verification: output prefix must equal a direct npz read of the same slice ---
    vrows = min(a.verify_rows, N)
    with open(dst, "rb") as f:
        f.seek(0)
        shape_o, _, dtype_o = read_npy_header(f)
        off = f.tell()
        chk_out = np.frombuffer(f.read(vrows * L * C * itemsize), dtype=dtype_o).reshape(vrows, L, C)
    with zipfile.ZipFile(a.npz) as zf:
        with zf.open(arr_name) as fp:
            read_npy_header(fp)
            left = skip_bytes
            while left:
                b = fp.read(min(CH, left))
                left -= len(b)
            chk_npz = np.frombuffer(fp.read(vrows * L * C * itemsize),
                                    dtype=arr_dtype).reshape(vrows, L, C)
    print(f"output     : shape={shape_o} dtype={dtype_o}")
    assert np.array_equal(chk_out, chk_npz), "VERIFY FAILED: output prefix != npz slice"
    print(f"verify     : first {vrows} trajectories identical to the npz slice  OK")
    print("next steps : register the dataset, then run the DM-1c statistics for this file")


if __name__ == "__main__":
    main()
