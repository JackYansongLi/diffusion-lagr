"""Read local data only. No implicit downloads or installations."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def small_hdf5(path, max_bytes=64 * 1024**2):
    """Load the bundled demo through h5py or an installed h5dump binary.

    The hard size ceiling prevents accidentally reading a full DNS file into
    a 16 GB machine. Production analysis should use iter_hdf5 instead.
    """
    path = Path(path)
    try:
        import h5py
    except ImportError:
        h5py = None
    if h5py is not None:
        with h5py.File(path, "r") as f:
            nbytes = sum(f[key].size * f[key].dtype.itemsize for key in ("train", "min", "max"))
            if nbytes > max_bytes:
                raise ValueError("small_hdf5 size limit exceeded; use streaming analysis")
            return {key: f[key][...] for key in ("train", "min", "max")}, "h5py"
    executable = shutil.which("h5dump")
    if not executable:
        raise RuntimeError("Install/activate h5py or h5dump before reading HDF5")
    result = {}
    for key in ("train", "min", "max"):
        header = subprocess.check_output([executable, "-H", "-d", key, str(path)], text=True)
        dtype_match = re.search(r"DATATYPE\s+H5T_IEEE_F(32|64)(LE|BE)", header)
        dims = re.search(r"DATASPACE\s+SIMPLE\s*\{\s*\(([^)]*)\)", header)
        if not dtype_match or (dims is None and "DATASPACE  SCALAR" not in header):
            raise ValueError("h5dump fallback supports only simple IEEE float datasets")
        shape = tuple(int(x.strip()) for x in dims[1].split(",")) if dims else ()
        dtype = np.dtype("<f" + str(int(dtype_match[1]) // 8))
        nbytes = int(np.prod(shape)) * dtype.itemsize
        if nbytes > max_bytes:
            raise ValueError("small_hdf5 size limit exceeded; use streaming analysis")
        with tempfile.TemporaryDirectory(prefix="diffusion-lagr-h5-") as temp:
            binary = Path(temp) / "data.bin"
            subprocess.run([executable, "-d", key, "-b", "LE", "-o", str(binary), str(path)],
                           check=True, stdout=subprocess.DEVNULL)
            if binary.stat().st_size != nbytes:
                raise ValueError("unexpected HDF5 extraction byte count")
            result[key] = np.fromfile(binary, dtype=dtype).reshape(shape)
    version = subprocess.check_output([executable, "-V"], text=True).strip()
    return result, version


def iter_hdf5(path, chunk_size=64, dataset="train"):
    """Yield (start index, normalized trajectories, minimum, maximum) in chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    import h5py
    with h5py.File(path, "r") as f:
        source, minimum, maximum = f[dataset], f["min"][...], f["max"][...]
        if len(source.shape) != 3 or source.shape[-1] not in (1, 3):
            raise ValueError("unexpected trajectory data shape")
        for start in range(0, len(source), chunk_size):
            yield start, source[start:start + chunk_size], minimum, maximum
