"""Loader for the CUDA kernels in sam3/csrc (driver-API launch, size independent).

Uses the shipped fatbin in sam3/csrc/prebuilt (SASS for sm_80 ... sm_120 + compute_80 PTX) when it matches the source;
otherwise compiles the source once per GPU arch with nvcc into ~/.cache/sam3-fast.
Rebuild the shipped fatbin after editing a kernel:  python -m sam3.fast_cuda
"""
import ctypes
import hashlib
import math
import os
import subprocess

import torch

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_PREBUILT = os.path.join(_CSRC, "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "sam3-fast")
_ARCHS = (80, 86, 87, 89, 90, 110, 120)
_SOURCES = ("flash_k256.cu",)
_MODS = {}


def _check(r, what):
    if r != 0:
        raise RuntimeError(f"CUDA driver error {r} in {what}")


def _nvcc(args, src, out):
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), *args, "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                    f"-I{cuda_home}/include", "-o", out + ".tmp", src], check=True)
    os.replace(out + ".tmp", out)


def _prebuilt_path(src_name):
    tag = hashlib.sha1(open(os.path.join(_CSRC, src_name), "rb").read()).hexdigest()[:16]
    return os.path.join(_PREBUILT, f"{os.path.splitext(src_name)[0]}_{tag}.fatbin")


def build_prebuilt():
    """Ship the kernels for the current sources: one fatbin per source in sam3/csrc/prebuilt."""
    os.makedirs(_PREBUILT, exist_ok=True)
    for f in os.listdir(_PREBUILT):
        os.remove(os.path.join(_PREBUILT, f))
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in _ARCHS] + ["-gencode=arch=compute_80,code=compute_80"]
    for src_name in _SOURCES:
        out = _prebuilt_path(src_name)
        _nvcc(["-fatbin", *gencode], os.path.join(_CSRC, src_name), out)
        print(out)


class Module:
    def __init__(self, src_name, names):
        torch.cuda.init()
        image = _prebuilt_path(src_name)
        if not os.path.exists(image):
            major, minor = torch.cuda.get_device_capability()
            src = os.path.join(_CSRC, src_name)
            tag = hashlib.sha1(open(src, "rb").read() + f"sm_{major}{minor}".encode()).hexdigest()[:16]
            image = os.path.join(_CACHE, f"{src_name}_{tag}.cubin")
            if not os.path.exists(image):
                os.makedirs(_CACHE, exist_ok=True)
                _nvcc(["-cubin", f"-arch=sm_{major}{minor}"], src, image)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        torch.zeros(1, device="cuda")
        self.module = ctypes.c_void_p()
        self._image = open(image, "rb").read()
        _check(self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image), "cuModuleLoadData")
        self.funcs = {}
        for n in names:
            f = ctypes.c_void_p()
            _check(self.lib.cuModuleGetFunction(ctypes.byref(f), self.module, n.encode()), n)
            self.funcs[n] = f

    def launch(self, name, grid, block, args, smem=0):
        """args: list of (kind, value), kind in p (tensor / int ptr), i (int32), l (int64), f (float32)"""
        ct = {"i": ctypes.c_int32, "l": ctypes.c_int64, "f": ctypes.c_float}
        holders = [ctypes.c_void_p(a.data_ptr() if isinstance(a, torch.Tensor) else int(a)) if t == "p" else ct[t](a) for t, a in args]
        arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        _check(self.lib.cuLaunchKernel(self.funcs[name], *grid, *block, smem, torch.cuda.current_stream().cuda_stream, arr, None), name)


def get(src_name, names):
    if src_name not in _MODS:
        _MODS[src_name] = Module(src_name, names)
    return _MODS[src_name]


def flash_k256v64(q, k, v, scale=None, variant="ff"):
    """q [B, Nq, 256], k [B, Nk, 256] (row stride arbitrary, last stride 1), v [B, Nk, 64] fp16 -> o [B, Nq, 64].
    variant "ff": fp32 accumulation, "hf": fp16-accumulate Q K^T, fp32 P V."""
    B, Nq, D = q.shape
    Nk = k.shape[1]
    assert D == 256 and v.shape[-1] == 64 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    if scale is None:
        scale = D ** -0.5
    m = get("flash_k256.cu", ["flash_k256v64_ff", "flash_k256v64_hf"])
    o = torch.empty((B, Nq, 64), device=q.device, dtype=q.dtype)
    m.launch(f"flash_k256v64_{variant}", ((Nq + 63) // 64, B, 1), (128, 1, 1),
             [("p", q), ("p", k), ("p", v), ("p", o), ("i", Nq), ("i", Nk), ("l", q.stride(0)), ("i", q.stride(1)), ("l", k.stride(0)), ("i", k.stride(1)),
              ("l", v.stride(0)), ("i", v.stride(1)), ("l", o.stride(0)), ("i", o.stride(1)), ("f", float(scale * math.log2(math.e)))])
    return o


if __name__ == "__main__":
    build_prebuilt()
