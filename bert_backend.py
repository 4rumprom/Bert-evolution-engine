"""Backend selection for the parametric BERT growth engine.

One code path, two array backends:

  - "numpy": the reference backend. Runs anywhere, float64,
    bit-exact against the stage-0..3 runners' arithmetic
    patterns (verified: tiled vs single-row einsum evaluation
    is bit-identical).
  - "jax":   the portable backend. The SAME engine code runs
    on CPU here and on GPU on the sponsor's PC (device is a
    config value). With precision="x64" the reference
    configuration enables jax_enable_x64 so cross-backend
    comparisons are meaningful. A GPU float32 performance
    mode (precision="x32") may differ from the float64
    reference at float tolerance; greedy decoders can flip
    on near-ties there. That is accepted and documented in
    the design record.

Host-side logic (breeding, culling, checkpointing, RNG) is
always numpy. Genomes are stored as numpy arrays and
converted at the evaluation boundary, so checkpoints are
backend-independent.

If backend="jax" is configured but jax is not importable,
get_backend falls back to numpy and records the fallback in
the returned namespace (`.fell_back`), so a run never dies
on a missing optional dependency; the fallback is printed
and written into the run results by the runner.
"""

from __future__ import annotations

import numpy as np


class Backend:
    def __init__(self, xp, name, device, precision, fell_back=False,
                 note=""):
        self.xp = xp
        self.name = name
        self.device = device
        self.precision = precision
        self.fell_back = fell_back
        self.note = note

    def from_numpy(self, a):
        a = np.asarray(a)
        if self.name == "jax":
            return self.xp.asarray(a)
        return a

    def to_numpy(self, a):
        if self.name == "jax":
            return np.asarray(a)
        return np.asarray(a)

    def describe(self):
        return {"backend": self.name, "device": self.device,
                "precision": self.precision,
                "fell_back": self.fell_back, "note": self.note}


def get_backend(name="numpy", device="cpu", precision="x64"):
    if name == "numpy":
        return Backend(np, "numpy", "cpu", precision,
                       note="numpy reference backend")
    if name == "jax":
        try:
            import jax
            import jax.numpy as jnp
        except Exception as e:  # pragma: no cover - env dependent
            return Backend(np, "numpy", "cpu", precision,
                           fell_back=True,
                           note=f"jax requested but not importable "
                                f"({type(e).__name__}: {e}); fell "
                                f"back to numpy")
        if precision == "x64":
            jax.config.update("jax_enable_x64", True)
        dev = "cpu"
        if device == "gpu":
            try:
                gpus = jax.devices("gpu")
                if gpus:
                    dev = "gpu"
                else:
                    return Backend(jnp, "jax", "cpu", precision,
                                   fell_back=True,
                                   note="gpu requested but no jax "
                                        "GPU device found; using CPU")
            except Exception:
                return Backend(jnp, "jax", "cpu", precision,
                               fell_back=True,
                               note="gpu requested but jax GPU "
                                    "backend unavailable; using CPU")
        return Backend(jnp, "jax", dev, precision,
                       note=f"jax on {dev}, x64 "
                            f"{'on' if precision == 'x64' else 'off'}")
    raise ValueError(f"unknown backend {name!r} (want numpy|jax)")
