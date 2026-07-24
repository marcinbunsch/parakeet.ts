# CUDA / Nvidia GPU support (Linux)

Goal: run this library on a Linux machine with an Nvidia RTX GPU, not just Apple
Silicon.

## Key finding: the model code likely needs no rewrite

MLX itself now has a CUDA backend, and so does the exact binding this repo uses.
`@mlx-node/core` (the mlx-node project) has an **experimental CUDA path** —
"device-agnostic eager fallbacks with no custom CUDA kernels yet — functional,
but not performance-tuned," inference-only. It runs MLX code on Nvidia GPUs
through MLX's own CUDA backend.

So the same `MxArray` API we already call is meant to run on CUDA unchanged. The
codebase is already set up for this:

- All MLX code is isolated in `src/mlx/`.
- Every file imports only the `MxArray` **type** — 9 files, ~30 tensor ops.
- The audio STFT/FFT is pure JS on the CPU anyway.

So "port 30 ops to a CUDA tensor library" is the wrong framing — that work is
already done inside MLX. The real work is getting the **native binding** to run
on **our box**.

## Our target: x86_64 desktop + RTX card

This is the decisive fact. **No prebuilt CUDA binary exists for x86_64 today.**
mlx-node's only validated CUDA target is `linux-arm64-gnu` (GB10 / DGX Spark, an
ARM Grace-Blackwell box). So the entry ticket is a source build.

### Step 1 — build `@mlx-node/core` from source with CUDA (the hard part)

Prerequisites:

- Nvidia driver >= 580
- CUDA 13 toolkit (`nvcc` on PATH)
- BLAS/LAPACK headers

Set the build's target arch to **our card's compute capability**, not their
`sm_121`:

| GPU            | Arch      |
| -------------- | --------- |
| RTX 30-series  | `sm_86`  (Ampere)  |
| RTX 40-series  | `sm_89`  (Ada)     |
| RTX 50-series  | `sm_120` (Blackwell) |

The build skips the Metal step automatically on Linux and emits
`mlx-core.linux-x64-gnu.node`.

This is the one genuinely uncertain step. It's an early proof-of-concept
(dependency pinned at `^0.0.7`; the CUDA path lives in a newer/experimental
build), so a clean x86_64 source build is plausible but not guaranteed. Budget
time for a compile fight.

### Step 2 — op-coverage shakeout

Once it builds, the model code should run **unchanged**. Run it and catch any
"not implemented on CUDA" throws. Our op set is basic (matmul, take, reshape,
transpose, slice, logSoftmax, pad, concatenate, LSTM gate arithmetic), so odds
are decent. Watch these spots:

- Conv via im2col (`take` / `reshape` / `matmul`) in `src/mlx/nn.ts`
- Cache slicing / concatenation in `src/mlx/cache.ts`

### Step 3 — correctness

Diff transcripts against a Mac run for numerical parity.

### Step 4 — performance

Slow at first: eager fallbacks, no tuned kernels. Fine for "it runs on my RTX,"
not yet for throughput.

### Step 5 — repo hygiene (small)

- Bump the dep off `^0.0.7` to the CUDA-capable build.
- Drop Apple-Silicon-only assumptions (`engines`, the `mlx` naming, README).
- Make the native dep platform-optional so a Mac install doesn't pull a CUDA
  binary and vice-versa.
- `ffmpeg` for audio is already cross-platform — no change.

## Decision rule

- **If the source build succeeds** → small project: build + shakeout + a bit of
  packaging. No rewrite.
- **If it fails or key ops are missing** → fallback is a separate backend:
  export Parakeet to ONNX and run via `onnxruntime-node` with the CUDA execution
  provider. That's a second implementation, weeks not hours, but it's the
  robust, well-trodden way to run this model on an RTX card.

Recommendation: timebox the mlx-node CUDA source build first, since success
there costs almost nothing in code. Treat ONNX as the fallback only if that
stalls.

## Sources

- mlx-node: https://github.com/mlx-node/mlx-node
- MLX CUDA backend (Awni Hannun): https://x.com/awnihannun/status/1948878861795819662
- MLX build/install docs: https://ml-explore.github.io/mlx/build/html/install.html
- MLX on CUDA discussion: https://github.com/ml-explore/mlx/discussions/2422
