# Running on CUDA (Linux) — dual-backend design

Goal: run this library live on **both** a Mac (Apple Silicon) and a Linux box
with an Nvidia RTX GPU, **without changing app code**. Same public API, same CLI,
same server — the GPU work is swapped underneath by platform.

Status: design + feasibility probe complete (2026-07-28). Nothing implemented
yet. Claims marked **[verified]** below were measured on this box; claims marked
**[unmeasured]** are still open.

## Chosen approach: abstract at the model boundary, offload CUDA to ONNX Runtime

We do **not** give this codebase a second tensor library. Instead we abstract at
the **model boundary**. The library is not one monolithic graph; it's:

1. Audio -> mel spectrogram — **already pure JS/CPU**
2. **Encoder** (conformer) forward — the heavy compute
3. A **decode loop** (TDT/RNNT greedy) that repeatedly calls the prediction
   network (LSTM) and joint network -> logits -> argmax
4. Logits -> tokens -> text — **tokenizer/alignment, already pure JS**

Only steps 2 and 3 touch tensors. Everything else — mel front-end, greedy loop,
argmax, tokenizer, alignment, CLI, server, streaming, mutex — is plain
TypeScript and stays identical on both platforms.

### The interface is two calls, not three

An earlier draft of this doc proposed `encode` / `predict` / `joint`. That is
wrong for the off-the-shelf ONNX exports: **prediction and joint are fused into
a single `decoder_joint` graph** and cannot be split without re-exporting.

```ts
interface ParakeetBackend {
  encode(mel):                        { data: Float32Array; shape: number[] }
  decodeStep(encFrame, token, state): { logits, state }
}
```

- `MlxBackend` — wraps `src/mlx/`. `decodeStep` fuses the existing
  `PredictNetwork.forward` + `JointNetwork.forward`, which `rnnt.ts:178-186`
  already calls back-to-back anyway.
- `OnnxBackend` — two `ort.InferenceSession`s, CUDA EP on Linux.

The LSTM state stays **opaque to the loop** — it's already passed straight
through in `decodeTDTGreedy`, so it types cleanly as a backend-specific handle.

The public API (`ParakeetTDT.transcribe`, streaming, `createParakeetRoutes`, the
CLI) never changes. Backend is chosen at load time / auto-detected by platform.

## Decision: MLX on Mac + ONNX on Linux

- **Mac** keeps the MLX/Metal path — fastest on Apple Silicon (unified memory).
- **Linux** uses ONNX Runtime + CUDA EP.

Cost: two backends to keep numerically in sync. Accepted for the performance
win.

## Verified: ONNX Runtime CUDA on this box

**[verified]** A plain `npm install onnxruntime-node` (v1.27.0) auto-downloads
the CUDA EP for linux-x64 — `libonnxruntime_providers_cuda.so`, 252 MB, no
install flags needed.

Its `NEEDED` list:

```
libcudart.so.13  libcublas.so.13  libcublasLt.so.13  libnvrtc.so.13
libcudnn.so.9    libcufft.so.12   libcurand.so.10    libcuda.so.1
```

ORT **1.27 moved its default GPU build to CUDA 13 + cuDNN 9**, which happens to
be exactly what the earlier MLX-CUDA build left on this box (`~/cuda-13` =
13.0.2, `~/cudnn` = 9.25). With

```
LD_LIBRARY_PATH=$HOME/buildprefix/lib:$HOME/cuda-13/lib64
```

**every dependency resolves — zero "not found".**

This cuts both ways: **ORT <= 1.26 wants CUDA 12**, so we are pinned to >= 1.27
unless we also install a CUDA 12 runtime.

Creating a CUDA session currently fails with exactly one error, and it is the
known local driver issue, not an ORT problem:

```
CUDA failure 804: forward compatibility was attempted on non supported HW
```

Kernel module 580.159.03 vs userspace 580.173.02 — reboot-gated.
**[unmeasured]** Everything GPU-side below is therefore verified only up to
session creation; all timings are CPU EP.

## Verified: the pre-exported model matches our checkpoint

`istupakov/parakeet-tdt-0.6b-v3-onnx` — `features_size: 128`,
`subsampling_factor: 8`. Ran `decoder_joint-model.onnx` on CPU with shapes taken
from our local `config.json`; correct on the first try:

| | local checkpoint | ONNX export |
|---|---|---|
| d_model | 1024 | `encoder_outputs` = 1024 |
| pred_hidden / rnn layers | 640 / 2 | states `[2, 1, 640]` |
| vocab + blank + durations | 8192 + 1 + 5 | logits width **8198** |

Graph I/O:

```
encoder:       audio_signal [1,128,T], length  ->  outputs [1,1024,T/8], encoded_lengths
decoder_joint: encoder_outputs [1,1024,1], targets, target_length,
               input_states_1/2 [2,1,640]
            -> outputs [1,1,1,8198], prednet_lengths, output_states_1/2
```

Note the encoder output is **channel-first** `[B, D, T]`, and `decoder_joint`
takes a single frame sliced along the last axis.

## Verified: the decode loop is cheap enough to stay on CPU

**[verified]** `decoder_joint` on the CPU EP averages **0.46 ms/step**. For 10 s
of audio (~125 encoder frames, ~250 TDT steps) that is ~115 ms total. The
encoder is the part worth offloading; keeping the autoregressive loop on CPU
avoids a GPU round-trip per token. Worth benchmarking both ways once CUDA runs.

## Streaming — the one real gap

The off-the-shelf ONNX encoder has **no cache tensors**: `audio_signal`,
`length` in; `outputs`, `encoded_lengths` out. But `StreamingParakeet`
(`src/mlx/parakeet.ts:476`) depends on per-layer `RotatingConformerCache`
threaded into `encoder.forward(mel, null, cache)`.

Important context: our Mac streaming is **not** a cache-aware *model*. It is the
offline model coerced at runtime — `setAttentionModel('rel_pos_local_attn',
contextSize)` (`parakeet.ts:522`) plus a rotating KV/conv cache. The checkpoint
config confirms it was never trained for this:

```
self_attention_model  rel_pos      att_context_size     [-1, -1]
att_context_style     regular      causal_downsampling  False
conv_context_size     None         conv_kernel_size     9
```

Fully non-causal, full context. So MLX streaming is already an approximation. We
are not preserving a gold standard — we need a *usable* streaming path.

### Option A — sliding-window re-encode (recommended)

No cache at all. Keep a rolling audio/mel buffer, re-run the whole encoder on
the trailing window each tick, reuse the existing finalized/draft split.

**[verified]** Real encoder ONNX, CPU EP only, i7-13700KF / 24 threads:

| window | encoder time | RTFx |
|---|---|---|
| 0.5 s | 46.6 ms | 10.7 |
| 1 s | 50.1 ms | 20.0 |
| 2 s | 60.6 ms | 33.0 |
| 5 s | 103.3 ms | 48.4 |
| 10 s | 180.2 ms | 55.5 |
| 20 s | 363.1 ms | 55.1 |

~40 ms fixed overhead + ~16 ms per second of audio. A 10 s window re-encoded
every 500 ms costs **180 ms per 500 ms tick — 36% of the real-time budget on CPU
alone**, before any GPU. Dynamic axes work down to 0.5 s (50 mel frames -> 7
output frames).

The strategic payoff: **sliding-window streaming needs only `encode()`.** No
cache tensors in the backend interface, so streaming behaves identically on MLX
and ONNX and Mac/Linux transcripts converge instead of diverging.
`RotatingConformerCache` becomes an optional MLX-only fast path, or goes away.

Cost: recompute, and higher per-tick latency than true cache streaming. Quality
is arguably *better* than local-attn — the window gets full attention, closer to
offline behaviour.

### Option B — export our checkpoint cache-aware via NeMo

Run NeMo's exporter with streaming params set, producing `cache_last_channel` /
`cache_last_time` / `cache_last_channel_len` graph I/O. Closest to today's MLX
semantics.

Risk: NeMo's cache-aware path expects `att_context_style: chunked_limited` +
`causal_downsampling: True` + causal conv context. Our checkpoint is `regular` /
non-causal on all three, so we'd be forcing the export machinery onto a model
shape it wasn't written for — and cache-aware ONNX export has a history of
dimension-mismatch and latency bugs (NeMo issues #6381, #5867). Also needs a
Python/NeMo toolchain this project doesn't have. Doable, but it reintroduces the
"uncertain payoff" the ONNX route was meant to eliminate.

Keep in reserve if Option A's per-tick latency proves too high in practice.

### Option C — ship a natively-streaming model on Linux

`altunenes/parakeet-rs` has a ready-made cache-aware streaming export.
**[verified]** graph I/O:

```
in:  processed_signal, processed_signal_length, prompt_index,
     cache_last_channel [24,1,56,1024], cache_last_time [24,1,1024,8],
     cache_last_channel_len [1]
out: encoded, encoded_len, cache_last_channel_next, cache_last_time_next,
     cache_last_channel_len_next
```

`att_context_size [56, 6]`, `chunk_size_output_frames: 7`, 24 layers x 1024 —
same architecture class as ours. Best actual streaming *quality*, since it is
trained for limited context.

But it is `nemotron-asr-streaming-multilingual-0.6b`: **vocab 13087 vs our
8192**, plus a `prompt_index` language-conditioning input we have no equivalent
for. Different tokenizer, different transcripts, a second model to ship, and
Mac/Linux parity is gone. Only worth it if streaming latency/quality outranks
cross-platform sameness.

### Option D — MLX-CUDA for the streaming path only

We already have this built (see note at the end). But it means two GPU stacks on
one box, two sets of CUDA libs, and the fragile source build back on the
critical path. Hard to justify given Option A's numbers.

## Work involved

1. **Define `ParakeetBackend`** (the 2-call interface above) and route the
   decode loops (`src/mlx/rnnt.ts`, `ctc.ts`) through it. Smaller than it looks:
   `decodeTDTGreedy` is already ~90% plain TS — argmax, entropy/confidence,
   duration stepping, `maxSymbols`, alignment all operate on plain numbers. MLX
   touches only four spots: the per-step encoder slice, the token input, the two
   forwards, and `softmax(...).toFloat32()`. Note the loop only commits hidden
   state on non-blank emission — that logic is loop-level and stays
   backend-agnostic.
2. **`MlxBackend`** — thin wrapper over current `src/mlx/` modules.
3. **`OnnxBackend`** — two `ort.InferenceSession`s fed/returning `Float32Array`
   + shape. CUDA EP on Linux. ONNX returns logits, so `softmax` moves to plain
   JS (argmax is invariant under it, but the entropy-based confidence needs
   probabilities).
4. **Model assets** — **done, confirmed matching** (see table above). Fetch from
   HF alongside the existing safetensors.
5. **Backend selection** — pick by platform at load, with an override.
6. **Streaming** — implement Option A generically on top of `encode()`.
7. **Parity check** — diff transcripts: MLX-on-Mac vs ONNX-on-Linux vs ONNX-CPU.
8. **Packaging** — `onnxruntime-node` as an optional/peer dep. Note the 252 MB
   CUDA EP downloads **by default**, so Mac-only installs want
   `--onnxruntime-node-install=skip`. `ffmpeg` audio is already cross-platform.

## Prerequisites on the Linux box (for ONNX + CUDA)

- `onnxruntime-node` **>= 1.27** (CUDA 13 + cuDNN 9). Earlier versions need
  CUDA 12 instead.
- CUDA 13.x runtime + cuDNN 9.x on `LD_LIBRARY_PATH`. Already satisfied here by
  `~/cuda-13` and `~/buildprefix/lib` from the MLX build.
- Nvidia driver whose **loaded kernel module matches the installed userspace**.
  This box has unattended-upgrades enabled and drifts; a reboot resyncs it.
- The RTX card only needs a supported compute capability — no per-arch compile,
  ORT ships prebuilt CUDA kernels.

## Note: mlx-node's CUDA backend

mlx-node's CUDA path is an early proof-of-concept (device-agnostic eager
fallbacks, no tuned kernels, validated on ARM64 GB10/DGX Spark, no x86_64
prebuilt -> source build required).

We **did** build it successfully on this box for the RTX 4060 Ti (sm_89) — the
addon loads and MLX brings up its CUDA backend. But it cost a full source build:
`MLX_CUDA_ARCHITECTURES=89`, a hand-assembled CUDA/cuDNN/BLAS prefix, a rustc
upgrade, and a 113 MB hand-copied `.node` artifact. ORT sidesteps all of that
and reuses NVIDIA's mature CUDA kernels, so it remains the preferred Linux path.
The MLX-CUDA build stays available as a fallback (Option D).

## Reproducing the probe

Probe artifacts live in the session scratchpad `ortprobe/`:
`encoder-model.onnx` + `.data`, `dj.onnx`, `stream_enc.onnx`, and the
`enc.mjs` / `dj.mjs` benchmarks. Re-run them against the CUDA EP after a driver
reboot to fill in the **[unmeasured]** GPU numbers. The external-data file must
keep its original name (`encoder-model.onnx.data`) or ORT fails to resolve
initializers.

## Sources

- onnx-asr (Parakeet via ONNX Runtime): https://github.com/istupakov/onnx-asr
- Parakeet TDT v3 ONNX: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
- Streaming/cache-aware exports: https://huggingface.co/altunenes/parakeet-rs
- ONNX Runtime CUDA EP matrix: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX Runtime Node.js: https://onnxruntime.ai/docs/get-started/with-javascript/node.html
- NVIDIA NeMo (reference Parakeet, ONNX export): https://github.com/NVIDIA/NeMo
- mlx-node (experimental CUDA): https://github.com/mlx-node/mlx-node
