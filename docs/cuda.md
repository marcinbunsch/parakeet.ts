# Running on CUDA (Linux) — dual-backend design

Goal: run this library live on **both** a Mac (Apple Silicon) and a Linux box
with an Nvidia RTX GPU, **without changing app code**. Same public API, same CLI,
same server — the GPU work is swapped underneath by platform.

**Status (2026-07-28): validated end-to-end on an RTX 4060 Ti.** Parakeet TDT
0.6b v3 transcribes on the GPU through ONNX Runtime with no MLX in the path.
Everything below marked "measured" was run on this box; nothing in this doc is
speculative unless it says so.

## Decision: MLX on Mac + ONNX Runtime on Linux

Two backends, best performance on each:

- **Mac** keeps the MLX/Metal path — fastest on Apple Silicon (unified memory).
- **Linux** uses ONNX Runtime + CUDA EP.

This was originally a bet. It is now a measurement: mlx-node's CUDA backend does
work, but it OOMs above ~45 s of audio and is ~3x slower than ONNX on the same
GPU (see "Why not mlx-node's CUDA backend" below).

## Architecture: abstract at the model boundary

We do **not** give this codebase a second tensor library. We abstract at the
**model boundary**. The library is not one monolithic graph; it's:

1. Audio -> mel spectrogram — CPU
2. **Encoder** (conformer) forward — the heavy compute
3. A **decode loop** (TDT/RNNT greedy) that repeatedly calls the prediction
   network (LSTM) and joint network -> logits -> argmax
4. Logits -> tokens -> text — tokenizer/alignment, already pure TypeScript

Only steps 2 and 3 touch tensors. Everything else — greedy loop, argmax,
tokenizer, alignment, CLI, server, mutex — is plain TypeScript and stays
identical on both platforms.

### The backend interface is 2 calls, not 3

An earlier draft of this doc proposed `encode` / `predict` / `joint`. That is
wrong for ONNX: the pre-exported model **fuses prediction and joint into a
single `decoder_joint` graph**, and it cannot be split without re-exporting.

```
encoder-model.onnx
  in : audio_signal [1, 128, T_mel], length [1]
  out: outputs [1, 1024, T], encoded_lengths [1]

decoder_joint-model.onnx
  in : encoder_outputs [1, 1024, 1], targets [1,1], target_length [1],
       input_states_1 [2,1,640], input_states_2 [2,1,640]
  out: outputs [1,1,1,8198], prednet_lengths [1],
       output_states_1, output_states_2
```

So the interface is:

```ts
interface ParakeetBackend {
  encode(mel):                        { data: Float32Array; shape: number[] }
  decodeStep(encFrame, token, state): { logits: Float32Array; state: State }
}
```

`MlxBackend` fuses its existing `predict` + `joint` behind one `decodeStep` —
trivial, since `src/mlx/rnnt.ts` already calls them back-to-back. The LSTM state
is opaque to the loop (it is only threaded through), so it types cleanly as a
backend-specific handle.

The decode loop itself barely changes: `decodeTDTGreedy` is already ~90% plain
TypeScript. MLX appears in only four places — the per-step encoder slice, the
token input, the two forwards, and `softmax(...).toFloat32()`.

## Measured performance (RTX 4060 Ti, 75.5 s of audio, warm)

| Config | Time | RTFx | Breakdown |
|---|---|---|---|
| **ONNX, encoder + decoder on CUDA** | **0.74 s** | **102x** | mel 326 ms · enc 166 ms · dec 223 ms |
| ONNX, encoder CUDA + decoder CPU | 0.87 s | 87x | enc 155 ms · dec 363 ms |
| ONNX, all CPU | 2.58 s | 29x | enc 1690 ms |

The GPU gives roughly **10x on the encoder** (1690 ms -> 166 ms). Running the
autoregressive decode loop on CUDA is modestly better than CPU (223 ms vs
363 ms) despite ~591 individual `session.run` round trips.

**The mel front-end is now the largest single cost (~44% of wall time).** The
next optimization target is the front-end, not the GPU work.

## Model assets — confirmed to match this checkpoint

`istupakov/parakeet-tdt-0.6b-v3-onnx` (also mirrored under
`altunenes/parakeet-rs/tdt/`), verified against the `config.json` this library
already loads:

| | local checkpoint | ONNX export |
|---|---|---|
| d_model | 1024 | `encoder_outputs` = 1024 |
| pred_hidden / layers | 640 / 2 | states `[2, 1, 640]` |
| vocab + blank + durations | 8192 + 1 + 5 | logits width **8198** |
| features / subsampling | 128 / 8 | `features_size` 128, `subsampling_factor` 8 |

Files: `encoder-model.onnx` (42 MB) + `encoder-model.onnx.data` (2.44 GB
external data — the name must be preserved or ORT cannot resolve initializers),
`decoder_joint-model.onnx` (72.5 MB), `vocab.txt` (8193 lines, blank at 8192),
`nemo128.onnx` (INT8 preprocessor — **we do not use this**, see below).

Decoding constants: durations `[0,1,2,3,4]`, blank id `8192`. The initial
`targets` value is the blank id, whose embedding row is `padding_idx` and
therefore zero — matching the MLX path's `lastToken = null` branch.

## Parity: the divergence is the mel front-end, not the model

The ONNX transcripts initially differed from MLX. Swapping front-ends against a
fixed encoder isolates the cause:

| Front-end | Encoder | sample-1 | sample-3 |
|---|---|---|---|
| `nemo128.onnx` (NVIDIA reference) | ONNX | "a go **then**" | "**uh**" |
| this repo's mel | ONNX | "a go **in**" | "**um**" |
| this repo's mel | MLX | "a go **in**" | "**um**" |

Same mel in -> same transcript out, on all three fixtures. **The ONNX encoder
and decoder_joint are numerically faithful to the MLX path.** A pure-JS port of
the repo's mel matches the MLX original at correlation 0.9999999977
(mean abs diff 5.3e-5) once degenerate bins are excluded.

Therefore: **use this repo's own mel front-end on both platforms and drop
`nemo128.onnx`.** That removes a dependency, removes an INT8 quantization step,
and makes the two backends agree.

### Known bug: 13 of 128 mel bins are dead

`computeMelFilterbanks` in `src/mlx/audio.ts` (inherited from parakeet-mlx) maps
mel points to FFT bins with `Math.floor(hz * (nFft + 1) / sr)`. At low
frequencies consecutive points collapse onto the same bin, producing **13
entirely all-zero filterbank rows** (mels 0,2,4,6,8,10,13,15,18,21,24,28,34).

Those rows are constant `log(1e-5)` across time. Per-feature normalization then
computes `(x - mean) / (std + 1e-5)` on a constant row — dividing ~0 by ~0 — so
the output is pure float rounding noise, amplified to O(0.1). Different
float32 reduction orders (MLX vs JS) produce different noise, and that noise is
what flips knife-edge tokens between backends.

A standard librosa/NeMo-style interpolated filterbank (triangles evaluated on
continuous frequencies, no `floor`) leaves only 1 dead row, and with it the ONNX
path converges *exactly* on NVIDIA's reference preprocessor output.

**Consequence for the test suite:** `test/integration/transcribe.test.ts`
expects "a go in" / "um" — which is what the *current, buggy* filterbank
produces. The reference preprocessor yields "a go then" / "uh". Those fixtures
record current MLX behaviour, not independent ground truth, so fixing the
filterbank will change 2 of 3 expected transcripts. Decide deliberately before
changing it.

(That test also hardcodes a Mac-only HF snapshot path,
`models--mlx-community--.../snapshots/<hash>`, so it cannot run on Linux, where
the CLI's downloader uses a flat cache layout.)

## Prerequisites on the Linux box — resolved

`onnxruntime-node` 1.27.0 auto-installs the CUDA EP for linux-x64 on a plain
`npm install` (252 MB `libonnxruntime_providers_cuda.so`). Its dependencies:

```
libcudart.so.13  libcublas.so.13  libcublasLt.so.13  libnvrtc.so.13
libcudnn.so.9    libcufft.so.12   libcurand.so.10    libcuda.so.1
```

**ORT 1.27 switched its default GPU build to CUDA 13 + cuDNN 9.** ORT <= 1.26
wants CUDA 12, so 1.27+ is required unless a CUDA 12 runtime is also installed.
On this box `~/cuda-13` (13.0.2) and `~/cudnn` (9.25) satisfy all of it:

```
LD_LIBRARY_PATH=$HOME/buildprefix/lib:$HOME/cuda-13/lib64
```

The RTX card needs only a supported compute capability — ORT ships prebuilt
CUDA kernels, so there is no per-arch compile and no `sm_XX` fight.

Packaging note: because the CUDA EP downloads by default, a Mac-only install
should pass `--onnxruntime-node-install=skip`.

## Streaming

### The existing MLX streaming path is broken on every platform

Before comparing backends: `StreamingParakeet` does not currently work at all,
and this is **not** CUDA- or Linux-specific.

`transcribeStream()` defaults to `keepOriginalAttention = false`, which calls
`encoder.setAttentionModel('rel_pos_local_attn', contextSize)`. In
`src/mlx/conformer.ts:184`:

```ts
setAttentionModel(name, contextSize = [256, 256]): void {
  const newAttn = this.buildAttention(name, contextSize);
  // Copy weights from old attention if possible
  // (In a real implementation we'd need to transfer parameters)
  this.selfAttn = newAttn;
}
```

It swaps in a **freshly constructed** attention module and never transfers the
trained weights, so the first streaming update dies with
`TypeError: Cannot read properties of undefined (reading 'transpose')` inside
`RelPositionMultiHeadLocalAttention.forward`. That is a plain code gap (the
comment says as much), independent of backend.

Passing `keepOriginalAttention = true` avoids that path but hits a second,
separate bug in the cached encoder: `[broadcast_shapes] Shapes (1,12,1024) and
(1,8,1024) cannot be broadcast`.

**Consequence:** there is no working streaming behaviour to preserve parity
with. We are free to design streaming for correctness rather than to replicate
the current MLX path, and the MLX streaming bugs need fixing on their own merits.

### Chosen approach: sliding-window re-encode (measured, works today)

The offline ONNX encoder exposes **no cache tensors** — its graph I/O is only
`audio_signal`/`length` -> `outputs`/`encoded_lengths`. Rather than obtain a
cache-aware graph, keep a bounded window of recent audio, re-encode it on each
update, and re-decode from the last committed boundary. Tokens older than a
`drop` margin are finalized; the tail is draft and may be revised.

Measured on 60 s of audio fed in 1 s chunks, RTX 4060 Ti:

| Window | Avg latency / update | p90 | Max | Headroom |
|---|---|---|---|---|
| 12 s | **98 ms** | 101 ms | 228 ms | ~10x real-time |
| 24 s | 159 ms | 183 ms | 229 ms | ~6x |
| 60 s (never slides) | 227 ms | 350 ms | 436 ms | ~4x |

At a 12 s window the split is mel 61 ms, encoder 29 ms, decode 7 ms. **The mel
front-end dominates streaming latency**, and the prototype recomputes it over
the whole window every update — computing only the new frames incrementally
should cut per-update latency roughly in half.

Why this is the right choice here:

- **No new model assets**, no NeMo export, no second checkpoint.
- **Better quality than cache-aware local attention**, because the encoder sees
  full bidirectional context within the window instead of a limited left/right
  context the model was never trained for.
- **It needs only `encode` + `decodeStep`** — the same 2-call backend interface
  as batch transcription, with no cache type to abstract. So streaming becomes
  backend-agnostic and works identically on MLX and ONNX, which also fixes the
  MLX streaming hole.

Cost: recompute. Each update re-encodes the whole window rather than only new
frames. The measurements above show that is affordable by a wide margin.

Known gap in the prototype: commit-boundary frame accounting is approximate and
drops a segment near the start. This is engineering to finish, not a limitation
of the approach — it reproduces identically with a non-sliding window, so it is
the finalize/draft split, not the window movement.

### Alternatives, if the window cost ever matters

The checkpoint is fully non-causal, so cache-aware streaming is not a simple
re-export:

```
self_attention_model  rel_pos          att_context_size    [-1, -1]
att_context_style     regular          causal_downsampling False
conv_context_size     None             conv_kernel_size    9
```

NeMo's cache-aware export path assumes `att_context_style: chunked_limited` plus
causal downsampling.

- **B. Custom NeMo cache-aware export** of this checkpoint with
  `att_context_size` set. Precedent exists (parakeet-rs published cache-aware
  graphs with `cache_last_channel` / `cache_last_time` /
  `cache_last_channel_len`), but nobody has published one for
  parakeet-tdt-0.6b-v3 — we would run the exporter ourselves, and the result is
  an approximation the model was not trained for.
- **C. A natively streaming checkpoint** such as
  `nemotron-3.5-asr-streaming-0.6b` (cache-aware, `att_context_size [56, 6]`,
  chunk 7 output frames). Ready-made and trained for streaming, but a different
  vocab (13087) plus an extra `prompt_index` input, so transcripts would not
  match the Mac path.

## Why not mlx-node's CUDA backend — measured

mlx-node's CUDA path *does* work: it builds from source for sm_89 and
transcribes all three fixtures correctly. It was rejected on measurements:

- **OOM ceiling around 45 s of audio.** 10 s -> 15x RTFx, 20 s -> 27x,
  30 s -> 34x, then 60 s and 75 s die with
  `cudaPeekAtLastError() failed: out of memory` on a 16 GB card. ONNX handles
  75 s at 102x on the same GPU. (`transcribe()` does accept `chunkDuration` to
  work around this; the CLI never passes it.)
- **A separate bug at 45 s**: `[squeeze] Cannot squeeze axis 1 with size 0`.
- **~3x slower** than ONNX at 30 s (34x vs 101x RTFx).

It also needs two environment fixes that ONNX does not:

1. `CUDA_HOME` / `CUDA_PATH` must point at the toolkit — MLX JIT-compiles
   kernels through NVRTC and needs the headers at runtime.
2. CUDA 13 moved the CCCL headers under `include/cccl/`. nvcc adds that path
   automatically; NVRTC does not, so JIT fails with
   `cannot open source file "cuda/std/tuple"`. Fixed with additive symlinks:
   `cd $CUDA_HOME/targets/x86_64-linux/include && ln -s cccl/cuda cuda`
   (likewise `cub`, `thrust`).

ORT sidesteps all of this and reuses NVIDIA's own tuned kernels.

## Implementation status

Built and verified on this box:

| Module | What it is |
|---|---|
| `src/backend.ts` | `ParakeetBackend` — `encode` + `decodeStep`, plus `EncoderLayout` so each backend reports its own memory order instead of paying for a transpose |
| `src/audio.ts` | Shared mel front-end, pure TypeScript, no tensor library |
| `src/decode.ts` | Shared greedy TDT / RNN-T loops over typed arrays |
| `src/model.ts` | `ParakeetModel` + `StreamingParakeet` — no tensor code at all |
| `src/onnx/backend.ts` | `OnnxBackend`, lazy `onnxruntime-node` import, CUDA EP |
| `src/onnx/parakeet.ts` | Loader for exported ONNX graphs |
| `src/mlx/backend.ts` | `MlxBackend` — adapts the existing MLX modules to the same interface |
| `src/mlx/load.ts` | Loads a safetensors checkpoint into `ParakeetModel` |

Both loaders return the **same** `ParakeetModel`:

```ts
import { fromLocal } from 'parakeet.ts/onnx';   // ONNX Runtime, CUDA on Linux
import { loadModel } from 'parakeet.ts/mlx';    // MLX, Apple Silicon

const model = await fromLocal('/path/to/onnx-model', { executionProvider: 'cuda' });
const result = await model.transcribe('audio.wav');
const stream = model.transcribeStream({ windowSeconds: 12 });
```

**Verified:** feeding identical features through both backends produces
byte-identical transcripts on all three fixtures — via the internal interface
and via the public API. Streaming, word timestamps, and chunked long-form
transcription all work on the ONNX/CUDA path.

`onnxruntime-node` is an `optionalDependency`, so a Mac-only install never pulls
the CUDA libraries. Note pnpm blocks its postinstall by default; the repo now
sets `pnpm.onlyBuiltDependencies` so the CUDA EP is actually fetched.

### Single API, and the filterbank default

The legacy `ParakeetTDT` / `ParakeetRNNT` / `ParakeetCTC` wrapper classes and the
old cache-aware `StreamingParakeet` have been **removed**. Both backends now load
into the one shared `ParakeetModel`, and the CLI and HTTP server drive it. CTC /
hybrid TDT-CTC checkpoints are no longer supported (the shared decode path and
the ONNX export are TDT / RNN-T only).

The shared path defaults to `filterbank: 'interpolated'`, because it is both
correct and self-consistent:

| Front-end | Backends agree? | sample-1 |
|---|---|---|
| `floor` (legacy MLX front-end, float noise in dead bins) | no — noise differs per backend | "a go in" |
| `floor` (shared, deterministic zeros in dead bins) | yes | "a go **golven**" |
| `interpolated` (NeMo reference) | yes | "a go then" |

The middle row is what makes the case: with the collapsed filterbank the model
emits the non-word "golven", and the only reason the old MLX path avoided it was
float rounding noise landing favourably. Pass `{ filterbank: 'floor' }` to either
loader to reproduce the legacy features exactly.

The two integration fixtures now assert the interpolated transcripts
("a go then", "uh give them a card"); the old buggy "a go in" / "um" strings are
gone with the legacy classes.

### Remaining work

- Make the streaming mel incremental — it currently recomputes the whole window
  each update and is ~60% of per-update latency.
- Finish the streaming commit-boundary accounting (a token can be mangled at a
  window boundary; `streaming.test.ts` fences this with a WER bound today).
- Fetch ONNX weights from HuggingFace rather than requiring a local directory.
- Re-add CTC / hybrid TDT-CTC to the shared path if a checkpoint needs it (would
  need a CTC `decodeStep` shape on the backend interface).

## Reproducing

Probe artifacts live in `~/parakeet-ortprobe/` (3.0 GB, models re-fetchable):

```
cd ~/parakeet-ortprobe
LD_LIBRARY_PATH=$HOME/buildprefix/lib:$HOME/cuda-13/lib64 EP=cuda node run2.mjs <file.wav>
```

`run2.mjs` is the zero-MLX pipeline, `mel.mjs` the pure-JS front-end,
`melcmp*.mjs` the parity harnesses. `EP` / `DEC_EP` select execution providers;
`MEL_FB=interp` switches to the interpolated filterbank.

## Sources

- onnx-asr (Parakeet via ONNX Runtime): https://github.com/istupakov/onnx-asr
- Parakeet TDT v3 ONNX export: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
- Cache-aware streaming ONNX exports: https://huggingface.co/altunenes/parakeet-rs
- ONNX Runtime CUDA EP matrix: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX Runtime Node.js: https://onnxruntime.ai/docs/get-started/with-javascript/node.html
- NVIDIA NeMo (reference Parakeet, ONNX export): https://github.com/NVIDIA/NeMo
- mlx-node (experimental CUDA): https://github.com/mlx-node/mlx-node
