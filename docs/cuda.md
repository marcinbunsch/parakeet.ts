# Running on CUDA (Linux) — dual-backend design

Goal: run this library live on **both** a Mac (Apple Silicon) and a Linux box
with an Nvidia RTX GPU, **without changing app code**. Same public API, same CLI,
same server — the GPU work is swapped underneath by platform.

## Chosen approach: abstract at the model boundary, offload CUDA to ONNX Runtime

We do **not** give this codebase a second tensor library, and we do **not** rely
on mlx-node's CUDA build (immature, painful to compile — see note at the end).

Instead we abstract at the **model boundary**. The library is not one monolithic
graph; it's:

1. Audio -> mel spectrogram — **already pure JS/CPU**
2. **Encoder** (conformer) forward — the heavy compute
3. A **decode loop** (TDT/RNNT greedy) that repeatedly calls:
   - **Prediction network** (LSTM, autoregressive over emitted tokens)
   - **Joint network** -> logits -> argmax
4. Logits -> tokens -> text — **tokenizer/alignment, already pure JS**

Only steps 2 and 3 touch tensors. Everything else — mel front-end, greedy loop,
argmax, tokenizer, alignment, CLI, server, streaming, mutex — is plain
TypeScript and stays identical on both platforms.

So the backend interface is tiny. A backend implements **three calls** returning
`{ data: Float32Array, shape: number[] }`:

```ts
interface ParakeetBackend {
  encode(mel):            EncoderOut       // conformer
  predict(tokens, state): { out, state }   // prediction net
  joint(encT, predU):     Logits           // joint net
}
```

- `MlxBackend` — wraps the existing `src/mlx/` modules. Mac, fast (Metal).
- `OnnxBackend` — wraps `onnxruntime-node` sessions. **CUDA execution provider
  on Linux.**

The public API (`ParakeetTDT.transcribe`, streaming, `createParakeetRoutes`, the
CLI) never changes. Backend is chosen at load time / auto-detected by platform.

## Decision: MLX on Mac + ONNX on Linux

Two backends, best performance on each:

- **Mac** keeps the MLX/Metal path — fastest on Apple Silicon (unified memory).
- **Linux** uses ONNX Runtime + CUDA EP.

Cost: two backends to keep numerically in sync. Accepted for the performance
win. (Alternatives considered: ONNX everywhere / drop MLX — simpler but slower
on Mac; ONNX default + MLX optional.)

## Why ONNX Runtime is the mature offload target

- `onnxruntime-node` is a single well-maintained npm package with a **CUDA EP**
  — no compiling MLX from source, no `sm_XX` arch fights, no mlx-node pain.
- We reimplement **nothing** of conv/attention. NeMo bakes those into the ONNX
  graph; ORT runs them with cuDNN kernels. That is the "offload" we want.
- Pre-exported Parakeet ONNX models already exist (e.g. the `onnx-asr` project
  and HuggingFace exports run exactly this model), so we may not need to run
  NeMo's exporter ourselves.
- ORT also runs on Mac (CPU/CoreML), so ONNX could be the only backend later if
  we ever want to drop MLX.

## Work involved

1. **Define `ParakeetBackend`** (the 3-call interface above) and route the
   existing decode loops (`src/mlx/rnnt.ts`, `ctc.ts`) through it. The loop
   logic, argmax, and slicing move to plain typed-array TS so they're
   backend-agnostic.
2. **`MlxBackend`** — thin wrapper over the current `src/mlx/` modules
   (encoder, prediction, joint forwards). Mostly re-exposing what exists.
3. **`OnnxBackend`** — three `ort.InferenceSession`s (encoder, prediction,
   joint), each `session.run()` fed/returning `Float32Array` + shape. Set the
   execution provider to CUDA on Linux.
4. **Model assets** — obtain/ship the ONNX weight files (encoder / prediction /
   joint) alongside the existing safetensors, or fetch from HF. Confirm they
   match the checkpoint this library already loads.
5. **Backend selection** — pick by platform at load, with an override.
6. **Parity check** — diff transcripts: MLX-on-Mac vs ONNX-on-Linux vs ONNX-CPU.
7. **Packaging** — `onnxruntime-node` as an optional/peer dep so a Mac-only
   install doesn't drag CUDA libs, and vice-versa. `ffmpeg` audio is already
   cross-platform.

## Prerequisites on the Linux box (for ONNX + CUDA)

- Nvidia driver + CUDA/cuDNN versions matching the `onnxruntime-node` build's
  CUDA EP requirements (check the installed ORT version's matrix).
- The RTX card only needs a supported compute capability — no per-arch compile,
  ORT ships prebuilt CUDA kernels.

## Note: why not mlx-node's CUDA backend

mlx-node does have an experimental CUDA path, but it is an early proof-of-concept
(device-agnostic eager fallbacks, no tuned kernels, validated only on ARM64
GB10/DGX Spark, no x86_64 prebuilt -> source build required). On an x86_64 + RTX
desktop it's a painful compile with uncertain payoff. The ONNX route sidesteps
all of that and reuses NVIDIA's own mature CUDA kernels.

## Sources

- onnx-asr (Parakeet via ONNX Runtime): https://github.com/istupakov/onnx-asr
- ONNX Runtime Node.js / execution providers: https://onnxruntime.ai/docs/get-started/with-javascript/node.html
- NVIDIA NeMo (reference Parakeet, ONNX export): https://github.com/NVIDIA/NeMo
- mlx-node (experimental CUDA): https://github.com/mlx-node/mlx-node
