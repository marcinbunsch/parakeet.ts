# Ideas — deferred / out of v1 scope

Things we deliberately did not build into v1 but might revisit. See `docs/api-plan.md` for what *is* in v1.

## Streaming protocol extensions

- **Live partial transcripts via WebSocket or SSE.** The data is already produced by `StreamingParakeet` (`finalizedTokens` / `draftTokens`), so it would just be a transport addition. Triggers: a real use case for showing in-progress transcripts to a user (chat UX, captions, voice agents).
- **Async iterator wrapper on `StreamingParakeet`.** Something like `for await (const { finalized, draft } of stream.feed(source)) { ... }`. Sugar over the imperative core. Wait until at least one client wants live partials.
- **Resumable / sessionful streams.** Request-id-keyed cache so a dropped client can reconnect and resume. Real engineering (storage, TTLs, IDs); only worth it if streaming becomes a primary product surface.

## Audio input flexibility

- **Non-PCM formats on the server.** Accept `audio/wav`, `audio/mp3`, `audio/webm; codecs=opus`, etc., by piping through ffmpeg server-side. Adds ffmpeg as a runtime dep for server users. Possibly ships as a separate adapter (`parakeet-mlx/server-ffmpeg`) to keep the core server lean.
- **Multi-format streaming PCM.** Negotiate sample rate and encoding via `Content-Type` parameters (s16le, 8k/16k/24k/48k). Pure-JS resampler server-side. Friendlier to browser AudioWorklet output, which is typically 48kHz.
- **File upload endpoint** (non-streaming). `POST /transcribe-file` with multipart upload, writes to temp file, runs existing `model.transcribe(path)`. Most of the code is there; deferred only because it overlaps with the CLI's job for long-form audio.

## CLI extensions

- **Subcommands.** `parakeet-mlx pull <model>` (pre-download), `parakeet-mlx serve` (built-in HTTP server). Add only if the bare form proves insufficient. Dispatch is ~10 lines; doesn't need a framework.
- **`--quiet` flag** to suppress stderr progress entirely. Today users can `2>/dev/null` if they want silence.
- **Output formats other than text/JSON.** SRT, VTT, plain timestamps. Easy to add given we always have the full `AlignedResult`.

## VAD and preprocessing

- **VAD integration.** Deliberately out of scope for this package. Belongs in a separate package (Silero via `onnxruntime-node`, or the marcinbunsch/vad fork). Composes cleanly via user code:
  ```ts
  for await (const frame of audioSource) {
    if (vad.isSpeech(frame)) stream.addAudio(frame);
  }
  ```
- **Noise suppression / AGC.** Same logic — separate package, composes by transforming the PCM stream before it reaches `addAudio`.

## Pooling and scaling

- **First-class `ParakeetPool`** class with `acquire`/`release` semantics, least-busy routing, lazy loading. Today: 5 lines of user code does round-robin, which is what the GPU constraint actually wants. Revisit only if multi-GPU or multi-process intra-package scheduling becomes a thing.
- **Cross-process model sharing.** MLX uses unified memory; in theory two processes could share weights via shared memory. Not supported by `@mlx-node/core` today. Massive engineering effort.

## Observability

- **Structured logging hook** on the server (per-request id, duration, audio length, mutex wait time). Today: server logs nothing by default; users can wrap routes themselves.
- **Metrics endpoint** (`GET /metrics`, Prometheus-style). Deferred until someone actually deploys this seriously.
- **Health endpoint** (`GET /healthz`). Trivial to add; revisit when a deployment story exists.

## Build / release

- **Dual CJS + ESM build.** Today: ESM only. Most consumers are fine. Revisit if a notable library/runtime can't consume ESM.
- **CI.** GitHub Actions running the integration test on a self-hosted Apple Silicon runner (the only place this code can actually run end-to-end).
- **Semver / changelog discipline.** Stays informal during 0.x.

## Speculative

- **Streaming transcription quality knobs** — expose `contextSize`, `depth` from `transcribeStream()` as factory options at the server level (currently SDK only).
- **Mid-stream language switching.** Parakeet TDT v3 is multilingual; might be possible to detect a switch and reset finalized tokens at a boundary. Research-y.
- **Word-level confidence scores.** Not produced by the current decoders, but could be added.
- **Speaker diarization.** Way out of scope; integrate via a separate package that takes `AlignedResult` + audio.
