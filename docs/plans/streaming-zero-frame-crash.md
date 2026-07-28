# Plan: fix streaming crash on undersized audio chunks

## Context for the executing agent

This is the `parakeet-mlx` / `parakeet.ts` package — a TypeScript port of NVIDIA
Parakeet ASR running on Apple Silicon via `@mlx-node/core` (MLX bindings). The
heavy math runs on MLX; several primitives (audio front-end, LSTM, conv) are
reimplemented in TS. You do not need any prior conversation context; everything
needed is below.

The streaming API lives in `src/mlx/parakeet.ts`:
- `StreamingParakeet.addAudio(audio: Float32Array)` — feed a PCM frame (16 kHz mono).
- `consumePcmStream(stream, asyncIterable)` — helper that calls `start()`,
  loops `addAudio`, then `stop()`, returning the final `AlignedResult`.

The compiled output in `dist/` is **vendored in the repo** (committed). After any
`src/` change you must rebuild: `npm run build` (runs `tsc`), and commit the
regenerated `dist/` files alongside the source.

## The bug

Feeding audio chunks that are too small to produce even one *subsampled* encoder
frame crashes the process with a native MLX abort. Observed messages (the exact
one depends on chunk size and which MLX op hits the zero-size tensor first):

```
libc++abi: terminating due to uncaught exception of type std::invalid_argument:
  [squeeze] Cannot squeeze axis 1 with size 0 which is not equal to 1.
```
```
libc++abi: terminating due to uncaught exception of type std::invalid_argument:
  [max] Cannot max reduce zero size array.
```

Because it is an uncaught C++ exception from the native addon, it **terminates
the whole process** — it cannot be caught with `try/catch` in JS. A streaming
server (`packages/server`) that forwards small/irregular PCM frames will crash.

### Root cause

In `StreamingParakeet.addAudio` (`src/mlx/parakeet.ts`, ~lines 563–607):

```ts
const subFactor = this.model.encoderConfig.subsamplingFactor;      // e.g. 8
const melFrames = Number(this.melBuffer.shape()[1]);
const usableMelFrames = Math.floor(melFrames / subFactor) * subFactor;

const melInput = this.melBuffer.slice(
  s(0, 0, 0),
  s(1, usableMelFrames, Number(this.melBuffer.shape()[2])),
);
const [features, lengths] = this.model.encoder.forward(melInput, null, this.cache);
```

When the accumulated `melBuffer` has fewer than `subFactor` frames,
`usableMelFrames` rounds down to **0**. `melInput` is then `[1, 0, nMels]`, and
`encoder.forward` runs on a zero-length sequence, aborting downstream (in the
subsampling conv, an attention `max`/softmax, or a `squeeze`).

A second, related edge: when `audioBuffer.length < hopLength`, `usableLen` is 0,
so `getLogMel` is called on an empty signal — which can also crash before we even
reach the encoder.

### Reproduction (fails on current HEAD)

```js
// node, from repo root, with the model cached locally
const { fromLocal } = await import("./dist/mlx/utils.js");
const { consumePcmStream } = await import("./dist/mlx/parakeet.js");
const model = fromLocal(MODEL_PATH);
async function* tiny() { for (let i = 0; i < 5; i++) yield new Float32Array(1024).fill(0.001); }
await consumePcmStream(model.transcribeStream(), tiny()); // <-- native abort
```

`MODEL_PATH` = the HF cache snapshot dir for `mlx-community/parakeet-tdt-0.6b-v3`
(see `test/integration/transcribe.test.ts` for the exact path pattern).

## Reference behavior

The upstream Python `parakeet_mlx` (vendored at `parakeet-mlx/parakeet_mlx/parakeet.py`,
`add_audio`, ~lines 999–1041) has **no explicit guard** either — it happens to be
driven with large enough chunks in practice. So there is nothing to port; the TS
port needs a defensive guard that Python lacks. The guard must not change
behavior for normal-sized chunks (the existing passing tests must stay green).

## The fix

Add early-return guards in `StreamingParakeet.addAudio` so a call that cannot yet
produce a usable encoder frame just **buffers and returns**, leaving all state
(`melBuffer`, `audioBuffer`, `_finalizedTokens`, `_draftTokens`, `decoderHidden`,
`lastToken`, `cache`) untouched. The next `addAudio` keeps accumulating until
there is enough audio.

Concrete steps in `addAudio`:

1. **Guard empty audio before `getLogMel`.** After computing `usableLen`, if
   `usableLen === 0`, `return` immediately (append-to-buffer already happened at
   the top; nothing else to do this call).

2. **Guard zero usable mel frames.** After computing `usableMelFrames`, if
   `usableMelFrames === 0`, `return` before slicing `melInput` / calling the
   encoder. Do **not** run the mel-buffer trim in this case — we want the frames
   to accumulate for the next call.

   Important ordering detail: in the current code the line
   `this.audioBuffer = this.audioBuffer.slice(usableLen)` and the `melBuffer`
   concatenation happen *before* the `usableMelFrames` computation, so those
   still run (correct — consumed audio has become mel frames buffered in
   `melBuffer`). Only the encode + decode + mel-trim tail must be skipped.

3. (Defensive, optional) After `const length = Number(lengths.toInt32()[0])`,
   if `length === 0`, `return` before the two-phase decode. With guard (2) in
   place this should be unreachable for the 0.6b model (8 mel frames → 1 encoder
   frame), but it is cheap insurance against other subsampling configs.

Keep the guards minimal and place a short comment explaining why (undersized
chunk → zero-length sequence → native MLX abort that JS cannot catch).

### Sketch (adapt to the real surrounding code — do not paste blindly)

```ts
const usableLen = Math.floor(this.audioBuffer.length / hopLen) * hopLen;
if (usableLen === 0) return; // not enough audio for a single STFT frame yet

// ... getLogMel, append to melBuffer, drop consumed audioBuffer ...

const usableMelFrames = Math.floor(melFrames / subFactor) * subFactor;
if (usableMelFrames === 0) return; // not enough mel frames for one subsampled frame

// ... existing encode + two-phase decode + mel-buffer trim ...
```

## Tests

Add a case to `test/integration/transcribe.test.ts` (or a new
`test/integration/streaming-robustness.test.ts`). Two assertions:

1. **No crash on tiny chunks.** Drive a real sample through very small frames
   (e.g. 512–2048 samples each) via `consumePcmStream` and assert it resolves
   without throwing and returns a non-empty transcript once enough audio has
   accumulated. This is the regression guard for this bug.

2. **Equivalence to normal chunking (sanity).** Optionally assert that feeding a
   sample as tiny frames yields the same (or a stable-prefix match of the)
   transcript as feeding it as one large frame — small chunks must not corrupt
   output, only defer it.

Note: these are integration tests; they need the model cached locally and
`ffmpeg` on PATH, matching the existing `transcribe.test.ts` setup. Run with
`npm run test:integration` (vitest). Redirect noisy output to a file and grep
with ASCII-only patterns (`FAIL`, `passed`, `failed`) — vitest prints Unicode
glyphs.

## Acceptance criteria

- The reproduction snippet above no longer crashes; it returns a transcript.
- New robustness test passes.
- All existing tests in `test/integration/transcribe.test.ts` still pass
  (6 tests: 3 non-streaming exact-match, 3 streaming stable-prefix).
- `npx tsc --noEmit` is clean.
- `dist/` rebuilt (`npm run build`) and committed with the `src/` change.

## Out of scope

- The known streaming **tail-accuracy drift** (transcripts differ from Python by
  a word or two at the very end) is a separate issue caused by a TS-vs-Python
  numerical-fidelity gap in the audio front-end / encoder, not by this crash. Do
  not try to fix it here.
