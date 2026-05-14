# parakeet-mlx — API Plan (v1)

Three usage modes shipped from a single npm package: **SDK**, **CLI**, and **HTTP server**.

## Decisions locked in

| # | Decision |
|---|---|
| Q1 | SDK: two-step explicit API (`fromPretrained` → `model.transcribe`); no top-level `transcribe()` helper. |
| Q2 | Streaming: keep imperative core; add `consumePcmStream` helper; expose `finalizedTokens` / `draftTokens` (and `finalizedResult` / `result` getters). |
| Q3 | Wire format for streamed PCM: strict `audio/pcm; rate=16000; channels=1; format=f32le`. |
| Q4 | Server: user instantiates the model and passes it in (`createParakeetRoutes({ model })`). |
| Q5 | Concurrency: per-instance async mutex in `BaseParakeet` — safe under concurrent `await`s, enables a round-robin pool with zero extra plumbing. |
| Q6 | CLI: single-purpose binary, no subcommands, uses `node:util.parseArgs`. |
| Q7 | Package name: `parakeet-mlx` (npm name confirmed available). |
| Q8 | Output schema: always emit the full `AlignedResult` (`text` + `sentences` with token timing). One schema for CLI `--json` and server `200`. No `--timestamps` flag. |
| Q9a | Server: `maxDurationSeconds` factory option, default **300s (5 min)**. Streaming is for short bursts; long-form goes through the CLI. |
| Q9b | Server: `idleTimeoutMs` factory option, default **30 s**. |
| Q9c | Server: silent disconnect drop. No resumable sessions. |
| Q10 | CLI auto-download: hand-rolled progress bar on stderr with `\r` redraw; falls back to a single "downloading..." line if `process.stderr.isTTY` is false. |
| Q11 | No first-class pool helper. The 5-line user-code pattern is documented; nothing in the SDK. |
| Q12 | Server error responses: JSON envelope `{ error: "<code>", message: "<human>" }`. |
| Q13 | v1 scope: see "Definition of done" at the bottom. Everything else is in `docs/ideas.md`. |

**Non-goals for this package:** VAD (use a separate package; e.g. a `@ricky0123/vad` fork). Non-PCM audio on the streaming endpoint. Resumable sessions. WebSocket / SSE. See `docs/ideas.md` for the full deferred list.

## Package shape

Single npm package `parakeet-mlx` with multiple entry points:

```
parakeet-mlx           → SDK (default, no extra deps)
parakeet-mlx (bin)     → CLI binary
parakeet-mlx/server    → server factory (requires hono)
```

`package.json` (additions/changes from current):

```json
{
  "name": "parakeet-mlx",
  "type": "module",
  "bin": { "parakeet-mlx": "./dist/cli.js" },
  "exports": {
    ".": "./dist/index.js",
    "./server": "./dist/server.js"
  },
  "peerDependencies": { "hono": "^4" },
  "peerDependenciesMeta": { "hono": { "optional": true } }
}
```

- **SDK users**: `npm i parakeet-mlx` → done.
- **CLI users**: same install, run `npx parakeet-mlx file.wav`.
- **Server users**: `npm i parakeet-mlx hono @hono/node-server` and import from `parakeet-mlx/server`.

README disclaimer: community TypeScript port, not affiliated with NVIDIA or the upstream Python `parakeet-mlx` project.

## 1. SDK

### Canonical usage

```ts
import { fromPretrained } from 'parakeet-mlx';

const model = await fromPretrained('mlx-community/parakeet-tdt-0.6b-v3');
const result = await model.transcribe('file.wav');
console.log(result.text);
```

Two steps, explicit. No hidden global model. Loading is expensive and the user should control when it happens.

### Concurrency safety

`BaseParakeet` carries a per-instance async mutex. All entry points that touch the encoder (`transcribe`, `transcribeStream`'s `addAudio`, etc.) acquire it. Two concurrent `await model.transcribe(...)` calls on the same instance serialize automatically — no silent GPU-state corruption.

Pooling for tail-latency / mixed-model serving is just an array of instances:

```ts
const pool = await Promise.all([
  fromPretrained('mlx-community/parakeet-tdt-0.6b-v3'),
  fromPretrained('mlx-community/parakeet-tdt-0.6b-v3'),
]);

let i = 0;
function next() { return pool[i++ % pool.length]; }
```

On a single Apple Silicon GPU, pooling buys lower tail latency under bursty load, not raw throughput (GPU is single-resource). For real throughput scaling, run multiple processes behind a load balancer.

### Streaming API

`StreamingParakeet` keeps its current imperative core:

```ts
const stream = model.transcribeStream();
stream.start();
stream.addAudio(pcmFrame);     // Float32Array, 16kHz mono
// ... repeat as audio arrives ...
stream.stop();
const final = stream.result;   // AlignedResult
```

New getters to expose what the model already tracks internally:

- `stream.finalizedTokens: AlignedToken[]` — committed, will not change.
- `stream.draftTokens: AlignedToken[]` — within the rotating context window; may be revised on subsequent `addAudio` calls.
- `stream.finalizedResult: AlignedResult` — finalized tokens only, safe to ship/persist incrementally.
- `stream.result: AlignedResult` — full transcript (finalized + draft), the "best current guess."

Helper for the common case of "I have a stream of PCM frames, give me the final transcript":

```ts
async function consumePcmStream(
  stream: StreamingParakeet,
  source: AsyncIterable<Float32Array>,
): Promise<AlignedResult>;
```

Takes the most general iterator shape (`AsyncIterable<Float32Array>`) so it works with Node streams, Web `ReadableStream`s, Hono request bodies, etc. Both CLI's `--stream` and server's `POST /transcribe` use this internally.

### Output schema

`AlignedResult` (already defined in `src/alignment.ts`) is the canonical shape returned by everything that produces a transcript:

```ts
type AlignedResult = {
  text: string;
  sentences: AlignedSentence[];
};

type AlignedSentence = {
  text: string;
  start: number;
  end: number;
  duration: number;
  tokens: AlignedToken[];
};

type AlignedToken = {
  id: number;
  text: string;
  start: number;
  duration: number;
};
```

Both CLI `--json` and server `200` emit this exact shape.

### Public exports

`src/index.ts` is already reasonably tight (no leakage of `nn`, `attention`, `conformer` internals). One small audit pass before publish, plus the new streaming getters and `consumePcmStream`.

## 2. CLI

Single-purpose binary. No subcommands.

```
parakeet-mlx file.wav
parakeet-mlx file.wav --json
parakeet-mlx file.wav --model mlx-community/parakeet-tdt-0.6b-v3
parakeet-mlx --stream < pcm_f32le_16k.raw
```

- File: `src/cli.ts`, registered as `bin` in `package.json`.
- Arg parser: **`node:util.parseArgs`** (Node 18.11+, zero deps).
- Default model: `mlx-community/parakeet-tdt-0.6b-v3`.
- `--stream`: read raw f32le mono 16kHz PCM from stdin (same wire format as the server's streaming POST — symmetric contract).
- **stdout** = transcript (plain text) or full `AlignedResult` JSON (with `--json`).
- **stderr** = progress: model download (with progress bar), model load, audio decode.
- **Auto-download** on first run; cache under `~/.cache/huggingface/hub/` (same dir `fromPretrained` already uses). Progress bar via `\r` line redraw on stderr; falls back to a single "downloading..." line if `process.stderr.isTTY` is false (CI, redirected logs).
- Exit codes: `0` success, `1` file/IO error, `2` model error.

## 3. HTTP server (`parakeet-mlx/server`)

Exported as a Hono route factory; the user owns the lifecycle:

```ts
import { Hono } from 'hono';
import { serve } from '@hono/node-server';
import { fromPretrained } from 'parakeet-mlx';
import { createParakeetRoutes } from 'parakeet-mlx/server';

const model = await fromPretrained('mlx-community/parakeet-tdt-0.6b-v3');
const app = new Hono();
app.route('/asr', createParakeetRoutes({
  model,
  // optional, with defaults shown:
  maxDurationSeconds: 300,
  idleTimeoutMs: 30_000,
}));

serve({ fetch: app.fetch, port: 8080 });
```

Hono is an **optional peer dependency** — only installed when a user actually wants the server. The SDK and CLI never import it.

### Streaming endpoint contract

```
POST /transcribe
Content-Type: audio/pcm; rate=16000; channels=1; format=f32le
Body: raw PCM frames, streamed (HTTP chunked transfer encoding)

200 OK
{ "text": "...", "sentences": [...] }
```

Server behavior:

1. Validate `Content-Type` — reject anything other than the exact format above with `415 Unsupported Media Type`.
2. Open a `StreamingParakeet` session on the provided model. The model's mutex guarantees only one session per instance runs at a time; concurrent HTTP requests queue automatically.
3. Iterate the request body as an `AsyncIterable<Float32Array>` (reinterpreting incoming bytes as Float32 views), feeding `consumePcmStream`.
4. On body close: `stream.stop()`, return `200 { text, sentences }`.
5. If the body exceeds `maxDurationSeconds` of audio (~7.2 MB/min at f32le 16kHz): return `413 Payload Too Large`.
6. If no bytes arrive for `idleTimeoutMs`: return `408 Request Timeout`.
7. On client disconnect mid-stream: stop the session, release the mutex, log, no response.

**Design intent:** the streaming endpoint is for short bursts (live capture, chat-style voice input). Long-form audio (lectures, podcasts, hour-long recordings) should go through the CLI. The 5-minute default cap codifies this.

### Error response shape

All non-2xx responses use a JSON envelope:

```json
{ "error": "unsupported_media_type", "message": "..." }
```

Error codes:

| HTTP | `error` | When |
|---|---|---|
| 415 | `unsupported_media_type` | `Content-Type` not exactly `audio/pcm; rate=16000; channels=1; format=f32le`. |
| 413 | `payload_too_large` | Audio body exceeds `maxDurationSeconds`. |
| 408 | `idle_timeout` | No bytes received for `idleTimeoutMs`. |
| 500 | `internal_error` | Unexpected exception in the pipeline. |

### Browser client example

```ts
await fetch('/asr/transcribe', {
  method: 'POST',
  headers: { 'Content-Type': 'audio/pcm; rate=16000; channels=1; format=f32le' },
  body: pcmReadableStream, // from AudioWorklet, resampled to 16kHz mono f32
  duplex: 'half',
});
```

Clients are responsible for resampling to 16kHz mono f32 (typically from 48kHz AudioWorklet output). ~20 lines of client code, deliberately not the server's problem.

### Why chunked POST (not SSE / not WebSocket)

| | Chunked POST | SSE | WebSocket |
|---|---|---|---|
| Direction | Client → server streaming, server → client single response | Server → client only | Bidirectional |
| Fits "stream in, transcript on finish" | ✅ | ❌ wrong direction | Overkill |
| Protocol overhead | Plain HTTP | Plain HTTP | Upgrade handshake |
| Proxies / CDNs | Just works | Just works | Often needs config |

Live partial transcripts during capture would justify WebSocket — see `docs/ideas.md`.

### Concurrency

The model's internal mutex (Q5) is the queue. The server doesn't add its own. Multiple concurrent HTTP requests just `await` on the same mutex and serialize cleanly. For higher throughput, run multiple processes (each owns a model) or instantiate a pool of models and route requests across them in user code.

## Definition of done (v1)

What ships when we declare v1 publishable to npm:

1. **SDK**: per-instance mutex on `BaseParakeet`, new streaming getters on `StreamingParakeet`, `consumePcmStream` helper, exports audit, package name → `parakeet-mlx`.
2. **CLI**: `parakeet-mlx file.wav`, `--json`, `--model`, `--stream`. Auto-download with progress bar (TTY) / one-shot line (non-TTY). Smoke-tested on a real wav.
3. **Server**: `createParakeetRoutes` with the streaming POST endpoint; defaults of 5 min cap, 30 s idle; JSON error envelope. Hono peer-optional. Smoke-tested with a local Hono app and a real PCM stream from `curl --data-binary @file -H "..."`.
4. **README**: 5-line SDK example, CLI usage section, server example with disclaimer.
5. Existing integration test still passes.

Suggested implementation order: SDK → CLI → Server. Each builds on the previous; the CLI validates SDK ergonomics; the server reuses the CLI's PCM-streaming helper.
