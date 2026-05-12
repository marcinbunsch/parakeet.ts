# parakeet-mlx.ts — API Plan

Three usage modes shipped from a single npm package: **SDK**, **CLI**, and **HTTP server**.

## Package shape

Single `parakeet-mlx` package with multiple entry points:

```
parakeet-mlx           → SDK (default, no extra deps)
parakeet-mlx (bin)     → CLI binary
parakeet-mlx/server    → server factory (requires hono)
```

In `package.json`:

```json
{
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

## 1. SDK

Foundation — most of this exists already. Gaps to close:

- **Public API surface** — audit `src/index.ts` and split `public` vs `internal`. Currently everything (nn, attention, conformer internals) is exported, which is too much.
- **High-level convenience** — `transcribe(audioPath, options)` that wraps load + mel + decode. Most SDK users won't want to touch the pipeline.
- **Streaming API** — confirm `StreamingParakeet` exposes a clean async iterator:
  ```ts
  for await (const chunk of stream.feed(pcm)) { ... }
  ```
- **Types** — ensure `AlignedResult`, `Sentence`, `AlignedToken` are first-class exports.
- **README** — minimal install + 5-line usage example.

## 2. CLI

`parakeet-mlx file.wav` → prints transcript to stdout.

- New file: `src/cli.ts`, registered as `bin` in `package.json`.
- Args: positional file path.
- Flags:
  - `--model <id>` (default `mlx-community/parakeet-tdt-0.6b-v3`)
  - `--json` (emit aligned JSON instead of plain text)
  - `--timestamps`
  - `--stream` (read PCM from stdin)
- Use `cac` or a hand-rolled arg parser — avoid yargs/commander weight.
- **stdout** = transcript; **stderr** = progress (model load, audio decode, download). Quiet by default.
- **Auto-download model on first run** — fetch default model from HF, cache under `~/.cache/huggingface/hub/`, then transcribe.
- Exit codes: `0` success, `1` file error, `2` model error.

## 3. HTTP server

Exported as a factory, not a running server, so the user owns the lifecycle:

```ts
import { createParakeetRoutes } from 'parakeet-mlx/server';
import { Hono } from 'hono';
import { fromPretrained } from 'parakeet-mlx';

const model = await fromPretrained('mlx-community/parakeet-tdt-0.6b-v3');
const app = new Hono();
app.route('/asr', createParakeetRoutes({ model }));
```

This way we don't pin a Node/Bun/Deno runtime adapter — the user picks `@hono/node-server`, `bun serve`, or whatever fits.

### Streaming contract — chunked HTTP POST

Goal: stream audio frames as they are captured, get a single transcript when the client closes the body. No WebSocket or SSE needed.

```
POST /transcribe
Content-Type: audio/pcm; rate=16000; channels=1; format=f32le
Body: raw PCM frames, streamed (HTTP chunked transfer encoding)

200 OK
{ "text": "...", "sentences": [...] }
```

**Server behavior:**

1. Read the request body as a stream.
2. Feed frames into `StreamingParakeet` as they arrive (model stays warm, processes incrementally).
3. When the client closes the body, finalize and respond with `{ text, sentences }` JSON.

**Browser client example:**

```ts
await fetch('/asr/transcribe', {
  method: 'POST',
  headers: { 'Content-Type': 'audio/pcm; rate=16000; channels=1; format=f32le' },
  body: readableStream, // from MediaRecorder / AudioWorklet
  duplex: 'half',
});
```

### Why chunked POST over WebSocket/SSE

| | Chunked POST | SSE | WebSocket |
|---|---|---|---|
| Direction | Client → server streaming, server → client single response | Server → client only | Bidirectional |
| Fits "stream in, transcript on finish" | ✅ | ❌ wrong direction | Overkill |
| Protocol overhead | Plain HTTP | Plain HTTP | Upgrade handshake |
| Proxies/CDNs | Just works | Just works | Often needs config |

If we later want **live partial transcripts** during capture (not just final), WebSocket becomes worthwhile — that's a future addition, not v1.

### Concurrency

MLX is GPU-bound. One model instance per process. Serialize requests via an internal queue. Document this clearly — operators wanting throughput run multiple processes behind a load balancer.

## Suggested implementation order

1. **SDK polish** — trim `index.ts` exports, add `transcribe()` convenience, document `StreamingParakeet` async iterator.
2. **CLI** — smallest surface, validates SDK ergonomics. Includes auto-download.
3. **Server** — reuses SDK's streaming. Hono as optional peer dep.
