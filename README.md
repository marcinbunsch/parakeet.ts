# parakeet-mlx

Community TypeScript port of [parakeet-mlx](https://github.com/senstens/parakeet-mlx) — Nvidia Parakeet ASR models running on Apple Silicon via MLX.

> **Not affiliated with NVIDIA or the upstream Python project.**

## Requirements

- Apple Silicon Mac (M1 or later)
- Node.js 18.11+
- [ffmpeg](https://ffmpeg.org/) in `$PATH` (for audio file decoding)

## Install

```sh
npm install parakeet-mlx
```

## Usage

### SDK

```ts
import { fromPretrained } from 'parakeet-mlx';

const model = await fromPretrained('mlx-community/parakeet-tdt-0.6b-v3');
const result = await model.transcribe('recording.wav');

console.log(result.text);
// "Hello world"

console.log(result.sentences);
// [{ text: 'Hello world', start: 0, end: 1.2, duration: 1.2, tokens: [...] }]
```

`fromPretrained` downloads the model weights from HuggingFace Hub on first run and caches them under `~/.cache/huggingface/hub/`. Subsequent calls load from cache.

```ts
// Track download progress
const model = await fromPretrained('mlx-community/parakeet-tdt-0.6b-v3', {
  onProgress(file, downloaded, total) {
    console.error(`${file}: ${Math.round(downloaded / total * 100)}%`);
  },
});

// Or load from a local directory
import { fromLocal } from 'parakeet-mlx';
const model = fromLocal('/path/to/model/dir');
```

#### Streaming

For real-time or incremental transcription, use `StreamingParakeet` directly or the `consumePcmStream` helper:

```ts
import { consumePcmStream } from 'parakeet-mlx';

const stream = model.transcribeStream();
const result = await consumePcmStream(stream, pcmFrameIterable);
// pcmFrameIterable: AsyncIterable<Float32Array> — 16kHz mono f32le
```

Or drive it manually for live feedback:

```ts
const stream = model.transcribeStream();
await stream.start();

for await (const chunk of pcmFrameIterable) {
  stream.addAudio(chunk);
  console.log(stream.result.text);        // best current guess (finalized + draft)
  console.log(stream.finalizedResult.text); // committed tokens only
}

stream.stop();
const final = stream.result;
```

#### Concurrency

Each model instance has an internal async mutex. Concurrent `await model.transcribe(...)` calls on the same instance are automatically serialized — no GPU-state corruption. For lower tail latency under bursty load, create a pool:

```ts
const pool = await Promise.all([
  fromPretrained('mlx-community/parakeet-tdt-0.6b-v3'),
  fromPretrained('mlx-community/parakeet-tdt-0.6b-v3'),
]);
let i = 0;
const next = () => pool[i++ % pool.length];
```

#### Output shape

All entry points return `AlignedResult`:

```ts
type AlignedResult = {
  text: string;
  sentences: AlignedSentence[];
};

type AlignedSentence = {
  text: string;
  start: number;    // seconds
  end: number;
  duration: number;
  tokens: AlignedToken[];
};

type AlignedToken = {
  id: number;
  text: string;
  start: number;    // seconds
  duration: number;
};
```

### CLI

```sh
# Transcribe a file (auto-downloads model on first run)
parakeet-mlx recording.wav

# JSON output with word-level timestamps
parakeet-mlx recording.wav --json

# Different model
parakeet-mlx recording.wav --model mlx-community/parakeet-tdt-0.6b-v3

# Stream raw f32le mono 16kHz PCM from stdin
ffmpeg -i recording.wav -f f32le -ar 16000 -ac 1 - | parakeet-mlx --stream
```

Options:

```
--model, -m <id>   HuggingFace repo ID or local directory (default: mlx-community/parakeet-tdt-0.6b-v3)
--json, -j         Output full AlignedResult JSON instead of plain text
--stream, -s       Read raw f32le mono 16kHz PCM from stdin
--help, -h         Show help
```

Exit codes: `0` success, `1` file/IO error, `2` model error.

### HTTP server

The `parakeet-mlx/server` entry point exports a [Hono](https://hono.dev/) route factory. Install the extra deps first:

```sh
npm install hono @hono/node-server
```

```ts
import { Hono } from 'hono';
import { serve } from '@hono/node-server';
import { fromPretrained } from 'parakeet-mlx';
import { createParakeetRoutes } from 'parakeet-mlx/server';

const model = await fromPretrained('mlx-community/parakeet-tdt-0.6b-v3');

const app = new Hono();
app.route('/asr', createParakeetRoutes({
  model,
  maxDurationSeconds: 300,  // default
  idleTimeoutMs: 30_000,    // default
}));

serve({ fetch: app.fetch, port: 8080 });
```

#### `POST /asr/transcribe`

Send raw f32le mono 16kHz PCM audio as the request body.

**Request headers:**
```
Content-Type: audio/pcm; rate=16000; channels=1; format=f32le
```

**Response `200`:** `AlignedResult` JSON

**Error responses:**
```json
{ "error": "<code>", "message": "<human-readable>" }
```

| Status | Error code | Cause |
|--------|------------|-------|
| 408 | `idle_timeout` | No bytes for `idleTimeoutMs` ms |
| 413 | `payload_too_large` | Audio exceeds `maxDurationSeconds` |
| 415 | `unsupported_media_type` | Wrong Content-Type |
| 500 | `internal_error` | Unexpected error |

## Available models

| Model | HuggingFace ID | Notes |
|-------|---------------|-------|
| Parakeet TDT 0.6B v2 | `mlx-community/parakeet-tdt-0.6b-v2` | |
| Parakeet TDT 0.6B v3 | `mlx-community/parakeet-tdt-0.6b-v3` | **Default** |
| Parakeet RNNT 0.6B | `mlx-community/parakeet-rnnt-0.6b` | |
| Parakeet CTC 0.6B | `mlx-community/parakeet-ctc-0.6b` | |
| Parakeet TDT-CTC 0.6B | `mlx-community/parakeet-tdt-ctc-0.6b` | |
| Parakeet TDT 1.1B | `mlx-community/parakeet-tdt-1.1b` | |

## License

MIT
