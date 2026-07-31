# parakeet.ts

TypeScript runtime for Nvidia Parakeet ASR models. Runs on Apple Silicon via MLX, or Nvidia GPUs (Linux) via ONNX Runtime.

> **Not affiliated with NVIDIA.**

## Requirements

- Node.js 18.11+
- [ffmpeg](https://ffmpeg.org/) in `$PATH` (for audio file decoding)
- A backend, chosen automatically by [`load()`](#usage):
  - **MLX** — Apple Silicon Mac (M1 or later). No extra install.
  - **ONNX Runtime** — everywhere else (CUDA on Linux/Nvidia, CPU otherwise).
    Install the optional `onnxruntime-node` package.

## Install

```sh
npm install parakeet.ts
```

## Backends

The model, audio front-end, decode loop, and alignment are backend-agnostic, and
`load()` picks the right backend for the machine — you normally don't choose one.
For explicit control, import a backend's loader directly. Every path returns the
same `ParakeetModel`, so all the examples below work unchanged.

| Backend | Runs on | `load()` picks it | Explicit import |
|---------|---------|-------------------|-----------------|
| **MLX** | Apple Silicon (Metal) | on Apple Silicon | `parakeet.ts/mlx` — `fromPretrained`, `fromLocal` |
| **ONNX Runtime** | Nvidia GPU (CUDA, Linux) / CPU | everywhere else | `parakeet.ts/onnx` — `fromLocal` |

Each backend has its own default checkpoint (different asset formats, same
transcripts). The CLI and HTTP server use the MLX backend. See
[docs/cuda.md](docs/cuda.md) for the dual-backend design and benchmarks.

## Usage

### SDK

```ts
import { load } from 'parakeet.ts';

const model = await load(); // detects the backend, downloads the model on first run
const result = await model.transcribe('recording.wav');

console.log(result.text);
// "Hello world"

console.log(result.sentences);
// [{ text: 'Hello world', start: 0, end: 1.2, duration: 1.2, tokens: [...] }]
```

`load()` is the canonical entry point. It selects the backend for the current
machine — **MLX** on Apple Silicon, **ONNX Runtime** everywhere else — and
downloads that backend's default checkpoint from HuggingFace Hub on first run,
caching under `~/.cache/huggingface/hub/`. Subsequent calls load from cache.

```ts
const model = await load({
  // backend:  'auto' (default) | 'mlx' | 'onnx'
  // model:    HF repo id or local directory (default: the backend's checkpoint)
  // cacheDir: override the HF cache root
  // executionProvider: ONNX only — 'cuda' | 'cpu' | ...
  // filterbank: 'interpolated' (default) | 'floor'
  onProgress(file, downloaded, total) {
    console.error(`${file}: ${Math.round((downloaded / total) * 100)}%`);
  },
});

// Ask which backend load() would choose here, without loading:
import { detectBackend } from 'parakeet.ts';
detectBackend(); // 'mlx' | 'onnx'
```

To pin a backend or load from a local directory instead of auto-selecting, use
the per-backend loaders — see [Backends](#backends) and
[ONNX Runtime backend](#onnx-runtime-backend):

```ts
import { fromPretrained, fromLocal } from 'parakeet.ts/mlx';

const a = await fromPretrained('mlx-community/parakeet-tdt-0.6b-v3');
const b = fromLocal('/path/to/mlx-model-dir');
```

#### Streaming

For real-time or incremental transcription, use `StreamingParakeet` directly or the `consumePcmStream` helper:

```ts
import { consumePcmStream } from 'parakeet.ts';

const stream = model.transcribeStream();
const result = await consumePcmStream(stream, pcmFrameIterable);
// pcmFrameIterable: AsyncIterable<Float32Array> — 16kHz mono f32le
```

`consumePcmStream` returns only once the source is exhausted. To surface partial
transcripts as audio arrives, drive it manually and read the getters after each
chunk:

```ts
const stream = model.transcribeStream();

for await (const chunk of pcmFrameIterable) {
  await stream.addAudio(chunk);              // async
  console.log(stream.result.text);           // best current guess (finalized + draft)
  console.log(stream.finalizedResult.text);  // committed tokens only — never revised
}

const final = stream.finish();               // commit the remaining draft
```

Streaming is backend-agnostic — the same `StreamingParakeet` runs on MLX and ONNX.

#### Concurrency

Each model instance has an internal async mutex. Concurrent `await model.transcribe(...)` calls on the same instance are automatically serialized — no GPU-state corruption. For lower tail latency under bursty load, create a pool:

```ts
const pool = await Promise.all([load(), load()]);
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

### ONNX Runtime backend

On anything other than Apple Silicon, `load()` uses [ONNX Runtime](https://onnxruntime.ai/)
(CUDA execution provider on Linux/Nvidia, CPU elsewhere). It needs the optional
runtime:

```sh
npm install onnxruntime-node
```

With that installed, `await load()` downloads the default ONNX checkpoint
([`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx))
and runs — no other setup. Pass the execution provider through if you want to
override it:

```ts
const model = await load({ executionProvider: 'cuda' }); // 'cuda' | 'cpu' | 'tensorrt' | 'coreml' | 'dml'
```

To use your own exported graphs, load a local directory containing
`encoder-model.onnx` (plus its `encoder-model.onnx.data` sidecar — keep the
filename), `decoder_joint-model.onnx`, and `vocab.txt`:

```ts
import { fromLocal } from 'parakeet.ts/onnx';

const model = await fromLocal('/path/to/onnx-model-dir', {
  executionProvider: 'cuda',        // default: 'cuda' on Linux/Windows x64, 'cpu' elsewhere
  decoderExecutionProvider: 'cpu',  // optional: run the small per-step decoder on CPU
});
```

Either way `model` is the same `ParakeetModel`, so streaming, concurrency, and
the [output shape](#output-shape) behave identically. Only TDT and RNN-T
checkpoints are available as ONNX exports.

### CLI

```sh
# Transcribe a file (auto-downloads model on first run)
parakeet recording.wav

# JSON output with word-level timestamps
parakeet recording.wav --json

# Different model
parakeet recording.wav --model mlx-community/parakeet-tdt-0.6b-v3

# Stream raw f32le mono 16kHz PCM from stdin
ffmpeg -i recording.wav -f f32le -ar 16000 -ac 1 - | parakeet --stream
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

The `parakeet.ts/mlx/server` entry point exports a [Hono](https://hono.dev/) route factory. Install the extra deps first:

```sh
npm install hono @hono/node-server
```

```ts
import { Hono } from 'hono';
import { serve } from '@hono/node-server';
import { fromPretrained } from 'parakeet.ts/mlx';
import { createParakeetRoutes } from 'parakeet.ts/mlx/server';

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

`load()` downloads a default checkpoint per backend — `mlx-community/parakeet-tdt-0.6b-v3`
(MLX) or [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
(ONNX). The table lists the MLX (`mlx-community`) checkpoints; pass any of them as
`load({ model })` or to `fromPretrained`. ONNX exports live in their own repos —
`load({ backend: 'onnx' })` fetches the default automatically, or point
`fromLocal` at a local directory of graphs.

## License

MIT
