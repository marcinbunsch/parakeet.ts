import { describe, it, expect } from "vitest"
import fs from "node:fs"
import path from "node:path"
import { fromLocal } from "../../src/onnx/index.js"
import { loadAudioRaw } from "../../src/audio.js"

/**
 * ONNX Runtime backend integration tests — the Linux/Nvidia path.
 *
 * Point PARAKEET_ONNX_MODEL at a directory holding the exported graphs
 * (encoder-model.onnx [+ .data], decoder_joint-model.onnx, vocab.txt).
 * The suite skips when that directory is absent so it stays inert on machines
 * that only run the MLX backend.
 *
 * The ONNX loader defaults to `filterbank: 'interpolated'`, which matches
 * NVIDIA's reference preprocessor. The MLX fixtures in transcribe.test.ts use
 * the legacy parakeet-mlx filterbank and therefore expect different words on
 * two samples ("a go in" / "um" there vs "a go then" / "uh" here) — see
 * docs/cuda.md. That divergence is the front-end, not the model.
 */
const MODEL_PATH = process.env["PARAKEET_ONNX_MODEL"]
  ?? path.join(process.env["HOME"] ?? "/tmp", "parakeet-onnx-model")

const INPUTS = path.join(import.meta.dirname, "inputs")

const SAMPLES = [
  { file: "sample-1.wav", expected: "alright lets give this a go then" },
  { file: "sample-2.wav", expected: "I absolutely hate small talk" },
  {
    file: "sample-3.wav",
    expected: "The best thing you can do is uh give them a card",
  },
]

function normalize(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .replace(/\s+/g, " ")
    .trim()
}

/** True when `prefix` is a word-wise prefix of `full`. */
function isStablePrefix(prefix: string, full: string): boolean {
  const a = prefix.split(" ")
  const b = full.split(" ")
  if (a.length > b.length) return false
  return a.every((w, i) => w === b[i])
}

const available = fs.existsSync(MODEL_PATH)
const suite = available ? describe : describe.skip

if (!available) {
  console.warn(
    `[onnx] skipping: no ONNX model at ${MODEL_PATH} (set PARAKEET_ONNX_MODEL)`,
  )
}

suite("onnx transcription", () => {
  for (const { file, expected } of SAMPLES) {
    it(`transcribes ${file}`, async () => {
      const model = await fromLocal(MODEL_PATH)
      try {
        const result = await model.transcribe(path.join(INPUTS, file))
        expect(normalize(result.text)).toBe(normalize(expected))
      } finally {
        await model.dispose()
      }
    }, 180_000)
  }

  it("produces monotonically increasing word timestamps", async () => {
    const model = await fromLocal(MODEL_PATH)
    try {
      const result = await model.transcribe(path.join(INPUTS, "sample-3.wav"))
      const tokens = result.sentences.flatMap(s => s.tokens)
      expect(tokens.length).toBeGreaterThan(0)
      for (let i = 1; i < tokens.length; i++) {
        expect(tokens[i].start).toBeGreaterThanOrEqual(tokens[i - 1].start)
      }
      expect(tokens[0].start).toBeGreaterThanOrEqual(0)
    } finally {
      await model.dispose()
    }
  }, 180_000)
})

suite("onnx streaming", () => {
  for (const { file, expected } of SAMPLES) {
    it(`streams ${file}`, async () => {
      const model = await fromLocal(MODEL_PATH)
      try {
        const pcm = loadAudioRaw(
          path.join(INPUTS, file),
          model.preprocessorConfig.sampleRate,
        )
        const stream = model.transcribeStream()
        const chunk = model.preprocessorConfig.sampleRate // 1s
        for (let o = 0; o < pcm.length; o += chunk) {
          await stream.addAudio(pcm.subarray(o, Math.min(o + chunk, pcm.length)))
        }
        const result = stream.finish()
        // streaming may lag the offline transcript, but must not contradict it
        expect(isStablePrefix(normalize(result.text), normalize(expected))).toBe(true)
      } finally {
        await model.dispose()
      }
    }, 180_000)
  }

  it("handles chunks far smaller than one encoder frame", async () => {
    const model = await fromLocal(MODEL_PATH)
    try {
      const pcm = loadAudioRaw(
        path.join(INPUTS, "sample-2.wav"),
        model.preprocessorConfig.sampleRate,
      )
      const stream = model.transcribeStream()
      // 256 samples = 16ms, well under the 1280 samples of one encoder frame
      for (let o = 0; o < pcm.length; o += 256) {
        await stream.addAudio(pcm.subarray(o, Math.min(o + 256, pcm.length)))
      }
      const result = stream.finish()
      expect(result.text.trim().length).toBeGreaterThan(0)
    } finally {
      await model.dispose()
    }
  }, 180_000)
})
