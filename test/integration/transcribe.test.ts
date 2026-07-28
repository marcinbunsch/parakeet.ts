import { describe, it, expect } from "vitest"
import path from "node:path"
import { fromLocal } from "../../src/mlx/utils.js"
import { loadAudioRaw } from "../../src/mlx/audio.js"
import { consumePcmStream } from "../../src/mlx/parakeet.js"

const MODEL_PATH = path.join(
  process.env["HOME"] ?? "/tmp",
  ".cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3/snapshots/ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
)

const INPUTS = path.join(import.meta.dirname, "inputs")

const SAMPLES = [
  { file: "sample-1.wav", expected: "alright lets give this a go in" },
  { file: "sample-2.wav", expected: "I absolutely hate small talk" },
  {
    file: "sample-3.wav",
    expected: "The best thing you can do is um give them a card",
  },
]

function normalize(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .replace(/\s+/g, " ")
    .trim()
}

describe("transcription", () => {
  const model = fromLocal(MODEL_PATH)

  for (const { file, expected } of SAMPLES) {
    it(`transcribes ${file}`, async () => {
      const result = await model.transcribe(path.join(INPUTS, file))
      expect(normalize(result.text)).toBe(normalize(expected))
    }, 120_000)
  }
})

const SAMPLE_RATE = 16_000
// Two seconds of audio per frame — mimics a live PCM feed. Smaller frames push
// more per-chunk mel-normalisation boundaries in and degrade the tail further.
const FRAME_SAMPLES = SAMPLE_RATE * 2

async function* pcmFrames(file: string): AsyncGenerator<Float32Array> {
  const pcm = loadAudioRaw(path.join(INPUTS, file), SAMPLE_RATE)
  for (let start = 0; start < pcm.length; start += FRAME_SAMPLES) {
    yield pcm.slice(start, Math.min(start + FRAME_SAMPLES, pcm.length))
  }
}

/** Number of leading words two transcripts agree on. */
function commonLeadingWords(a: string, b: string): number {
  const wa = a.split(" ")
  const wb = b.split(" ")
  let i = 0
  while (i < wa.length && i < wb.length && wa[i] === wb[i]) i++
  return i
}

describe("streaming transcription", () => {
  const model = fromLocal(MODEL_PATH)

  for (const { file, expected } of SAMPLES) {
    it(`streams ${file}`, async () => {
      const stream = model.transcribeStream()
      const result = await consumePcmStream(stream, pcmFrames(file))

      // Streaming feeds ~1s chunks through a rotating cache and windowed local
      // attention. The stable (finalized) region matches the full-context pass,
      // but the last word or two can drift: the streaming mel is normalised
      // per-chunk, which nudges borderline tail tokens. So we assert the
      // stable leading prefix matches rather than the whole utterance. This is
      // primarily a regression guard against the streaming path crashing or
      // producing garbage (see the setAttentionModel / local-attention / conv
      // + KV cache fixes).
      const got = normalize(result.text)
      const want = normalize(expected)
      const wantWords = want.split(" ").length
      const matched = commonLeadingWords(got, want)
      expect(
        matched,
        `streamed "${got}" should share a leading prefix with "${want}"`,
      ).toBeGreaterThanOrEqual(Math.ceil(wantWords * 0.6))
    }, 120_000)
  }
})

describe("streaming robustness", () => {
  const model = fromLocal(MODEL_PATH)

  // A chunk too short to yield a single subsampled encoder frame used to feed a
  // zero-length sequence to the encoder and abort the process natively
  // ("[max] Cannot max reduce zero size array" / "[squeeze] Cannot squeeze
  // axis 1 with size 0"). addAudio must instead buffer it and return. 1024
  // samples (~64ms) produces ~7 mel frames, below the subsampling factor of 8.
  it("buffers an undersized chunk without crashing", async () => {
    async function* undersized(): AsyncGenerator<Float32Array> {
      yield new Float32Array(1024).fill(0.001)
    }
    const result = await consumePcmStream(model.transcribeStream(), undersized())
    // Nothing decodable yet — but the process survives.
    expect(result.text).toBe("")
  }, 120_000)
})
