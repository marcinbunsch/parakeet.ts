import { describe, it, expect } from "vitest"
import path from "node:path"
import { fromLocal } from "../../src/utils.js"

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
