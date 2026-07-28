import { describe, it, expect } from "vitest"
import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import { fromLocal } from "../../src/mlx/index.js"

/** Resolve the cached MLX checkpoint dir without pinning a snapshot hash. */
function findModelDir(): string | null {
  const base = path.join(
    os.homedir(),
    ".cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3/snapshots",
  )
  if (!fs.existsSync(base)) return null
  for (const snap of fs.readdirSync(base)) {
    const dir = path.join(base, snap)
    if (fs.existsSync(path.join(dir, "config.json"))) return dir
  }
  return null
}

const MODEL_DIR = findModelDir()
const INPUTS = path.join(import.meta.dirname, "inputs")

// Expected transcripts use the `interpolated` filterbank (NVIDIA's reference
// preprocessor), which is the default for the shared ParakeetModel path. The
// old `floor` filterbank produced "a go in" / "um" — float noise in 13 dead mel
// bins tipping knife-edge tokens; see docs/cuda.md.
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

const d = MODEL_DIR ? describe : describe.skip

d("transcription", () => {
  const model = fromLocal(MODEL_DIR as string)

  for (const { file, expected } of SAMPLES) {
    it(`transcribes ${file}`, async () => {
      const result = await model.transcribe(path.join(INPUTS, file))
      expect(normalize(result.text)).toBe(normalize(expected))
    }, 120_000)
  }
})
