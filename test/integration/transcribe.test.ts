/**
 * Transcription — runs against the one backend selected by PARAKEET_BACKEND
 * (MLX on a Mac, ONNX on Linux). Same assertions either way.
 */
import { describe, it, expect, beforeAll } from "vitest"
import path from "node:path"
import type { ParakeetModel } from "../../src/index.js"
import { backend, INPUTS, SAMPLES, normalize } from "./support.js"

const d = backend.canRun ? describe : describe.skip
if (!backend.canRun) {
  // eslint-disable-next-line no-console
  console.warn(`transcribe.test [${backend.name}]: skipped (${backend.reason})`)
}

d(`transcription (${backend.name})`, () => {
  let model: ParakeetModel
  beforeAll(async () => { model = await backend.make() })

  for (const { file, text } of SAMPLES) {
    it(`transcribes ${file}`, async () => {
      const result = await model.transcribe(path.join(INPUTS, file))
      expect(normalize(result.text)).toBe(normalize(text))
    }, 120_000)
  }
})
