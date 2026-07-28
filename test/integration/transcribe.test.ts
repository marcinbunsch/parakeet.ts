/**
 * Transcription — run over every backend this machine can load (MLX and/or
 * ONNX). Each backend self-skips when its native lib or checkpoint is missing.
 */
import { describe, it, expect, beforeAll } from "vitest"
import path from "node:path"
import type { ParakeetModel } from "../../src/index.js"
import { BACKENDS, INPUTS, SAMPLES, normalize } from "./support.js"

for (const backend of BACKENDS) {
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
}
