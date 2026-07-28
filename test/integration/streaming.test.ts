/**
 * Streaming (sliding-window) — run over every backend this machine can load.
 *
 * The same StreamingParakeet drives MLX and ONNX, so these properties must hold
 * on both. Ground truth is the backend's own one-shot `transcribePcm` on the
 * same audio (streaming's job is to converge to the offline result), not a
 * hand-written fixture.
 */
import { describe, it, expect, beforeAll } from "vitest"
import path from "node:path"
import type { ParakeetModel } from "../../src/index.js"
import {
  BACKENDS, INPUTS, normalize, wordErrorRate, concatWithGaps, loadPcm, runStream,
} from "./support.js"

for (const backend of BACKENDS) {
  const d = backend.canRun ? describe : describe.skip
  if (!backend.canRun) {
    // eslint-disable-next-line no-console
    console.warn(`streaming.test [${backend.name}]: skipped (${backend.reason})`)
  }

  d(`streaming (sliding-window, ${backend.name})`, () => {
    let model: ParakeetModel
    let sr: number

    beforeAll(async () => {
      model = await backend.make()
      sr = model.preprocessorConfig.sampleRate
    })

    it("chunked stream equals one-shot when the window never slides", async () => {
      const pcm = loadPcm("sample-1.wav", sr)
      const oneShot = (await model.transcribePcm(pcm)).text
      // 12 s window > clip length: identical bidirectional context, so exact.
      const run = await runStream(model, pcm, 12, Math.floor(0.5 * sr))

      expect(normalize(run.final)).toBe(normalize(oneShot))
      expect(normalize(oneShot)).toContain("give this a go")
    }, 120_000)

    it("emits partial results before the stream ends", async () => {
      const pcm = loadPcm("sample-1.wav", sr)
      const run = await runStream(model, pcm, 12, Math.floor(0.25 * sr))
      expect(run.tokensBeforeEnd).toBeGreaterThan(0)
    }, 120_000)

    it("never rewrites finalized text (prefix-stable commits)", async () => {
      const pcm = concatWithGaps(
        ["sample-1.wav", "sample-2.wav", "sample-3.wav"].map((f) => loadPcm(f, sr)),
        Math.floor(0.3 * sr),
      )
      // 5 s window over ~7 s audio forces the window to slide.
      const run = await runStream(model, pcm, 5, Math.floor(0.5 * sr))

      let prev = ""
      for (const snap of run.finalizedSnapshots) {
        expect(normalize(snap).startsWith(normalize(prev))).toBe(true)
        prev = snap
      }
      expect(normalize(prev).length).toBeGreaterThan(0)
    }, 180_000)

    it("finish() commits the draft tail exactly once", async () => {
      const pcm = loadPcm("sample-1.wav", sr)
      const stream = model.transcribeStream({ windowSeconds: 12 })
      const chunk = Math.floor(0.5 * sr)
      for (let i = 0; i < pcm.length; i += chunk) {
        await stream.addAudio(pcm.subarray(i, Math.min(i + chunk, pcm.length)))
      }
      const draftBefore = stream.draftTokens.length
      const finalized = stream.finish()

      expect(draftBefore).toBeGreaterThan(0)
      expect(stream.draftTokens.length).toBe(0)
      expect(stream.finalizedTokens.length).toBeGreaterThan(0)
      expect(normalize(stream.finalizedResult.text)).toBe(normalize(finalized.text))
    }, 120_000)

    it("converges to one-shot within a small WER while the window slides", async () => {
      const pcm = concatWithGaps(
        ["sample-1.wav", "sample-2.wav", "sample-3.wav"].map((f) => loadPcm(f, sr)),
        Math.floor(0.3 * sr),
      )
      const oneShot = (await model.transcribePcm(pcm)).text
      // 4 s window forces several slides across the ~7 s input.
      const run = await runStream(model, pcm, 4, Math.floor(0.25 * sr))

      // Boundary frame-accounting is approximate (see docs/cuda.md "Known gap"):
      // a token can be mangled at a commit boundary. Tolerate a little; tighten
      // this bound when that accounting is fixed.
      expect(wordErrorRate(oneShot, run.final)).toBeLessThanOrEqual(0.15)
    }, 240_000)
  })
}
