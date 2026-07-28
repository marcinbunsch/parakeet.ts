/**
 * Streaming transcription — the shared sliding-window path (src/model.ts).
 *
 * Runs on the MLX backend here; the same StreamingParakeet drives the ONNX/CUDA
 * backend on Linux, so these are the properties both must satisfy.
 *
 * Ground truth is the model's own one-shot `transcribePcm` on the same audio,
 * not a hand-written fixture: streaming's job is to *converge to the offline
 * result*, so the offline result is the reference. (See docs/cuda.md — the
 * legacy `transcribe.test.ts` fixtures encode the old `floor` filterbank; this
 * path uses the `interpolated` default and produces "a go then" / "uh".)
 */
import { describe, it, expect, beforeAll } from "vitest"
import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import { fromLocal, loadAudioRaw } from "../../src/mlx/index.js"
import type { ParakeetModel } from "../../src/model.js"

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

const INPUTS = path.join(import.meta.dirname, "inputs")
const MODEL_DIR = findModelDir()

function norm(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim()
}

/** Word-level error rate (Levenshtein over tokens) between two strings. */
function wordErrorRate(ref: string, hyp: string): number {
  const r = norm(ref).split(" ").filter(Boolean)
  const h = norm(hyp).split(" ").filter(Boolean)
  const d: number[][] = Array.from({ length: r.length + 1 }, () => new Array(h.length + 1).fill(0))
  for (let i = 0; i <= r.length; i++) d[i][0] = i
  for (let j = 0; j <= h.length; j++) d[0][j] = j
  for (let i = 1; i <= r.length; i++) {
    for (let j = 1; j <= h.length; j++) {
      const cost = r[i - 1] === h[j - 1] ? 0 : 1
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    }
  }
  return r.length === 0 ? (h.length === 0 ? 0 : 1) : d[r.length][h.length] / r.length
}

/** Concatenate PCM buffers with a silence gap between each. */
function concatWithGaps(parts: Float32Array[], gapSamples: number): Float32Array {
  const gap = new Float32Array(gapSamples)
  const total = parts.reduce((n, p) => n + p.length + gap.length, 0)
  const out = new Float32Array(total)
  let o = 0
  for (const p of parts) {
    out.set(p, o); o += p.length
    out.set(gap, o); o += gap.length
  }
  return out
}

interface StreamRun {
  final: string
  finalizedSnapshots: string[]  // finalizedResult.text after each chunk
  tokensBeforeEnd: number       // finalized+draft tokens seen before the last chunk
}

/** Feed `pcm` to a fresh stream in fixed chunks, recording behaviour. */
async function runStream(
  model: ParakeetModel,
  pcm: Float32Array,
  windowSeconds: number,
  chunkSamples: number,
): Promise<StreamRun> {
  const stream = model.transcribeStream({ windowSeconds })
  const finalizedSnapshots: string[] = []
  let tokensBeforeEnd = 0
  for (let i = 0; i < pcm.length; i += chunkSamples) {
    const last = i + chunkSamples >= pcm.length
    await stream.addAudio(pcm.subarray(i, Math.min(i + chunkSamples, pcm.length)))
    finalizedSnapshots.push(stream.finalizedResult.text)
    if (!last) {
      tokensBeforeEnd = Math.max(
        tokensBeforeEnd,
        stream.finalizedTokens.length + stream.draftTokens.length,
      )
    }
  }
  return { final: stream.finish().text, finalizedSnapshots, tokensBeforeEnd }
}

const d = MODEL_DIR ? describe : describe.skip
if (!MODEL_DIR) {
  // eslint-disable-next-line no-console
  console.warn("streaming.test: MLX checkpoint not cached; skipping")
}

d("streaming (sliding-window)", () => {
  let model: ParakeetModel
  let sr: number

  beforeAll(() => {
    model = fromLocal(MODEL_DIR as string, { filterbank: "interpolated" })
    sr = model.preprocessorConfig.sampleRate
  })

  it("chunked stream equals one-shot when the window never slides", async () => {
    const pcm = loadAudioRaw(path.join(INPUTS, "sample-1.wav"), sr)
    const oneShot = (await model.transcribePcm(pcm)).text
    // 12 s window > clip length: identical bidirectional context, so exact.
    const run = await runStream(model, pcm, 12, Math.floor(0.5 * sr))

    expect(norm(run.final)).toBe(norm(oneShot))
    expect(norm(oneShot)).toContain("give this a go")
  }, 120_000)

  it("emits partial results before the stream ends", async () => {
    const pcm = loadAudioRaw(path.join(INPUTS, "sample-1.wav"), sr)
    const run = await runStream(model, pcm, 12, Math.floor(0.25 * sr))
    expect(run.tokensBeforeEnd).toBeGreaterThan(0)
  }, 120_000)

  it("never rewrites finalized text (prefix-stable commits)", async () => {
    const pcm = concatWithGaps(
      ["sample-1.wav", "sample-2.wav", "sample-3.wav"]
        .map((f) => loadAudioRaw(path.join(INPUTS, f), sr)),
      Math.floor(0.3 * sr),
    )
    // 5 s window over ~7 s audio forces the window to slide.
    const run = await runStream(model, pcm, 5, Math.floor(0.5 * sr))

    let prev = ""
    for (const snap of run.finalizedSnapshots) {
      expect(norm(snap).startsWith(norm(prev))).toBe(true)
      prev = snap
    }
    // Some commits actually happened over the run.
    expect(norm(prev).length).toBeGreaterThan(0)
  }, 180_000)

  it("finish() commits the draft tail exactly once", async () => {
    const pcm = loadAudioRaw(path.join(INPUTS, "sample-1.wav"), sr)
    const stream = model.transcribeStream({ windowSeconds: 12 })
    const chunk = Math.floor(0.5 * sr)
    for (let i = 0; i < pcm.length; i += chunk) {
      await stream.addAudio(pcm.subarray(i, Math.min(i + chunk, pcm.length)))
    }
    const draftBefore = stream.draftTokens.length
    const finalized = stream.finish()

    expect(draftBefore).toBeGreaterThan(0)          // there was an uncommitted tail
    expect(stream.draftTokens.length).toBe(0)        // finish drained it
    expect(stream.finalizedTokens.length).toBeGreaterThan(0)
    // finish() returns the finalized result and is idempotent for reads.
    expect(norm(stream.finalizedResult.text)).toBe(norm(finalized.text))
  }, 120_000)

  it("converges to one-shot within a small WER while the window slides", async () => {
    const files = ["sample-1.wav", "sample-2.wav", "sample-3.wav"]
    const pcm = concatWithGaps(files.map((f) => loadAudioRaw(path.join(INPUTS, f), sr)), Math.floor(0.3 * sr))
    const oneShot = (await model.transcribePcm(pcm)).text
    // 4 s window forces several slides across the ~7 s input.
    const run = await runStream(model, pcm, 4, Math.floor(0.25 * sr))

    const wer = wordErrorRate(oneShot, run.final)
    // Boundary frame-accounting is approximate (see docs/cuda.md "Known gap"):
    // a token can be mangled at a commit boundary. Tolerate a little; tighten
    // this bound when that accounting is fixed.
    expect(wer).toBeLessThanOrEqual(0.15)
  }, 240_000)
})
