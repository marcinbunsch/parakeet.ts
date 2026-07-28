/**
 * Shared support for the backend-parameterized integration suites.
 *
 * The transcribe and streaming tests are backend-agnostic: they only need a
 * `ParakeetModel`. We describe each available backend once here and run the same
 * suites over every backend that can actually load on this machine — MLX on
 * Apple Silicon, ONNX Runtime on Linux/CUDA, or both. A backend that can't load
 * (missing native lib or model assets) reports `canRun: false` and its suite is
 * skipped, so one `vitest run` adapts to whatever the box has.
 *
 * Backends are imported dynamically so importing this module never hard-fails on
 * a platform that lacks one of the native libraries.
 */
import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import { loadAudioRaw } from "../../src/index.js"
import type { ParakeetModel } from "../../src/index.js"

export interface TestBackend {
  name: string
  canRun: boolean
  reason?: string
  make: () => Promise<ParakeetModel>
}

export const INPUTS = path.join(import.meta.dirname, "inputs")

// Expected transcripts use the `interpolated` front-end (NVIDIA's reference
// preprocessor), the shared default. Both backends are expected to produce these
// — that equivalence is exactly what backend-parity.test.ts pins.
export const SAMPLES = [
  { file: "sample-1.wav", text: "alright lets give this a go then" },
  { file: "sample-2.wav", text: "I absolutely hate small talk" },
  { file: "sample-3.wav", text: "The best thing you can do is uh give them a card" },
]

export function normalize(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim()
}

/** Word-level error rate (Levenshtein over tokens) between two strings. */
export function wordErrorRate(ref: string, hyp: string): number {
  const r = normalize(ref).split(" ").filter(Boolean)
  const h = normalize(hyp).split(" ").filter(Boolean)
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

export function loadPcm(file: string, sr: number): Float32Array {
  return loadAudioRaw(path.join(INPUTS, file), sr)
}

/** Concatenate PCM buffers with a silence gap between each. */
export function concatWithGaps(parts: Float32Array[], gapSamples: number): Float32Array {
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

export interface StreamRun {
  final: string
  finalizedSnapshots: string[]  // finalizedResult.text after each chunk
  tokensBeforeEnd: number       // finalized+draft tokens seen before the last chunk
}

/** Feed `pcm` to a fresh stream in fixed chunks, recording behaviour. */
export async function runStream(
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

function findMlxDir(): string | null {
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

function hasOnnxExports(dir: string | null): dir is string {
  if (!dir) return false
  const enc = ["encoder-model.onnx", "encoder.onnx"].some((n) => fs.existsSync(path.join(dir, n)))
  const dec = ["decoder_joint-model.onnx", "decoder_joint.onnx"].some((n) => fs.existsSync(path.join(dir, n)))
  return enc && dec
}

async function mlxBackend(): Promise<TestBackend> {
  const dir = findMlxDir()
  let load: ((d: string, o: { filterbank: "interpolated" }) => ParakeetModel) | null = null
  try {
    load = (await import("../../src/mlx/index.js")).fromLocal
  } catch { /* @mlx-node/core not loadable on this platform */ }
  const canRun = !!dir && !!load
  return {
    name: "mlx",
    canRun,
    reason: !dir ? "MLX checkpoint not cached" : !load ? "@mlx-node/core not loadable" : undefined,
    make: async () => load!(dir as string, { filterbank: "interpolated" }),
  }
}

async function onnxBackend(): Promise<TestBackend> {
  const dir = process.env["PARAKEET_ONNX_DIR"] ?? null
  let load: ((d: string, o: { filterbank: "interpolated" }) => Promise<ParakeetModel>) | null = null
  let runtimeOk = true
  try {
    load = (await import("../../src/onnx/index.js")).fromLocal
    await import("onnxruntime-node")  // ensure the native lib is present
  } catch {
    runtimeOk = false
  }
  const canRun = hasOnnxExports(dir) && !!load && runtimeOk
  return {
    name: "onnx",
    canRun,
    reason: !dir
      ? "PARAKEET_ONNX_DIR unset"
      : !hasOnnxExports(dir) ? "ONNX exports missing"
      : !runtimeOk ? "onnxruntime-node not loadable" : undefined,
    make: async () => load!(dir as string, { filterbank: "interpolated" }),
  }
}

/** Every backend this machine could possibly run, with availability resolved. */
export const BACKENDS: TestBackend[] = [await mlxBackend(), await onnxBackend()]
