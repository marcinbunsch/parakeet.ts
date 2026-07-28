/**
 * Shared support for the integration suites.
 *
 * There is ONE set of tests. Which backend they run against is chosen by the
 * PARAKEET_BACKEND env var (defaulting to the platform's native backend):
 *
 *   PARAKEET_BACKEND=mlx   vitest run   # Apple Silicon   (pnpm test:mac)
 *   PARAKEET_BACKEND=onnx  vitest run   # Linux / Nvidia  (pnpm test:linux)
 *
 * The chosen backend is resolved once here. If its native lib or model assets
 * aren't present, `backend.canRun` is false and the suites skip with a reason.
 * The backend module is imported dynamically so this never hard-fails on a
 * platform lacking the other backend's native library.
 */
import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import { loadAudioRaw } from "../../src/index.js"
import type { ParakeetModel } from "../../src/index.js"

export type BackendName = "mlx" | "onnx"

export const BACKEND: BackendName =
  (process.env["PARAKEET_BACKEND"] as BackendName | undefined) ??
  (process.platform === "darwin" ? "mlx" : "onnx")

export interface TestBackend {
  name: BackendName
  canRun: boolean
  reason?: string
  make: () => Promise<ParakeetModel>
}

export const INPUTS = path.join(import.meta.dirname, "inputs")

// One expected transcript set, asserted for whichever backend runs. Both
// backends producing these (MLX on a Mac, ONNX on Linux) IS the cross-backend
// parity guarantee. Uses the `interpolated` front-end (NVIDIA reference).
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
  finalizedSnapshots: string[]
  tokensBeforeEnd: number
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

function findOnnxDir(): string | null {
  const env = process.env["PARAKEET_ONNX_DIR"] ?? process.env["PARAKEET_ONNX_MODEL"]
  return env ?? null
}

function hasOnnxExports(dir: string | null): dir is string {
  if (!dir) return false
  const enc = ["encoder-model.onnx", "encoder.onnx"].some((n) => fs.existsSync(path.join(dir, n)))
  const dec = ["decoder_joint-model.onnx", "decoder_joint.onnx"].some((n) => fs.existsSync(path.join(dir, n)))
  return enc && dec
}

async function resolveBackend(): Promise<TestBackend> {
  if (BACKEND === "mlx") {
    const dir = findMlxDir()
    let load: ((d: string, o: { filterbank: "interpolated" }) => ParakeetModel) | null = null
    try {
      load = (await import("../../src/mlx/index.js")).fromLocal
    } catch { /* @mlx-node/core not loadable here */ }
    return {
      name: "mlx",
      canRun: !!dir && !!load,
      reason: !dir ? "MLX checkpoint not cached" : !load ? "@mlx-node/core not loadable" : undefined,
      make: async () => load!(dir as string, { filterbank: "interpolated" }),
    }
  }

  const dir = findOnnxDir()
  let load: ((d: string, o: { filterbank: "interpolated" }) => Promise<ParakeetModel>) | null = null
  let runtimeOk = true
  try {
    load = (await import("../../src/onnx/index.js")).fromLocal
    await import("onnxruntime-node")
  } catch {
    runtimeOk = false
  }
  return {
    name: "onnx",
    canRun: hasOnnxExports(dir) && !!load && runtimeOk,
    reason: !dir
      ? "PARAKEET_ONNX_DIR unset"
      : !hasOnnxExports(dir) ? "ONNX exports missing"
      : !runtimeOk ? "onnxruntime-node not loadable" : undefined,
    make: async () => load!(dir as string, { filterbank: "interpolated" }),
  }
}

/** The single backend this run tests against, availability resolved. */
export const backend: TestBackend = await resolveBackend()
