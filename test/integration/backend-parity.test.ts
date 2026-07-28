/**
 * Backend parity — the core promise of the dual-backend design.
 *
 * The MLX and ONNX backends load the *same* checkpoint and drive the *same*
 * shared front-end and decode loop. Given identical audio they must produce the
 * same transcript, byte for byte, and the same token-id sequence. This is the
 * one test that would catch the ONNX encoder or decoder_joint drifting away from
 * the MLX reference (or a front-end change silently affecting only one path).
 *
 * It needs all three of:
 *   - the MLX safetensors checkpoint cached (mlx-community/parakeet-tdt-0.6b-v3),
 *   - a directory of ONNX exports in PARAKEET_ONNX_DIR (encoder-model.onnx +
 *     decoder_joint-model.onnx + vocab.txt),
 *   - a working onnxruntime-node (its native lib, CUDA EP on Linux).
 * so it runs on the Linux/CUDA box and skips elsewhere (e.g. on a Mac where the
 * onnxruntime-node native build was not installed).
 */
import { describe, it, expect, beforeAll } from "vitest"
import fs from "node:fs"
import path from "node:path"
import os from "node:os"
import { fromLocal as mlxFromLocal } from "../../src/mlx/index.js"
import { fromLocal as onnxFromLocal } from "../../src/onnx/index.js"
import type { ParakeetModel } from "../../src/model.js"

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

const MLX_DIR = findMlxDir()
const ONNX_DIR = process.env["PARAKEET_ONNX_DIR"] ?? null

// onnxruntime-node loads a native library; on a Mac dev box its build is often
// skipped, so importing it throws. Treat that as "ONNX backend unavailable".
let ONNX_RUNTIME_OK = false
try {
  await import("onnxruntime-node")
  ONNX_RUNTIME_OK = true
} catch {
  ONNX_RUNTIME_OK = false
}

const CAN_RUN = !!MLX_DIR && hasOnnxExports(ONNX_DIR) && ONNX_RUNTIME_OK
if (!CAN_RUN) {
  // eslint-disable-next-line no-console
  console.warn(
    "backend-parity.test: needs the MLX checkpoint, PARAKEET_ONNX_DIR with ONNX " +
    "exports, and a working onnxruntime-node; skipping.",
  )
}

const INPUTS = path.join(import.meta.dirname, "inputs")
const SAMPLES = ["sample-1.wav", "sample-2.wav", "sample-3.wav"]

const d = CAN_RUN ? describe : describe.skip

d("backend parity (MLX vs ONNX)", () => {
  let mlx: ParakeetModel
  let onnx: ParakeetModel

  beforeAll(async () => {
    // Same interpolated front-end on both sides, so the mel fed to each encoder
    // is identical by construction; any divergence is the encoder/decoder graph.
    mlx = mlxFromLocal(MLX_DIR as string, { filterbank: "interpolated" })
    onnx = await onnxFromLocal(ONNX_DIR as string, { filterbank: "interpolated" })
  })

  for (const file of SAMPLES) {
    it(`produces the same transcript for ${file}`, async () => {
      const a = await mlx.transcribe(path.join(INPUTS, file))
      const b = await onnx.transcribe(path.join(INPUTS, file))

      // Byte-identical text.
      expect(b.text).toBe(a.text)
      // ...and the same token-id sequence, which is stronger than the string
      // (catches identical detokenizations of different ids).
      expect(b.tokens.map((t) => t.id)).toEqual(a.tokens.map((t) => t.id))
    }, 180_000)
  }
})
