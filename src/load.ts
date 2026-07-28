/**
 * Platform-detecting loader — the entry point most callers should use.
 *
 *   import { load } from 'parakeet.ts';
 *   const model = await load();          // picks the backend, fetches the model
 *   const result = await model.transcribe('audio.wav');
 *
 * Backend selection:
 *   - Apple Silicon  -> MLX / Metal
 *   - everything else -> ONNX Runtime (CUDA execution provider on Linux/Nvidia,
 *     CPU otherwise)
 *
 * Each backend has its own default checkpoint on the Hub, because the two need
 * different asset formats (safetensors vs exported ONNX graphs). Both produce
 * the same transcripts — see docs/cuda.md.
 */
import type { ParakeetModel } from './model.js';

export type BackendKind = 'auto' | 'mlx' | 'onnx';

export interface LoadOptions {
  /** Force a backend. Default 'auto'. */
  backend?: BackendKind;
  /**
   * HF repo id or local directory. Defaults to the chosen backend's standard
   * checkpoint.
   */
  model?: string;
  /** Override the HF cache root. */
  cacheDir?: string;
  /** Called as each file downloads. */
  onProgress?: (file: string, downloaded: number, total: number) => void;
  /**
   * Mel filterbank. 'interpolated' (default) matches NVIDIA's reference
   * preprocessor; 'floor' reproduces the legacy parakeet-mlx filterbank.
   */
  filterbank?: 'floor' | 'interpolated';
  /** ONNX only: override the execution provider (default: CUDA where available). */
  executionProvider?: 'cuda' | 'cpu' | 'tensorrt' | 'coreml' | 'dml';
}

/** Standard checkpoint per backend — different asset formats, same transcripts. */
export const DEFAULT_MODELS = {
  mlx: 'mlx-community/parakeet-tdt-0.6b-v3',
  onnx: 'istupakov/parakeet-tdt-0.6b-v3-onnx',
} as const;

/** Which backend `load()` would choose on this machine. */
export function detectBackend(): 'mlx' | 'onnx' {
  return process.platform === 'darwin' && process.arch === 'arm64' ? 'mlx' : 'onnx';
}

/**
 * Load Parakeet using the backend native to this platform, downloading the
 * model on first use.
 */
export async function load(options: LoadOptions = {}): Promise<ParakeetModel> {
  const kind = !options.backend || options.backend === 'auto'
    ? detectBackend()
    : options.backend;

  if (kind === 'mlx') {
    const { fromPretrained } = await import('./mlx/load.js');
    return fromPretrained(options.model ?? DEFAULT_MODELS.mlx, {
      cacheDir: options.cacheDir,
      onProgress: options.onProgress,
      filterbank: options.filterbank,
    });
  }

  const { fromPretrained } = await import('./onnx/parakeet.js');
  return fromPretrained(options.model ?? DEFAULT_MODELS.onnx, {
    cacheDir: options.cacheDir,
    onProgress: options.onProgress,
    filterbank: options.filterbank,
    executionProvider: options.executionProvider,
  });
}
