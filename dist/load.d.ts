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
export declare const DEFAULT_MODELS: {
    readonly mlx: "mlx-community/parakeet-tdt-0.6b-v3";
    readonly onnx: "istupakov/parakeet-tdt-0.6b-v3-onnx";
};
/** Which backend `load()` would choose on this machine. */
export declare function detectBackend(): 'mlx' | 'onnx';
/**
 * Load Parakeet using the backend native to this platform, downloading the
 * model on first use.
 */
export declare function load(options?: LoadOptions): Promise<ParakeetModel>;
//# sourceMappingURL=load.d.ts.map