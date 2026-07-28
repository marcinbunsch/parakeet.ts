import { ExecutionProvider } from './backend.js';
import { ParakeetModel } from '../model.js';
export interface OnnxModelOptions {
    executionProvider?: ExecutionProvider;
    /**
     * Execution provider for the autoregressive decode step. Defaults to the same
     * as `executionProvider`; 'cpu' is a reasonable alternative since the step is
     * small and CPU avoids per-step device round trips.
     */
    decoderExecutionProvider?: ExecutionProvider;
    /**
     * 'interpolated' (default) matches NVIDIA's reference preprocessor.
     * 'floor' reproduces parakeet-mlx's filterbank, which collapses 13 of 128 mel
     * bins to zero — see docs/cuda.md.
     */
    filterbank?: 'floor' | 'interpolated';
    logSeverityLevel?: number;
}
/**
 * Load an ONNX Parakeet model from a directory of exported graphs.
 *
 * Expected contents:
 *   encoder-model.onnx        (+ encoder-model.onnx.data for external weights)
 *   decoder_joint-model.onnx
 *   vocab.txt                 (or config.json with joint.vocabulary)
 *   config.json               (optional; a full NeMo config supplies preprocessor settings)
 */
export declare function fromLocal(dir: string, options?: OnnxModelOptions): Promise<ParakeetModel>;
/** Default ONNX export on the Hub, matching the checkpoint the MLX path loads. */
export declare const DEFAULT_ONNX_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx";
export interface OnnxPretrainedOptions extends OnnxModelOptions {
    /** Override the HF cache root. */
    cacheDir?: string;
    /** Called as each file downloads. */
    onProgress?: (file: string, downloaded: number, total: number) => void;
}
/**
 * Load an ONNX Parakeet model, downloading it from the HuggingFace Hub on first
 * use and caching it under the standard HF cache directory.
 *
 * `hfIdOrPath` may also be a local directory, in which case it is used as-is.
 */
export declare function fromPretrained(hfIdOrPath?: string, options?: OnnxPretrainedOptions): Promise<ParakeetModel>;
//# sourceMappingURL=parakeet.d.ts.map