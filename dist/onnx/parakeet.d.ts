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
//# sourceMappingURL=parakeet.d.ts.map