import { ParakeetModel } from '../model.js';
export interface MlxModelOptions {
    /**
     * 'interpolated' (default) matches NVIDIA's reference preprocessor.
     * 'floor' reproduces parakeet-mlx's filterbank, which collapses 13 of 128 mel
     * bins to zero — see docs/cuda.md.
     */
    filterbank?: 'floor' | 'interpolated';
}
/**
 * Load a safetensors Parakeet checkpoint from a local directory.
 *
 * @param dir directory holding `config.json` and `model.safetensors`
 */
export declare function fromLocal(dir: string, options?: MlxModelOptions): ParakeetModel;
export interface FromPretrainedOptions extends MlxModelOptions {
    cacheDir?: string;
    onProgress?: (file: string, downloaded: number, total: number) => void;
}
/**
 * Load a Parakeet checkpoint from a HuggingFace Hub repo or a local directory.
 *
 * @param hfIdOrPath HuggingFace repo id (e.g. "mlx-community/parakeet-tdt-0.6b-v3")
 *                   or a local directory path.
 */
export declare function fromPretrained(hfIdOrPath: string, options?: FromPretrainedOptions): Promise<ParakeetModel>;
//# sourceMappingURL=load.d.ts.map