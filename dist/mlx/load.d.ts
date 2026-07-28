import { ParakeetModel } from '../model.js';
export interface MlxModelOptions {
    /**
     * 'interpolated' (default) matches NVIDIA's reference preprocessor.
     * 'floor' reproduces parakeet-mlx's filterbank — see docs/cuda.md.
     */
    filterbank?: 'floor' | 'interpolated';
}
/**
 * Load a safetensors Parakeet checkpoint and wrap it in `ParakeetModel`.
 *
 * @param dir directory holding `config.json` and `model.safetensors`
 */
export declare function fromLocal(dir: string, options?: MlxModelOptions): ParakeetModel;
//# sourceMappingURL=load.d.ts.map