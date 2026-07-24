/**
 * Model loading utilities.
 *
 * Supports:
 *  - Local directories (e.g. "./models/parakeet-tdt-0.6b-v3")
 *  - HuggingFace Hub repos (e.g. "mlx-community/parakeet-tdt-0.6b-v3")
 *
 * Weight files are SafeTensors format (model.safetensors).
 * Configuration is read from config.json which follows the NeMo format used
 * by the original parakeet-mlx Python project.
 */
import { BaseParakeet } from './parakeet.js';
/**
 * Load a Parakeet model from a local directory.
 */
export declare function fromLocal(modelDir: string): BaseParakeet;
/**
 * Load a Parakeet model from a HuggingFace Hub repo or local directory.
 *
 * @param hfIdOrPath - HuggingFace repo ID (e.g. "mlx-community/parakeet-tdt-0.6b-v3")
 *                     or local directory path.
 * @param options.cacheDir - Override default HF cache directory.
 * @param options.onProgress - Called with (downloaded, total) bytes during file downloads.
 */
export declare function fromPretrained(hfIdOrPath: string, options?: {
    cacheDir?: string;
    onProgress?: (file: string, downloaded: number, total: number) => void;
}): Promise<BaseParakeet>;
//# sourceMappingURL=utils.d.ts.map