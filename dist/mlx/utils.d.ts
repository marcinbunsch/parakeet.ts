/**
 * Low-level model-loading helpers for the MLX backend.
 *
 *  - `loadSafetensors` parses a SafeTensors weight file into a WeightMap.
 *  - `downloadFromHub` fetches a single file from the HuggingFace CDN with an
 *    optional progress callback, caching it under the HF hub layout.
 *
 * The public loaders (`fromLocal` / `fromPretrained`) live in `./load.ts`, which
 * builds a backend-agnostic `ParakeetModel` from these pieces.
 */
import type { WeightMap } from './nn.js';
/**
 * Parse a safetensors file and return a WeightMap.
 * The safetensors format: 8 bytes (header length LE uint64) + JSON header + data.
 * Uses fd-based random access to support files larger than the 2 GiB Buffer limit.
 */
export declare function loadSafetensors(filePath: string): WeightMap;
export declare function downloadFromHub(repoId: string, filename: string, cacheDir?: string, onProgress?: (downloaded: number, total: number) => void): Promise<string>;
//# sourceMappingURL=utils.d.ts.map