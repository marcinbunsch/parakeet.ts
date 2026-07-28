/** Standard checkpoint per backend — different asset formats, same transcripts. */
export const DEFAULT_MODELS = {
    mlx: 'mlx-community/parakeet-tdt-0.6b-v3',
    onnx: 'istupakov/parakeet-tdt-0.6b-v3-onnx',
};
/** Which backend `load()` would choose on this machine. */
export function detectBackend() {
    return process.platform === 'darwin' && process.arch === 'arm64' ? 'mlx' : 'onnx';
}
/**
 * Load Parakeet using the backend native to this platform, downloading the
 * model on first use.
 */
export async function load(options = {}) {
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
//# sourceMappingURL=load.js.map