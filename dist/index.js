// parakeet.ts — shared, backend-agnostic API
//
// The model class, audio front-end, decode loops, alignment and tokenizer are
// all backend-independent. Pick a backend by importing its loader:
//
//   import { fromLocal } from 'parakeet.ts/onnx';   // ONNX Runtime (CUDA on Linux)
//   import { fromLocal } from 'parakeet.ts/mlx';    // MLX (Apple Silicon)
//
// Both return the same `ParakeetModel`.
// Zero-config entry point: picks the backend for this platform and fetches the
// model on first use.
export { load, detectBackend, DEFAULT_MODELS } from './load.js';
export { downloadFromHub, downloadRepoFiles, defaultCacheDir, repoDir } from './hub.js';
export { ParakeetModel, StreamingParakeet, consumePcmStream } from './model.js';
export { decodeTDTGreedy, decodeRNNTGreedy } from './decode.js';
export { getLogMel, loadAudioRaw, getWindow, makePreprocessArgs, computeMelFilterbanks, computeMelFilterbanksInterpolated, } from './audio.js';
export { makeAlignedToken, makeAlignedSentence, makeAlignedResult, tokensToSentences, sentencesToResult, } from './alignment.js';
export { decode } from './tokenizer.js';
//# sourceMappingURL=index.js.map