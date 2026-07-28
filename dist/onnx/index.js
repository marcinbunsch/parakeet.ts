// parakeet.ts/onnx — ONNX Runtime backend (CUDA on Linux/Nvidia, CPU elsewhere)
export { fromLocal } from './parakeet.js';
export { OnnxBackend, defaultExecutionProvider } from './backend.js';
// Shared API, re-exported for convenience
export { ParakeetModel, StreamingParakeet, decodeTDTGreedy, decodeRNNTGreedy, getLogMel, loadAudioRaw, makePreprocessArgs, computeMelFilterbanks, computeMelFilterbanksInterpolated, tokensToSentences, sentencesToResult, decode, } from '../index.js';
//# sourceMappingURL=index.js.map