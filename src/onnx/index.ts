// parakeet.ts/onnx — ONNX Runtime backend (CUDA on Linux/Nvidia, CPU elsewhere)

export { fromLocal } from './parakeet.js';
export type { OnnxModelOptions } from './parakeet.js';

export { OnnxBackend, defaultExecutionProvider } from './backend.js';
export type { OnnxBackendOptions, ExecutionProvider } from './backend.js';

// Shared API, re-exported for convenience
export {
  ParakeetModel,
  StreamingParakeet,
  consumePcmStream,
  decodeTDTGreedy,
  decodeRNNTGreedy,
  getLogMel,
  loadAudioRaw,
  makePreprocessArgs,
  computeMelFilterbanks,
  computeMelFilterbanksInterpolated,
  tokensToSentences,
  sentencesToResult,
  decode,
} from '../index.js';
export type {
  ParakeetBackend,
  EncoderOutput,
  TranscribeOptions,
  StreamOptions,
  DecoderState,
  PreprocessArgs,
  LogMel,
  AlignedToken,
  AlignedSentence,
  AlignedResult,
  SentenceConfig,
} from '../index.js';
