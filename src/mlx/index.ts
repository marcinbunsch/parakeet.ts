// parakeet.ts/mlx — MLX backend (Apple Silicon)
//
// Loads a safetensors checkpoint and returns the shared, backend-agnostic
// `ParakeetModel` — the same class the ONNX backend returns.

export { fromLocal, fromPretrained } from './load.js';
export type { MlxModelOptions, FromPretrainedOptions } from './load.js';
export { MlxBackend } from './backend.js';

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
