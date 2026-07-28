// parakeet.ts/mlx — MLX backend (Apple Silicon)

// Model loading (legacy classes)
export { fromPretrained, fromLocal } from './utils.js';

// Shared-backend loading — returns the backend-agnostic ParakeetModel
export { fromLocal as loadModel } from './load.js';
export type { MlxModelOptions } from './load.js';
export { MlxBackend } from './backend.js';

// Model classes
export {
  BaseParakeet,
  ParakeetTDT,
  ParakeetRNNT,
  ParakeetCTC,
  ParakeetTDTCTC,
  StreamingParakeet,
  // Arg types
  ParakeetTDTArgs,
  ParakeetRNNTArgs,
  ParakeetCTCArgs,
  ParakeetTDTCTCArgs,
  // Decoding
  DecodingConfig,
  DecodingStrategy,
  Greedy,
  Beam,
  greedy,
  beam,
  defaultDecodingConfig,
  TranscribeOptions,
  consumePcmStream,
} from './parakeet.js';

// Shared types (re-exported for convenience)
export {
  AlignedToken,
  AlignedSentence,
  AlignedResult,
  SentenceConfig,
  makeAlignedToken,
  makeAlignedSentence,
  makeAlignedResult,
  tokensToSentences,
  sentencesToResult,
} from '../alignment.js';

// Tokenizer
export { decode } from '../tokenizer.js';

// Audio utilities (for advanced use)
export { loadAudio, loadAudioRaw, getLogMel, makePreprocessArgs, PreprocessArgs } from './audio.js';
