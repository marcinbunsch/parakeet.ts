// parakeet.ts — shared, backend-agnostic API
//
// The model class, audio front-end, decode loops, alignment and tokenizer are
// all backend-independent. Pick a backend by importing its loader:
//
//   import { fromLocal } from 'parakeet.ts/onnx';   // ONNX Runtime (CUDA on Linux)
//   import { fromLocal } from 'parakeet.ts/mlx';    // MLX (Apple Silicon)
//
// Both return the same `ParakeetModel`.

export { ParakeetModel, StreamingParakeet, consumePcmStream } from './model.js';
export type { ParakeetModelArgs, TranscribeOptions, StreamOptions } from './model.js';

export type {
  ParakeetBackend,
  EncoderOutput,
  EncoderLayout,
  DecodeStepResult,
  DecoderStateHandle,
} from './backend.js';

export { decodeTDTGreedy, decodeRNNTGreedy } from './decode.js';
export type { DecoderState, GreedyOptions } from './decode.js';

export {
  getLogMel,
  loadAudioRaw,
  getWindow,
  makePreprocessArgs,
  computeMelFilterbanks,
  computeMelFilterbanksInterpolated,
} from './audio.js';
export type { PreprocessArgs, LogMel } from './audio.js';

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
} from './alignment.js';

export { decode } from './tokenizer.js';
