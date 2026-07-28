/**
 * Backend-agnostic greedy decoding.
 *
 * These loops previously lived inside the MLX backend and operated on MxArray.
 * They are plain TypeScript over typed arrays now, and drive any
 * `ParakeetBackend`. Semantics (blank handling, duration stepping, maxSymbols
 * guard, entropy confidence, timestamps) are unchanged from the MLX versions in
 * `src/mlx/rnnt.ts`.
 */
import type { ParakeetBackend, EncoderOutput, DecoderStateHandle } from './backend.js';
import { makeAlignedToken, AlignedToken } from './alignment.js';
import { decode as decodeTokens } from './tokenizer.js';

export interface DecoderState {
  lastToken: number | null;
  hiddenState: DecoderStateHandle | null;
}

/** Copy encoder frame `t` into `out`, honouring the backend's layout. */
function readFrame(enc: EncoderOutput, t: number, out: Float32Array): Float32Array {
  if (enc.layout === 'time-major') {
    const base = t * enc.dim;
    for (let d = 0; d < enc.dim; d++) out[d] = enc.data[base + d];
  } else {
    for (let d = 0; d < enc.dim; d++) out[d] = enc.data[d * enc.stride + t];
  }
  return out;
}

/** Softmax in place over `n` values starting at `offset`, returning the array. */
function softmax(src: Float32Array, offset: number, n: number): Float32Array {
  const out = new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) { const v = src[offset + i]; if (v > max) max = v; }
  let sum = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(src[offset + i] - max); out[i] = e; sum += e; }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

function argmax(a: Float32Array): number {
  let best = 0, bv = a[0];
  for (let i = 1; i < a.length; i++) if (a[i] > bv) { bv = a[i]; best = i; }
  return best;
}

/** Normalized-entropy confidence, matching the MLX implementation. */
function confidenceOf(probs: Float32Array): number {
  let entropy = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i] + 1e-10;
    entropy -= p * Math.log(p);
  }
  return 1.0 - entropy / Math.log(probs.length);
}

export interface GreedyOptions {
  vocabulary: string[];
  maxSymbols: number | null;
  timeRatio: number;
  /** Absolute time offset (seconds) added to emitted token starts. */
  timeOffset?: number;
  /** Decode only frames in [from, to). Defaults to the whole encoder output. */
  from?: number;
  to?: number;
}

/**
 * Greedy TDT decoding. `durations` is the TDT duration table, e.g. [0,1,2,3,4].
 */
export async function decodeTDTGreedy(
  backend: ParakeetBackend,
  enc: EncoderOutput,
  durations: number[],
  state: DecoderState,
  opts: GreedyOptions,
): Promise<{ tokens: AlignedToken[]; state: DecoderState }> {
  const { vocabulary, maxSymbols, timeRatio, timeOffset = 0 } = opts;
  const from = opts.from ?? 0;
  const to = Math.min(opts.to ?? enc.frames, enc.frames);
  const vocabSize = vocabulary.length;

  const tokens: AlignedToken[] = [];
  const frame = new Float32Array(enc.dim);
  let { lastToken, hiddenState } = state;
  let step = from;
  let newSymbols = 0;

  while (step < to) {
    readFrame(enc, step, frame);
    const { logits, state: nextState } = await backend.decodeStep(frame, lastToken, hiddenState);

    const tokenProbs = softmax(logits, 0, vocabSize + 1);
    const durationProbs = softmax(logits, vocabSize + 1, durations.length);

    const predToken = argmax(tokenProbs);
    const decision = argmax(durationProbs);
    const isBlank = predToken === vocabSize;

    if (!isBlank) {
      tokens.push(makeAlignedToken(
        predToken,
        decodeTokens([predToken], vocabulary),
        timeOffset + step * timeRatio,
        durations[decision] * timeRatio,
        confidenceOf(tokenProbs),
      ));
      lastToken = predToken;
      hiddenState = nextState;
    }

    step += durations[decision];
    newSymbols += 1;

    if (durations[decision] !== 0) {
      newSymbols = 0;
    } else if (maxSymbols !== null && newSymbols >= maxSymbols) {
      step += 1;
      newSymbols = 0;
    }
  }

  return { tokens, state: { lastToken, hiddenState } };
}

/** Greedy RNN-T decoding (no duration head). */
export async function decodeRNNTGreedy(
  backend: ParakeetBackend,
  enc: EncoderOutput,
  state: DecoderState,
  opts: GreedyOptions,
): Promise<{ tokens: AlignedToken[]; state: DecoderState }> {
  const { vocabulary, maxSymbols, timeRatio, timeOffset = 0 } = opts;
  const from = opts.from ?? 0;
  const to = Math.min(opts.to ?? enc.frames, enc.frames);
  const vocabSize = vocabulary.length;

  const tokens: AlignedToken[] = [];
  const frame = new Float32Array(enc.dim);
  let { lastToken, hiddenState } = state;
  let step = from;
  let newSymbols = 0;

  while (step < to) {
    readFrame(enc, step, frame);
    const { logits, state: nextState } = await backend.decodeStep(frame, lastToken, hiddenState);

    const tokenProbs = softmax(logits, 0, vocabSize + 1);
    const predToken = argmax(tokenProbs);

    if (predToken !== vocabSize) {
      tokens.push(makeAlignedToken(
        predToken,
        decodeTokens([predToken], vocabulary),
        timeOffset + step * timeRatio,
        timeRatio,
        confidenceOf(tokenProbs),
      ));
      lastToken = predToken;
      hiddenState = nextState;
      newSymbols += 1;
      if (maxSymbols !== null && newSymbols >= maxSymbols) {
        step += 1;
        newSymbols = 0;
      }
    } else {
      step += 1;
      newSymbols = 0;
    }
  }

  return { tokens, state: { lastToken, hiddenState } };
}
