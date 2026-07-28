/**
 * Backend-agnostic Parakeet model.
 *
 * This class contains no tensor code at all. It owns the shared audio
 * front-end, the greedy decode loops, chunking and streaming, and drives
 * whatever `ParakeetBackend` it is given — MLX on Apple Silicon, ONNX Runtime
 * with the CUDA execution provider on Linux/Nvidia.
 */
import type { ParakeetBackend, EncoderOutput } from './backend.js';
import { getLogMel, loadAudioRaw, PreprocessArgs } from './audio.js';
import { decodeTDTGreedy, decodeRNNTGreedy, DecoderState } from './decode.js';
import {
  AlignedToken, AlignedResult, SentenceConfig, tokensToSentences, sentencesToResult,
  mergeLongestContiguous, mergeLongestCommonSubsequence,
} from './alignment.js';

export interface TranscribeOptions {
  /** Split audio into chunks of this many seconds. Unset = single pass. */
  chunkDuration?: number;
  /** Overlap between chunks, seconds. Default 15. */
  overlapDuration?: number;
  onChunk?: (processed: number, total: number) => void;
  sentence?: SentenceConfig;
}

export interface ParakeetModelArgs {
  backend: ParakeetBackend;
  preprocessor: PreprocessArgs;
  vocabulary: string[];
  /** TDT duration table, or null for plain RNN-T. */
  durations: number[] | null;
  maxSymbols: number | null;
  subsamplingFactor: number;
}

export class ParakeetModel {
  readonly backend: ParakeetBackend;
  readonly preprocessorConfig: PreprocessArgs;
  readonly vocabulary: string[];
  readonly durations: number[] | null;
  readonly maxSymbols: number | null;
  readonly subsamplingFactor: number;

  constructor(args: ParakeetModelArgs) {
    this.backend = args.backend;
    this.preprocessorConfig = args.preprocessor;
    this.vocabulary = args.vocabulary;
    this.durations = args.durations;
    this.maxSymbols = args.maxSymbols;
    this.subsamplingFactor = args.subsamplingFactor;
  }

  /** Seconds of audio per encoder frame. */
  get timeRatio(): number {
    return (this.subsamplingFactor / this.preprocessorConfig.sampleRate)
      * this.preprocessorConfig.hopLength;
  }

  /** Run the shared mel front-end and the backend encoder. */
  async encodePcm(pcm: Float32Array): Promise<EncoderOutput> {
    const { data, nMels, numFrames } = getLogMel(pcm, this.preprocessorConfig);
    return this.backend.encode(data, nMels, numFrames);
  }

  /** Greedy-decode an encoder output, threading decoder state. */
  async decode(
    enc: EncoderOutput,
    state: DecoderState = { lastToken: null, hiddenState: null },
    range?: { from?: number; to?: number; timeOffset?: number },
  ): Promise<{ tokens: AlignedToken[]; state: DecoderState }> {
    const opts = {
      vocabulary: this.vocabulary,
      maxSymbols: this.maxSymbols,
      timeRatio: this.timeRatio,
      ...range,
    };
    return this.durations
      ? decodeTDTGreedy(this.backend, enc, this.durations, state, opts)
      : decodeRNNTGreedy(this.backend, enc, state, opts);
  }

  async transcribe(file: string, options: TranscribeOptions = {}): Promise<AlignedResult> {
    const { chunkDuration, overlapDuration = 15.0, onChunk, sentence } = options;
    const sr = this.preprocessorConfig.sampleRate;
    const pcm = loadAudioRaw(file, sr);
    return this.transcribePcm(pcm, options);
  }

  /** Transcribe raw mono float32 PCM at the model's sample rate. */
  async transcribePcm(pcm: Float32Array, options: TranscribeOptions = {}): Promise<AlignedResult> {
    const { chunkDuration, overlapDuration = 15.0, onChunk, sentence } = options;
    const sr = this.preprocessorConfig.sampleRate;

    if (chunkDuration === undefined || pcm.length / sr <= chunkDuration) {
      const enc = await this.encodePcm(pcm);
      const { tokens } = await this.decode(enc);
      return sentencesToResult(tokensToSentences(tokens, sentence));
    }

    const chunkSamples = Math.floor(chunkDuration * sr);
    const overlapSamples = Math.floor(overlapDuration * sr);
    let allTokens: AlignedToken[] = [];

    for (let start = 0; start < pcm.length; start += chunkSamples - overlapSamples) {
      const end = Math.min(start + chunkSamples, pcm.length);
      if (onChunk) onChunk(end, pcm.length);
      if (end - start < this.preprocessorConfig.hopLength) break;

      const enc = await this.encodePcm(pcm.subarray(start, end));
      const { tokens } = await this.decode(
        enc, { lastToken: null, hiddenState: null }, { timeOffset: start / sr },
      );

      if (allTokens.length > 0) {
        try {
          allTokens = mergeLongestContiguous(allTokens, tokens, overlapDuration);
        } catch {
          allTokens = mergeLongestCommonSubsequence(allTokens, tokens, overlapDuration);
        }
      } else {
        allTokens = tokens;
      }
    }

    return sentencesToResult(tokensToSentences(allTokens, sentence));
  }

  transcribeStream(options: StreamOptions = {}): StreamingParakeet {
    return new StreamingParakeet(this, options);
  }

  async dispose(): Promise<void> {
    await this.backend.dispose?.();
  }
}

// ---------------------------------------------------------------------------
// Streaming — sliding-window re-encode
// ---------------------------------------------------------------------------

export interface StreamOptions {
  /** Max seconds of audio re-encoded per update. Default 12. */
  windowSeconds?: number;
  /**
   * Encoder frames at the tail left uncommitted (revisable). Larger = better
   * quality at the edge, more churn in `draftTokens`. Default 12.
   */
  dropFrames?: number;
  sentence?: SentenceConfig;
}

/**
 * Streaming transcription by sliding-window re-encode.
 *
 * No encoder cache is involved: a bounded window of recent audio is re-encoded
 * on each update and re-decoded from the last committed frame. Tokens older
 * than `dropFrames` from the window end are finalized; the tail is draft and
 * may be revised. This keeps the encoder's full bidirectional context inside
 * the window, which suits an offline-trained checkpoint better than limited
 * left/right context, and it needs nothing from the backend beyond `encode`
 * and `decodeStep` — so it works identically on MLX and ONNX.
 */
export class StreamingParakeet {
  private readonly model: ParakeetModel;
  private readonly windowSamples: number;
  private readonly dropFrames: number;
  private readonly sentence?: SentenceConfig;

  private audio: Float32Array = new Float32Array(0);
  /** Absolute encoder-frame index corresponding to `audio[0]`. */
  private windowStartFrame = 0;
  /** Absolute encoder frame already committed. */
  private finalizedUpTo = 0;
  private state: DecoderState = { lastToken: null, hiddenState: null };
  private _finalized: AlignedToken[] = [];
  private _draft: AlignedToken[] = [];

  constructor(model: ParakeetModel, options: StreamOptions = {}) {
    this.model = model;
    const sr = model.preprocessorConfig.sampleRate;
    this.windowSamples = Math.floor((options.windowSeconds ?? 12.0) * sr);
    this.dropFrames = options.dropFrames ?? 12;
    this.sentence = options.sentence;
  }

  private get samplesPerFrame(): number {
    return this.model.preprocessorConfig.hopLength * this.model.subsamplingFactor;
  }

  /** Committed tokens — will not be revised. */
  get finalizedTokens(): AlignedToken[] { return [...this._finalized]; }
  /** Tokens in the revisable tail. */
  get draftTokens(): AlignedToken[] { return [...this._draft]; }

  get finalizedResult(): AlignedResult {
    return sentencesToResult(tokensToSentences(this._finalized, this.sentence));
  }

  /** Best current guess: committed + draft. */
  get result(): AlignedResult {
    return sentencesToResult(
      tokensToSentences([...this._finalized, ...this._draft], this.sentence),
    );
  }

  async addAudio(pcm: Float32Array): Promise<void> {
    const combined = new Float32Array(this.audio.length + pcm.length);
    combined.set(this.audio);
    combined.set(pcm, this.audio.length);
    this.audio = combined;

    // Not enough audio yet for a single subsampled encoder frame. Running the
    // front-end / encoder on a sub-frame window aborts natively (e.g. "[squeeze]
    // Cannot squeeze axis 1 with size 0") and cannot be caught in JS — keep
    // buffering until a later call has enough. Harmless for normal chunk sizes.
    const minSamples = this.model.preprocessorConfig.nFft + this.samplesPerFrame;
    if (this.audio.length < minSamples) return;

    // Trim in whole encoder frames so absolute frame accounting stays exact.
    const spf = this.samplesPerFrame;
    if (this.audio.length > this.windowSamples) {
      const dropSamples = Math.floor((this.audio.length - this.windowSamples) / spf) * spf;
      if (dropSamples > 0) {
        this.audio = this.audio.slice(dropSamples);
        this.windowStartFrame += dropSamples / spf;
      }
    }

    const enc = await this.model.encodePcm(this.audio);
    const absEnd = this.windowStartFrame + enc.frames;
    const commitUntil = Math.max(this.finalizedUpTo, absEnd - this.dropFrames);
    const timeOffset = this.windowStartFrame * this.model.timeRatio;

    if (commitUntil > this.finalizedUpTo) {
      const { tokens, state } = await this.model.decode(enc, this.state, {
        from: this.finalizedUpTo - this.windowStartFrame,
        to: commitUntil - this.windowStartFrame,
        timeOffset,
      });
      this._finalized.push(...tokens);
      this.state = state;
      this.finalizedUpTo = commitUntil;
    }

    // Speculative tail; its state is intentionally discarded.
    const { tokens: draft } = await this.model.decode(enc, this.state, {
      from: commitUntil - this.windowStartFrame,
      to: enc.frames,
      timeOffset,
    });
    this._draft = draft;
  }

  /** Commit the remaining draft. Call once the audio stream ends. */
  finish(): AlignedResult {
    this._finalized.push(...this._draft);
    this._draft = [];
    return this.finalizedResult;
  }
}

/**
 * Feed an async iterable of raw PCM frames (Float32Array, mono at the model's
 * sample rate) into a streaming session and return the final AlignedResult.
 *
 * The common "consume and wait" pattern behind the CLI `--stream` flag and the
 * HTTP server's transcribe endpoint.
 */
export async function consumePcmStream(
  stream: StreamingParakeet,
  source: AsyncIterable<Float32Array>,
): Promise<AlignedResult> {
  for await (const chunk of source) {
    await stream.addAudio(chunk);
  }
  return stream.finish();
}
