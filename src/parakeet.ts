import { MxArray } from '@mlx-node/core';
import { Module, WeightMap } from './nn.js';
import { Conformer, ConformerArgs } from './conformer.js';
import {
  PredictNetwork,
  JointNetwork,
  PredictArgs,
  JointArgs,
  DecoderState,
  decodeTDTGreedy,
  decodeRNNTGreedy,
} from './rnnt.js';
import { ConvASRDecoder, ConvASRDecoderArgs, AuxCTCArgs, decodeCTCGreedy } from './ctc.js';
import { PreprocessArgs, loadAudio, getLogMel } from './audio.js';
import {
  AlignedToken,
  AlignedResult,
  SentenceConfig,
  tokensToSentences,
  sentencesToResult,
  mergeLongestContiguous,
  mergeLongestCommonSubsequence,
} from './alignment.js';
import { ConformerCache, RotatingConformerCache } from './cache.js';

function s(...dims: number[]): BigInt64Array {
  return BigInt64Array.from(dims.map(BigInt));
}

// ---------------------------------------------------------------------------
// Decoding / sentence configs
// ---------------------------------------------------------------------------

export interface Greedy {
  type: 'greedy';
}

export interface Beam {
  type: 'beam';
  beamSize: number;
  lengthPenalty: number;
  patience: number;
  durationReward: number;
}

export type DecodingStrategy = Greedy | Beam;

export function greedy(): Greedy {
  return { type: 'greedy' };
}

export function beam(options: Partial<Omit<Beam, 'type'>> = {}): Beam {
  return {
    type: 'beam',
    beamSize: options.beamSize ?? 5,
    lengthPenalty: options.lengthPenalty ?? 1.0,
    patience: options.patience ?? 1.0,
    durationReward: options.durationReward ?? 0.7,
  };
}

export interface DecodingConfig {
  decoding: DecodingStrategy;
  sentence: SentenceConfig;
}

export function defaultDecodingConfig(): DecodingConfig {
  return { decoding: greedy(), sentence: {} };
}

// ---------------------------------------------------------------------------
// Arg types (mirror Python dataclasses)
// ---------------------------------------------------------------------------

export interface TDTDecodingArgs {
  modelType: string;
  durations: number[];
  greedy: Record<string, unknown> | null;
}

export interface RNNTDecodingArgs {
  greedy: Record<string, unknown> | null;
}

export interface CTCDecodingArgs {
  greedy: Record<string, unknown> | null;
}

export interface ParakeetTDTArgs {
  preprocessor: PreprocessArgs;
  encoder: ConformerArgs;
  decoder: PredictArgs;
  joint: JointArgs;
  decoding: TDTDecodingArgs;
}

export interface ParakeetRNNTArgs {
  preprocessor: PreprocessArgs;
  encoder: ConformerArgs;
  decoder: PredictArgs;
  joint: JointArgs;
  decoding: RNNTDecodingArgs;
}

export interface ParakeetCTCArgs {
  preprocessor: PreprocessArgs;
  encoder: ConformerArgs;
  decoder: ConvASRDecoderArgs;
  decoding: CTCDecodingArgs;
}

export interface ParakeetTDTCTCArgs extends ParakeetTDTArgs {
  auxCtc: AuxCTCArgs;
}

// ---------------------------------------------------------------------------
// Transcription options
// ---------------------------------------------------------------------------

export interface TranscribeOptions {
  decodingConfig?: DecodingConfig;
  chunkDuration?: number;
  overlapDuration?: number;
  onChunk?: (current: number, total: number) => void;
}

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------

export abstract class BaseParakeet extends Module {
  readonly preprocessorConfig: PreprocessArgs;
  readonly encoderConfig: ConformerArgs;
  readonly encoder: Conformer;

  constructor(preprocessorConfig: PreprocessArgs, encoderConfig: ConformerArgs) {
    super();
    this.preprocessorConfig = preprocessorConfig;
    this.encoderConfig = encoderConfig;
    this.encoder = new Conformer(encoderConfig);
  }

  get timeRatio(): number {
    return (
      (this.encoderConfig.subsamplingFactor /
        this.preprocessorConfig.sampleRate) *
      this.preprocessorConfig.hopLength
    );
  }

  abstract generate(mel: MxArray, decodingConfig?: DecodingConfig): AlignedResult[];

  async transcribe(
    path: string,
    options: TranscribeOptions = {},
  ): Promise<AlignedResult> {
    const {
      decodingConfig = defaultDecodingConfig(),
      chunkDuration,
      overlapDuration = 15.0,
      onChunk,
    } = options;

    const audioData = loadAudio(path, this.preprocessorConfig.sampleRate);
    const audioLen = Number(audioData.shape()[0]);

    if (chunkDuration === undefined || audioLen / this.preprocessorConfig.sampleRate <= chunkDuration) {
      const mel = getLogMel(audioData, this.preprocessorConfig);
      return this.generate(mel, decodingConfig)[0];
    }

    const chunkSamples = Math.floor(chunkDuration * this.preprocessorConfig.sampleRate);
    const overlapSamples = Math.floor(overlapDuration * this.preprocessorConfig.sampleRate);

    let allTokens: AlignedToken[] = [];
    const audioDataF32 = audioData.toFloat32();

    for (let start = 0; start < audioDataF32.length; start += chunkSamples - overlapSamples) {
      const end = Math.min(start + chunkSamples, audioDataF32.length);

      if (onChunk) onChunk(end, audioDataF32.length);

      if (end - start < this.preprocessorConfig.hopLength) break;

      const chunkF32 = audioDataF32.slice(start, end);
      const chunkAudio = MxArray.fromFloat32(chunkF32, s(chunkF32.length));
      const chunkMel = getLogMel(chunkAudio, this.preprocessorConfig);
      const chunkResult = this.generate(chunkMel, decodingConfig)[0];

      const chunkOffset = start / this.preprocessorConfig.sampleRate;
      for (const sentence of chunkResult.sentences) {
        for (const token of sentence.tokens) {
          token.start += chunkOffset;
          (token as any).end = token.start + token.duration;
        }
      }

      if (allTokens.length > 0) {
        try {
          allTokens = mergeLongestContiguous(allTokens, chunkResult.tokens, overlapDuration);
        } catch {
          allTokens = mergeLongestCommonSubsequence(allTokens, chunkResult.tokens, overlapDuration);
        }
      } else {
        allTokens = chunkResult.tokens;
      }
    }

    return sentencesToResult(tokensToSentences(allTokens, decodingConfig.sentence));
  }

  transcribeStream(
    contextSize: [number, number] = [256, 256],
    depth = 1,
    decodingConfig: DecodingConfig = defaultDecodingConfig(),
    keepOriginalAttention = false,
  ): StreamingParakeet {
    return new StreamingParakeet(this, contextSize, depth, decodingConfig, keepOriginalAttention);
  }

  loadWeights(weights: WeightMap, prefix = ''): void {
    this.encoder.loadWeights(weights, prefix ? `${prefix}.encoder` : 'encoder');
  }
}

// ---------------------------------------------------------------------------
// ParakeetTDT
// ---------------------------------------------------------------------------

export class ParakeetTDT extends BaseParakeet {
  readonly vocabulary: string[];
  readonly durations: number[];
  readonly maxSymbols: number | null;
  readonly decoder: PredictNetwork;
  readonly joint: JointNetwork;

  constructor(args: ParakeetTDTArgs) {
    super(args.preprocessor, args.encoder);
    if (args.decoding.modelType !== 'tdt') throw new Error('Model must be a TDT model');

    this.vocabulary = args.joint.vocabulary;
    this.durations = args.decoding.durations;
    this.maxSymbols =
      args.decoding.greedy?.['max_symbols'] != null
        ? Number(args.decoding.greedy['max_symbols'])
        : null;

    this.decoder = new PredictNetwork(args.decoder);
    this.joint = new JointNetwork(args.joint);
  }

  decode(
    features: MxArray,
    lengths: MxArray,
    states: DecoderState[],
    decodingConfig: DecodingConfig = defaultDecodingConfig(),
  ): [Array<AlignedToken[]>, DecoderState[]] {
    const B = Number(features.shape()[0]);
    const effectiveStates = states.length === B ? states : Array.from({ length: B }, () => ({
      lastToken: null,
      hiddenState: null,
    }));

    return decodeTDTGreedy(
      features,
      lengths,
      this.decoder,
      this.joint,
      this.vocabulary,
      this.durations,
      this.maxSymbols,
      effectiveStates,
      this.timeRatio,
    ) as any; // dynamic import makes typing tricky
  }

  generate(mel: MxArray, decodingConfig: DecodingConfig = defaultDecodingConfig()): AlignedResult[] {
    if (mel.ndim() === 2) mel = mel.expandDims(0);

    const [features, lengths] = this.encoder.forward(mel, null, null);
    features.eval();
    lengths.eval();

    const B = Number(features.shape()[0]);
    const initStates: DecoderState[] = Array.from({ length: B }, () => ({ lastToken: null, hiddenState: null }));
    const [result] = this.decode(features, lengths, initStates, decodingConfig);

    return (result as AlignedToken[][]).map(tokens =>
      sentencesToResult(tokensToSentences(tokens, decodingConfig.sentence)),
    );
  }

  loadWeights(weights: WeightMap, prefix = ''): void {
    super.loadWeights(weights, prefix);
    this.decoder.loadWeights(weights, prefix ? `${prefix}.decoder` : 'decoder');
    this.joint.loadWeights(weights, prefix ? `${prefix}.joint` : 'joint');
  }
}

// ---------------------------------------------------------------------------
// ParakeetRNNT
// ---------------------------------------------------------------------------

export class ParakeetRNNT extends BaseParakeet {
  readonly vocabulary: string[];
  readonly maxSymbols: number | null;
  readonly decoder: PredictNetwork;
  readonly joint: JointNetwork;

  constructor(args: ParakeetRNNTArgs) {
    super(args.preprocessor, args.encoder);
    this.vocabulary = args.joint.vocabulary;
    this.maxSymbols =
      args.decoding.greedy?.['max_symbols'] != null
        ? Number(args.decoding.greedy['max_symbols'])
        : null;

    this.decoder = new PredictNetwork(args.decoder);
    this.joint = new JointNetwork(args.joint);
  }

  decode(
    features: MxArray,
    lengths: MxArray,
    states: DecoderState[],
    decodingConfig: DecodingConfig = defaultDecodingConfig(),
  ): [Array<AlignedToken[]>, DecoderState[]] {
    const B = Number(features.shape()[0]);
    const effectiveStates = states.length === B ? states : Array.from({ length: B }, () => ({
      lastToken: null,
      hiddenState: null,
    }));

    return decodeRNNTGreedy(
      features,
      lengths,
      this.decoder,
      this.joint,
      this.vocabulary,
      this.maxSymbols,
      effectiveStates,
      this.timeRatio,
    ) as any;
  }

  generate(mel: MxArray, decodingConfig: DecodingConfig = defaultDecodingConfig()): AlignedResult[] {
    if (mel.ndim() === 2) mel = mel.expandDims(0);
    const [features, lengths] = this.encoder.forward(mel, null, null);
    features.eval();
    lengths.eval();

    const B = Number(features.shape()[0]);
    const initStates: DecoderState[] = Array.from({ length: B }, () => ({ lastToken: null, hiddenState: null }));
    const [result] = this.decode(features, lengths, initStates, decodingConfig);

    return (result as AlignedToken[][]).map(tokens =>
      sentencesToResult(tokensToSentences(tokens, decodingConfig.sentence)),
    );
  }

  loadWeights(weights: WeightMap, prefix = ''): void {
    super.loadWeights(weights, prefix);
    this.decoder.loadWeights(weights, prefix ? `${prefix}.decoder` : 'decoder');
    this.joint.loadWeights(weights, prefix ? `${prefix}.joint` : 'joint');
  }
}

// ---------------------------------------------------------------------------
// ParakeetCTC
// ---------------------------------------------------------------------------

export class ParakeetCTC extends BaseParakeet {
  readonly vocabulary: string[];
  readonly ctcDecoder: ConvASRDecoder;

  constructor(args: ParakeetCTCArgs) {
    super(args.preprocessor, args.encoder);
    this.vocabulary = args.decoder.vocabulary;
    this.ctcDecoder = new ConvASRDecoder(args.decoder);
  }

  decode(
    features: MxArray,
    lengths: MxArray,
    decodingConfig: DecodingConfig = defaultDecodingConfig(),
  ): Array<AlignedToken[]> {
    return decodeCTCGreedy(features, lengths, this.ctcDecoder, this.vocabulary, this.timeRatio);
  }

  generate(mel: MxArray, decodingConfig: DecodingConfig = defaultDecodingConfig()): AlignedResult[] {
    if (mel.ndim() === 2) mel = mel.expandDims(0);
    const [features, lengths] = this.encoder.forward(mel, null, null);
    features.eval();
    lengths.eval();

    const result = this.decode(features, lengths, decodingConfig);
    return result.map(tokens =>
      sentencesToResult(tokensToSentences(tokens, decodingConfig.sentence)),
    );
  }

  loadWeights(weights: WeightMap, prefix = ''): void {
    super.loadWeights(weights, prefix);
    this.ctcDecoder.loadWeights(weights, prefix ? `${prefix}.decoder` : 'decoder');
  }
}

// ---------------------------------------------------------------------------
// ParakeetTDTCTC (TDT model with auxiliary CTC head)
// ---------------------------------------------------------------------------

export class ParakeetTDTCTC extends ParakeetTDT {
  readonly ctcDecoder: ConvASRDecoder;

  constructor(args: ParakeetTDTCTCArgs) {
    super(args);
    this.ctcDecoder = new ConvASRDecoder(args.auxCtc.decoder);
  }

  override loadWeights(weights: WeightMap, prefix = ''): void {
    super.loadWeights(weights, prefix);
    this.ctcDecoder.loadWeights(weights, prefix ? `${prefix}.ctc_decoder` : 'ctc_decoder');
  }
}

// ---------------------------------------------------------------------------
// StreamingParakeet
// ---------------------------------------------------------------------------

export class StreamingParakeet {
  private readonly model: BaseParakeet;
  private readonly contextSize: [number, number];
  private readonly depth: number;
  private readonly decodingConfig: DecodingConfig;
  private readonly keepOriginalAttention: boolean;

  private cache: ConformerCache[];
  private audioBuffer: Float32Array;
  private melBuffer: MxArray | null = null;
  private decoderHidden: [MxArray, MxArray] | null = null;
  private lastToken: number | null = null;
  private finalizedTokens: AlignedToken[] = [];
  private draftTokens: AlignedToken[] = [];

  constructor(
    model: BaseParakeet,
    contextSize: [number, number],
    depth: number,
    decodingConfig: DecodingConfig,
    keepOriginalAttention: boolean,
  ) {
    this.model = model;
    this.contextSize = contextSize;
    this.depth = depth;
    this.decodingConfig = decodingConfig;
    this.keepOriginalAttention = keepOriginalAttention;

    this.cache = model.encoder.layers.map(
      () => new RotatingConformerCache(contextSize[0], contextSize[1] * depth),
    );
    this.audioBuffer = new Float32Array(0);
  }

  get keepSize(): number {
    return this.contextSize[0];
  }

  get dropSize(): number {
    return this.contextSize[1] * this.depth;
  }

  start(): void {
    if (!this.keepOriginalAttention) {
      this.model.encoder.setAttentionModel('rel_pos_local_attn', this.contextSize);
    }
  }

  stop(): void {
    if (!this.keepOriginalAttention) {
      this.model.encoder.setAttentionModel('rel_pos');
    }
  }

  get result(): AlignedResult {
    return sentencesToResult(
      tokensToSentences(
        [...this.finalizedTokens, ...this.draftTokens],
        this.decodingConfig.sentence,
      ),
    );
  }

  addAudio(audio: Float32Array): void {
    // Append to buffer
    const combined = new Float32Array(this.audioBuffer.length + audio.length);
    combined.set(this.audioBuffer);
    combined.set(audio, this.audioBuffer.length);
    this.audioBuffer = combined;

    const hopLen = this.model.preprocessorConfig.hopLength;
    const usableLen = Math.floor(this.audioBuffer.length / hopLen) * hopLen;

    const usableAudio = MxArray.fromFloat32(this.audioBuffer.slice(0, usableLen), s(usableLen));
    const mel = getLogMel(usableAudio, this.model.preprocessorConfig);

    // mel: [1, frames, nMels]
    if (this.melBuffer === null) {
      this.melBuffer = mel;
    } else {
      this.melBuffer = MxArray.concatenate(this.melBuffer, mel, 1);
    }

    this.audioBuffer = this.audioBuffer.slice(usableLen);

    const subFactor = this.model.encoderConfig.subsamplingFactor;
    const melFrames = Number(this.melBuffer.shape()[1]);
    const usableMelFrames = Math.floor(melFrames / subFactor) * subFactor;

    const melInput = this.melBuffer.slice(
      s(0, 0, 0),
      s(1, usableMelFrames, Number(this.melBuffer.shape()[2])),
    );

    const [features, lengths] = this.model.encoder.forward(melInput, null, this.cache);
    features.eval();
    lengths.eval();

    const length = Number(lengths.toInt32()[0]);
    const finalizedLength = Math.max(0, length - this.dropSize);

    // Trim mel buffer to keep only drop_size worth of subsampled frames
    const leftover = melFrames - usableMelFrames;
    const keepMel = this.dropSize * subFactor + leftover;
    this.melBuffer = this.melBuffer.slice(
      s(0, Math.max(0, melFrames - keepMel), 0),
      s(1, melFrames, Number(this.melBuffer.shape()[2])),
    );

    if (this.model instanceof ParakeetTDT || this.model instanceof ParakeetRNNT) {
      const finLengths = MxArray.fromInt32(new Int32Array([finalizedLength]), s(1));
      const initState: DecoderState = { lastToken: this.lastToken, hiddenState: this.decoderHidden };

      const [finTokens, finStates] = (this.model as ParakeetTDT).decode(features, finLengths, [initState]);
      this.decoderHidden = finStates[0].hiddenState;
      this.lastToken = finTokens[0].length > 0 ? finTokens[0][finTokens[0].length - 1].id : this.lastToken;

      const draftInput = features.slice(
        s(0, finalizedLength, 0),
        s(1, length, Number(features.shape()[2])),
      );
      const draftLengths = MxArray.fromInt32(new Int32Array([length - finalizedLength]), s(1));
      const draftState: DecoderState = { lastToken: this.lastToken, hiddenState: this.decoderHidden };
      const [draftTokens] = (this.model as ParakeetTDT).decode(draftInput, draftLengths, [draftState]);

      this.finalizedTokens.push(...(finTokens[0] as AlignedToken[]));
      this.draftTokens = draftTokens[0] as AlignedToken[];

    } else if (this.model instanceof ParakeetCTC) {
      const finLengths = MxArray.fromInt32(new Int32Array([finalizedLength]), s(1));
      const finTokens = (this.model as ParakeetCTC).decode(features, finLengths);

      const draftInput = features.slice(
        s(0, finalizedLength, 0),
        s(1, length, Number(features.shape()[2])),
      );
      const draftLengths = MxArray.fromInt32(new Int32Array([length - finalizedLength]), s(1));
      const draftTokens = (this.model as ParakeetCTC).decode(draftInput, draftLengths);

      this.finalizedTokens.push(...(finTokens[0] as AlignedToken[]));
      this.draftTokens = draftTokens[0] as AlignedToken[];
    }
  }
}
