import { MxArray } from '@mlx-node/core';
import { Module, WeightMap } from './nn.js';
import { Conformer, ConformerArgs } from './conformer.js';
import { PredictNetwork, JointNetwork, PredictArgs, JointArgs, DecoderState } from './rnnt.js';
import { ConvASRDecoder, ConvASRDecoderArgs, AuxCTCArgs } from './ctc.js';
import { PreprocessArgs } from './audio.js';
import { AlignedToken, AlignedResult, SentenceConfig } from '../alignment.js';
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
export declare function greedy(): Greedy;
export declare function beam(options?: Partial<Omit<Beam, 'type'>>): Beam;
export interface DecodingConfig {
    decoding: DecodingStrategy;
    sentence: SentenceConfig;
}
export declare function defaultDecodingConfig(): DecodingConfig;
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
export interface TranscribeOptions {
    decodingConfig?: DecodingConfig;
    chunkDuration?: number;
    overlapDuration?: number;
    onChunk?: (current: number, total: number) => void;
}
export declare abstract class BaseParakeet extends Module {
    readonly preprocessorConfig: PreprocessArgs;
    readonly encoderConfig: ConformerArgs;
    readonly encoder: Conformer;
    private readonly _mutex;
    constructor(preprocessorConfig: PreprocessArgs, encoderConfig: ConformerArgs);
    /** Acquire the encoder mutex. Used internally by transcribe/transcribeStream. */
    _acquireMutex(): Promise<() => void>;
    get timeRatio(): number;
    abstract generate(mel: MxArray, decodingConfig?: DecodingConfig): AlignedResult[];
    transcribe(path: string, options?: TranscribeOptions): Promise<AlignedResult>;
    private _transcribeInner;
    transcribeStream(contextSize?: [number, number], depth?: number, decodingConfig?: DecodingConfig, keepOriginalAttention?: boolean): StreamingParakeet;
    loadWeights(weights: WeightMap, prefix?: string): void;
}
export declare class ParakeetTDT extends BaseParakeet {
    readonly vocabulary: string[];
    readonly durations: number[];
    readonly maxSymbols: number | null;
    readonly decoder: PredictNetwork;
    readonly joint: JointNetwork;
    constructor(args: ParakeetTDTArgs);
    decode(features: MxArray, lengths: MxArray, states: DecoderState[], decodingConfig?: DecodingConfig): [Array<AlignedToken[]>, DecoderState[]];
    generate(mel: MxArray, decodingConfig?: DecodingConfig): AlignedResult[];
    loadWeights(weights: WeightMap, prefix?: string): void;
}
export declare class ParakeetRNNT extends BaseParakeet {
    readonly vocabulary: string[];
    readonly maxSymbols: number | null;
    readonly decoder: PredictNetwork;
    readonly joint: JointNetwork;
    constructor(args: ParakeetRNNTArgs);
    decode(features: MxArray, lengths: MxArray, states: DecoderState[], decodingConfig?: DecodingConfig): [Array<AlignedToken[]>, DecoderState[]];
    generate(mel: MxArray, decodingConfig?: DecodingConfig): AlignedResult[];
    loadWeights(weights: WeightMap, prefix?: string): void;
}
export declare class ParakeetCTC extends BaseParakeet {
    readonly vocabulary: string[];
    readonly ctcDecoder: ConvASRDecoder;
    constructor(args: ParakeetCTCArgs);
    decode(features: MxArray, lengths: MxArray, decodingConfig?: DecodingConfig): Array<AlignedToken[]>;
    generate(mel: MxArray, decodingConfig?: DecodingConfig): AlignedResult[];
    loadWeights(weights: WeightMap, prefix?: string): void;
}
export declare class ParakeetTDTCTC extends ParakeetTDT {
    readonly ctcDecoder: ConvASRDecoder;
    constructor(args: ParakeetTDTCTCArgs);
    loadWeights(weights: WeightMap, prefix?: string): void;
}
export declare class StreamingParakeet {
    private readonly model;
    private readonly contextSize;
    private readonly depth;
    private readonly decodingConfig;
    private readonly keepOriginalAttention;
    private cache;
    private audioBuffer;
    private melBuffer;
    private decoderHidden;
    private lastToken;
    private _finalizedTokens;
    private _draftTokens;
    private _releaseMutex;
    constructor(model: BaseParakeet, contextSize: [number, number], depth: number, decodingConfig: DecodingConfig, keepOriginalAttention: boolean);
    get keepSize(): number;
    get dropSize(): number;
    start(): Promise<void>;
    stop(): void;
    /** Committed tokens — will not be revised on future addAudio calls. */
    get finalizedTokens(): AlignedToken[];
    /** Tokens in the rotating context window — may be revised on subsequent addAudio calls. */
    get draftTokens(): AlignedToken[];
    /** AlignedResult built from finalized tokens only — safe to persist incrementally. */
    get finalizedResult(): AlignedResult;
    /** Full transcript (finalized + draft) — best current guess. */
    get result(): AlignedResult;
    addAudio(audio: Float32Array): void;
}
/**
 * Feed an async iterable of raw PCM frames (Float32Array, 16kHz mono) into a
 * StreamingParakeet session and return the final AlignedResult.
 *
 * This is the common "consume and wait" pattern used by both the CLI --stream
 * flag and the HTTP server's POST /transcribe endpoint.
 */
export declare function consumePcmStream(stream: StreamingParakeet, source: AsyncIterable<Float32Array>): Promise<AlignedResult>;
//# sourceMappingURL=parakeet.d.ts.map