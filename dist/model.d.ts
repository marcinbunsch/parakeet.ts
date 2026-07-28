/**
 * Backend-agnostic Parakeet model.
 *
 * This class contains no tensor code at all. It owns the shared audio
 * front-end, the greedy decode loops, chunking and streaming, and drives
 * whatever `ParakeetBackend` it is given — MLX on Apple Silicon, ONNX Runtime
 * with the CUDA execution provider on Linux/Nvidia.
 */
import type { ParakeetBackend, EncoderOutput } from './backend.js';
import { PreprocessArgs } from './audio.js';
import { DecoderState } from './decode.js';
import { AlignedToken, AlignedResult, SentenceConfig } from './alignment.js';
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
export declare class ParakeetModel {
    readonly backend: ParakeetBackend;
    readonly preprocessorConfig: PreprocessArgs;
    readonly vocabulary: string[];
    readonly durations: number[] | null;
    readonly maxSymbols: number | null;
    readonly subsamplingFactor: number;
    constructor(args: ParakeetModelArgs);
    /** Seconds of audio per encoder frame. */
    get timeRatio(): number;
    /** Run the shared mel front-end and the backend encoder. */
    encodePcm(pcm: Float32Array): Promise<EncoderOutput>;
    /** Greedy-decode an encoder output, threading decoder state. */
    decode(enc: EncoderOutput, state?: DecoderState, range?: {
        from?: number;
        to?: number;
        timeOffset?: number;
    }): Promise<{
        tokens: AlignedToken[];
        state: DecoderState;
    }>;
    transcribe(file: string, options?: TranscribeOptions): Promise<AlignedResult>;
    /** Transcribe raw mono float32 PCM at the model's sample rate. */
    transcribePcm(pcm: Float32Array, options?: TranscribeOptions): Promise<AlignedResult>;
    transcribeStream(options?: StreamOptions): StreamingParakeet;
    dispose(): Promise<void>;
}
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
export declare class StreamingParakeet {
    private readonly model;
    private readonly windowSamples;
    private readonly dropFrames;
    private readonly sentence?;
    private audio;
    /** Absolute encoder-frame index corresponding to `audio[0]`. */
    private windowStartFrame;
    /** Absolute encoder frame already committed. */
    private finalizedUpTo;
    private state;
    private _finalized;
    private _draft;
    constructor(model: ParakeetModel, options?: StreamOptions);
    private get samplesPerFrame();
    /** Committed tokens — will not be revised. */
    get finalizedTokens(): AlignedToken[];
    /** Tokens in the revisable tail. */
    get draftTokens(): AlignedToken[];
    get finalizedResult(): AlignedResult;
    /** Best current guess: committed + draft. */
    get result(): AlignedResult;
    addAudio(pcm: Float32Array): Promise<void>;
    /** Commit the remaining draft. Call once the audio stream ends. */
    finish(): AlignedResult;
}
//# sourceMappingURL=model.d.ts.map