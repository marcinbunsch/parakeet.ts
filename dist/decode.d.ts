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
import { AlignedToken } from './alignment.js';
export interface DecoderState {
    lastToken: number | null;
    hiddenState: DecoderStateHandle | null;
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
export declare function decodeTDTGreedy(backend: ParakeetBackend, enc: EncoderOutput, durations: number[], state: DecoderState, opts: GreedyOptions): Promise<{
    tokens: AlignedToken[];
    state: DecoderState;
}>;
/** Greedy RNN-T decoding (no duration head). */
export declare function decodeRNNTGreedy(backend: ParakeetBackend, enc: EncoderOutput, state: DecoderState, opts: GreedyOptions): Promise<{
    tokens: AlignedToken[];
    state: DecoderState;
}>;
//# sourceMappingURL=decode.d.ts.map