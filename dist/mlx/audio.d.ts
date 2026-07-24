import { MxArray } from '@mlx-node/core';
export interface PreprocessArgs {
    sampleRate: number;
    normalize: string;
    windowSize: number;
    windowStride: number;
    window: string;
    features: number;
    nFft: number;
    dither: number;
    padTo: number;
    padValue: number;
    preemph: number | null;
    magPower: number;
    winLength: number;
    hopLength: number;
    filterbanks: MxArray;
}
export declare function makePreprocessArgs(args: {
    sampleRate: number;
    normalize: string;
    windowSize: number;
    windowStride: number;
    window: string;
    features: number;
    nFft: number;
    dither: number;
    padTo?: number;
    padValue?: number;
    preemph?: number | null;
    magPower?: number;
}): PreprocessArgs;
/**
 * Load an audio file using ffmpeg and return a Float32Array at the given sample rate.
 * Output is normalized to [-1, 1].
 */
export declare function loadAudioRaw(filename: string, samplingRate: number): Float32Array;
/** Load audio as MxArray (float32) */
export declare function loadAudio(filename: string, samplingRate: number): MxArray;
/**
 * Compute log-mel spectrogram from a 1D audio signal.
 * Input: MxArray of shape [T] (float32)
 * Output: MxArray of shape [1, frames, nMels]
 */
export declare function getLogMel(audio: MxArray, args: PreprocessArgs): MxArray;
//# sourceMappingURL=audio.d.ts.map