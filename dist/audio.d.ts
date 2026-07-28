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
    /** [features, nFft/2+1], row-major */
    filterbanks: Float32Array;
}
/** Log-mel output, laid out [nMels, numFrames] — the encoder's `audio_signal` order. */
export interface LogMel {
    data: Float32Array;
    nMels: number;
    numFrames: number;
}
/** Decode any ffmpeg-readable file to mono float32 PCM in [-1, 1]. */
export declare function loadAudioRaw(filename: string, samplingRate: number): Float32Array;
export declare function getWindow(name: string, size: number): Float32Array;
/**
 * Filterbank as parakeet-mlx builds it: mel points snapped to FFT bins with
 * `floor`. NOTE: at 16 kHz / nFft 512 / 128 mels this collapses 13 rows to all
 * zero — see docs/cuda.md. Kept as the default for bit-compatibility with the
 * existing MLX path and the current test fixtures.
 */
export declare function computeMelFilterbanks(sr: number, nFft: number, nMels: number, fMin: number, fMax: number): Float32Array;
/**
 * librosa/NeMo-style filterbank: triangles evaluated on continuous frequencies,
 * so no rows collapse. Matches NVIDIA's reference preprocessor. Opt in via
 * `makePreprocessArgs({ filterbank: 'interpolated' })`.
 */
export declare function computeMelFilterbanksInterpolated(sr: number, nFft: number, nMels: number, fMin: number, fMax: number): Float32Array;
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
    /** 'floor' (default, parakeet-mlx compatible) or 'interpolated' (NeMo reference) */
    filterbank?: 'floor' | 'interpolated';
}): PreprocessArgs;
/** Log-mel spectrogram from mono PCM. Output is [nMels, numFrames]. */
export declare function getLogMel(signalIn: Float32Array, args: PreprocessArgs): LogMel;
//# sourceMappingURL=audio.d.ts.map