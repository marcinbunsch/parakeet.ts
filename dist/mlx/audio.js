import { spawnSync } from 'node:child_process';
import { MxArray } from '@mlx-node/core';
export function makePreprocessArgs(args) {
    const winLength = Math.round(args.windowSize * args.sampleRate);
    const hopLength = Math.round(args.windowStride * args.sampleRate);
    const filterbanks = computeMelFilterbanks(args.sampleRate, args.nFft, args.features, 0, args.sampleRate / 2);
    return {
        ...args,
        padTo: args.padTo ?? 0,
        padValue: args.padValue ?? 0,
        preemph: args.preemph !== undefined ? args.preemph : 0.97,
        magPower: args.magPower ?? 2.0,
        winLength,
        hopLength,
        filterbanks,
    };
}
/**
 * Load an audio file using ffmpeg and return a Float32Array at the given sample rate.
 * Output is normalized to [-1, 1].
 */
export function loadAudioRaw(filename, samplingRate) {
    const result = spawnSync('ffmpeg', [
        '-nostdin', '-i', filename,
        '-threads', '0',
        '-f', 's16le',
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        '-ar', String(samplingRate),
        '-',
    ], { maxBuffer: 512 * 1024 * 1024 });
    if (result.error)
        throw new Error(`FFmpeg not found: ${result.error.message}`);
    if (result.status !== 0) {
        throw new Error(`FFmpeg failed: ${result.stderr?.toString() ?? 'unknown error'}`);
    }
    const buf = result.stdout;
    const int16 = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768.0;
    }
    return float32;
}
/** Load audio as MxArray (float32) */
export function loadAudio(filename, samplingRate) {
    const raw = loadAudioRaw(filename, samplingRate);
    return MxArray.fromFloat32(raw, BigInt64Array.from([BigInt(raw.length)]));
}
// ---- Window functions (CPU-side) ----
function hanningWindow(size) {
    const w = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size);
    }
    return w;
}
function hammingWindow(size) {
    const w = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / size);
    }
    return w;
}
function blackmanWindow(size) {
    const w = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        w[i] =
            0.42 -
                0.5 * Math.cos((2 * Math.PI * i) / size) +
                0.08 * Math.cos((4 * Math.PI * i) / size);
    }
    return w;
}
function bartlettWindow(size) {
    const w = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        w[i] = 1.0 - Math.abs((2 * i - size) / size);
    }
    return w;
}
function getWindow(name, size) {
    switch (name.toLowerCase()) {
        case 'hann':
        case 'hanning':
            return hanningWindow(size);
        case 'hamming':
            return hammingWindow(size);
        case 'blackman':
            return blackmanWindow(size);
        case 'bartlett':
            return bartlettWindow(size);
        default:
            return hanningWindow(size);
    }
}
// ---- Real FFT (Cooley-Tukey, radix-2) ----
/** Compute real-valued FFT. Returns interleaved [re0, im0, re1, im1, ...] of length nFft/2+1 pairs. */
function rfft(input, nFft) {
    // Zero-pad or truncate to nFft
    const x = new Float32Array(nFft);
    x.set(input.subarray(0, Math.min(input.length, nFft)));
    // In-place Cooley-Tukey FFT (complex, using real+imaginary interleaved)
    const N = nFft;
    const re = new Float32Array(N);
    const im = new Float32Array(N);
    re.set(x);
    // Bit-reversal permutation
    let j = 0;
    for (let i = 1; i < N; i++) {
        let bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }
    // FFT butterfly
    for (let len = 2; len <= N; len <<= 1) {
        const ang = (-2 * Math.PI) / len;
        const wRe = Math.cos(ang);
        const wIm = Math.sin(ang);
        for (let i = 0; i < N; i += len) {
            let curRe = 1.0;
            let curIm = 0.0;
            for (let k = 0; k < len / 2; k++) {
                const uRe = re[i + k];
                const uIm = im[i + k];
                const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
                const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
                re[i + k] = uRe + vRe;
                im[i + k] = uIm + vIm;
                re[i + k + len / 2] = uRe - vRe;
                im[i + k + len / 2] = uIm - vIm;
                const newCurRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = newCurRe;
            }
        }
    }
    // Return only first nFft/2+1 complex values (positive frequencies)
    const numBins = Math.floor(N / 2) + 1;
    const out = new Float32Array(numBins * 2);
    for (let i = 0; i < numBins; i++) {
        out[i * 2] = re[i];
        out[i * 2 + 1] = im[i];
    }
    return out;
}
/** Compute STFT magnitude spectrum. Returns [numFrames, nFft/2+1] as Float32Array. */
function computeStft(signal, nFft, hopLength, winLength, window, magPower) {
    // Reflect-pad signal by nFft/2 on each side
    const padding = Math.floor(nFft / 2);
    const padded = new Float32Array(signal.length + 2 * padding);
    // Reflect at the start
    for (let i = 0; i < padding; i++) {
        padded[padding - 1 - i] = signal[i + 1] ?? 0;
    }
    // Copy signal
    padded.set(signal, padding);
    // Reflect at the end
    for (let i = 0; i < padding; i++) {
        const srcIdx = signal.length - 2 - i;
        padded[padding + signal.length + i] = signal[Math.max(0, srcIdx)] ?? 0;
    }
    const numBins = Math.floor(nFft / 2) + 1;
    const numFrames = Math.floor((padded.length - winLength) / hopLength) + 1;
    // Pad window to nFft with zeros at the END (matches Python parakeet-mlx behavior)
    const win = new Float32Array(nFft);
    if (winLength <= nFft) {
        win.set(window.subarray(0, winLength));
    }
    else {
        win.set(window.subarray(0, nFft));
    }
    const magnitude = new Float32Array(numFrames * numBins);
    for (let t = 0; t < numFrames; t++) {
        const start = t * hopLength;
        const frame = new Float32Array(nFft);
        // Apply window
        for (let i = 0; i < nFft; i++) {
            frame[i] = (padded[start + i] ?? 0) * win[i];
        }
        const spectrum = rfft(frame, nFft);
        // Python parakeet-mlx uses |re| + |im| (L1 norm), NOT sqrt(re^2 + im^2)
        if (magPower === 1.0) {
            for (let k = 0; k < numBins; k++) {
                const re = spectrum[k * 2];
                const im = spectrum[k * 2 + 1];
                magnitude[t * numBins + k] = Math.abs(re) + Math.abs(im);
            }
        }
        else {
            for (let k = 0; k < numBins; k++) {
                const re = spectrum[k * 2];
                const im = spectrum[k * 2 + 1];
                const mag = Math.abs(re) + Math.abs(im);
                magnitude[t * numBins + k] = Math.pow(mag, magPower);
            }
        }
    }
    return magnitude; // [numFrames, numBins]
}
// ---- Mel filterbank computation ----
/** Compute mel filterbank matrix [nMels, nFft/2+1]. */
function computeMelFilterbanks(sr, nFft, nMels, fMin, fMax) {
    const numBins = Math.floor(nFft / 2) + 1;
    const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
    const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);
    const melMin = hzToMel(fMin);
    const melMax = hzToMel(fMax);
    // nMels+2 mel points (including endpoints)
    const melPoints = new Float32Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) {
        melPoints[i] = melMin + (i / (nMels + 1)) * (melMax - melMin);
    }
    // Convert back to Hz, then to FFT bin index
    const hzPoints = melPoints.map(m => melToHz(m));
    const binPoints = hzPoints.map(hz => Math.floor((hz * (nFft + 1)) / sr));
    // Build filterbank [nMels, numBins]
    const fb = new Float32Array(nMels * numBins);
    for (let m = 0; m < nMels; m++) {
        const fLeft = binPoints[m];
        const fCenter = binPoints[m + 1];
        const fRight = binPoints[m + 2];
        // Rising slope
        for (let k = fLeft; k < fCenter; k++) {
            if (k >= 0 && k < numBins) {
                fb[m * numBins + k] = (k - fLeft) / Math.max(1, fCenter - fLeft);
            }
        }
        // Falling slope
        for (let k = fCenter; k <= fRight; k++) {
            if (k >= 0 && k < numBins) {
                fb[m * numBins + k] = (fRight - k) / Math.max(1, fRight - fCenter);
            }
        }
    }
    // Apply Slaney normalization (divide by mel width in Hz)
    for (let m = 0; m < nMels; m++) {
        const hzLeft = hzPoints[m];
        const hzRight = hzPoints[m + 2];
        const norm = 2.0 / Math.max(1e-8, hzRight - hzLeft);
        for (let k = 0; k < numBins; k++) {
            fb[m * numBins + k] *= norm;
        }
    }
    return MxArray.fromFloat32(fb, BigInt64Array.from([BigInt(nMels), BigInt(numBins)]));
}
/**
 * Compute log-mel spectrogram from a 1D audio signal.
 * Input: MxArray of shape [T] (float32)
 * Output: MxArray of shape [1, frames, nMels]
 */
export function getLogMel(audio, args) {
    // Bring to CPU
    let signal = audio.toFloat32();
    const origLen = signal.length;
    // Pad to padTo
    if (args.padTo > 0 && origLen < args.padTo) {
        const padded = new Float32Array(args.padTo).fill(args.padValue);
        padded.set(signal);
        signal = padded;
    }
    // Pre-emphasis filter
    if (args.preemph !== null) {
        const preemphd = new Float32Array(signal.length);
        preemphd[0] = signal[0];
        for (let i = 1; i < signal.length; i++) {
            preemphd[i] = signal[i] - args.preemph * signal[i - 1];
        }
        signal = preemphd;
    }
    // Window function
    const win = getWindow(args.window, args.winLength);
    // STFT magnitude: [numFrames, numBins]
    const magnitude = computeStft(signal, args.nFft, args.hopLength, args.winLength, win, args.magPower);
    const numBins = Math.floor(args.nFft / 2) + 1;
    const numFrames = magnitude.length / numBins;
    // Convert to MxArray [numFrames, numBins]
    const magArray = MxArray.fromFloat32(magnitude, BigInt64Array.from([BigInt(numFrames), BigInt(numBins)]));
    // Mel filterbank: [nMels, numBins] @ magArray.T → [nMels, numFrames]
    // magArray: [numFrames, numBins] → transpose → [numBins, numFrames]
    // filterbanks: [nMels, numBins]
    // result: [nMels, numFrames]
    const magT = magArray.transpose(new Int32Array([1, 0])); // [numBins, numFrames]
    const melSpec = args.filterbanks.matmul(magT); // [nMels, numFrames]
    // Log(mel + 1e-5)
    const logMel = melSpec.addScalar(1e-5).log();
    // Normalize
    let normalizedMel;
    if (args.normalize === 'per_feature') {
        const mean = logMel.mean(new Int32Array([1]), true); // [nMels, 1]
        const std = logMel.std(new Int32Array([1]), true); // [nMels, 1]
        normalizedMel = logMel.sub(mean).div(std.addScalar(1e-5));
    }
    else {
        const mean = logMel.mean();
        const std = logMel.std();
        normalizedMel = logMel.sub(mean).div(std.addScalar(1e-5));
    }
    // Transpose: [nMels, numFrames] → [numFrames, nMels]
    const transposed = normalizedMel.transpose(new Int32Array([1, 0]));
    // Add batch dimension: [1, numFrames, nMels]
    return transposed.expandDims(0);
}
//# sourceMappingURL=audio.js.map