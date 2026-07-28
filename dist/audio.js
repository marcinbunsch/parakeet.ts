/**
 * Backend-agnostic audio front-end — pure TypeScript, no tensor library.
 *
 * This is a port of the MLX front-end in `src/mlx/audio.ts`, with the MxArray
 * tail (filterbank matmul, log, per-feature normalize) rewritten over plain
 * typed arrays so it runs identically under the MLX and ONNX backends.
 *
 * It intentionally reproduces parakeet-mlx's magnitude convention
 * (`|re| + |im|`, not `sqrt(re^2 + im^2)`) so both backends see the same
 * features. See docs/cuda.md for the parity analysis.
 */
import { spawnSync } from 'node:child_process';
// ---------------------------------------------------------------------------
// Audio loading
// ---------------------------------------------------------------------------
/** Decode any ffmpeg-readable file to mono float32 PCM in [-1, 1]. */
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
    for (let i = 0; i < int16.length; i++)
        float32[i] = int16[i] / 32768.0;
    return float32;
}
// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------
function hanningWindow(size) {
    const w = new Float32Array(size);
    for (let i = 0; i < size; i++)
        w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size);
    return w;
}
function hammingWindow(size) {
    const w = new Float32Array(size);
    for (let i = 0; i < size; i++)
        w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / size);
    return w;
}
function blackmanWindow(size) {
    const w = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        w[i] = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / size) + 0.08 * Math.cos((4 * Math.PI * i) / size);
    }
    return w;
}
function bartlettWindow(size) {
    const w = new Float32Array(size);
    for (let i = 0; i < size; i++)
        w[i] = 1.0 - Math.abs((2 * i - size) / size);
    return w;
}
export function getWindow(name, size) {
    switch (name.toLowerCase()) {
        case 'hann':
        case 'hanning': return hanningWindow(size);
        case 'hamming': return hammingWindow(size);
        case 'blackman': return blackmanWindow(size);
        case 'bartlett': return bartlettWindow(size);
        default: return hanningWindow(size);
    }
}
// ---------------------------------------------------------------------------
// FFT / STFT
// ---------------------------------------------------------------------------
/** Radix-2 Cooley-Tukey FFT. Returns interleaved [re, im] for the first nFft/2+1 bins. */
function rfft(input, nFft, re, im, out) {
    const N = nFft;
    re.set(input.subarray(0, Math.min(input.length, N)));
    if (input.length < N)
        re.fill(0, input.length);
    im.fill(0);
    let j = 0;
    for (let i = 1; i < N; i++) {
        let bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            const tr = re[i];
            re[i] = re[j];
            re[j] = tr;
            const ti = im[i];
            im[i] = im[j];
            im[j] = ti;
        }
    }
    for (let len = 2; len <= N; len <<= 1) {
        const ang = (-2 * Math.PI) / len;
        const wRe = Math.cos(ang), wIm = Math.sin(ang);
        const half = len >> 1;
        for (let i = 0; i < N; i += len) {
            let curRe = 1.0, curIm = 0.0;
            for (let k = 0; k < half; k++) {
                const uRe = re[i + k], uIm = im[i + k];
                const aRe = re[i + k + half], aIm = im[i + k + half];
                const vRe = aRe * curRe - aIm * curIm;
                const vIm = aRe * curIm + aIm * curRe;
                re[i + k] = uRe + vRe;
                im[i + k] = uIm + vIm;
                re[i + k + half] = uRe - vRe;
                im[i + k + half] = uIm - vIm;
                const n = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = n;
            }
        }
    }
    const numBins = (N >> 1) + 1;
    for (let i = 0; i < numBins; i++) {
        out[i * 2] = re[i];
        out[i * 2 + 1] = im[i];
    }
}
/** STFT magnitude, [numFrames, nFft/2+1]. */
function computeStft(signal, nFft, hopLength, winLength, window, magPower) {
    const padding = Math.floor(nFft / 2);
    const padded = new Float32Array(signal.length + 2 * padding);
    for (let i = 0; i < padding; i++)
        padded[padding - 1 - i] = signal[i + 1] ?? 0;
    padded.set(signal, padding);
    for (let i = 0; i < padding; i++) {
        const srcIdx = signal.length - 2 - i;
        padded[padding + signal.length + i] = signal[Math.max(0, srcIdx)] ?? 0;
    }
    const numBins = Math.floor(nFft / 2) + 1;
    const numFrames = Math.max(0, Math.floor((padded.length - winLength) / hopLength) + 1);
    // window is zero-padded to nFft at the END (matches parakeet-mlx)
    const win = new Float32Array(nFft);
    win.set(window.subarray(0, Math.min(winLength, nFft)));
    const magnitude = new Float32Array(numFrames * numBins);
    const frame = new Float32Array(nFft);
    const re = new Float32Array(nFft);
    const im = new Float32Array(nFft);
    const spectrum = new Float32Array(numBins * 2);
    for (let t = 0; t < numFrames; t++) {
        const start = t * hopLength;
        for (let i = 0; i < nFft; i++)
            frame[i] = (padded[start + i] ?? 0) * win[i];
        rfft(frame, nFft, re, im, spectrum);
        const base = t * numBins;
        if (magPower === 1.0) {
            for (let k = 0; k < numBins; k++) {
                magnitude[base + k] = Math.abs(spectrum[k * 2]) + Math.abs(spectrum[k * 2 + 1]);
            }
        }
        else {
            for (let k = 0; k < numBins; k++) {
                const mag = Math.abs(spectrum[k * 2]) + Math.abs(spectrum[k * 2 + 1]);
                magnitude[base + k] = Math.pow(mag, magPower);
            }
        }
    }
    return { magnitude, numFrames, numBins };
}
// ---------------------------------------------------------------------------
// Mel filterbanks
// ---------------------------------------------------------------------------
/**
 * Filterbank as parakeet-mlx builds it: mel points snapped to FFT bins with
 * `floor`. NOTE: at 16 kHz / nFft 512 / 128 mels this collapses 13 rows to all
 * zero — see docs/cuda.md. Kept as the default for bit-compatibility with the
 * existing MLX path and the current test fixtures.
 */
export function computeMelFilterbanks(sr, nFft, nMels, fMin, fMax) {
    const numBins = Math.floor(nFft / 2) + 1;
    const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
    const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);
    const melMin = hzToMel(fMin), melMax = hzToMel(fMax);
    const melPoints = new Float32Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++)
        melPoints[i] = melMin + (i / (nMels + 1)) * (melMax - melMin);
    const hzPoints = melPoints.map(m => melToHz(m));
    const binPoints = hzPoints.map(hz => Math.floor((hz * (nFft + 1)) / sr));
    const fb = new Float32Array(nMels * numBins);
    for (let m = 0; m < nMels; m++) {
        const fLeft = binPoints[m], fCenter = binPoints[m + 1], fRight = binPoints[m + 2];
        for (let k = fLeft; k < fCenter; k++) {
            if (k >= 0 && k < numBins)
                fb[m * numBins + k] = (k - fLeft) / Math.max(1, fCenter - fLeft);
        }
        for (let k = fCenter; k <= fRight; k++) {
            if (k >= 0 && k < numBins)
                fb[m * numBins + k] = (fRight - k) / Math.max(1, fRight - fCenter);
        }
    }
    for (let m = 0; m < nMels; m++) {
        const norm = 2.0 / Math.max(1e-8, hzPoints[m + 2] - hzPoints[m]);
        for (let k = 0; k < numBins; k++)
            fb[m * numBins + k] *= norm;
    }
    return fb;
}
/**
 * librosa/NeMo-style filterbank: triangles evaluated on continuous frequencies,
 * so no rows collapse. Matches NVIDIA's reference preprocessor. Opt in via
 * `makePreprocessArgs({ filterbank: 'interpolated' })`.
 */
export function computeMelFilterbanksInterpolated(sr, nFft, nMels, fMin, fMax) {
    const numBins = Math.floor(nFft / 2) + 1;
    const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
    const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);
    const melMin = hzToMel(fMin), melMax = hzToMel(fMax);
    const hzPoints = new Float64Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) {
        hzPoints[i] = melToHz(melMin + (i / (nMels + 1)) * (melMax - melMin));
    }
    const fftFreqs = new Float64Array(numBins);
    for (let k = 0; k < numBins; k++)
        fftFreqs[k] = (k * sr) / nFft;
    const fb = new Float32Array(nMels * numBins);
    for (let m = 0; m < nMels; m++) {
        const l = hzPoints[m], c = hzPoints[m + 1], r = hzPoints[m + 2];
        const norm = 2.0 / Math.max(1e-8, r - l);
        for (let k = 0; k < numBins; k++) {
            const f = fftFreqs[k];
            const up = (f - l) / Math.max(1e-8, c - l);
            const dn = (r - f) / Math.max(1e-8, r - c);
            fb[m * numBins + k] = Math.max(0, Math.min(up, dn)) * norm;
        }
    }
    return fb;
}
// ---------------------------------------------------------------------------
// Log-mel
// ---------------------------------------------------------------------------
export function makePreprocessArgs(args) {
    const winLength = Math.round(args.windowSize * args.sampleRate);
    const hopLength = Math.round(args.windowStride * args.sampleRate);
    const build = args.filterbank === 'interpolated'
        ? computeMelFilterbanksInterpolated
        : computeMelFilterbanks;
    const filterbanks = build(args.sampleRate, args.nFft, args.features, 0, args.sampleRate / 2);
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
/** Log-mel spectrogram from mono PCM. Output is [nMels, numFrames]. */
export function getLogMel(signalIn, args) {
    let signal = signalIn;
    if (args.padTo > 0 && signal.length < args.padTo) {
        const padded = new Float32Array(args.padTo).fill(args.padValue);
        padded.set(signal);
        signal = padded;
    }
    if (args.preemph !== null) {
        const p = new Float32Array(signal.length);
        p[0] = signal[0];
        for (let i = 1; i < signal.length; i++)
            p[i] = signal[i] - args.preemph * signal[i - 1];
        signal = p;
    }
    const win = getWindow(args.window, args.winLength);
    const { magnitude, numFrames, numBins } = computeStft(signal, args.nFft, args.hopLength, args.winLength, win, args.magPower);
    const nMels = args.features;
    const fb = args.filterbanks;
    // mel[m, t] = log(sum_k fb[m, k] * mag[t, k] + 1e-5)
    const out = new Float32Array(nMels * numFrames);
    for (let m = 0; m < nMels; m++) {
        const fbRow = m * numBins;
        const outRow = m * numFrames;
        for (let t = 0; t < numFrames; t++) {
            let acc = 0;
            const magRow = t * numBins;
            for (let k = 0; k < numBins; k++)
                acc += fb[fbRow + k] * magnitude[magRow + k];
            out[outRow + t] = Math.log(acc + 1e-5);
        }
    }
    if (args.normalize === 'per_feature') {
        for (let m = 0; m < nMels; m++) {
            const base = m * numFrames;
            let mean = 0;
            for (let t = 0; t < numFrames; t++)
                mean += out[base + t];
            mean /= numFrames;
            let varr = 0;
            for (let t = 0; t < numFrames; t++) {
                const d = out[base + t] - mean;
                varr += d * d;
            }
            const inv = 1 / (Math.sqrt(varr / numFrames) + 1e-5);
            for (let t = 0; t < numFrames; t++)
                out[base + t] = (out[base + t] - mean) * inv;
        }
    }
    else {
        let mean = 0;
        for (let i = 0; i < out.length; i++)
            mean += out[i];
        mean /= out.length;
        let varr = 0;
        for (let i = 0; i < out.length; i++) {
            const d = out[i] - mean;
            varr += d * d;
        }
        const inv = 1 / (Math.sqrt(varr / out.length) + 1e-5);
        for (let i = 0; i < out.length; i++)
            out[i] = (out[i] - mean) * inv;
    }
    return { data: out, nMels, numFrames };
}
//# sourceMappingURL=audio.js.map