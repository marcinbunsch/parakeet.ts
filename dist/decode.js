import { makeAlignedToken } from './alignment.js';
import { decode as decodeTokens } from './tokenizer.js';
/** Copy encoder frame `t` into `out`, honouring the backend's layout. */
function readFrame(enc, t, out) {
    if (enc.layout === 'time-major') {
        const base = t * enc.dim;
        for (let d = 0; d < enc.dim; d++)
            out[d] = enc.data[base + d];
    }
    else {
        for (let d = 0; d < enc.dim; d++)
            out[d] = enc.data[d * enc.stride + t];
    }
    return out;
}
/** Softmax in place over `n` values starting at `offset`, returning the array. */
function softmax(src, offset, n) {
    const out = new Float32Array(n);
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
        const v = src[offset + i];
        if (v > max)
            max = v;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) {
        const e = Math.exp(src[offset + i] - max);
        out[i] = e;
        sum += e;
    }
    for (let i = 0; i < n; i++)
        out[i] /= sum;
    return out;
}
function argmax(a) {
    let best = 0, bv = a[0];
    for (let i = 1; i < a.length; i++)
        if (a[i] > bv) {
            bv = a[i];
            best = i;
        }
    return best;
}
/** Normalized-entropy confidence, matching the MLX implementation. */
function confidenceOf(probs) {
    let entropy = 0;
    for (let i = 0; i < probs.length; i++) {
        const p = probs[i] + 1e-10;
        entropy -= p * Math.log(p);
    }
    return 1.0 - entropy / Math.log(probs.length);
}
/**
 * Greedy TDT decoding. `durations` is the TDT duration table, e.g. [0,1,2,3,4].
 */
export async function decodeTDTGreedy(backend, enc, durations, state, opts) {
    const { vocabulary, maxSymbols, timeRatio, timeOffset = 0 } = opts;
    const from = opts.from ?? 0;
    const to = Math.min(opts.to ?? enc.frames, enc.frames);
    const vocabSize = vocabulary.length;
    const tokens = [];
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
            tokens.push(makeAlignedToken(predToken, decodeTokens([predToken], vocabulary), timeOffset + step * timeRatio, durations[decision] * timeRatio, confidenceOf(tokenProbs)));
            lastToken = predToken;
            hiddenState = nextState;
        }
        step += durations[decision];
        newSymbols += 1;
        if (durations[decision] !== 0) {
            newSymbols = 0;
        }
        else if (maxSymbols !== null && newSymbols >= maxSymbols) {
            step += 1;
            newSymbols = 0;
        }
    }
    return { tokens, state: { lastToken, hiddenState } };
}
/** Greedy RNN-T decoding (no duration head). */
export async function decodeRNNTGreedy(backend, enc, state, opts) {
    const { vocabulary, maxSymbols, timeRatio, timeOffset = 0 } = opts;
    const from = opts.from ?? 0;
    const to = Math.min(opts.to ?? enc.frames, enc.frames);
    const vocabSize = vocabulary.length;
    const tokens = [];
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
            tokens.push(makeAlignedToken(predToken, decodeTokens([predToken], vocabulary), timeOffset + step * timeRatio, timeRatio, confidenceOf(tokenProbs)));
            lastToken = predToken;
            hiddenState = nextState;
            newSymbols += 1;
            if (maxSymbols !== null && newSymbols >= maxSymbols) {
                step += 1;
                newSymbols = 0;
            }
        }
        else {
            step += 1;
            newSymbols = 0;
        }
    }
    return { tokens, state: { lastToken, hiddenState } };
}
//# sourceMappingURL=decode.js.map