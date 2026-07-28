import { MxArray } from '@mlx-node/core';
import { Module } from './nn.js';
import { Conformer } from './conformer.js';
import { PredictNetwork, JointNetwork, decodeTDTGreedy, decodeRNNTGreedy, } from './rnnt.js';
import { ConvASRDecoder, decodeCTCGreedy } from './ctc.js';
import { loadAudio, getLogMel } from './audio.js';
import { tokensToSentences, sentencesToResult, mergeLongestContiguous, mergeLongestCommonSubsequence, } from '../alignment.js';
import { RotatingConformerCache } from './cache.js';
function s(...dims) {
    return BigInt64Array.from(dims.map(BigInt));
}
export function greedy() {
    return { type: 'greedy' };
}
export function beam(options = {}) {
    return {
        type: 'beam',
        beamSize: options.beamSize ?? 5,
        lengthPenalty: options.lengthPenalty ?? 1.0,
        patience: options.patience ?? 1.0,
        durationReward: options.durationReward ?? 0.7,
    };
}
export function defaultDecodingConfig() {
    return { decoding: greedy(), sentence: {} };
}
// ---------------------------------------------------------------------------
// Async mutex (per-instance concurrency safety)
// ---------------------------------------------------------------------------
class AsyncMutex {
    _locked = false;
    _queue = [];
    acquire() {
        return new Promise(resolve => {
            if (!this._locked) {
                this._locked = true;
                resolve(this._release.bind(this));
            }
            else {
                this._queue.push(() => resolve(this._release.bind(this)));
            }
        });
    }
    _release() {
        if (this._queue.length > 0) {
            this._queue.shift()();
        }
        else {
            this._locked = false;
        }
    }
}
// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------
export class BaseParakeet extends Module {
    preprocessorConfig;
    encoderConfig;
    encoder;
    _mutex = new AsyncMutex();
    constructor(preprocessorConfig, encoderConfig) {
        super();
        this.preprocessorConfig = preprocessorConfig;
        this.encoderConfig = encoderConfig;
        this.encoder = new Conformer(encoderConfig);
    }
    /** Acquire the encoder mutex. Used internally by transcribe/transcribeStream. */
    _acquireMutex() {
        return this._mutex.acquire();
    }
    get timeRatio() {
        return ((this.encoderConfig.subsamplingFactor /
            this.preprocessorConfig.sampleRate) *
            this.preprocessorConfig.hopLength);
    }
    async transcribe(path, options = {}) {
        const release = await this._acquireMutex();
        try {
            return await this._transcribeInner(path, options);
        }
        finally {
            release();
        }
    }
    async _transcribeInner(path, options = {}) {
        const { decodingConfig = defaultDecodingConfig(), chunkDuration, overlapDuration = 15.0, onChunk, } = options;
        const audioData = loadAudio(path, this.preprocessorConfig.sampleRate);
        const audioLen = Number(audioData.shape()[0]);
        if (chunkDuration === undefined || audioLen / this.preprocessorConfig.sampleRate <= chunkDuration) {
            const mel = getLogMel(audioData, this.preprocessorConfig);
            return this.generate(mel, decodingConfig)[0];
        }
        const chunkSamples = Math.floor(chunkDuration * this.preprocessorConfig.sampleRate);
        const overlapSamples = Math.floor(overlapDuration * this.preprocessorConfig.sampleRate);
        let allTokens = [];
        const audioDataF32 = audioData.toFloat32();
        for (let start = 0; start < audioDataF32.length; start += chunkSamples - overlapSamples) {
            const end = Math.min(start + chunkSamples, audioDataF32.length);
            if (onChunk)
                onChunk(end, audioDataF32.length);
            if (end - start < this.preprocessorConfig.hopLength)
                break;
            const chunkF32 = audioDataF32.slice(start, end);
            const chunkAudio = MxArray.fromFloat32(chunkF32, s(chunkF32.length));
            const chunkMel = getLogMel(chunkAudio, this.preprocessorConfig);
            const chunkResult = this.generate(chunkMel, decodingConfig)[0];
            const chunkOffset = start / this.preprocessorConfig.sampleRate;
            for (const sentence of chunkResult.sentences) {
                for (const token of sentence.tokens) {
                    token.start += chunkOffset;
                    token.end = token.start + token.duration;
                }
            }
            if (allTokens.length > 0) {
                try {
                    allTokens = mergeLongestContiguous(allTokens, chunkResult.tokens, overlapDuration);
                }
                catch {
                    allTokens = mergeLongestCommonSubsequence(allTokens, chunkResult.tokens, overlapDuration);
                }
            }
            else {
                allTokens = chunkResult.tokens;
            }
        }
        return sentencesToResult(tokensToSentences(allTokens, decodingConfig.sentence));
    }
    transcribeStream(contextSize = [256, 256], depth = 1, decodingConfig = defaultDecodingConfig(), keepOriginalAttention = false) {
        return new StreamingParakeet(this, contextSize, depth, decodingConfig, keepOriginalAttention);
    }
    loadWeights(weights, prefix = '') {
        this.encoder.loadWeights(weights, prefix ? `${prefix}.encoder` : 'encoder');
    }
}
// ---------------------------------------------------------------------------
// ParakeetTDT
// ---------------------------------------------------------------------------
export class ParakeetTDT extends BaseParakeet {
    vocabulary;
    durations;
    maxSymbols;
    decoder;
    joint;
    constructor(args) {
        super(args.preprocessor, args.encoder);
        if (args.decoding.modelType !== 'tdt')
            throw new Error('Model must be a TDT model');
        this.vocabulary = args.joint.vocabulary;
        this.durations = args.decoding.durations;
        this.maxSymbols =
            args.decoding.greedy?.['max_symbols'] != null
                ? Number(args.decoding.greedy['max_symbols'])
                : null;
        this.decoder = new PredictNetwork(args.decoder);
        this.joint = new JointNetwork(args.joint);
    }
    decode(features, lengths, states, decodingConfig = defaultDecodingConfig()) {
        const B = Number(features.shape()[0]);
        const effectiveStates = states.length === B ? states : Array.from({ length: B }, () => ({
            lastToken: null,
            hiddenState: null,
        }));
        return decodeTDTGreedy(features, lengths, this.decoder, this.joint, this.vocabulary, this.durations, this.maxSymbols, effectiveStates, this.timeRatio); // dynamic import makes typing tricky
    }
    generate(mel, decodingConfig = defaultDecodingConfig()) {
        if (mel.ndim() === 2)
            mel = mel.expandDims(0);
        const [features, lengths] = this.encoder.forward(mel, null, null);
        features.eval();
        lengths.eval();
        const B = Number(features.shape()[0]);
        const initStates = Array.from({ length: B }, () => ({ lastToken: null, hiddenState: null }));
        const [result] = this.decode(features, lengths, initStates, decodingConfig);
        return result.map(tokens => sentencesToResult(tokensToSentences(tokens, decodingConfig.sentence)));
    }
    loadWeights(weights, prefix = '') {
        super.loadWeights(weights, prefix);
        this.decoder.loadWeights(weights, prefix ? `${prefix}.decoder` : 'decoder');
        this.joint.loadWeights(weights, prefix ? `${prefix}.joint` : 'joint');
    }
}
// ---------------------------------------------------------------------------
// ParakeetRNNT
// ---------------------------------------------------------------------------
export class ParakeetRNNT extends BaseParakeet {
    vocabulary;
    maxSymbols;
    decoder;
    joint;
    constructor(args) {
        super(args.preprocessor, args.encoder);
        this.vocabulary = args.joint.vocabulary;
        this.maxSymbols =
            args.decoding.greedy?.['max_symbols'] != null
                ? Number(args.decoding.greedy['max_symbols'])
                : null;
        this.decoder = new PredictNetwork(args.decoder);
        this.joint = new JointNetwork(args.joint);
    }
    decode(features, lengths, states, decodingConfig = defaultDecodingConfig()) {
        const B = Number(features.shape()[0]);
        const effectiveStates = states.length === B ? states : Array.from({ length: B }, () => ({
            lastToken: null,
            hiddenState: null,
        }));
        return decodeRNNTGreedy(features, lengths, this.decoder, this.joint, this.vocabulary, this.maxSymbols, effectiveStates, this.timeRatio);
    }
    generate(mel, decodingConfig = defaultDecodingConfig()) {
        if (mel.ndim() === 2)
            mel = mel.expandDims(0);
        const [features, lengths] = this.encoder.forward(mel, null, null);
        features.eval();
        lengths.eval();
        const B = Number(features.shape()[0]);
        const initStates = Array.from({ length: B }, () => ({ lastToken: null, hiddenState: null }));
        const [result] = this.decode(features, lengths, initStates, decodingConfig);
        return result.map(tokens => sentencesToResult(tokensToSentences(tokens, decodingConfig.sentence)));
    }
    loadWeights(weights, prefix = '') {
        super.loadWeights(weights, prefix);
        this.decoder.loadWeights(weights, prefix ? `${prefix}.decoder` : 'decoder');
        this.joint.loadWeights(weights, prefix ? `${prefix}.joint` : 'joint');
    }
}
// ---------------------------------------------------------------------------
// ParakeetCTC
// ---------------------------------------------------------------------------
export class ParakeetCTC extends BaseParakeet {
    vocabulary;
    ctcDecoder;
    constructor(args) {
        super(args.preprocessor, args.encoder);
        this.vocabulary = args.decoder.vocabulary;
        this.ctcDecoder = new ConvASRDecoder(args.decoder);
    }
    decode(features, lengths, decodingConfig = defaultDecodingConfig()) {
        return decodeCTCGreedy(features, lengths, this.ctcDecoder, this.vocabulary, this.timeRatio);
    }
    generate(mel, decodingConfig = defaultDecodingConfig()) {
        if (mel.ndim() === 2)
            mel = mel.expandDims(0);
        const [features, lengths] = this.encoder.forward(mel, null, null);
        features.eval();
        lengths.eval();
        const result = this.decode(features, lengths, decodingConfig);
        return result.map(tokens => sentencesToResult(tokensToSentences(tokens, decodingConfig.sentence)));
    }
    loadWeights(weights, prefix = '') {
        super.loadWeights(weights, prefix);
        this.ctcDecoder.loadWeights(weights, prefix ? `${prefix}.decoder` : 'decoder');
    }
}
// ---------------------------------------------------------------------------
// ParakeetTDTCTC (TDT model with auxiliary CTC head)
// ---------------------------------------------------------------------------
export class ParakeetTDTCTC extends ParakeetTDT {
    ctcDecoder;
    constructor(args) {
        super(args);
        this.ctcDecoder = new ConvASRDecoder(args.auxCtc.decoder);
    }
    loadWeights(weights, prefix = '') {
        super.loadWeights(weights, prefix);
        this.ctcDecoder.loadWeights(weights, prefix ? `${prefix}.ctc_decoder` : 'ctc_decoder');
    }
}
// ---------------------------------------------------------------------------
// StreamingParakeet
// ---------------------------------------------------------------------------
export class StreamingParakeet {
    model;
    contextSize;
    depth;
    decodingConfig;
    keepOriginalAttention;
    cache;
    audioBuffer;
    melBuffer = null;
    decoderHidden = null;
    lastToken = null;
    _finalizedTokens = [];
    _draftTokens = [];
    _releaseMutex = null;
    constructor(model, contextSize, depth, decodingConfig, keepOriginalAttention) {
        this.model = model;
        this.contextSize = contextSize;
        this.depth = depth;
        this.decodingConfig = decodingConfig;
        this.keepOriginalAttention = keepOriginalAttention;
        this.cache = model.encoder.layers.map(() => new RotatingConformerCache(contextSize[0], contextSize[1] * depth));
        this.audioBuffer = new Float32Array(0);
    }
    get keepSize() {
        return this.contextSize[0];
    }
    get dropSize() {
        return this.contextSize[1] * this.depth;
    }
    async start() {
        this._releaseMutex = await this.model._acquireMutex();
        if (!this.keepOriginalAttention) {
            this.model.encoder.setAttentionModel('rel_pos_local_attn', this.contextSize);
        }
    }
    stop() {
        if (!this.keepOriginalAttention) {
            this.model.encoder.setAttentionModel('rel_pos');
        }
        if (this._releaseMutex) {
            this._releaseMutex();
            this._releaseMutex = null;
        }
    }
    /** Committed tokens — will not be revised on future addAudio calls. */
    get finalizedTokens() {
        return [...this._finalizedTokens];
    }
    /** Tokens in the rotating context window — may be revised on subsequent addAudio calls. */
    get draftTokens() {
        return [...this._draftTokens];
    }
    /** AlignedResult built from finalized tokens only — safe to persist incrementally. */
    get finalizedResult() {
        return sentencesToResult(tokensToSentences(this._finalizedTokens, this.decodingConfig.sentence));
    }
    /** Full transcript (finalized + draft) — best current guess. */
    get result() {
        return sentencesToResult(tokensToSentences([...this._finalizedTokens, ...this._draftTokens], this.decodingConfig.sentence));
    }
    addAudio(audio) {
        // Append to buffer
        const combined = new Float32Array(this.audioBuffer.length + audio.length);
        combined.set(this.audioBuffer);
        combined.set(audio, this.audioBuffer.length);
        this.audioBuffer = combined;
        const hopLen = this.model.preprocessorConfig.hopLength;
        const usableLen = Math.floor(this.audioBuffer.length / hopLen) * hopLen;
        // Not enough audio for a single STFT frame yet — keep buffering. Feeding an
        // empty signal to getLogMel would abort natively (uncatchable in JS).
        if (usableLen === 0)
            return;
        const usableAudio = MxArray.fromFloat32(this.audioBuffer.slice(0, usableLen), s(usableLen));
        const mel = getLogMel(usableAudio, this.model.preprocessorConfig);
        // mel: [1, frames, nMels]
        if (this.melBuffer === null) {
            this.melBuffer = mel;
        }
        else {
            this.melBuffer = MxArray.concatenate(this.melBuffer, mel, 1);
        }
        this.audioBuffer = this.audioBuffer.slice(usableLen);
        const subFactor = this.model.encoderConfig.subsamplingFactor;
        const melFrames = Number(this.melBuffer.shape()[1]);
        const usableMelFrames = Math.floor(melFrames / subFactor) * subFactor;
        // Fewer than one subsampled frame available — keep the mel frames buffered
        // for the next call. Running the encoder on a zero-length sequence aborts
        // natively (e.g. "[squeeze] Cannot squeeze axis 1 with size 0", "[max]
        // Cannot max reduce zero size array") and cannot be caught in JS.
        if (usableMelFrames === 0)
            return;
        const melInput = this.melBuffer.slice(s(0, 0, 0), s(1, usableMelFrames, Number(this.melBuffer.shape()[2])));
        const [features, lengths] = this.model.encoder.forward(melInput, null, this.cache);
        features.eval();
        lengths.eval();
        const length = Number(lengths.toInt32()[0]);
        const finalizedLength = Math.max(0, length - this.dropSize);
        // Trim mel buffer to keep only drop_size worth of subsampled frames
        const leftover = melFrames - usableMelFrames;
        const keepMel = this.dropSize * subFactor + leftover;
        this.melBuffer = this.melBuffer.slice(s(0, Math.max(0, melFrames - keepMel), 0), s(1, melFrames, Number(this.melBuffer.shape()[2])));
        if (this.model instanceof ParakeetTDT || this.model instanceof ParakeetRNNT) {
            const finLengths = MxArray.fromInt32(new Int32Array([finalizedLength]), s(1));
            const initState = { lastToken: this.lastToken, hiddenState: this.decoderHidden };
            const [finTokens, finStates] = this.model.decode(features, finLengths, [initState]);
            this.decoderHidden = finStates[0].hiddenState;
            this.lastToken = finTokens[0].length > 0 ? finTokens[0][finTokens[0].length - 1].id : this.lastToken;
            const draftInput = features.slice(s(0, finalizedLength, 0), s(1, length, Number(features.shape()[2])));
            const draftLengths = MxArray.fromInt32(new Int32Array([length - finalizedLength]), s(1));
            const draftState = { lastToken: this.lastToken, hiddenState: this.decoderHidden };
            const [draftTokens] = this.model.decode(draftInput, draftLengths, [draftState]);
            this._finalizedTokens.push(...finTokens[0]);
            this._draftTokens = draftTokens[0];
        }
        else if (this.model instanceof ParakeetCTC) {
            const finLengths = MxArray.fromInt32(new Int32Array([finalizedLength]), s(1));
            const finTokens = this.model.decode(features, finLengths);
            const draftInput = features.slice(s(0, finalizedLength, 0), s(1, length, Number(features.shape()[2])));
            const draftLengths = MxArray.fromInt32(new Int32Array([length - finalizedLength]), s(1));
            const draftTokens = this.model.decode(draftInput, draftLengths);
            this._finalizedTokens.push(...finTokens[0]);
            this._draftTokens = draftTokens[0];
        }
    }
}
// ---------------------------------------------------------------------------
// consumePcmStream helper
// ---------------------------------------------------------------------------
/**
 * Feed an async iterable of raw PCM frames (Float32Array, 16kHz mono) into a
 * StreamingParakeet session and return the final AlignedResult.
 *
 * This is the common "consume and wait" pattern used by both the CLI --stream
 * flag and the HTTP server's POST /transcribe endpoint.
 */
export async function consumePcmStream(stream, source) {
    await stream.start();
    try {
        for await (const chunk of source) {
            stream.addAudio(chunk);
        }
    }
    finally {
        stream.stop();
    }
    return stream.result;
}
//# sourceMappingURL=parakeet.js.map