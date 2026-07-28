import { getLogMel, loadAudioRaw } from './audio.js';
import { decodeTDTGreedy, decodeRNNTGreedy } from './decode.js';
import { tokensToSentences, sentencesToResult, mergeLongestContiguous, mergeLongestCommonSubsequence, } from './alignment.js';
export class ParakeetModel {
    backend;
    preprocessorConfig;
    vocabulary;
    durations;
    maxSymbols;
    subsamplingFactor;
    constructor(args) {
        this.backend = args.backend;
        this.preprocessorConfig = args.preprocessor;
        this.vocabulary = args.vocabulary;
        this.durations = args.durations;
        this.maxSymbols = args.maxSymbols;
        this.subsamplingFactor = args.subsamplingFactor;
    }
    /** Seconds of audio per encoder frame. */
    get timeRatio() {
        return (this.subsamplingFactor / this.preprocessorConfig.sampleRate)
            * this.preprocessorConfig.hopLength;
    }
    /** Run the shared mel front-end and the backend encoder. */
    async encodePcm(pcm) {
        const { data, nMels, numFrames } = getLogMel(pcm, this.preprocessorConfig);
        return this.backend.encode(data, nMels, numFrames);
    }
    /** Greedy-decode an encoder output, threading decoder state. */
    async decode(enc, state = { lastToken: null, hiddenState: null }, range) {
        const opts = {
            vocabulary: this.vocabulary,
            maxSymbols: this.maxSymbols,
            timeRatio: this.timeRatio,
            ...range,
        };
        return this.durations
            ? decodeTDTGreedy(this.backend, enc, this.durations, state, opts)
            : decodeRNNTGreedy(this.backend, enc, state, opts);
    }
    async transcribe(file, options = {}) {
        const { chunkDuration, overlapDuration = 15.0, onChunk, sentence } = options;
        const sr = this.preprocessorConfig.sampleRate;
        const pcm = loadAudioRaw(file, sr);
        return this.transcribePcm(pcm, options);
    }
    /** Transcribe raw mono float32 PCM at the model's sample rate. */
    async transcribePcm(pcm, options = {}) {
        const { chunkDuration, overlapDuration = 15.0, onChunk, sentence } = options;
        const sr = this.preprocessorConfig.sampleRate;
        if (chunkDuration === undefined || pcm.length / sr <= chunkDuration) {
            const enc = await this.encodePcm(pcm);
            const { tokens } = await this.decode(enc);
            return sentencesToResult(tokensToSentences(tokens, sentence));
        }
        const chunkSamples = Math.floor(chunkDuration * sr);
        const overlapSamples = Math.floor(overlapDuration * sr);
        let allTokens = [];
        for (let start = 0; start < pcm.length; start += chunkSamples - overlapSamples) {
            const end = Math.min(start + chunkSamples, pcm.length);
            if (onChunk)
                onChunk(end, pcm.length);
            if (end - start < this.preprocessorConfig.hopLength)
                break;
            const enc = await this.encodePcm(pcm.subarray(start, end));
            const { tokens } = await this.decode(enc, { lastToken: null, hiddenState: null }, { timeOffset: start / sr });
            if (allTokens.length > 0) {
                try {
                    allTokens = mergeLongestContiguous(allTokens, tokens, overlapDuration);
                }
                catch {
                    allTokens = mergeLongestCommonSubsequence(allTokens, tokens, overlapDuration);
                }
            }
            else {
                allTokens = tokens;
            }
        }
        return sentencesToResult(tokensToSentences(allTokens, sentence));
    }
    transcribeStream(options = {}) {
        return new StreamingParakeet(this, options);
    }
    async dispose() {
        await this.backend.dispose?.();
    }
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
export class StreamingParakeet {
    model;
    windowSamples;
    dropFrames;
    sentence;
    audio = new Float32Array(0);
    /** Absolute encoder-frame index corresponding to `audio[0]`. */
    windowStartFrame = 0;
    /** Absolute encoder frame already committed. */
    finalizedUpTo = 0;
    state = { lastToken: null, hiddenState: null };
    _finalized = [];
    _draft = [];
    constructor(model, options = {}) {
        this.model = model;
        const sr = model.preprocessorConfig.sampleRate;
        this.windowSamples = Math.floor((options.windowSeconds ?? 12.0) * sr);
        this.dropFrames = options.dropFrames ?? 12;
        this.sentence = options.sentence;
    }
    get samplesPerFrame() {
        return this.model.preprocessorConfig.hopLength * this.model.subsamplingFactor;
    }
    /** Committed tokens — will not be revised. */
    get finalizedTokens() { return [...this._finalized]; }
    /** Tokens in the revisable tail. */
    get draftTokens() { return [...this._draft]; }
    get finalizedResult() {
        return sentencesToResult(tokensToSentences(this._finalized, this.sentence));
    }
    /** Best current guess: committed + draft. */
    get result() {
        return sentencesToResult(tokensToSentences([...this._finalized, ...this._draft], this.sentence));
    }
    async addAudio(pcm) {
        const combined = new Float32Array(this.audio.length + pcm.length);
        combined.set(this.audio);
        combined.set(pcm, this.audio.length);
        this.audio = combined;
        // Not enough audio yet for a single subsampled encoder frame. Running the
        // front-end / encoder on a sub-frame window aborts natively (e.g. "[squeeze]
        // Cannot squeeze axis 1 with size 0") and cannot be caught in JS — keep
        // buffering until a later call has enough. Harmless for normal chunk sizes.
        const minSamples = this.model.preprocessorConfig.nFft + this.samplesPerFrame;
        if (this.audio.length < minSamples)
            return;
        // Trim in whole encoder frames so absolute frame accounting stays exact.
        const spf = this.samplesPerFrame;
        if (this.audio.length > this.windowSamples) {
            const dropSamples = Math.floor((this.audio.length - this.windowSamples) / spf) * spf;
            if (dropSamples > 0) {
                this.audio = this.audio.slice(dropSamples);
                this.windowStartFrame += dropSamples / spf;
            }
        }
        const enc = await this.model.encodePcm(this.audio);
        const absEnd = this.windowStartFrame + enc.frames;
        const commitUntil = Math.max(this.finalizedUpTo, absEnd - this.dropFrames);
        const timeOffset = this.windowStartFrame * this.model.timeRatio;
        if (commitUntil > this.finalizedUpTo) {
            const { tokens, state } = await this.model.decode(enc, this.state, {
                from: this.finalizedUpTo - this.windowStartFrame,
                to: commitUntil - this.windowStartFrame,
                timeOffset,
            });
            this._finalized.push(...tokens);
            this.state = state;
            this.finalizedUpTo = commitUntil;
        }
        // Speculative tail; its state is intentionally discarded.
        const { tokens: draft } = await this.model.decode(enc, this.state, {
            from: commitUntil - this.windowStartFrame,
            to: enc.frames,
            timeOffset,
        });
        this._draft = draft;
    }
    /** Commit the remaining draft. Call once the audio stream ends. */
    finish() {
        this._finalized.push(...this._draft);
        this._draft = [];
        return this.finalizedResult;
    }
}
/**
 * Feed an async iterable of raw PCM frames (Float32Array, mono at the model's
 * sample rate) into a streaming session and return the final AlignedResult.
 *
 * The common "consume and wait" pattern behind the CLI `--stream` flag and the
 * HTTP server's transcribe endpoint.
 */
export async function consumePcmStream(stream, source) {
    for await (const chunk of source) {
        await stream.addAudio(chunk);
    }
    return stream.finish();
}
//# sourceMappingURL=model.js.map