let ortModule = null;
async function loadOrt() {
    if (ortModule)
        return ortModule;
    try {
        const mod = await import('onnxruntime-node');
        ortModule = (mod.default ?? mod);
        return ortModule;
    }
    catch (err) {
        throw new Error('The ONNX backend requires the optional dependency "onnxruntime-node". ' +
            'Install it with `npm install onnxruntime-node`. ' +
            `Original error: ${err.message}`);
    }
}
export function defaultExecutionProvider() {
    if (process.platform === 'darwin')
        return 'cpu';
    if (process.arch === 'x64' && (process.platform === 'linux' || process.platform === 'win32')) {
        return 'cuda';
    }
    return 'cpu';
}
export class OnnxBackend {
    name;
    encoderDim;
    ort;
    encoder;
    decoderJoint;
    predRnnLayers;
    predHidden;
    zeroState;
    /** Reused scratch for the single-element int32 inputs. */
    targetLength;
    constructor(ort, encoder, decoderJoint, encoderDim, predRnnLayers, predHidden, name) {
        this.ort = ort;
        this.encoder = encoder;
        this.decoderJoint = decoderJoint;
        this.encoderDim = encoderDim;
        this.predRnnLayers = predRnnLayers;
        this.predHidden = predHidden;
        this.name = name;
        const size = predRnnLayers * predHidden;
        this.zeroState = {
            s1: new ort.Tensor('float32', new Float32Array(size), [predRnnLayers, 1, predHidden]),
            s2: new ort.Tensor('float32', new Float32Array(size), [predRnnLayers, 1, predHidden]),
        };
        this.targetLength = new ort.Tensor('int32', Int32Array.from([1]), [1]);
    }
    static async create(opts) {
        const ort = await loadOrt();
        const ep = opts.executionProvider ?? defaultExecutionProvider();
        const decEp = opts.decoderExecutionProvider ?? ep;
        const sev = opts.logSeverityLevel ?? 3;
        const encoder = await ort.InferenceSession.create(opts.encoderPath, {
            executionProviders: [ep], logSeverityLevel: sev,
        });
        const decoderJoint = await ort.InferenceSession.create(opts.decoderJointPath, {
            executionProviders: [decEp], logSeverityLevel: sev,
        });
        // encoderDim is discovered on first encode; probe it cheaply from the
        // decoder_joint graph instead by running a 1-frame step is overkill, so we
        // take it from options-free defaults and correct it after the first encode.
        const backend = new OnnxBackend(ort, encoder, decoderJoint, 0, opts.predRnnLayers ?? 2, opts.predHidden ?? 640, decEp === ep ? `onnx-${ep}` : `onnx-${ep}/${decEp}`);
        return backend;
    }
    async encode(mel, nMels, numFrames) {
        const feeds = {
            audio_signal: new this.ort.Tensor('float32', mel, [1, nMels, numFrames]),
            length: new this.ort.Tensor('int64', BigInt64Array.from([BigInt(numFrames)]), [1]),
        };
        const out = await this.encoder.run(feeds);
        const encoded = out['outputs'];
        const frames = Number(out['encoded_lengths'].data[0]);
        const dim = encoded.dims[1];
        // `encoderDim` is readonly to callers but discovered here on first use.
        this.encoderDim = dim;
        return {
            data: encoded.data,
            dim,
            frames,
            stride: encoded.dims[2],
            layout: 'dim-major',
        };
    }
    async decodeStep(encFrame, token, state) {
        const st = state ?? this.zeroState;
        // The blank id doubles as the prediction network's padding_idx, whose
        // embedding row is zero — this is the "start of sequence" input.
        const tokenId = token ?? this.blankId;
        const feeds = {
            encoder_outputs: new this.ort.Tensor('float32', encFrame, [1, encFrame.length, 1]),
            targets: new this.ort.Tensor('int32', Int32Array.from([tokenId]), [1, 1]),
            target_length: this.targetLength,
            input_states_1: st.s1,
            input_states_2: st.s2,
        };
        const r = await this.decoderJoint.run(feeds);
        return {
            logits: r['outputs'].data,
            state: { s1: r['output_states_1'], s2: r['output_states_2'] },
        };
    }
    /** Blank/padding token id, set by the model wrapper (vocabulary length). */
    blankId = 0;
    async dispose() {
        await this.encoder.release?.();
        await this.decoderJoint.release?.();
    }
}
//# sourceMappingURL=backend.js.map