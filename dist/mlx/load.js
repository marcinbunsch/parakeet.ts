/**
 * Loading MLX-weight Parakeet checkpoints into the backend-agnostic
 * `ParakeetModel`.
 *
 * This is the single MLX entry point: it builds the Conformer encoder, the
 * prediction network and the joint network directly from `config.json`, loads
 * the safetensors weights into them, and wraps them behind `MlxBackend` so the
 * MLX path drives the same decode loop, tokenizer, alignment and audio
 * front-end as the ONNX path. MLX and ONNX therefore produce identical output
 * from identical features.
 *
 * TDT and RNN-T checkpoints are supported (this is also all the ONNX export
 * covers). CTC / hybrid TDT-CTC checkpoints are not.
 */
import fs from 'node:fs';
import path from 'node:path';
import { Conformer } from './conformer.js';
import { PredictNetwork, JointNetwork } from './rnnt.js';
import { MlxBackend } from './backend.js';
import { loadSafetensors, downloadFromHub } from './utils.js';
import { ParakeetModel } from '../model.js';
import { makePreprocessArgs } from '../audio.js';
/** Build the MLX modules and metadata from a parsed NeMo-style config. */
function buildFromConfig(config) {
    const modelDefaults = config['model_defaults'] ?? {};
    const encoderRaw = config['encoder'];
    const decoderRaw = config['decoder'];
    const jointRaw = config['joint'];
    const decodingRaw = config['decoding'] ?? {};
    if (!jointRaw || !decoderRaw?.['prednet']) {
        throw new Error('Only TDT and RNN-T checkpoints are supported by the MLX loader ' +
            '(config needs decoder.prednet and a joint network). ' +
            'CTC / hybrid TDT-CTC checkpoints are not supported.');
    }
    const attContextSize = encoderRaw['att_context_size'] ?? null;
    const encoderArgs = {
        featIn: encoderRaw['feat_in'],
        nLayers: encoderRaw['n_layers'],
        dModel: encoderRaw['d_model'],
        nHeads: encoderRaw['n_heads'],
        ffExpansionFactor: encoderRaw['ff_expansion_factor'],
        subsamplingFactor: encoderRaw['subsampling_factor'],
        selfAttentionModel: encoderRaw['self_attention_model'],
        subsampling: encoderRaw['subsampling'],
        convKernelSize: encoderRaw['conv_kernel_size'],
        subsamplingConvChannels: encoderRaw['subsampling_conv_channels'],
        posEmbMaxLen: encoderRaw['pos_emb_max_len'],
        causalDownsampling: encoderRaw['causal_downsampling'] ?? false,
        useBias: encoderRaw['use_bias'] ?? true,
        xscaling: encoderRaw['xscaling'] ?? false,
        subsamplingConvChunkingFactor: encoderRaw['subsampling_conv_chunking_factor'] ?? 1,
        attContextSize,
    };
    const prednetRaw = decoderRaw['prednet'] ?? {};
    const predictArgs = {
        blankAsPad: decoderRaw['blank_as_pad'],
        vocabSize: decoderRaw['vocab_size'],
        prednet: {
            predHidden: prednetRaw['pred_hidden'],
            predRnnLayers: prednetRaw['pred_rnn_layers'],
            rnnHiddenSize: prednetRaw['rnn_hidden_size'],
        },
    };
    const jointnetRaw = jointRaw['jointnet'] ?? {};
    const jointArgs = {
        numClasses: jointRaw['num_classes'],
        vocabulary: jointRaw['vocabulary'],
        jointnet: {
            jointHidden: jointnetRaw['joint_hidden'],
            activation: jointnetRaw['activation'],
            encoderHidden: jointnetRaw['encoder_hidden'],
            predHidden: jointnetRaw['pred_hidden'],
        },
        numExtraOutputs: jointRaw['num_extra_outputs'] ?? 0,
    };
    const durations = modelDefaults['tdt_durations'] ?? null;
    const greedy = decodingRaw['greedy'];
    const maxSymbols = greedy?.['max_symbols'] != null ? Number(greedy['max_symbols']) : null;
    return {
        encoder: new Conformer(encoderArgs),
        predict: new PredictNetwork(predictArgs),
        joint: new JointNetwork(jointArgs),
        vocabulary: jointArgs.vocabulary,
        durations,
        maxSymbols,
        encoderDim: encoderArgs.dModel,
        subsamplingFactor: encoderArgs.subsamplingFactor,
    };
}
/** Assemble a `ParakeetModel` from a config, a weights file, and options. */
function assemble(config, weightsPath, options) {
    const built = buildFromConfig(config);
    const weights = loadSafetensors(weightsPath);
    built.encoder.loadWeights(weights, 'encoder');
    built.predict.loadWeights(weights, 'decoder');
    built.joint.loadWeights(weights, 'joint');
    const pre = config['preprocessor'];
    const preprocessor = makePreprocessArgs({
        sampleRate: pre['sample_rate'],
        normalize: pre['normalize'],
        windowSize: pre['window_size'],
        windowStride: pre['window_stride'],
        window: pre['window'],
        features: pre['features'],
        nFft: pre['n_fft'],
        dither: pre['dither'],
        padTo: pre['pad_to'] ?? 0,
        padValue: pre['pad_value'] ?? 0,
        preemph: pre['preemph'],
        magPower: pre['mag_power'] ?? 2.0,
        filterbank: options.filterbank ?? 'interpolated',
    });
    const backend = new MlxBackend({
        encoder: built.encoder,
        predict: built.predict,
        joint: built.joint,
        encoderDim: built.encoderDim,
    });
    return new ParakeetModel({
        backend,
        preprocessor,
        vocabulary: built.vocabulary,
        durations: built.durations,
        maxSymbols: built.maxSymbols,
        subsamplingFactor: built.subsamplingFactor,
    });
}
/**
 * Load a safetensors Parakeet checkpoint from a local directory.
 *
 * @param dir directory holding `config.json` and `model.safetensors`
 */
export function fromLocal(dir, options = {}) {
    const configPath = path.join(dir, 'config.json');
    const weightsPath = path.join(dir, 'model.safetensors');
    if (!fs.existsSync(configPath))
        throw new Error(`config.json not found in ${dir}`);
    if (!fs.existsSync(weightsPath))
        throw new Error(`model.safetensors not found in ${dir}`);
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    return assemble(config, weightsPath, options);
}
/**
 * Load a Parakeet checkpoint from a HuggingFace Hub repo or a local directory.
 *
 * @param hfIdOrPath HuggingFace repo id (e.g. "mlx-community/parakeet-tdt-0.6b-v3")
 *                   or a local directory path.
 */
export async function fromPretrained(hfIdOrPath, options = {}) {
    if (fs.existsSync(hfIdOrPath) && fs.statSync(hfIdOrPath).isDirectory()) {
        return fromLocal(hfIdOrPath, options);
    }
    const makeProgress = (file) => options.onProgress
        ? (downloaded, total) => options.onProgress(file, downloaded, total)
        : undefined;
    const configPath = await downloadFromHub(hfIdOrPath, 'config.json', options.cacheDir, makeProgress('config.json'));
    const weightsPath = await downloadFromHub(hfIdOrPath, 'model.safetensors', options.cacheDir, makeProgress('model.safetensors'));
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    return assemble(config, weightsPath, options);
}
//# sourceMappingURL=load.js.map