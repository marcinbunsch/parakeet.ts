/**
 * Loading ONNX-exported Parakeet models.
 *
 * The model class itself is backend-agnostic (`ParakeetModel`); this file only
 * knows how to find the exported graphs and read the checkpoint metadata.
 */
import fs from 'node:fs';
import path from 'node:path';
import { OnnxBackend } from './backend.js';
import { ParakeetModel } from '../model.js';
import { makePreprocessArgs } from '../audio.js';
const DEFAULTS = {
    sampleRate: 16000,
    normalize: 'per_feature',
    windowSize: 0.025,
    windowStride: 0.01,
    window: 'hann',
    features: 128,
    nFft: 512,
    dither: 0,
    preemph: 0.97,
    magPower: 2.0,
    subsamplingFactor: 8,
    durations: [0, 1, 2, 3, 4],
    maxSymbols: 10,
};
function readVocabulary(dir, config) {
    const vocabPath = path.join(dir, 'vocab.txt');
    if (fs.existsSync(vocabPath)) {
        // lines are "<token> <id>"; the final entry is the blank symbol, which we
        // drop so that `vocabulary.length` equals the blank id
        const lines = fs.readFileSync(vocabPath, 'utf8').split('\n').filter(l => l.length > 0);
        const tokens = lines.map(l => {
            const sp = l.lastIndexOf(' ');
            return sp === -1 ? l : l.slice(0, sp);
        });
        return tokens.slice(0, -1);
    }
    const joint = config['joint'];
    if (joint?.['vocabulary'])
        return joint['vocabulary'];
    throw new Error(`No vocabulary found: expected vocab.txt in ${dir} or joint.vocabulary in config.json`);
}
/**
 * Load an ONNX Parakeet model from a directory of exported graphs.
 *
 * Expected contents:
 *   encoder-model.onnx        (+ encoder-model.onnx.data for external weights)
 *   decoder_joint-model.onnx
 *   vocab.txt                 (or config.json with joint.vocabulary)
 *   config.json               (optional; a full NeMo config supplies preprocessor settings)
 */
export async function fromLocal(dir, options = {}) {
    const pick = (...names) => {
        for (const n of names) {
            const p = path.join(dir, n);
            if (fs.existsSync(p))
                return p;
        }
        throw new Error(`None of ${names.join(', ')} found in ${dir}`);
    };
    const encoderPath = pick('encoder-model.onnx', 'encoder.onnx');
    const decoderJointPath = pick('decoder_joint-model.onnx', 'decoder_joint.onnx');
    const configPath = path.join(dir, 'config.json');
    const config = fs.existsSync(configPath)
        ? JSON.parse(fs.readFileSync(configPath, 'utf8'))
        : {};
    const pre = config['preprocessor'] ?? {};
    const preprocessor = makePreprocessArgs({
        sampleRate: pre['sample_rate'] ?? DEFAULTS.sampleRate,
        normalize: pre['normalize'] ?? DEFAULTS.normalize,
        windowSize: pre['window_size'] ?? DEFAULTS.windowSize,
        windowStride: pre['window_stride'] ?? DEFAULTS.windowStride,
        window: pre['window'] ?? DEFAULTS.window,
        features: pre['features']
            ?? config['features_size']
            ?? DEFAULTS.features,
        nFft: pre['n_fft'] ?? DEFAULTS.nFft,
        dither: pre['dither'] ?? DEFAULTS.dither,
        padTo: pre['pad_to'] ?? 0,
        padValue: pre['pad_value'] ?? 0,
        preemph: pre['preemph'] !== undefined ? pre['preemph'] : DEFAULTS.preemph,
        magPower: pre['mag_power'] ?? DEFAULTS.magPower,
        filterbank: options.filterbank ?? 'interpolated',
    });
    const vocabulary = readVocabulary(dir, config);
    const modelDefaults = config['model_defaults'] ?? {};
    const decoding = config['decoding'] ?? {};
    const modelType = config['model_type'] ?? decoding['model_type'] ?? 'tdt';
    const durations = modelDefaults['tdt_durations']
        ?? (modelType.includes('tdt') ? DEFAULTS.durations : null);
    const greedy = decoding['greedy'];
    const maxSymbols = greedy?.['max_symbols'] != null
        ? Number(greedy['max_symbols'])
        : DEFAULTS.maxSymbols;
    const subsamplingFactor = config['subsampling_factor']
        ?? config['encoder']?.['subsampling_factor']
        ?? DEFAULTS.subsamplingFactor;
    const backendOpts = {
        encoderPath,
        decoderJointPath,
        executionProvider: options.executionProvider,
        decoderExecutionProvider: options.decoderExecutionProvider,
        logSeverityLevel: options.logSeverityLevel,
    };
    const backend = await OnnxBackend.create(backendOpts);
    backend.blankId = vocabulary.length;
    return new ParakeetModel({
        backend, preprocessor, vocabulary, durations, maxSymbols, subsamplingFactor,
    });
}
//# sourceMappingURL=parakeet.js.map