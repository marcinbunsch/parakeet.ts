/**
 * Loading MLX-weight Parakeet models into the shared, backend-agnostic
 * `ParakeetModel`.
 *
 * This sits alongside the original `src/mlx/utils.ts` loaders, which return the
 * legacy `ParakeetTDT` / `ParakeetRNNT` classes. Those still work; this path
 * gives you the same model driving the shared decode loop, so MLX and ONNX
 * produce identical output from identical features.
 */
import fs from 'node:fs';
import path from 'node:path';
import { MlxBackend } from './backend.js';
import { ParakeetModel } from '../model.js';
import { makePreprocessArgs } from '../audio.js';
import { fromLocal as loadMlxModel } from './utils.js';
import { ParakeetTDT, ParakeetRNNT } from './parakeet.js';
/**
 * Load a safetensors Parakeet checkpoint and wrap it in `ParakeetModel`.
 *
 * @param dir directory holding `config.json` and `model.safetensors`
 */
export function fromLocal(dir, options = {}) {
    const configPath = path.join(dir, 'config.json');
    if (!fs.existsSync(configPath))
        throw new Error(`config.json not found in ${dir}`);
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    const legacy = loadMlxModel(dir);
    if (!(legacy instanceof ParakeetTDT) && !(legacy instanceof ParakeetRNNT)) {
        throw new Error('Only TDT and RNN-T models are supported by the shared backend path; ' +
            'use src/mlx/utils.ts fromLocal for CTC models.');
    }
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
    const encoderCfg = config['encoder'];
    const backend = new MlxBackend({
        encoder: legacy.encoder,
        predict: legacy.decoder,
        joint: legacy.joint,
        encoderDim: encoderCfg['d_model'],
    });
    const durations = legacy instanceof ParakeetTDT ? legacy.durations : null;
    return new ParakeetModel({
        backend,
        preprocessor,
        vocabulary: legacy.vocabulary,
        durations,
        maxSymbols: legacy.maxSymbols,
        subsamplingFactor: encoderCfg['subsampling_factor'],
    });
}
//# sourceMappingURL=load.js.map