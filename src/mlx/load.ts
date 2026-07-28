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

export interface MlxModelOptions {
  /**
   * 'interpolated' (default) matches NVIDIA's reference preprocessor.
   * 'floor' reproduces parakeet-mlx's filterbank — see docs/cuda.md.
   */
  filterbank?: 'floor' | 'interpolated';
}

/**
 * Load a safetensors Parakeet checkpoint and wrap it in `ParakeetModel`.
 *
 * @param dir directory holding `config.json` and `model.safetensors`
 */
export function fromLocal(dir: string, options: MlxModelOptions = {}): ParakeetModel {
  const configPath = path.join(dir, 'config.json');
  if (!fs.existsSync(configPath)) throw new Error(`config.json not found in ${dir}`);
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8')) as Record<string, unknown>;

  const legacy = loadMlxModel(dir);
  if (!(legacy instanceof ParakeetTDT) && !(legacy instanceof ParakeetRNNT)) {
    throw new Error(
      'Only TDT and RNN-T models are supported by the shared backend path; ' +
      'use src/mlx/utils.ts fromLocal for CTC models.',
    );
  }

  const pre = config['preprocessor'] as Record<string, unknown>;
  const preprocessor = makePreprocessArgs({
    sampleRate: pre['sample_rate'] as number,
    normalize: pre['normalize'] as string,
    windowSize: pre['window_size'] as number,
    windowStride: pre['window_stride'] as number,
    window: pre['window'] as string,
    features: pre['features'] as number,
    nFft: pre['n_fft'] as number,
    dither: pre['dither'] as number,
    padTo: (pre['pad_to'] as number) ?? 0,
    padValue: (pre['pad_value'] as number) ?? 0,
    preemph: pre['preemph'] as number | null,
    magPower: (pre['mag_power'] as number) ?? 2.0,
    filterbank: options.filterbank ?? 'interpolated',
  });

  const encoderCfg = config['encoder'] as Record<string, unknown>;
  const backend = new MlxBackend({
    encoder: legacy.encoder,
    predict: legacy.decoder,
    joint: legacy.joint,
    encoderDim: encoderCfg['d_model'] as number,
  });

  const durations = legacy instanceof ParakeetTDT ? legacy.durations : null;

  return new ParakeetModel({
    backend,
    preprocessor,
    vocabulary: legacy.vocabulary,
    durations,
    maxSymbols: legacy.maxSymbols,
    subsamplingFactor: encoderCfg['subsampling_factor'] as number,
  });
}
