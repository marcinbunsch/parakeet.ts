/**
 * Loading ONNX-exported Parakeet models.
 *
 * The model class itself is backend-agnostic (`ParakeetModel`); this file only
 * knows how to find the exported graphs and read the checkpoint metadata.
 */
import fs from 'node:fs';
import path from 'node:path';
import { OnnxBackend, OnnxBackendOptions, ExecutionProvider } from './backend.js';
import { ParakeetModel } from '../model.js';
import { makePreprocessArgs } from '../audio.js';

export interface OnnxModelOptions {
  executionProvider?: ExecutionProvider;
  /**
   * Execution provider for the autoregressive decode step. Defaults to the same
   * as `executionProvider`; 'cpu' is a reasonable alternative since the step is
   * small and CPU avoids per-step device round trips.
   */
  decoderExecutionProvider?: ExecutionProvider;
  /**
   * 'interpolated' (default) matches NVIDIA's reference preprocessor.
   * 'floor' reproduces parakeet-mlx's filterbank, which collapses 13 of 128 mel
   * bins to zero — see docs/cuda.md.
   */
  filterbank?: 'floor' | 'interpolated';
  logSeverityLevel?: number;
}

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

function readVocabulary(dir: string, config: Record<string, unknown>): string[] {
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
  const joint = config['joint'] as Record<string, unknown> | undefined;
  if (joint?.['vocabulary']) return joint['vocabulary'] as string[];
  throw new Error(
    `No vocabulary found: expected vocab.txt in ${dir} or joint.vocabulary in config.json`,
  );
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
export async function fromLocal(
  dir: string,
  options: OnnxModelOptions = {},
): Promise<ParakeetModel> {
  const pick = (...names: string[]): string => {
    for (const n of names) {
      const p = path.join(dir, n);
      if (fs.existsSync(p)) return p;
    }
    throw new Error(`None of ${names.join(', ')} found in ${dir}`);
  };

  const encoderPath = pick('encoder-model.onnx', 'encoder.onnx');
  const decoderJointPath = pick('decoder_joint-model.onnx', 'decoder_joint.onnx');

  const configPath = path.join(dir, 'config.json');
  const config: Record<string, unknown> = fs.existsSync(configPath)
    ? JSON.parse(fs.readFileSync(configPath, 'utf8'))
    : {};

  const pre = (config['preprocessor'] as Record<string, unknown> | undefined) ?? {};
  const preprocessor = makePreprocessArgs({
    sampleRate: (pre['sample_rate'] as number) ?? DEFAULTS.sampleRate,
    normalize: (pre['normalize'] as string) ?? DEFAULTS.normalize,
    windowSize: (pre['window_size'] as number) ?? DEFAULTS.windowSize,
    windowStride: (pre['window_stride'] as number) ?? DEFAULTS.windowStride,
    window: (pre['window'] as string) ?? DEFAULTS.window,
    features: (pre['features'] as number)
      ?? (config['features_size'] as number)
      ?? DEFAULTS.features,
    nFft: (pre['n_fft'] as number) ?? DEFAULTS.nFft,
    dither: (pre['dither'] as number) ?? DEFAULTS.dither,
    padTo: (pre['pad_to'] as number) ?? 0,
    padValue: (pre['pad_value'] as number) ?? 0,
    preemph: pre['preemph'] !== undefined ? (pre['preemph'] as number | null) : DEFAULTS.preemph,
    magPower: (pre['mag_power'] as number) ?? DEFAULTS.magPower,
    filterbank: options.filterbank ?? 'interpolated',
  });

  const vocabulary = readVocabulary(dir, config);

  const modelDefaults = (config['model_defaults'] as Record<string, unknown> | undefined) ?? {};
  const decoding = (config['decoding'] as Record<string, unknown> | undefined) ?? {};
  const modelType = (config['model_type'] as string) ?? (decoding['model_type'] as string) ?? 'tdt';
  const durations = (modelDefaults['tdt_durations'] as number[] | undefined)
    ?? (modelType.includes('tdt') ? DEFAULTS.durations : null);

  const greedy = decoding['greedy'] as Record<string, unknown> | undefined;
  const maxSymbols = greedy?.['max_symbols'] != null
    ? Number(greedy['max_symbols'])
    : DEFAULTS.maxSymbols;

  const subsamplingFactor = (config['subsampling_factor'] as number)
    ?? ((config['encoder'] as Record<string, unknown> | undefined)?.['subsampling_factor'] as number)
    ?? DEFAULTS.subsamplingFactor;

  const backendOpts: OnnxBackendOptions = {
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
