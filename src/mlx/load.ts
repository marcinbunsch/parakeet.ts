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
import { Conformer, ConformerArgs } from './conformer.js';
import { PredictNetwork, JointNetwork, PredictArgs, JointArgs } from './rnnt.js';
import { MlxBackend } from './backend.js';
import { loadSafetensors, downloadFromHub } from './utils.js';
import { ParakeetModel } from '../model.js';
import { makePreprocessArgs } from '../audio.js';

export interface MlxModelOptions {
  /**
   * 'interpolated' (default) matches NVIDIA's reference preprocessor.
   * 'floor' reproduces parakeet-mlx's filterbank, which collapses 13 of 128 mel
   * bins to zero — see docs/cuda.md.
   */
  filterbank?: 'floor' | 'interpolated';
}

interface BuiltModel {
  encoder: Conformer;
  predict: PredictNetwork;
  joint: JointNetwork;
  vocabulary: string[];
  durations: number[] | null;
  maxSymbols: number | null;
  encoderDim: number;
  subsamplingFactor: number;
}

/** Build the MLX modules and metadata from a parsed NeMo-style config. */
function buildFromConfig(config: Record<string, unknown>): BuiltModel {
  const modelDefaults = (config['model_defaults'] as Record<string, unknown>) ?? {};
  const encoderRaw = config['encoder'] as Record<string, unknown>;
  const decoderRaw = config['decoder'] as Record<string, unknown>;
  const jointRaw = config['joint'] as Record<string, unknown> | undefined;
  const decodingRaw = (config['decoding'] as Record<string, unknown>) ?? {};

  if (!jointRaw || !decoderRaw?.['prednet']) {
    throw new Error(
      'Only TDT and RNN-T checkpoints are supported by the MLX loader ' +
      '(config needs decoder.prednet and a joint network). ' +
      'CTC / hybrid TDT-CTC checkpoints are not supported.',
    );
  }

  const attContextSize = (encoderRaw['att_context_size'] as [number, number] | null) ?? null;
  const encoderArgs: ConformerArgs = {
    featIn: encoderRaw['feat_in'] as number,
    nLayers: encoderRaw['n_layers'] as number,
    dModel: encoderRaw['d_model'] as number,
    nHeads: encoderRaw['n_heads'] as number,
    ffExpansionFactor: encoderRaw['ff_expansion_factor'] as number,
    subsamplingFactor: encoderRaw['subsampling_factor'] as number,
    selfAttentionModel: encoderRaw['self_attention_model'] as string,
    subsampling: encoderRaw['subsampling'] as string,
    convKernelSize: encoderRaw['conv_kernel_size'] as number,
    subsamplingConvChannels: encoderRaw['subsampling_conv_channels'] as number,
    posEmbMaxLen: encoderRaw['pos_emb_max_len'] as number,
    causalDownsampling: (encoderRaw['causal_downsampling'] as boolean) ?? false,
    useBias: (encoderRaw['use_bias'] as boolean) ?? true,
    xscaling: (encoderRaw['xscaling'] as boolean) ?? false,
    subsamplingConvChunkingFactor: (encoderRaw['subsampling_conv_chunking_factor'] as number) ?? 1,
    attContextSize,
  };

  const prednetRaw = (decoderRaw['prednet'] as Record<string, unknown>) ?? {};
  const predictArgs: PredictArgs = {
    blankAsPad: decoderRaw['blank_as_pad'] as boolean,
    vocabSize: decoderRaw['vocab_size'] as number,
    prednet: {
      predHidden: prednetRaw['pred_hidden'] as number,
      predRnnLayers: prednetRaw['pred_rnn_layers'] as number,
      rnnHiddenSize: prednetRaw['rnn_hidden_size'] as number | undefined,
    },
  };

  const jointnetRaw = (jointRaw['jointnet'] as Record<string, unknown>) ?? {};
  const jointArgs: JointArgs = {
    numClasses: jointRaw['num_classes'] as number,
    vocabulary: jointRaw['vocabulary'] as string[],
    jointnet: {
      jointHidden: jointnetRaw['joint_hidden'] as number,
      activation: jointnetRaw['activation'] as string,
      encoderHidden: jointnetRaw['encoder_hidden'] as number,
      predHidden: jointnetRaw['pred_hidden'] as number,
    },
    numExtraOutputs: (jointRaw['num_extra_outputs'] as number) ?? 0,
  };

  const durations = (modelDefaults['tdt_durations'] as number[] | undefined) ?? null;
  const greedy = decodingRaw['greedy'] as Record<string, unknown> | undefined;
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
function assemble(
  config: Record<string, unknown>,
  weightsPath: string,
  options: MlxModelOptions,
): ParakeetModel {
  const built = buildFromConfig(config);

  const weights = loadSafetensors(weightsPath);
  built.encoder.loadWeights(weights, 'encoder');
  built.predict.loadWeights(weights, 'decoder');
  built.joint.loadWeights(weights, 'joint');

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
export function fromLocal(dir: string, options: MlxModelOptions = {}): ParakeetModel {
  const configPath = path.join(dir, 'config.json');
  const weightsPath = path.join(dir, 'model.safetensors');
  if (!fs.existsSync(configPath)) throw new Error(`config.json not found in ${dir}`);
  if (!fs.existsSync(weightsPath)) throw new Error(`model.safetensors not found in ${dir}`);

  const config = JSON.parse(fs.readFileSync(configPath, 'utf8')) as Record<string, unknown>;
  return assemble(config, weightsPath, options);
}

export interface FromPretrainedOptions extends MlxModelOptions {
  cacheDir?: string;
  onProgress?: (file: string, downloaded: number, total: number) => void;
}

/**
 * Load a Parakeet checkpoint from a HuggingFace Hub repo or a local directory.
 *
 * @param hfIdOrPath HuggingFace repo id (e.g. "mlx-community/parakeet-tdt-0.6b-v3")
 *                   or a local directory path.
 */
export async function fromPretrained(
  hfIdOrPath: string,
  options: FromPretrainedOptions = {},
): Promise<ParakeetModel> {
  if (fs.existsSync(hfIdOrPath) && fs.statSync(hfIdOrPath).isDirectory()) {
    return fromLocal(hfIdOrPath, options);
  }

  const makeProgress = (file: string) =>
    options.onProgress
      ? (downloaded: number, total: number) => options.onProgress!(file, downloaded, total)
      : undefined;

  const configPath = await downloadFromHub(hfIdOrPath, 'config.json', options.cacheDir, makeProgress('config.json'));
  const weightsPath = await downloadFromHub(hfIdOrPath, 'model.safetensors', options.cacheDir, makeProgress('model.safetensors'));

  const config = JSON.parse(fs.readFileSync(configPath, 'utf8')) as Record<string, unknown>;
  return assemble(config, weightsPath, options);
}
