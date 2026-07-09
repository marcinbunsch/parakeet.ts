/**
 * Model loading utilities.
 *
 * Supports:
 *  - Local directories (e.g. "./models/parakeet-tdt-0.6b-v3")
 *  - HuggingFace Hub repos (e.g. "mlx-community/parakeet-tdt-0.6b-v3")
 *
 * Weight files are SafeTensors format (model.safetensors).
 * Configuration is read from config.json which follows the NeMo format used
 * by the original parakeet-mlx Python project.
 */

import fs from 'node:fs';
import path from 'node:path';
import type { WeightMap } from './nn.js';
import {
  BaseParakeet,
  ParakeetTDT,
  ParakeetTDTArgs,
  ParakeetTDTCTC,
  ParakeetTDTCTCArgs,
  ParakeetRNNT,
  ParakeetRNNTArgs,
  ParakeetCTC,
  ParakeetCTCArgs,
} from './parakeet.js';
import { makePreprocessArgs } from './audio.js';

// ---------------------------------------------------------------------------
// SafeTensors loading (simple header + data parser)
// ---------------------------------------------------------------------------

interface SafeTensorHeader {
  [name: string]: {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
  };
}

/**
 * Parse a safetensors file and return a WeightMap.
 * The safetensors format: 8 bytes (header length LE uint64) + JSON header + data.
 * Uses fd-based random access to support files larger than the 2 GiB Buffer limit.
 */
function loadSafetensors(filePath: string): WeightMap {
  const fd = fs.openSync(filePath, 'r');

  try {
    // Read the 8-byte header length
    const lenBuf = Buffer.allocUnsafe(8);
    fs.readSync(fd, lenBuf, 0, 8, 0);
    const headerLen = Number(new DataView(lenBuf.buffer, lenBuf.byteOffset, 8).getBigUint64(0, true));

    // Read the JSON header
    const headerBuf = Buffer.allocUnsafe(headerLen);
    fs.readSync(fd, headerBuf, 0, headerLen, 8);
    const header: SafeTensorHeader = JSON.parse(headerBuf.toString('utf8'));

    const dataStart = 8 + headerLen;
    const map: WeightMap = new Map();

    for (const [name, meta] of Object.entries(header)) {
      if (name === '__metadata__') continue;

      const [start, end] = meta.data_offsets;
      const byteLen = end - start;
      const rawSlice = Buffer.allocUnsafe(byteLen);
      fs.readSync(fd, rawSlice, 0, byteLen, dataStart + start);

      const shape = meta.shape;
      let data: Float32Array;

      switch (meta.dtype) {
        case 'F32':
          data = new Float32Array(rawSlice.buffer, rawSlice.byteOffset, byteLen / 4);
          break;
        case 'BF16':
          data = bf16ToF32(rawSlice);
          break;
        case 'F16':
          data = f16ToF32(rawSlice);
          break;
        case 'I32':
          data = new Float32Array(new Int32Array(rawSlice.buffer, rawSlice.byteOffset, byteLen / 4));
          break;
        default:
          continue;
      }

      map.set(name, { data: new Float32Array(data), shape });
    }

    return map;
  } finally {
    fs.closeSync(fd);
  }
}

function bf16ToF32(buf: Buffer): Float32Array {
  const n = buf.byteLength / 2;
  const out = new Float32Array(n);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let i = 0; i < n; i++) {
    const bits = view.getUint16(i * 2, true);
    // BF16: sign[1] + exponent[8] + mantissa[7] → F32: shift mantissa left by 16
    const f32bits = bits << 16;
    const f32view = new DataView(new ArrayBuffer(4));
    f32view.setInt32(0, f32bits, true);
    out[i] = f32view.getFloat32(0, true);
  }
  return out;
}

function f16ToF32(buf: Buffer): Float32Array {
  const n = buf.byteLength / 2;
  const out = new Float32Array(n);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let i = 0; i < n; i++) {
    const h = view.getUint16(i * 2, true);
    const sign = (h >> 15) & 1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;

    if (exp === 0) {
      out[i] = sign ? -0 : 0;
    } else if (exp === 0x1f) {
      out[i] = mant ? NaN : sign ? -Infinity : Infinity;
    } else {
      const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
      out[i] = sign ? -f : f;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// HuggingFace Hub download
// ---------------------------------------------------------------------------

async function downloadFromHub(
  repoId: string,
  filename: string,
  cacheDir?: string,
  onProgress?: (downloaded: number, total: number) => void,
): Promise<string> {
  const effectiveCacheDir = cacheDir ?? path.join(
    process.env['HOME'] ?? '/tmp',
    '.cache',
    'huggingface',
    'hub',
  );

  const modelDir = path.join(effectiveCacheDir, repoId.replace('/', '--'));
  fs.mkdirSync(modelDir, { recursive: true });

  const localPath = path.join(modelDir, filename);
  if (fs.existsSync(localPath)) {
    return localPath;
  }

  // Download directly from HuggingFace CDN (Node 18+ has built-in fetch)
  const url = `https://huggingface.co/${repoId}/resolve/main/${filename}`;
  const response = await fetch(url, {
    headers: { 'User-Agent': 'parakeet-mlx/1.0.0' },
  });

  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }

  const total = parseInt(response.headers.get('content-length') ?? '0', 10);

  if (onProgress && response.body) {
    // Stream with progress reporting
    const tmpPath = `${localPath}.tmp`;
    const writeStream = fs.createWriteStream(tmpPath);
    let downloaded = 0;

    const reader = response.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        writeStream.write(value);
        downloaded += value.byteLength;
        onProgress(downloaded, total);
      }
    } finally {
      reader.releaseLock();
    }

    await new Promise<void>((resolve, reject) => {
      writeStream.end((err?: Error | null) => (err ? reject(err) : resolve()));
    });
    fs.renameSync(tmpPath, localPath);
  } else {
    const arrayBuffer = await response.arrayBuffer();
    fs.writeFileSync(localPath, Buffer.from(arrayBuffer));
  }

  return localPath;
}

// ---------------------------------------------------------------------------
// Config parsing (NeMo → parakeet-mlx format)
// ---------------------------------------------------------------------------

function parseConfig(config: Record<string, unknown>): BaseParakeet {
  const target = config['target'] as string;
  const modelDefaults = (config['model_defaults'] as Record<string, unknown>) ?? {};
  const preprocessorRaw = config['preprocessor'] as Record<string, unknown>;
  const encoderRaw = config['encoder'] as Record<string, unknown>;
  const decoderRaw = config['decoder'] as Record<string, unknown>;
  const jointRaw = config['joint'] as Record<string, unknown>;
  const decodingRaw = config['decoding'] as Record<string, unknown>;

  // Build preprocessor args
  const preprocessor = makePreprocessArgs({
    sampleRate: preprocessorRaw['sample_rate'] as number,
    normalize: preprocessorRaw['normalize'] as string,
    windowSize: preprocessorRaw['window_size'] as number,
    windowStride: preprocessorRaw['window_stride'] as number,
    window: preprocessorRaw['window'] as string,
    features: preprocessorRaw['features'] as number,
    nFft: preprocessorRaw['n_fft'] as number,
    dither: preprocessorRaw['dither'] as number,
    padTo: (preprocessorRaw['pad_to'] as number) ?? 0,
    padValue: (preprocessorRaw['pad_value'] as number) ?? 0,
    preemph: preprocessorRaw['preemph'] as number | null,
    magPower: (preprocessorRaw['mag_power'] as number) ?? 2.0,
  });

  // Build encoder args
  const attContextSize = (encoderRaw['att_context_size'] as [number, number] | null) ?? null;
  const encoder = {
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

  const tdtDurations = modelDefaults['tdt_durations'] as number[] | undefined;

  if (
    target === 'nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel' &&
    tdtDurations != null
  ) {
    // TDT model
    const prednetRaw = (decoderRaw['prednet'] as Record<string, unknown>) ?? {};
    const jointnetRaw = (jointRaw['jointnet'] as Record<string, unknown>) ?? {};

    const args: ParakeetTDTArgs = {
      preprocessor,
      encoder,
      decoder: {
        blankAsPad: decoderRaw['blank_as_pad'] as boolean,
        vocabSize: decoderRaw['vocab_size'] as number,
        prednet: {
          predHidden: prednetRaw['pred_hidden'] as number,
          predRnnLayers: prednetRaw['pred_rnn_layers'] as number,
          rnnHiddenSize: prednetRaw['rnn_hidden_size'] as number | undefined,
        },
      },
      joint: {
        numClasses: jointRaw['num_classes'] as number,
        vocabulary: jointRaw['vocabulary'] as string[],
        jointnet: {
          jointHidden: jointnetRaw['joint_hidden'] as number,
          activation: jointnetRaw['activation'] as string,
          encoderHidden: jointnetRaw['encoder_hidden'] as number,
          predHidden: jointnetRaw['pred_hidden'] as number,
        },
        numExtraOutputs: (jointRaw['num_extra_outputs'] as number) ?? 0,
      },
      decoding: {
        modelType: (decodingRaw['model_type'] as string) ?? 'tdt',
        durations: tdtDurations,
        greedy: (decodingRaw['greedy'] as Record<string, unknown>) ?? null,
      },
    };
    return new ParakeetTDT(args);
  }

  if (
    target === 'nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel' &&
    tdtDurations != null
  ) {
    // TDT-CTC model
    const prednetRaw = (decoderRaw['prednet'] as Record<string, unknown>) ?? {};
    const jointnetRaw = (jointRaw['jointnet'] as Record<string, unknown>) ?? {};
    const auxCtcRaw = (config['aux_ctc'] as Record<string, unknown>) ?? {};
    const auxDecoderRaw = (auxCtcRaw['decoder'] as Record<string, unknown>) ?? {};

    const args: ParakeetTDTCTCArgs = {
      preprocessor,
      encoder,
      decoder: {
        blankAsPad: decoderRaw['blank_as_pad'] as boolean,
        vocabSize: decoderRaw['vocab_size'] as number,
        prednet: {
          predHidden: prednetRaw['pred_hidden'] as number,
          predRnnLayers: prednetRaw['pred_rnn_layers'] as number,
          rnnHiddenSize: prednetRaw['rnn_hidden_size'] as number | undefined,
        },
      },
      joint: {
        numClasses: jointRaw['num_classes'] as number,
        vocabulary: jointRaw['vocabulary'] as string[],
        jointnet: {
          jointHidden: jointnetRaw['joint_hidden'] as number,
          activation: jointnetRaw['activation'] as string,
          encoderHidden: jointnetRaw['encoder_hidden'] as number,
          predHidden: jointnetRaw['pred_hidden'] as number,
        },
        numExtraOutputs: (jointRaw['num_extra_outputs'] as number) ?? 0,
      },
      decoding: {
        modelType: (decodingRaw['model_type'] as string) ?? 'tdt',
        durations: tdtDurations,
        greedy: (decodingRaw['greedy'] as Record<string, unknown>) ?? null,
      },
      auxCtc: {
        decoder: {
          featIn: auxDecoderRaw['feat_in'] as number,
          numClasses: auxDecoderRaw['num_classes'] as number,
          vocabulary: auxDecoderRaw['vocabulary'] as string[],
        },
      },
    };
    return new ParakeetTDTCTC(args);
  }

  if (
    target === 'nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel' &&
    tdtDurations == null
  ) {
    // RNNT model
    const prednetRaw = (decoderRaw['prednet'] as Record<string, unknown>) ?? {};
    const jointnetRaw = (jointRaw['jointnet'] as Record<string, unknown>) ?? {};

    const args: ParakeetRNNTArgs = {
      preprocessor,
      encoder,
      decoder: {
        blankAsPad: decoderRaw['blank_as_pad'] as boolean,
        vocabSize: decoderRaw['vocab_size'] as number,
        prednet: {
          predHidden: prednetRaw['pred_hidden'] as number,
          predRnnLayers: prednetRaw['pred_rnn_layers'] as number,
          rnnHiddenSize: prednetRaw['rnn_hidden_size'] as number | undefined,
        },
      },
      joint: {
        numClasses: jointRaw['num_classes'] as number,
        vocabulary: jointRaw['vocabulary'] as string[],
        jointnet: {
          jointHidden: jointnetRaw['joint_hidden'] as number,
          activation: jointnetRaw['activation'] as string,
          encoderHidden: jointnetRaw['encoder_hidden'] as number,
          predHidden: jointnetRaw['pred_hidden'] as number,
        },
        numExtraOutputs: (jointRaw['num_extra_outputs'] as number) ?? 0,
      },
      decoding: {
        greedy: (decodingRaw['greedy'] as Record<string, unknown>) ?? null,
      },
    };
    return new ParakeetRNNT(args);
  }

  if (target === 'nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE') {
    // CTC model
    const args: ParakeetCTCArgs = {
      preprocessor,
      encoder,
      decoder: {
        featIn: decoderRaw['feat_in'] as number,
        numClasses: decoderRaw['num_classes'] as number,
        vocabulary: decoderRaw['vocabulary'] as string[],
      },
      decoding: {
        greedy: (decodingRaw['greedy'] as Record<string, unknown>) ?? null,
      },
    };
    return new ParakeetCTC(args);
  }

  throw new Error(`Unsupported model target: ${target}`);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Load a Parakeet model from a local directory.
 */
export function fromLocal(modelDir: string): BaseParakeet {
  const configPath = path.join(modelDir, 'config.json');
  const weightsPath = path.join(modelDir, 'model.safetensors');

  if (!fs.existsSync(configPath)) throw new Error(`config.json not found in ${modelDir}`);
  if (!fs.existsSync(weightsPath)) throw new Error(`model.safetensors not found in ${modelDir}`);

  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  const model = parseConfig(config);

  const weights = loadSafetensors(weightsPath);
  model.loadWeights(weights);

  return model;
}

/**
 * Load a Parakeet model from a HuggingFace Hub repo or local directory.
 *
 * @param hfIdOrPath - HuggingFace repo ID (e.g. "mlx-community/parakeet-tdt-0.6b-v3")
 *                     or local directory path.
 * @param options.cacheDir - Override default HF cache directory.
 * @param options.onProgress - Called with (downloaded, total) bytes during file downloads.
 */
export async function fromPretrained(
  hfIdOrPath: string,
  options: { cacheDir?: string; onProgress?: (file: string, downloaded: number, total: number) => void } = {},
): Promise<BaseParakeet> {
  // Check if it's a local path
  if (fs.existsSync(hfIdOrPath) && fs.statSync(hfIdOrPath).isDirectory()) {
    return fromLocal(hfIdOrPath);
  }

  const makeProgress = (file: string) =>
    options.onProgress
      ? (downloaded: number, total: number) => options.onProgress!(file, downloaded, total)
      : undefined;

  // Download from HuggingFace Hub
  const configPath = await downloadFromHub(hfIdOrPath, 'config.json', options.cacheDir, makeProgress('config.json'));
  const weightsPath = await downloadFromHub(hfIdOrPath, 'model.safetensors', options.cacheDir, makeProgress('model.safetensors'));

  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  const model = parseConfig(config);

  const weights = loadSafetensors(weightsPath);
  model.loadWeights(weights);

  return model;
}
