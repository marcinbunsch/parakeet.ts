/**
 * ONNX Runtime backend — CUDA execution provider on Linux/Nvidia.
 *
 * Two sessions: the conformer encoder, and the fused prediction+joint
 * `decoder_joint` graph. `onnxruntime-node` is loaded lazily so that a
 * Mac-only install never has to resolve it.
 *
 * Graph contract (parakeet-tdt-0.6b-v3 exports):
 *   encoder       : audio_signal [1,nMels,T_mel], length [1]
 *                -> outputs [1,D,T], encoded_lengths [1]
 *   decoder_joint : encoder_outputs [1,D,1], targets [1,1], target_length [1],
 *                   input_states_1 [L,1,H], input_states_2 [L,1,H]
 *                -> outputs [1,1,1,C], output_states_1, output_states_2
 */
import type { ParakeetBackend, EncoderOutput, DecodeStepResult, DecoderStateHandle } from '../backend.js';

// Minimal structural types so we don't need a hard dependency on the ORT types.
interface OrtTensor { data: Float32Array | Int32Array | BigInt64Array; dims: readonly number[]; }
interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>>;
  release?(): Promise<void>;
}
interface OrtModule {
  Tensor: new (type: string, data: ArrayLike<number> | BigInt64Array, dims: number[]) => OrtTensor;
  InferenceSession: {
    create(path: string | Uint8Array, options?: Record<string, unknown>): Promise<OrtSession>;
  };
}

export type ExecutionProvider = 'cuda' | 'cpu' | 'tensorrt' | 'coreml' | 'dml';

export interface OnnxBackendOptions {
  encoderPath: string;
  decoderJointPath: string;
  /** Execution provider for the encoder. Default: 'cuda' on linux/win x64, else 'cpu'. */
  executionProvider?: ExecutionProvider;
  /**
   * Execution provider for the autoregressive decode step. Defaults to the same
   * as `executionProvider`. CPU is a reasonable choice — the step is small
   * (~0.5 ms) and CPU avoids per-step device round trips.
   */
  decoderExecutionProvider?: ExecutionProvider;
  /** LSTM layers in the prediction network (state dim 0). Default 2. */
  predRnnLayers?: number;
  /** LSTM hidden size (state dim 2). Default 640. */
  predHidden?: number;
  /** ORT log severity: 0 verbose .. 4 fatal. Default 3 (error). */
  logSeverityLevel?: number;
}

let ortModule: OrtModule | null = null;

async function loadOrt(): Promise<OrtModule> {
  if (ortModule) return ortModule;
  try {
    const mod = await import('onnxruntime-node');
    ortModule = ((mod as unknown as { default?: OrtModule }).default ?? mod) as OrtModule;
    return ortModule;
  } catch (err) {
    throw new Error(
      'The ONNX backend requires the optional dependency "onnxruntime-node". ' +
      'Install it with `npm install onnxruntime-node`. ' +
      `Original error: ${(err as Error).message}`,
    );
  }
}

export function defaultExecutionProvider(): ExecutionProvider {
  if (process.platform === 'darwin') return 'cpu';
  if (process.arch === 'x64' && (process.platform === 'linux' || process.platform === 'win32')) {
    return 'cuda';
  }
  return 'cpu';
}

/** Decoder state as ORT tensors — opaque to the decode loop. */
interface OnnxState {
  s1: OrtTensor;
  s2: OrtTensor;
}

export class OnnxBackend implements ParakeetBackend {
  readonly name: string;
  readonly encoderDim: number;

  private readonly ort: OrtModule;
  private readonly encoder: OrtSession;
  private readonly decoderJoint: OrtSession;
  private readonly predRnnLayers: number;
  private readonly predHidden: number;
  private readonly zeroState: OnnxState;
  /** Reused scratch for the single-element int32 inputs. */
  private readonly targetLength: OrtTensor;

  private constructor(
    ort: OrtModule,
    encoder: OrtSession,
    decoderJoint: OrtSession,
    encoderDim: number,
    predRnnLayers: number,
    predHidden: number,
    name: string,
  ) {
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

  static async create(opts: OnnxBackendOptions): Promise<OnnxBackend> {
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
    const backend = new OnnxBackend(
      ort, encoder, decoderJoint,
      0,
      opts.predRnnLayers ?? 2,
      opts.predHidden ?? 640,
      decEp === ep ? `onnx-${ep}` : `onnx-${ep}/${decEp}`,
    );
    return backend;
  }

  async encode(mel: Float32Array, nMels: number, numFrames: number): Promise<EncoderOutput> {
    const feeds = {
      audio_signal: new this.ort.Tensor('float32', mel, [1, nMels, numFrames]),
      length: new this.ort.Tensor('int64', BigInt64Array.from([BigInt(numFrames)]), [1]),
    };
    const out = await this.encoder.run(feeds);
    const encoded = out['outputs'];
    const frames = Number(out['encoded_lengths'].data[0]);
    const dim = encoded.dims[1];
    // `encoderDim` is readonly to callers but discovered here on first use.
    (this as { encoderDim: number }).encoderDim = dim;

    return {
      data: encoded.data as Float32Array,
      dim,
      frames,
      stride: encoded.dims[2],
      layout: 'dim-major',
    };
  }

  async decodeStep(
    encFrame: Float32Array,
    token: number | null,
    state: DecoderStateHandle | null,
  ): Promise<DecodeStepResult> {
    const st = (state as OnnxState | null) ?? this.zeroState;
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
      logits: r['outputs'].data as Float32Array,
      state: { s1: r['output_states_1'], s2: r['output_states_2'] } satisfies OnnxState,
    };
  }

  /** Blank/padding token id, set by the model wrapper (vocabulary length). */
  blankId = 0;

  async dispose(): Promise<void> {
    await this.encoder.release?.();
    await this.decoderJoint.release?.();
  }
}
