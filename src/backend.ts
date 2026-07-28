/**
 * The backend boundary.
 *
 * The library is abstracted at the *model* boundary, not the tensor boundary:
 * everything except the encoder forward and the prediction+joint step is plain
 * TypeScript and shared by all backends.
 *
 * Two calls are enough. The ONNX export fuses the prediction network and the
 * joint network into a single `decoder_joint` graph that cannot be split, so
 * `decodeStep` is the unit of work — the MLX backend simply calls its own
 * `predict` and `joint` back to back behind it.
 */

/**
 * Which axis is contiguous in `EncoderOutput.data`.
 * ONNX emits [1, dim, time] ("dim-major"); the MLX encoder emits
 * [1, time, dim] ("time-major"). Declaring it avoids a transpose copy.
 */
export type EncoderLayout = 'dim-major' | 'time-major';

/** Encoder output for a single utterance, flattened row-major. */
export interface EncoderOutput {
  /** length dim * timeStride */
  data: Float32Array;
  /** encoder hidden size */
  dim: number;
  /** number of valid time frames */
  frames: number;
  /**
   * Allocated extent of the time axis, which can exceed `frames` when the graph
   * pads. Used as the row stride for 'dim-major'.
   */
  stride: number;
  layout: EncoderLayout;
}

/**
 * Opaque, backend-owned decoder state (the prediction network's LSTM state).
 * The decode loop only threads it through; it never inspects it.
 */
export type DecoderStateHandle = unknown;

export interface DecodeStepResult {
  /** joint logits: vocab + blank + durations */
  logits: Float32Array;
  state: DecoderStateHandle;
}

export interface ParakeetBackend {
  /** Human-readable backend id, e.g. "mlx" or "onnx-cuda". */
  readonly name: string;

  /** Encoder hidden size. */
  readonly encoderDim: number;

  /** Run the conformer encoder over a log-mel spectrogram laid out [nMels, numFrames]. */
  encode(mel: Float32Array, nMels: number, numFrames: number): Promise<EncoderOutput>;

  /**
   * One autoregressive decode step.
   * @param encFrame  a single encoder frame, length `encoderDim`
   * @param token     previously emitted token id, or `null` at sequence start
   * @param state     previous decoder state, or `null` at sequence start
   */
  decodeStep(
    encFrame: Float32Array,
    token: number | null,
    state: DecoderStateHandle | null,
  ): Promise<DecodeStepResult>;

  /** Release any native resources. */
  dispose?(): void | Promise<void>;
}
