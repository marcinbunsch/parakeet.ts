/**
 * MLX backend adapter.
 *
 * Wraps the existing `src/mlx/` modules behind the shared `ParakeetBackend`
 * interface so the MLX path and the ONNX path run the *same* decode loop,
 * tokenizer, alignment and audio front-end.
 *
 * The ONNX export fuses prediction and joint into one graph, so `decodeStep`
 * calls this backend's `PredictNetwork` and `JointNetwork` back to back.
 */
import { MxArray } from '@mlx-node/core';
import type {
  ParakeetBackend, EncoderOutput, DecodeStepResult, DecoderStateHandle,
} from '../backend.js';
import { Conformer } from './conformer.js';
import { PredictNetwork, JointNetwork } from './rnnt.js';

function s(...dims: number[]): BigInt64Array {
  return BigInt64Array.from(dims.map(BigInt));
}

/** MLX decoder state: the prediction network's [h, c]. */
type MlxState = [MxArray, MxArray];

export class MlxBackend implements ParakeetBackend {
  readonly name = 'mlx';
  readonly encoderDim: number;

  private readonly encoder: Conformer;
  private readonly predict: PredictNetwork;
  private readonly joint: JointNetwork;

  constructor(args: {
    encoder: Conformer;
    predict: PredictNetwork;
    joint: JointNetwork;
    encoderDim: number;
  }) {
    this.encoder = args.encoder;
    this.predict = args.predict;
    this.joint = args.joint;
    this.encoderDim = args.encoderDim;
  }

  /**
   * @param mel log-mel laid out [nMels, numFrames] (the shared front-end's order)
   */
  async encode(mel: Float32Array, nMels: number, numFrames: number): Promise<EncoderOutput> {
    // The MLX conformer wants [batch, frames, nMels]; the shared front-end
    // produces [nMels, frames], so transpose on the way in.
    const t = new Float32Array(mel.length);
    for (let m = 0; m < nMels; m++) {
      const row = m * numFrames;
      for (let f = 0; f < numFrames; f++) t[f * nMels + m] = mel[row + f];
    }
    const melArray = MxArray.fromFloat32(t, s(1, numFrames, nMels));

    const [features, lengths] = this.encoder.forward(melArray, null, null);
    features.eval();
    lengths.eval();

    const shape = Array.from(features.shape(), Number); // [1, T, D]
    const frames = Number(lengths.toInt32()[0]);

    return {
      data: features.toFloat32(),
      dim: shape[2],
      frames,
      stride: shape[1],
      layout: 'time-major',
    };
  }

  async decodeStep(
    encFrame: Float32Array,
    token: number | null,
    state: DecoderStateHandle | null,
  ): Promise<DecodeStepResult> {
    const y = token !== null
      ? MxArray.fromInt32(new Int32Array([token]), s(1, 1))
      : null;

    const [decOut, nextState] = this.predict.forward(y, (state as MlxState | null) ?? null);

    // predict: [1, 1, predHidden] -> [1, 1, 1, predHidden]
    const decExpand = decOut.expandDims(1);
    // encoder frame -> [1, 1, 1, encoderDim]
    const enc = MxArray.fromFloat32(encFrame, s(1, 1, 1, encFrame.length));

    const jointOut = this.joint.forward(enc, decExpand);
    jointOut.eval();

    return { logits: jointOut.toFloat32(), state: nextState };
  }
}
