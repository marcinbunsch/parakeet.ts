import { MxArray } from '@mlx-node/core';
import { Module, WeightMap, Linear, Embedding, LSTM, relu, sigmoid, softmax } from './nn.js';
import { makeAlignedToken, AlignedToken } from '../alignment.js';
import { decode } from '../tokenizer.js';

function s(...dims: number[]): BigInt64Array {
  return BigInt64Array.from(dims.map(BigInt));
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface PredictNetworkArgs {
  predHidden: number;
  predRnnLayers: number;
  rnnHiddenSize?: number;
}

export interface JointNetworkArgs {
  jointHidden: number;
  activation: string;
  encoderHidden: number;
  predHidden: number;
}

export interface PredictArgs {
  blankAsPad: boolean;
  vocabSize: number;
  prednet: PredictNetworkArgs;
}

export interface JointArgs {
  numClasses: number;
  vocabulary: string[];
  jointnet: JointNetworkArgs;
  numExtraOutputs?: number;
}

// ---------------------------------------------------------------------------
// Predict (decoder) network — embedding + LSTM
// ---------------------------------------------------------------------------

export class PredictNetwork extends Module {
  embed: Embedding;
  decRnn: LSTM;
  readonly predHidden: number;

  constructor(args: PredictArgs) {
    super();
    this.predHidden = args.prednet.predHidden;
    const vocabSize = args.blankAsPad ? args.vocabSize + 1 : args.vocabSize;
    const rnnHidden = args.prednet.rnnHiddenSize ?? args.prednet.predHidden;

    this.embed = new Embedding(vocabSize, args.prednet.predHidden);
    this.decRnn = new LSTM(args.prednet.predHidden, rnnHidden, args.prednet.predRnnLayers);
  }

  /**
   * y: [batch, 1] token ids or null for blank start
   * hc: LSTM hidden/cell state or null
   * Returns [output: [batch, 1, predHidden], [h, c]]
   */
  forward(
    y: MxArray | null,
    hc: [MxArray, MxArray] | null,
  ): [MxArray, [MxArray, MxArray]] {
    let embeddedY: MxArray;

    if (y !== null) {
      embeddedY = this.embed.forward(y); // [batch, 1, predHidden]
    } else {
      const batch = hc ? Number(hc[0].shape()[1]) : 1;
      embeddedY = MxArray.zeros(s(batch, 1, this.predHidden), null);
    }

    return this.decRnn.forward(embeddedY, hc);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.embed.loadWeights(weights, `${prefix}.prediction.embed`);
    this.decRnn.loadWeights(weights, `${prefix}.prediction.dec_rnn.lstm`);
  }
}

// ---------------------------------------------------------------------------
// Joint network — combines encoder and decoder outputs
// ---------------------------------------------------------------------------

export class JointNetwork extends Module {
  pred: Linear;
  enc: Linear;
  activation: (x: MxArray) => MxArray;
  jointOut: Linear;
  readonly numClasses: number;

  constructor(args: JointArgs) {
    super();
    const numExtraOutputs = args.numExtraOutputs ?? 0;
    this.numClasses = args.numClasses + 1 + numExtraOutputs;

    this.pred = new Linear(args.jointnet.predHidden, args.jointnet.jointHidden);
    this.enc = new Linear(args.jointnet.encoderHidden, args.jointnet.jointHidden);
    this.jointOut = new Linear(args.jointnet.jointHidden, this.numClasses);

    const act = args.jointnet.activation.toLowerCase();
    if (act === 'relu') {
      this.activation = relu;
    } else if (act === 'sigmoid') {
      this.activation = sigmoid;
    } else {
      this.activation = (x: MxArray) => x.tanh();
    }
  }

  forward(enc: MxArray, pred: MxArray): MxArray {
    // enc:  [batch, encSeq, 1, encoderHidden]
    // pred: [batch, 1, predSeq, predHidden]
    const encP = this.enc.forward(enc);   // [batch, encSeq, 1, jointHidden]
    const predP = this.pred.forward(pred); // [batch, 1, predSeq, jointHidden]

    // Broadcast addition: [batch, encSeq, predSeq, jointHidden]
    const x = encP.add(predP);
    const activated = this.activation(x);
    return this.jointOut.forward(activated);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.pred.loadWeights(weights, `${prefix}.pred`);
    this.enc.loadWeights(weights, `${prefix}.enc`);
    // joint_net is a list: [activation, Identity, Linear]
    // The Linear is at index 2 in Python (0=activation, 1=Identity, 2=Linear)
    this.jointOut.loadWeights(weights, `${prefix}.joint_net.2`);
  }
}

// ---------------------------------------------------------------------------
// Decoding helpers
// ---------------------------------------------------------------------------

export interface DecoderState {
  lastToken: number | null;
  hiddenState: [MxArray, MxArray] | null;
}

/** Greedy TDT decoder. Returns decoded token list per batch element. */
export function decodeTDTGreedy(
  features: MxArray,     // [batch, seqLen, encoderDim]
  lengths: MxArray,      // [batch]
  decoder: PredictNetwork,
  joint: JointNetwork,
  vocabulary: string[],
  durations: number[],
  maxSymbols: number | null,
  states: DecoderState[],
  timeRatio: number,
): [Array<AlignedToken[]>, DecoderState[]] {
  const fShape = features.shape();
  const B = Number(fShape[0]);

  const results: Array<AlignedToken[]> = [];
  const newStates: DecoderState[] = [];

  for (let b = 0; b < B; b++) {
    const hypothesis: AlignedToken[] = [];
    const feature = features.slice(s(b, 0, 0), s(b + 1, Number(fShape[1]), Number(fShape[2])));
    const length = Number(lengths.toInt32()[b]);

    let step = 0;
    let newSymbols = 0;
    let { lastToken, hiddenState } = states[b];

    while (step < length) {
      const yInput = lastToken !== null
        ? MxArray.fromInt32(new Int32Array([lastToken]), s(1, 1))
        : null;

      const [decOut, [newH, newC]] = decoder.forward(yInput, hiddenState);
      // decOut: [1, 1, predHidden] — expand for joint: [1, 1, 1, predHidden]
      const decExpand = decOut.expandDims(1);

      // encoder feature at current step: [1, 1, encoderDim] → [1, 1, 1, encoderDim]
      const encStep = feature.slice(s(0, step, 0), s(1, step + 1, Number(feature.shape()[2]))).expandDims(2);

      const jointOut = joint.forward(encStep, decExpand);
      // jointOut: [1, 1, 1, numClasses]

      const vocabSize = vocabulary.length;
      const numClasses = Number(jointOut.shape()[3]);
      const numDurations = durations.length;

      const tokenLogits = jointOut
        .slice(s(0, 0, 0, 0), s(1, 1, 1, vocabSize + 1))
        .squeeze(new Int32Array([0, 1, 2]));

      const durationLogits = jointOut
        .slice(s(0, 0, 0, vocabSize + 1), s(1, 1, 1, numClasses))
        .squeeze(new Int32Array([0, 1, 2]));

      const tokenProbs = softmax(tokenLogits, 0).toFloat32();
      const durationProbs = softmax(durationLogits, 0).toFloat32();

      // Find argmax for token and duration
      let predToken = 0;
      let maxTokenProb = tokenProbs[0];
      for (let i = 1; i < tokenProbs.length; i++) {
        if (tokenProbs[i] > maxTokenProb) {
          maxTokenProb = tokenProbs[i];
          predToken = i;
        }
      }

      let decision = 0;
      let maxDurProb = durationProbs[0];
      for (let i = 1; i < durationProbs.length; i++) {
        if (durationProbs[i] > maxDurProb) {
          maxDurProb = durationProbs[i];
          decision = i;
        }
      }

      const isBlank = predToken === vocabSize;

      if (!isBlank) {
        // Confidence using entropy
        let entropy = 0;
        for (let i = 0; i < tokenProbs.length; i++) {
          const p = tokenProbs[i] + 1e-10;
          entropy -= p * Math.log(p);
        }
        const maxEntropy = Math.log(tokenProbs.length);
        const confidence = 1.0 - entropy / maxEntropy;

        hypothesis.push(makeAlignedToken(
          predToken,
          decode([predToken], vocabulary),
          step * timeRatio,
          durations[decision] * timeRatio,
          confidence,
        ));
        lastToken = predToken;
        hiddenState = [newH, newC];
      }

      step += durations[decision];
      newSymbols += 1;

      if (durations[decision] !== 0) {
        newSymbols = 0;
      } else if (maxSymbols !== null && newSymbols >= maxSymbols) {
        step += 1;
        newSymbols = 0;
      }
    }

    results.push(hypothesis);
    newStates.push({ lastToken, hiddenState });
  }

  return [results, newStates];
}

/** Greedy RNNT decoder */
export function decodeRNNTGreedy(
  features: MxArray,
  lengths: MxArray,
  decoder: PredictNetwork,
  joint: JointNetwork,
  vocabulary: string[],
  maxSymbols: number | null,
  states: DecoderState[],
  timeRatio: number,
): [Array<Array<import('../alignment.js').AlignedToken>>, DecoderState[]] {
  const fShape = features.shape();
  const B = Number(fShape[0]);

  const results: Array<AlignedToken[]> = [];
  const newStates: DecoderState[] = [];

  for (let b = 0; b < B; b++) {
    const hypothesis: AlignedToken[] = [];
    const feature = features.slice(s(b, 0, 0), s(b + 1, Number(fShape[1]), Number(fShape[2])));
    const length = Number(lengths.toInt32()[b]);

    let step = 0;
    let newSymbols = 0;
    let { lastToken, hiddenState } = states[b];

    while (step < length) {
      const yInput = lastToken !== null
        ? MxArray.fromInt32(new Int32Array([lastToken]), s(1, 1))
        : null;

      const [decOut, [newH, newC]] = decoder.forward(yInput, hiddenState);
      const decExpand = decOut.expandDims(1);
      const encStep = feature.slice(s(0, step, 0), s(1, step + 1, Number(feature.shape()[2]))).expandDims(2);

      const jointOut = joint.forward(encStep, decExpand);

      const vocabSize = vocabulary.length;
      const tokenLogits = jointOut
        .slice(s(0, 0, 0, 0), s(1, 1, 1, vocabSize + 1))
        .squeeze(new Int32Array([0, 1, 2]));

      const tokenProbs = softmax(tokenLogits, 0).toFloat32();
      let predToken = 0;
      let maxP = tokenProbs[0];
      for (let i = 1; i < tokenProbs.length; i++) {
        if (tokenProbs[i] > maxP) { maxP = tokenProbs[i]; predToken = i; }
      }

      const isBlank = predToken === vocabSize;

      if (!isBlank) {
        let entropy = 0;
        for (let i = 0; i < tokenProbs.length; i++) {
          const p = tokenProbs[i] + 1e-10;
          entropy -= p * Math.log(p);
        }
        const confidence = 1.0 - entropy / Math.log(tokenProbs.length);

        hypothesis.push(makeAlignedToken(
          predToken,
          decode([predToken], vocabulary),
          step * timeRatio,
          timeRatio,
          confidence,
        ));
        lastToken = predToken;
        hiddenState = [newH, newC];
        newSymbols += 1;
        if (maxSymbols !== null && newSymbols >= maxSymbols) {
          step += 1;
          newSymbols = 0;
        }
      } else {
        step += 1;
        newSymbols = 0;
      }
    }

    results.push(hypothesis);
    newStates.push({ lastToken, hiddenState });
  }

  return [results, newStates];
}
