import { MxArray } from '@mlx-node/core';
import {
  Module,
  WeightMap,
  Linear,
  LayerNorm,
  BatchNorm,
  Conv1d,
  Conv2d,
  silu,
  relu,
  glu,
} from './nn.js';
import {
  MultiHeadAttention,
  RelPositionMultiHeadAttention,
  RelPositionMultiHeadLocalAttention,
  RelPositionalEncoding,
  LocalRelPositionalEncoding,
} from './attention.js';
import { ConformerCache } from './cache.js';

function s(...dims: number[]): BigInt64Array {
  return BigInt64Array.from(dims.map(BigInt));
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface ConformerArgs {
  featIn: number;
  nLayers: number;
  dModel: number;
  nHeads: number;
  ffExpansionFactor: number;
  subsamplingFactor: number;
  selfAttentionModel: string;
  subsampling: string;
  convKernelSize: number;
  subsamplingConvChannels: number;
  posEmbMaxLen: number;
  causalDownsampling?: boolean;
  useBias?: boolean;
  xscaling?: boolean;
  subsamplingConvChunkingFactor?: number;
  attContextSize?: [number, number] | null;
}

// ---------------------------------------------------------------------------
// Feed-forward block
// ---------------------------------------------------------------------------

class FeedForward extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(dModel: number, dFf: number, useBias: boolean) {
    super();
    this.linear1 = new Linear(dModel, dFf, useBias);
    this.linear2 = new Linear(dFf, dModel, useBias);
  }

  forward(x: MxArray): MxArray {
    return this.linear2.forward(silu(this.linear1.forward(x)));
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.linear1.loadWeights(weights, `${prefix}.linear1`);
    this.linear2.loadWeights(weights, `${prefix}.linear2`);
  }
}

// ---------------------------------------------------------------------------
// Convolution block
// ---------------------------------------------------------------------------

class Convolution extends Module {
  readonly padding: number;
  pointwiseConv1: Conv1d;
  depthwiseConv: Conv1d;
  batchNorm: BatchNorm;
  pointwiseConv2: Conv1d;

  constructor(args: ConformerArgs) {
    super();
    const useBias = args.useBias ?? true;
    this.padding = Math.floor((args.convKernelSize - 1) / 2);

    this.pointwiseConv1 = new Conv1d(args.dModel, args.dModel * 2, 1, 1, 0, 1, useBias);
    this.depthwiseConv = new Conv1d(args.dModel, args.dModel, args.convKernelSize, 1, 0, args.dModel, useBias);
    this.batchNorm = new BatchNorm(args.dModel);
    this.pointwiseConv2 = new Conv1d(args.dModel, args.dModel, 1, 1, 0, 1, useBias);
  }

  forward(x: MxArray, cache: ConformerCache | null): MxArray {
    x = this.pointwiseConv1.forward(x);
    x = glu(x, 2); // split along last axis, gate with sigmoid

    if (cache !== null) {
      x = cache.updateAndFetchConv(x, this.padding);
    } else {
      x = x.pad(new Int32Array([0, 0, this.padding, this.padding, 0, 0]), 0.0);
    }

    x = this.depthwiseConv.forward(x);
    x = this.batchNorm.forward(x);
    x = silu(x);
    x = this.pointwiseConv2.forward(x);
    return x;
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.pointwiseConv1.loadWeights(weights, `${prefix}.pointwise_conv1`);
    this.depthwiseConv.loadWeights(weights, `${prefix}.depthwise_conv`);
    this.batchNorm.loadWeights(weights, `${prefix}.batch_norm`);
    this.pointwiseConv2.loadWeights(weights, `${prefix}.pointwise_conv2`);
  }
}

// ---------------------------------------------------------------------------
// Conformer block
// ---------------------------------------------------------------------------

type AttentionModel = 'rel_pos' | 'rel_pos_local_attn' | 'normal';

export class ConformerBlock extends Module {
  normFF1: LayerNorm;
  ff1: FeedForward;

  normSelfAtt: LayerNorm;
  selfAttn: MultiHeadAttention | RelPositionMultiHeadAttention | RelPositionMultiHeadLocalAttention;

  normConv: LayerNorm;
  conv: Convolution;

  normFF2: LayerNorm;
  ff2: FeedForward;

  normOut: LayerNorm;

  private readonly args: ConformerArgs;

  constructor(args: ConformerArgs) {
    super();
    this.args = args;
    const useBias = args.useBias ?? true;
    const ffDim = args.dModel * args.ffExpansionFactor;

    this.normFF1 = new LayerNorm(args.dModel);
    this.ff1 = new FeedForward(args.dModel, ffDim, useBias);

    this.normSelfAtt = new LayerNorm(args.dModel);
    this.selfAttn = this.buildAttention(args.selfAttentionModel, args.attContextSize ?? null);

    this.normConv = new LayerNorm(args.dModel);
    this.conv = new Convolution(args);

    this.normFF2 = new LayerNorm(args.dModel);
    this.ff2 = new FeedForward(args.dModel, ffDim, useBias);

    this.normOut = new LayerNorm(args.dModel);
  }

  private buildAttention(
    name: string,
    contextSize: [number, number] | null,
  ): MultiHeadAttention | RelPositionMultiHeadAttention | RelPositionMultiHeadLocalAttention {
    const useBias = this.args.useBias ?? true;
    if (name === 'rel_pos') {
      return new RelPositionMultiHeadAttention(this.args.nHeads, this.args.dModel, useBias);
    } else if (name === 'rel_pos_local_attn') {
      return new RelPositionMultiHeadLocalAttention(
        this.args.nHeads,
        this.args.dModel,
        useBias,
        contextSize ?? [256, 256],
      );
    } else {
      return new MultiHeadAttention(this.args.nHeads, this.args.dModel, true);
    }
  }

  setAttentionModel(name: AttentionModel, contextSize: [number, number] = [256, 256]): void {
    const oldAttn = this.selfAttn;
    const newAttn = this.buildAttention(name, contextSize);

    // Transfer the already-loaded parameters into the new module. The rel_pos
    // and rel_pos_local_attn variants share an identical parameter set (the
    // local class extends the global one), so weights carry over directly.
    if (
      newAttn instanceof RelPositionMultiHeadAttention &&
      oldAttn instanceof RelPositionMultiHeadAttention
    ) {
      newAttn.copyWeightsFrom(oldAttn);
    } else if (
      newAttn instanceof MultiHeadAttention &&
      oldAttn instanceof MultiHeadAttention
    ) {
      newAttn.copyWeightsFrom(oldAttn);
    } else {
      throw new Error(
        `Cannot switch attention model between incompatible types (weights would be lost)`,
      );
    }

    this.selfAttn = newAttn;
  }

  forward(
    x: MxArray,
    posEmb: MxArray | null,
    mask: MxArray | null,
    cache: ConformerCache | null,
  ): MxArray {
    // FF1
    x = x.add(this.ff1.forward(this.normFF1.forward(x)).mulScalar(0.5));

    // Self-attention
    const xNorm = this.normSelfAtt.forward(x);
    x = x.add(this.selfAttn.forward(xNorm, xNorm, xNorm, posEmb, mask, cache));

    // Convolution
    x = x.add(this.conv.forward(this.normConv.forward(x), cache));

    // FF2
    x = x.add(this.ff2.forward(this.normFF2.forward(x)).mulScalar(0.5));

    return this.normOut.forward(x);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.normFF1.loadWeights(weights, `${prefix}.norm_feed_forward1`);
    this.ff1.loadWeights(weights, `${prefix}.feed_forward1`);
    this.normSelfAtt.loadWeights(weights, `${prefix}.norm_self_att`);
    this.selfAttn.loadWeights(weights, `${prefix}.self_attn`);
    this.normConv.loadWeights(weights, `${prefix}.norm_conv`);
    this.conv.loadWeights(weights, `${prefix}.conv`);
    this.normFF2.loadWeights(weights, `${prefix}.norm_feed_forward2`);
    this.ff2.loadWeights(weights, `${prefix}.feed_forward2`);
    this.normOut.loadWeights(weights, `${prefix}.norm_out`);
  }
}

// ---------------------------------------------------------------------------
// DW-striding subsampling (Conv2D-based)
// ---------------------------------------------------------------------------

class DwStridingSubsampling extends Module {
  private readonly samplingNum: number;
  private readonly stride = 2;
  private readonly kernelSize = 3;
  private readonly padding: number;

  convLayers: Array<Conv2d | null>; // null = ReLU placeholder
  out: Linear;

  constructor(args: ConformerArgs) {
    super();
    this.padding = Math.floor((this.kernelSize - 1) / 2);
    this.samplingNum = Math.round(Math.log2(args.subsamplingFactor));

    const convChannels = args.subsamplingConvChannels;

    // Compute final frequency dimension
    let finalFreqDim = args.featIn;
    for (let i = 0; i < this.samplingNum; i++) {
      finalFreqDim =
        Math.floor((finalFreqDim + 2 * this.padding - this.kernelSize) / this.stride) + 1;
    }

    // Build convolution layers
    this.convLayers = [];
    let inCh = 1;

    // First layer: standard conv2d
    this.convLayers.push(new Conv2d(inCh, convChannels, this.kernelSize, this.stride, this.padding, 1, true));
    this.convLayers.push(null); // ReLU

    inCh = convChannels;
    for (let i = 1; i < this.samplingNum; i++) {
      // Depthwise
      this.convLayers.push(new Conv2d(inCh, inCh, this.kernelSize, this.stride, this.padding, inCh, true));
      // Pointwise
      this.convLayers.push(new Conv2d(inCh, convChannels, 1, 1, 0, 1, true));
      this.convLayers.push(null); // ReLU
    }

    this.out = new Linear(convChannels * finalFreqDim, args.dModel);
  }

  forward(x: MxArray, lengths: MxArray): [MxArray, MxArray] {
    // x: [batch, seq, mel] → [batch, 1, seq, mel] then to MLX NHWC: [batch, seq, mel, 1]
    const xShape = x.shape();
    const batch = Number(xShape[0]);

    // lengths update
    let outLengths = lengths;
    for (let i = 0; i < this.samplingNum; i++) {
      const pad = this.padding;
      const k = this.kernelSize;
      const st = this.stride;
      // floor((len + 2*pad - k) / stride) + 1
      outLengths = outLengths
        .addScalar(2 * pad - k)
        .divScalar(st)
        .floor()
        .addScalar(1);
    }
    outLengths = outLengths.astype(3); // 3 = int32

    // Reshape x: [batch, seq, mel] → [batch, seq, mel, 1] (NHWC with C=1)
    let cur = x.expandDims(3); // [batch, seq, mel, 1]

    for (const layer of this.convLayers) {
      if (layer === null) {
        cur = relu(cur);
      } else {
        cur = layer.forward(cur);
      }
    }

    // cur: [batch, outSeq, outMel, convChannels] (NHWC layout)
    // Python flattens as [batch, outSeq, convChannels*outMel] (C-major, F-minor):
    // it transposes NHWC -> NCHW then swapaxes(1,2) -> [B, T', C', F'] before
    // reshaping. To match that memory ordering (which the Linear weights expect),
    // transpose channel axis before freq axis here.
    const cShape = cur.shape();
    const outSeq = Number(cShape[1]);
    const outMel = Number(cShape[2]);
    const ch = Number(cShape[3]);

    cur = cur
      .transpose(new Int32Array([0, 1, 3, 2]))
      .reshape(s(batch, outSeq, ch * outMel));
    cur = this.out.forward(cur);

    return [cur, outLengths];
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    let convIdx = 0;
    for (const layer of this.convLayers) {
      if (layer !== null) {
        layer.loadWeights(weights, `${prefix}.conv.${convIdx}`);
      }
      convIdx++;
    }
    this.out.loadWeights(weights, `${prefix}.out`);
  }
}

// ---------------------------------------------------------------------------
// Conformer encoder
// ---------------------------------------------------------------------------

export class Conformer extends Module {
  readonly args: ConformerArgs;
  posEnc: RelPositionalEncoding | LocalRelPositionalEncoding | null;
  preEncode: DwStridingSubsampling | Linear;
  layers: ConformerBlock[];

  constructor(args: ConformerArgs) {
    super();
    this.args = args;

    const selfAttModel = args.selfAttentionModel;
    const ctxSize = args.attContextSize ?? null;

    if (selfAttModel === 'rel_pos') {
      this.posEnc = new RelPositionalEncoding(args.dModel, args.posEmbMaxLen, args.xscaling ?? false);
    } else if (selfAttModel === 'rel_pos_local_attn') {
      this.posEnc = new LocalRelPositionalEncoding(
        args.dModel,
        args.posEmbMaxLen,
        args.xscaling ?? false,
        ctxSize ?? [256, 256],
      );
    } else {
      this.posEnc = null;
    }

    if (args.subsamplingFactor > 1) {
      if (args.subsampling === 'dw_striding' && !(args.causalDownsampling ?? false)) {
        this.preEncode = new DwStridingSubsampling(args);
      } else {
        throw new Error('Only dw_striding non-causal subsampling is supported');
      }
    } else {
      this.preEncode = new Linear(args.featIn, args.dModel);
    }

    this.layers = Array.from({ length: args.nLayers }, () => new ConformerBlock(args));
  }

  setAttentionModel(
    name: AttentionModel,
    contextSize: [number, number] = [256, 256],
  ): void {
    if (name === 'rel_pos') {
      this.posEnc = new RelPositionalEncoding(this.args.dModel, this.args.posEmbMaxLen, this.args.xscaling ?? false);
    } else if (name === 'rel_pos_local_attn') {
      this.posEnc = new LocalRelPositionalEncoding(
        this.args.dModel,
        this.args.posEmbMaxLen,
        this.args.xscaling ?? false,
        contextSize,
      );
    } else {
      this.posEnc = null;
    }

    for (const layer of this.layers) {
      layer.setAttentionModel(name, contextSize);
    }
  }

  forward(
    x: MxArray, // [batch, seq, mel]
    lengths: MxArray | null,
    cache: Array<ConformerCache | null> | null,
  ): [MxArray, MxArray] {
    const xShape = x.shape();
    const batch = Number(xShape[0]);
    const seq = Number(xShape[1]);

    if (lengths === null) {
      const lenData = new Int32Array(batch).fill(seq);
      lengths = MxArray.fromInt32(lenData, BigInt64Array.from([BigInt(batch)]));
    }

    let outLengths: MxArray;

    if (this.preEncode instanceof DwStridingSubsampling) {
      [x, outLengths] = this.preEncode.forward(x, lengths);
    } else {
      x = this.preEncode.forward(x);
      outLengths = lengths;
    }

    const effectiveCache = cache ?? new Array(this.layers.length).fill(null);

    let posEmb: MxArray | null = null;
    if (this.posEnc !== null) {
      const offset = effectiveCache[0]?.offset ?? 0;
      [x, posEmb] = this.posEnc.forward(x, offset);
    }

    for (let i = 0; i < this.layers.length; i++) {
      x = this.layers[i].forward(x, posEmb, null, effectiveCache[i]);
    }

    return [x, outLengths];
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    if (this.preEncode instanceof DwStridingSubsampling) {
      this.preEncode.loadWeights(weights, `${prefix}.pre_encode`);
    } else {
      (this.preEncode as Linear).loadWeights(weights, `${prefix}.pre_encode`);
    }

    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].loadWeights(weights, `${prefix}.layers.${i}`);
    }

    // pos_enc has no learnable weights (pe is computed; pos_bias_{u,v} live in attention)
  }
}
