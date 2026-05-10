/**
 * Neural network primitives implemented on top of @mlx-node/core MxArray.
 *
 * Design notes:
 *  - Each layer stores its weights as MxArray fields.
 *  - Conv1d and Conv2d are implemented via im2col (take + reshape + matmul),
 *    keeping all computation on-device.
 *  - LSTM is implemented with manual gate arithmetic so no native LSTM kernel
 *    is required.
 *  - Weight loading uses a flat Map<string, {data, shape}> that mirrors the
 *    safetensors key structure produced by mlx-python's tree_flatten.
 */

import { MxArray } from '@mlx-node/core';

// ---------------------------------------------------------------------------
// Weight map types
// ---------------------------------------------------------------------------

export interface TensorEntry {
  data: Float32Array;
  shape: number[];
}
export type WeightMap = Map<string, TensorEntry>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function s(...dims: number[]): BigInt64Array {
  return BigInt64Array.from(dims.map(BigInt));
}

function loadParam(weights: WeightMap, key: string): MxArray {
  const entry = weights.get(key);
  if (!entry) throw new Error(`Weight not found: "${key}"`);
  return MxArray.fromFloat32(entry.data, s(...entry.shape));
}

function tryLoadParam(weights: WeightMap, key: string): MxArray | null {
  const entry = weights.get(key);
  if (!entry) return null;
  return MxArray.fromFloat32(entry.data, s(...entry.shape));
}

// ---------------------------------------------------------------------------
// Module base
// ---------------------------------------------------------------------------

export abstract class Module {
  abstract loadWeights(weights: WeightMap, prefix: string): void;
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

export class Linear extends Module {
  weight!: MxArray; // [out, in]
  bias: MxArray | null = null;

  constructor(public inFeatures: number, public outFeatures: number, public useBias = true) {
    super();
  }

  forward(x: MxArray): MxArray {
    // x: [..., in] → [..., out]
    const out = x.matmul(this.weight.transpose(new Int32Array([1, 0])));
    return this.bias ? out.add(this.bias) : out;
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.weight = loadParam(weights, `${prefix}.weight`);
    if (this.useBias) {
      this.bias = tryLoadParam(weights, `${prefix}.bias`);
    }
  }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

export class Embedding extends Module {
  weight!: MxArray; // [num_embeddings, embedding_dim]

  constructor(public numEmbeddings: number, public embeddingDim: number) {
    super();
  }

  forward(indices: MxArray): MxArray {
    // indices: [...] (int32) → [..., embedding_dim]
    return this.weight.take(indices, 0);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.weight = loadParam(weights, `${prefix}.weight`);
  }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

export class LayerNorm extends Module {
  weight!: MxArray; // [features]
  bias!: MxArray;   // [features]

  constructor(public features: number, public eps = 1e-5) {
    super();
  }

  forward(x: MxArray): MxArray {
    // x: [..., features]
    const ndim = x.ndim();
    const axes = new Int32Array([ndim - 1]);
    const mean = x.mean(axes, true);
    const variance = x.var(axes, true);
    const norm = x.sub(mean).div(variance.addScalar(this.eps).sqrt());
    return norm.mul(this.weight).add(this.bias);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.weight = loadParam(weights, `${prefix}.weight`);
    this.bias = loadParam(weights, `${prefix}.bias`);
  }
}

// ---------------------------------------------------------------------------
// BatchNorm (inference mode: uses running stats)
// ---------------------------------------------------------------------------

export class BatchNorm extends Module {
  weight!: MxArray;        // [features]
  bias!: MxArray;          // [features]
  runningMean!: MxArray;   // [features]
  runningVar!: MxArray;    // [features]

  constructor(public features: number, public eps = 1e-5) {
    super();
  }

  forward(x: MxArray): MxArray {
    // x: [batch, seq, features] (after Conv1d in Conformer)
    const norm = x.sub(this.runningMean).div(this.runningVar.addScalar(this.eps).sqrt());
    return norm.mul(this.weight).add(this.bias);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.weight = loadParam(weights, `${prefix}.weight`);
    this.bias = loadParam(weights, `${prefix}.bias`);
    this.runningMean = loadParam(weights, `${prefix}.running_mean`);
    this.runningVar = loadParam(weights, `${prefix}.running_var`);
  }
}

// ---------------------------------------------------------------------------
// Conv1d — implemented via im2col + matmul, all on-device
//
// MLX weight layout for Conv1d: [out_ch, kernel_size, in_ch]
// Input layout: [batch, seq, in_ch]
// Output layout: [batch, out_seq, out_ch]
// ---------------------------------------------------------------------------

export class Conv1d extends Module {
  weight!: MxArray; // [out_ch, kernel_size, in_ch]
  bias: MxArray | null = null;

  constructor(
    public inChannels: number,
    public outChannels: number,
    public kernelSize: number,
    public stride = 1,
    public padding = 0,
    public groups = 1,
    public useBias = true,
  ) {
    super();
  }

  forward(x: MxArray): MxArray {
    return conv1d(x, this.weight, this.bias, this.stride, this.padding, this.groups);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.weight = loadParam(weights, `${prefix}.weight`);
    if (this.useBias) {
      this.bias = tryLoadParam(weights, `${prefix}.bias`);
    }
  }
}

/** Im2col-based Conv1d. weight: [out_ch, kernel_size, in_ch] */
export function conv1d(
  x: MxArray,
  weight: MxArray,
  bias: MxArray | null,
  stride: number,
  padding: number,
  groups: number,
): MxArray {
  const xShape = x.shape();
  const batch = Number(xShape[0]);
  const seq = Number(xShape[1]);
  const inCh = Number(xShape[2]);

  const wShape = weight.shape();
  const outCh = Number(wShape[0]);
  const kernelSize = Number(wShape[1]);

  // Pad the sequence dimension
  let xPad = x;
  if (padding > 0) {
    xPad = x.pad(new Int32Array([0, 0, padding, padding, 0, 0]), 0.0);
  }
  const paddedSeq = Number(xPad.shape()[1]);
  const outLen = Math.floor((paddedSeq - kernelSize) / stride) + 1;

  if (groups === 1) {
    // Standard conv: build gather indices [outLen * kernelSize]
    const idxData = new Int32Array(outLen * kernelSize);
    for (let t = 0; t < outLen; t++) {
      for (let k = 0; k < kernelSize; k++) {
        idxData[t * kernelSize + k] = t * stride + k;
      }
    }
    const idx = MxArray.fromInt32(idxData, s(outLen * kernelSize));

    // Gather along seq axis: [batch, outLen*kernelSize, inCh]
    const gathered = xPad.take(idx, 1);

    // Reshape to [batch, outLen, kernelSize, inCh]
    const unfolded4d = gathered.reshape(s(batch, outLen, kernelSize, inCh));

    // Reshape to [batch, outLen, kernelSize * inCh]
    const unfolded = unfolded4d.reshape(s(batch, outLen, kernelSize * inCh));

    // weight: [outCh, kernelSize, inCh] → transpose → [kernelSize, inCh, outCh]
    //       → reshape → [kernelSize*inCh, outCh]
    const wT = weight
      .transpose(new Int32Array([1, 2, 0]))
      .reshape(s(kernelSize * inCh, outCh));

    let out = unfolded.matmul(wT); // [batch, outLen, outCh]
    if (bias) out = out.add(bias);
    return out;
  }

  if (groups === inCh && outCh === inCh) {
    // Depthwise conv: each output channel depends on exactly one input channel
    const idxData = new Int32Array(outLen * kernelSize);
    for (let t = 0; t < outLen; t++) {
      for (let k = 0; k < kernelSize; k++) {
        idxData[t * kernelSize + k] = t * stride + k;
      }
    }
    const idx = MxArray.fromInt32(idxData, s(outLen * kernelSize));

    // Gather: [batch, outLen*kernelSize, inCh]
    const gathered = xPad.take(idx, 1);
    // Reshape: [batch, outLen, kernelSize, inCh]
    const unfolded4d = gathered.reshape(s(batch, outLen, kernelSize, inCh));

    // weight: [outCh, kernelSize, 1] → squeeze last dim → [inCh, kernelSize]
    //   then transpose → [kernelSize, inCh]
    const wSquzd = weight.squeeze(new Int32Array([2]));       // [inCh, kernelSize]
    const wT = wSquzd.transpose(new Int32Array([1, 0]));       // [kernelSize, inCh]

    // Element-wise multiply + sum over kernel axis
    const weighted = unfolded4d.mul(wT);            // [batch, outLen, kernelSize, inCh]
    let out = weighted.sum(new Int32Array([2]));     // [batch, outLen, inCh]
    if (bias) out = out.add(bias);
    return out;
  }

  throw new Error(`conv1d: groups=${groups} not supported (inCh=${inCh}, outCh=${outCh})`);
}

// ---------------------------------------------------------------------------
// Conv2d — same im2col strategy, in 2D
//
// MLX weight layout: [out_ch, kH, kW, in_ch]
// Input layout (MLX NHW C): [batch, H, W, in_ch]
// ---------------------------------------------------------------------------

export class Conv2d extends Module {
  weight!: MxArray;
  bias: MxArray | null = null;

  constructor(
    public inChannels: number,
    public outChannels: number,
    public kernelSize: number,
    public stride = 1,
    public padding = 0,
    public groups = 1,
    public useBias = true,
  ) {
    super();
  }

  forward(x: MxArray): MxArray {
    return conv2d(x, this.weight, this.bias, this.stride, this.padding, this.groups);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.weight = loadParam(weights, `${prefix}.weight`);
    if (this.useBias) {
      this.bias = tryLoadParam(weights, `${prefix}.bias`);
    }
  }
}

/** Im2col-based Conv2d. weight: [out_ch, kH, kW, in_ch], input: [batch, H, W, in_ch] */
export function conv2d(
  x: MxArray,
  weight: MxArray,
  bias: MxArray | null,
  stride: number,
  padding: number,
  groups: number,
): MxArray {
  const xShape = x.shape();
  const batch = Number(xShape[0]);
  const H = Number(xShape[1]);
  const W = Number(xShape[2]);
  const inCh = Number(xShape[3]);

  const wShape = weight.shape();
  const outCh = Number(wShape[0]);
  const kH = Number(wShape[1]);
  const kW = Number(wShape[2]);

  // Pad
  let xPad = x;
  if (padding > 0) {
    xPad = x.pad(new Int32Array([0, 0, padding, padding, padding, padding, 0, 0]), 0.0);
  }
  const pH = Number(xPad.shape()[1]);
  const pW = Number(xPad.shape()[2]);
  const outH = Math.floor((pH - kH) / stride) + 1;
  const outW = Math.floor((pW - kW) / stride) + 1;

  // Build gather indices for height and width dimensions
  const numPositions = outH * outW * kH * kW;
  const hIdx = new Int32Array(numPositions);
  const wIdx = new Int32Array(numPositions);

  let pos = 0;
  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      for (let kh = 0; kh < kH; kh++) {
        for (let kw = 0; kw < kW; kw++) {
          hIdx[pos] = oh * stride + kh;
          wIdx[pos] = ow * stride + kw;
          pos++;
        }
      }
    }
  }

  // We need to gather at (h, w) pairs. Build flat indices into the H*W grid.
  const hwIdx = new Int32Array(numPositions);
  for (let i = 0; i < numPositions; i++) {
    hwIdx[i] = hIdx[i] * pW + wIdx[i];
  }

  // Reshape x to [batch, H*W, inCh] then gather
  const xFlat = xPad.reshape(s(batch, pH * pW, inCh));
  const gathered = xFlat.take(MxArray.fromInt32(hwIdx, s(numPositions)), 1);
  // gathered: [batch, outH*outW*kH*kW, inCh]

  let out: MxArray;

  if (groups > 1) {
    // Depthwise conv2d: weight shape is [outCh, kH, kW, 1]
    // gathered: [batch, outH*outW*kH*kW, inCh]
    // → [batch, outH*outW, kH*kW, inCh] → transpose → [batch, outH*outW, inCh, kH*kW]
    const unfolded4d = gathered.reshape(s(batch, outH * outW, kH * kW, inCh));
    const unfoldedT = unfolded4d.transpose(new Int32Array([0, 1, 3, 2]));
    // weight: [outCh, kH, kW, 1] → [1, 1, outCh, kH*kW]
    const wBcast = weight.reshape(s(1, 1, outCh, kH * kW));
    // element-wise mul + sum over kernel dim → [batch, outH*outW, inCh]
    out = unfoldedT.mul(wBcast).sum(new Int32Array([3]));
  } else {
    // Standard conv2d
    // Reshape to [batch, outH*outW, kH*kW*inCh]
    const unfolded = gathered.reshape(s(batch, outH * outW, kH * kW * inCh));
    // weight: [outCh, kH, kW, inCh] → reshape → [outCh, kH*kW*inCh]
    //       → transpose → [kH*kW*inCh, outCh]
    const wFlat = weight.reshape(s(outCh, kH * kW * inCh));
    const wT = wFlat.transpose(new Int32Array([1, 0]));
    out = unfolded.matmul(wT); // [batch, outH*outW, outCh]
  }

  // Reshape to [batch, outH, outW, outCh]
  out = out.reshape(s(batch, outH, outW, outCh));

  if (bias) out = out.add(bias);
  return out;
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

export function sigmoid(x: MxArray): MxArray {
  // 1 / (1 + exp(-x))
  return x.negative().exp().addScalar(1.0).reciprocal();
}

export function silu(x: MxArray): MxArray {
  return x.mul(sigmoid(x));
}

export function relu(x: MxArray): MxArray {
  return x.maximum(MxArray.zeros(x.shape(), null));
}

export function tanh(x: MxArray): MxArray {
  return x.tanh();
}

export function glu(x: MxArray, axis: number): MxArray {
  // Split in half along axis, apply sigmoid to the second half
  const shape = x.shape();
  const dim = Number(shape[axis]);
  const half = dim / 2;

  const starts1 = new Array(shape.length).fill(0n) as bigint[];
  const stops1 = shape.map((d, i) => (i === axis ? BigInt(half) : d));
  const starts2 = new Array(shape.length).fill(0n) as bigint[];
  starts2[axis] = BigInt(half);
  const stops2 = shape.map(d => d);

  const a = x.slice(BigInt64Array.from(starts1), BigInt64Array.from(stops1));
  const b = x.slice(BigInt64Array.from(starts2), BigInt64Array.from(stops2));
  return a.mul(sigmoid(b));
}

export function softmax(x: MxArray, axis: number): MxArray {
  const maxVal = x.max(new Int32Array([axis]), true);
  const shifted = x.sub(maxVal);
  const expd = shifted.exp();
  const sumExp = expd.sum(new Int32Array([axis]), true);
  return expd.div(sumExp);
}

export function logSoftmax(x: MxArray, axis: number): MxArray {
  return x.logSoftmax(axis);
}

// ---------------------------------------------------------------------------
// LSTM — manual gate implementation
//
// MLX LSTM weight keys (per layer):
//   lstm.{i}.Wx: [4*hidden, input]  (gates ordered: i, f, g, o)
//   lstm.{i}.Wh: [4*hidden, hidden]
//   lstm.{i}.b:  [4*hidden]
// ---------------------------------------------------------------------------

export class LSTMLayer extends Module {
  Wx!: MxArray; // [4*hidden, input]
  Wh!: MxArray; // [4*hidden, hidden]
  b: MxArray | null = null;

  constructor(public inputSize: number, public hiddenSize: number, public useBias = true) {
    super();
  }

  /** x: [seq, batch, input], h: [batch, hidden] | null, c: [batch, hidden] | null */
  forward(
    x: MxArray,
    h: MxArray | null,
    c: MxArray | null,
  ): [MxArray, MxArray] {
    // x: [seq, batch, input]
    const xShape = x.shape();
    const seqLen = Number(xShape[0]);
    const batch = Number(xShape[1]);
    const H = this.hiddenSize;

    // Initialize h, c if null
    if (h === null) h = MxArray.zeros(s(batch, H), null);
    if (c === null) c = MxArray.zeros(s(batch, H), null);

    // Precompute input projections for all time steps: [seq, batch, 4H]
    // x: [seq, batch, input] → reshape [seq*batch, input]
    const xFlat = x.reshape(s(seqLen * batch, this.inputSize));
    // Wx: [4H, input] → xFlat @ Wx.T → [seq*batch, 4H]
    let xProj = xFlat.matmul(this.Wx.transpose(new Int32Array([1, 0])));
    if (this.b) xProj = xProj.add(this.b);
    // Reshape: [seq, batch, 4H]
    const xProjSeq = xProj.reshape(s(seqLen, batch, 4 * H));

    const allH: MxArray[] = [];
    const allC: MxArray[] = [];

    for (let t = 0; t < seqLen; t++) {
      // Get x projection for this step: [batch, 4H]
      const xt = xProjSeq.slice(s(t, 0, 0), s(t + 1, batch, 4 * H)).squeeze(new Int32Array([0]));

      // Hidden projection: h @ Wh.T → [batch, 4H]
      const hProj = h.matmul(this.Wh.transpose(new Int32Array([1, 0])));

      const gates = xt.add(hProj); // [batch, 4H]

      // Split gates: i, f, g, o each [batch, H]
      const gate = (idx: number) =>
        gates.slice(s(0, idx * H), s(batch, (idx + 1) * H));

      const gi = sigmoid(gate(0)); // input gate
      const gf = sigmoid(gate(1)); // forget gate
      const gg = gate(2).tanh();   // cell gate
      const go = sigmoid(gate(3)); // output gate

      c = gf.mul(c!).add(gi.mul(gg)); // new cell
      h = go.mul(c.tanh());           // new hidden

      // Force evaluation to avoid unbounded graph growth
      h.eval();
      c.eval();

      allH.push(h);
      allC.push(c);
    }

    // Stack: [seq, batch, H]
    const hStack = MxArray.stack(allH, 0);
    // We return all hidden states and the last c
    return [hStack, c];
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.Wx = loadParam(weights, `${prefix}.Wx`);
    this.Wh = loadParam(weights, `${prefix}.Wh`);
    if (this.useBias) {
      this.b = tryLoadParam(weights, `${prefix}.bias`) ?? tryLoadParam(weights, `${prefix}.b`);
    }
  }
}

export class LSTM extends Module {
  layers: LSTMLayer[];

  constructor(
    inputSize: number,
    hiddenSize: number,
    numLayers: number,
    useBias = true,
  ) {
    super();
    this.layers = [];
    for (let i = 0; i < numLayers; i++) {
      this.layers.push(new LSTMLayer(i === 0 ? inputSize : hiddenSize, hiddenSize, useBias));
    }
  }

  /**
   * x: [batch, seq, input] (batch_first=true as in Python code)
   * Returns [output: [batch, seq, hidden], [h: [numLayers, batch, H], c: [numLayers, batch, H]]]
   */
  forward(
    x: MxArray,
    hc: [MxArray, MxArray] | null = null,
  ): [MxArray, [MxArray, MxArray]] {
    // x: [batch, seq, input] → [seq, batch, input]
    let current = x.transpose(new Int32Array([1, 0, 2]));

    const xShape = x.shape();
    const batch = Number(xShape[0]);
    const H = this.layers[0].hiddenSize;

    const nextHList: MxArray[] = [];
    const nextCList: MxArray[] = [];

    for (let i = 0; i < this.layers.length; i++) {
      const h = hc ? hc[0].slice(s(i, 0, 0), s(i + 1, batch, H)).squeeze(new Int32Array([0])) : null;
      const c = hc ? hc[1].slice(s(i, 0, 0), s(i + 1, batch, H)).squeeze(new Int32Array([0])) : null;

      const [allH, lastC] = this.layers[i].forward(current, h, c);
      const seqLen = Number(allH.shape()[0]);
      const lastH = allH.slice(s(seqLen - 1, 0, 0), allH.shape()).squeeze(new Int32Array([0]));

      current = allH; // pass all hidden states to next layer
      nextHList.push(lastH);
      nextCList.push(lastC);
    }

    // output: [seq, batch, H] → [batch, seq, H]
    const output = current.transpose(new Int32Array([1, 0, 2]));
    const finalH = MxArray.stack(nextHList, 0); // [numLayers, batch, H]
    const finalC = MxArray.stack(nextCList, 0); // [numLayers, batch, H]

    return [output, [finalH, finalC]];
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].loadWeights(weights, `${prefix}.${i}`);
    }
  }
}
