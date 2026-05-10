import { MxArray } from '@mlx-node/core';
import { Module, Linear, WeightMap, softmax, sigmoid } from './nn.js';
import { ConformerCache } from './cache.js';

function s(...dims: number[]): BigInt64Array {
  return BigInt64Array.from(dims.map(BigInt));
}

// ---------------------------------------------------------------------------
// Scaled dot-product attention (standard)
// ---------------------------------------------------------------------------

function scaledDotProductAttention(
  q: MxArray, // [batch, heads, q_seq, head_dim]
  k: MxArray, // [batch, heads, k_seq, head_dim]
  v: MxArray, // [batch, heads, k_seq, head_dim]
  scale: number,
  mask: MxArray | null,
): MxArray {
  // scores: [batch, heads, q_seq, k_seq]
  let scores = q.matmul(k.transpose(new Int32Array([0, 1, 3, 2]))).mulScalar(scale);

  if (mask !== null) {
    // mask is additive (0 or -inf) or boolean — add directly
    scores = scores.add(mask);
  }

  const attn = softmax(scores, 3);
  return attn.matmul(v); // [batch, heads, q_seq, head_dim]
}

// ---------------------------------------------------------------------------
// Standard multi-head attention
// ---------------------------------------------------------------------------

export class MultiHeadAttention extends Module {
  linearQ: Linear;
  linearK: Linear;
  linearV: Linear;
  linearOut: Linear;

  readonly nHead: number;
  readonly headDim: number;
  readonly scale: number;

  constructor(nHead: number, nFeat: number, bias = true) {
    super();
    this.nHead = nHead;
    this.headDim = Math.floor(nFeat / nHead);
    this.scale = Math.pow(this.headDim, -0.5);

    this.linearQ = new Linear(nFeat, nFeat, bias);
    this.linearK = new Linear(nFeat, nFeat, bias);
    this.linearV = new Linear(nFeat, nFeat, bias);
    this.linearOut = new Linear(nFeat, nFeat, bias);
  }

  forward(
    q: MxArray,
    k: MxArray,
    v: MxArray,
    posEmb: MxArray | null,
    mask: MxArray | null,
    cache: ConformerCache | null,
  ): MxArray {
    const Qp = this.linearQ.forward(q);
    const Kp = this.linearK.forward(k);
    const Vp = this.linearV.forward(v);

    const qShape = Qp.shape();
    const batch = Number(qShape[0]);
    const qSeq = Number(qShape[1]);
    const kSeq = Number(Kp.shape()[1]);

    const Qr = Qp.reshape(s(batch, qSeq, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));
    let Kr = Kp.reshape(s(batch, kSeq, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));
    let Vr = Vp.reshape(s(batch, kSeq, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));

    if (cache) {
      [Kr, Vr] = cache.updateAndFetchKV(Kr, Vr);
    }

    const o = scaledDotProductAttention(Qr, Kr, Vr, this.scale, mask);
    const out = o.transpose(new Int32Array([0, 2, 1, 3])).reshape(s(batch, qSeq, this.headDim * this.nHead));
    return this.linearOut.forward(out);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.linearQ.loadWeights(weights, `${prefix}.linear_q`);
    this.linearK.loadWeights(weights, `${prefix}.linear_k`);
    this.linearV.loadWeights(weights, `${prefix}.linear_v`);
    this.linearOut.loadWeights(weights, `${prefix}.linear_out`);
  }
}

// ---------------------------------------------------------------------------
// Relative-position multi-head attention
// ---------------------------------------------------------------------------

export class RelPositionMultiHeadAttention extends Module {
  linearQ: Linear;
  linearK: Linear;
  linearV: Linear;
  linearOut: Linear;
  linearPos: Linear;

  posBiasU!: MxArray; // [nHead, headDim]
  posBiasV!: MxArray; // [nHead, headDim]

  readonly nHead: number;
  readonly headDim: number;
  readonly scale: number;

  constructor(nHead: number, nFeat: number, bias = true) {
    super();
    this.nHead = nHead;
    this.headDim = Math.floor(nFeat / nHead);
    this.scale = Math.pow(this.headDim, -0.5);

    this.linearQ = new Linear(nFeat, nFeat, bias);
    this.linearK = new Linear(nFeat, nFeat, bias);
    this.linearV = new Linear(nFeat, nFeat, bias);
    this.linearOut = new Linear(nFeat, nFeat, bias);
    this.linearPos = new Linear(nFeat, nFeat, false);
  }

  private relShift(x: MxArray): MxArray {
    // x: [B, H, Tq, posLen]
    const xShape = x.shape();
    const B = Number(xShape[0]);
    const H = Number(xShape[1]);
    const Tq = Number(xShape[2]);
    const posLen = Number(xShape[3]);

    // Pad: [B, H, Tq, posLen+1]
    const padded = x.pad(new Int32Array([0, 0, 0, 0, 0, 0, 1, 0]), 0.0);
    // Reshape: [B, H, posLen+1, Tq]
    const reshaped = padded.reshape(s(B, H, posLen + 1, Tq));
    // Slice off first row: [B, H, posLen, Tq]
    const sliced = reshaped.slice(s(0, 0, 1, 0), s(B, H, posLen + 1, Tq));
    // Reshape back: [B, H, Tq, posLen]
    return sliced.reshape(s(B, H, Tq, posLen));
  }

  forward(
    q: MxArray,
    k: MxArray,
    v: MxArray,
    posEmb: MxArray | null,
    mask: MxArray | null,
    cache: ConformerCache | null,
  ): MxArray {
    if (posEmb === null) throw new Error('posEmb is required for RelPositionMultiHeadAttention');

    const Qp = this.linearQ.forward(q);
    const Kp = this.linearK.forward(k);
    const Vp = this.linearV.forward(v);
    const P = this.linearPos.forward(posEmb);

    const qShape = Qp.shape();
    const batch = Number(qShape[0]);
    const qSeq = Number(qShape[1]);
    const kSeq = Number(Kp.shape()[1]);
    const pBatch = Number(P.shape()[0]);
    const posLen = Number(P.shape()[1]);

    // posEmb broadcast if batch > 1
    const Pb = pBatch === 1 && batch > 1
      ? P.broadcastTo(s(batch, posLen, Number(P.shape()[2])))
      : P;

    const Qr = Qp.reshape(s(batch, qSeq, this.nHead, this.headDim));
    const Qu = Qr.add(this.posBiasU).transpose(new Int32Array([0, 2, 1, 3]));
    const Qv = Qr.add(this.posBiasV).transpose(new Int32Array([0, 2, 1, 3]));

    let Kr = Kp.reshape(s(batch, kSeq, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));
    let Vr = Vp.reshape(s(batch, kSeq, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));
    const Pr = Pb.reshape(s(batch, posLen, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));

    if (cache) {
      [Kr, Vr] = cache.updateAndFetchKV(Kr, Vr);
    }

    const kLen = Number(Kr.shape()[2]);

    // Content-based scores: [batch, heads, qSeq, kLen]
    const matrixAC = Qu.matmul(Kr.transpose(new Int32Array([0, 1, 3, 2]))).mulScalar(this.scale);

    // Position-based scores: [batch, heads, qSeq, posLen] → rel_shift → [batch, heads, qSeq, kLen]
    let matrixBD = Qv.matmul(Pr.transpose(new Int32Array([0, 1, 3, 2])));
    matrixBD = this.relShift(matrixBD);
    matrixBD = matrixBD
      .slice(s(0, 0, 0, 0), s(batch, this.nHead, qSeq, kLen))
      .mulScalar(this.scale);

    let scores = matrixAC.add(matrixBD);

    if (mask !== null) {
      scores = scores.add(mask);
    }

    const attn = softmax(scores, 3);
    const o = attn.matmul(Vr);
    const out = o.transpose(new Int32Array([0, 2, 1, 3])).reshape(s(batch, qSeq, -1 /* nHead*headDim */));
    // Note: -1 not supported by reshape; use actual size:
    const outReshaped = o.transpose(new Int32Array([0, 2, 1, 3])).reshape(s(batch, qSeq, this.nHead * this.headDim));
    return this.linearOut.forward(outReshaped);
  }

  loadWeights(weights: WeightMap, prefix: string): void {
    this.linearQ.loadWeights(weights, `${prefix}.linear_q`);
    this.linearK.loadWeights(weights, `${prefix}.linear_k`);
    this.linearV.loadWeights(weights, `${prefix}.linear_v`);
    this.linearOut.loadWeights(weights, `${prefix}.linear_out`);
    this.linearPos.loadWeights(weights, `${prefix}.linear_pos`);

    const entry = weights.get(`${prefix}.pos_bias_u`);
    if (entry) {
      this.posBiasU = MxArray.fromFloat32(entry.data, BigInt64Array.from(entry.shape.map(BigInt)));
    } else {
      this.posBiasU = MxArray.zeros(s(this.nHead, this.headDim), null);
    }
    const entryV = weights.get(`${prefix}.pos_bias_v`);
    if (entryV) {
      this.posBiasV = MxArray.fromFloat32(entryV.data, BigInt64Array.from(entryV.shape.map(BigInt)));
    } else {
      this.posBiasV = MxArray.zeros(s(this.nHead, this.headDim), null);
    }
  }
}

// ---------------------------------------------------------------------------
// Local relative-position attention
// Uses standard attention with a local window mask (no custom Metal kernel)
// ---------------------------------------------------------------------------

export class RelPositionMultiHeadLocalAttention extends RelPositionMultiHeadAttention {
  readonly contextSize: [number, number];

  constructor(
    nHead: number,
    nFeat: number,
    bias = true,
    contextSize: [number, number] = [256, 256],
  ) {
    super(nHead, nFeat, bias);
    this.contextSize = contextSize;
  }

  forward(
    q: MxArray,
    k: MxArray,
    v: MxArray,
    posEmb: MxArray | null,
    mask: MxArray | null,
    cache: ConformerCache | null,
  ): MxArray {
    if (posEmb === null) throw new Error('posEmb is required');

    const Qp = this.linearQ.forward(q);
    const Kp = this.linearK.forward(k);
    const Vp = this.linearV.forward(v);
    const P = this.linearPos.forward(posEmb);

    const qShape = Qp.shape();
    const batch = Number(qShape[0]);
    const qSeq = Number(qShape[1]);
    const kSeq = Number(Kp.shape()[1]);
    const posLen = Number(P.shape()[1]);

    const Pb = Number(P.shape()[0]) === 1 && batch > 1
      ? P.broadcastTo(s(batch, posLen, Number(P.shape()[2])))
      : P;

    const Qr = Qp.reshape(s(batch, qSeq, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));
    let Kr = Kp.reshape(s(batch, kSeq, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));
    let Vr = Vp.reshape(s(batch, kSeq, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));
    const Pr = Pb.reshape(s(batch, posLen, this.nHead, this.headDim)).transpose(new Int32Array([0, 2, 1, 3]));

    if (cache) {
      [Kr, Vr] = cache.updateAndFetchKV(Kr, Vr);
    }

    const kLen = Number(Kr.shape()[2]);
    const [leftCtx, rightCtx] = this.contextSize;

    // Content scores
    const Qu = Qr.add(this.posBiasU.expandDims(1));
    const Qv = Qr.add(this.posBiasV.expandDims(1));

    const matrixAC = Qu.matmul(Kr.transpose(new Int32Array([0, 1, 3, 2]))).mulScalar(this.scale);

    // Position scores — simplified: use relative positions from posEmb
    const matrixBD = Qv.matmul(Pr.transpose(new Int32Array([0, 1, 3, 2]))).mulScalar(this.scale);

    let scores = matrixAC; // start with content scores

    // Build local attention mask — positions outside [t-left, t+right] get -inf
    // We build a [1, 1, qSeq, kLen] mask
    const maskData = new Float32Array(qSeq * kLen).fill(-Infinity);
    for (let qi = 0; qi < qSeq; qi++) {
      // Align q to the right side of k (k might be longer due to cache)
      const kOffset = kLen - qSeq;
      const ki_start = Math.max(0, kOffset + qi - leftCtx);
      const ki_end = Math.min(kLen, kOffset + qi + rightCtx + 1);
      for (let ki = ki_start; ki < ki_end; ki++) {
        maskData[qi * kLen + ki] = 0.0;
      }
    }
    const localMask = MxArray.fromFloat32(maskData, s(1, 1, qSeq, kLen));
    scores = scores.add(localMask);

    if (mask !== null) {
      scores = scores.add(mask);
    }

    // Add position bias where in range
    scores = scores.add(matrixBD);

    const attn = softmax(scores, 3);
    const o = attn.matmul(Vr);
    const outReshaped = o.transpose(new Int32Array([0, 2, 1, 3])).reshape(s(batch, qSeq, this.nHead * this.headDim));
    return this.linearOut.forward(outReshaped);
  }
}

// ---------------------------------------------------------------------------
// Positional encodings
// ---------------------------------------------------------------------------

export class RelPositionalEncoding extends Module {
  readonly dModel: number;
  maxLen: number;
  readonly scaleInput: boolean;
  readonly scale: number;
  pe!: MxArray; // [1, 2*maxLen-1, dModel]

  constructor(dModel: number, maxLen = 5000, scaleInput = true) {
    super();
    this.dModel = dModel;
    this.maxLen = maxLen;
    this.scaleInput = scaleInput;
    this.scale = scaleInput ? Math.sqrt(dModel) : 1.0;
    this.calculatePE();
  }

  calculatePE(): void {
    const totalLen = 2 * this.maxLen - 1;
    const pe = new Float32Array(totalLen * this.dModel);

    for (let i = 0; i < totalLen; i++) {
      const pos = this.maxLen - 1 - i; // from (maxLen-1) down to -(maxLen-1)
      for (let j = 0; j < this.dModel; j += 2) {
        const expFactor = Math.exp((-j * Math.log(10000.0)) / this.dModel);
        pe[i * this.dModel + j] = Math.sin(pos * expFactor);
        if (j + 1 < this.dModel) {
          pe[i * this.dModel + j + 1] = Math.cos(pos * expFactor);
        }
      }
    }

    this.pe = MxArray.fromFloat32(pe, BigInt64Array.from([1n, BigInt(totalLen), BigInt(this.dModel)]));
  }

  forward(x: MxArray, offset = 0): [MxArray, MxArray] {
    const inputLen = Number(x.shape()[1]) + offset;

    if (inputLen > this.maxLen) {
      this.maxLen = inputLen + 1;
      this.calculatePE();
    }

    const scaledX = x.mulScalar(this.scale);

    const bufferLen = Number(this.pe.shape()[1]);
    const startIdx = Math.floor(bufferLen / 2) - (inputLen - 1);
    const endIdx = Math.floor(bufferLen / 2) + (inputLen - 1) + 1;

    const posEmb = this.pe.slice(s(0, startIdx, 0), s(1, endIdx, this.dModel));
    return [scaledX, posEmb];
  }

  loadWeights(_weights: WeightMap, _prefix: string): void {
    // PE is computed, not learned (though pos_bias_u/v in attention are learned)
  }
}

export class LocalRelPositionalEncoding extends Module {
  readonly dModel: number;
  readonly maxLen: number;
  readonly scaleInput: boolean;
  readonly scale: number;
  readonly leftContext: number;
  readonly rightContext: number;
  pe!: MxArray; // [1, leftCtx+rightCtx+1, dModel]

  constructor(
    dModel: number,
    maxLen = 5000,
    scaleInput = true,
    contextSize: [number, number] = [256, 256],
  ) {
    super();
    this.dModel = dModel;
    this.maxLen = maxLen;
    this.scaleInput = scaleInput;
    this.scale = scaleInput ? Math.sqrt(dModel) : 1.0;
    [this.leftContext, this.rightContext] = contextSize;
    this.calculatePE();
  }

  calculatePE(): void {
    const totalLen = this.leftContext + this.rightContext + 1;
    const pe = new Float32Array(totalLen * this.dModel);

    for (let i = 0; i < totalLen; i++) {
      const pos = this.leftContext - i; // from leftCtx down to -rightCtx
      for (let j = 0; j < this.dModel; j += 2) {
        const expFactor = Math.exp((-j * Math.log(10000.0)) / this.dModel);
        pe[i * this.dModel + j] = Math.sin(pos * expFactor);
        if (j + 1 < this.dModel) {
          pe[i * this.dModel + j + 1] = Math.cos(pos * expFactor);
        }
      }
    }

    this.pe = MxArray.fromFloat32(pe, BigInt64Array.from([1n, BigInt(totalLen), BigInt(this.dModel)]));
  }

  forward(x: MxArray, _offset = 0): [MxArray, MxArray] {
    const scaledX = x.mulScalar(this.scale);
    return [scaledX, this.pe];
  }

  loadWeights(_weights: WeightMap, _prefix: string): void {}
}
