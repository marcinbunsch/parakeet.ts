import { MxArray } from '@mlx-node/core';
import { Module, Linear, softmax } from './nn.js';
function s(...dims) {
    return BigInt64Array.from(dims.map(BigInt));
}
// ---------------------------------------------------------------------------
// Scaled dot-product attention (standard)
// ---------------------------------------------------------------------------
function scaledDotProductAttention(q, // [batch, heads, q_seq, head_dim]
k, // [batch, heads, k_seq, head_dim]
v, // [batch, heads, k_seq, head_dim]
scale, mask) {
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
    linearQ;
    linearK;
    linearV;
    linearOut;
    nHead;
    headDim;
    scale;
    constructor(nHead, nFeat, bias = true) {
        super();
        this.nHead = nHead;
        this.headDim = Math.floor(nFeat / nHead);
        this.scale = Math.pow(this.headDim, -0.5);
        this.linearQ = new Linear(nFeat, nFeat, bias);
        this.linearK = new Linear(nFeat, nFeat, bias);
        this.linearV = new Linear(nFeat, nFeat, bias);
        this.linearOut = new Linear(nFeat, nFeat, bias);
    }
    forward(q, k, v, posEmb, mask, cache) {
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
    loadWeights(weights, prefix) {
        this.linearQ.loadWeights(weights, `${prefix}.linear_q`);
        this.linearK.loadWeights(weights, `${prefix}.linear_k`);
        this.linearV.loadWeights(weights, `${prefix}.linear_v`);
        this.linearOut.loadWeights(weights, `${prefix}.linear_out`);
    }
    /** Copy the loaded parameters from another module of the same shape. */
    copyWeightsFrom(other) {
        this.linearQ.weight = other.linearQ.weight;
        this.linearQ.bias = other.linearQ.bias;
        this.linearK.weight = other.linearK.weight;
        this.linearK.bias = other.linearK.bias;
        this.linearV.weight = other.linearV.weight;
        this.linearV.bias = other.linearV.bias;
        this.linearOut.weight = other.linearOut.weight;
        this.linearOut.bias = other.linearOut.bias;
    }
}
// ---------------------------------------------------------------------------
// Relative-position multi-head attention
// ---------------------------------------------------------------------------
export class RelPositionMultiHeadAttention extends Module {
    linearQ;
    linearK;
    linearV;
    linearOut;
    linearPos;
    posBiasU; // [nHead, headDim]
    posBiasV; // [nHead, headDim]
    nHead;
    headDim;
    scale;
    constructor(nHead, nFeat, bias = true) {
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
    relShift(x) {
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
    forward(q, k, v, posEmb, mask, cache) {
        if (posEmb === null)
            throw new Error('posEmb is required for RelPositionMultiHeadAttention');
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
    loadWeights(weights, prefix) {
        this.linearQ.loadWeights(weights, `${prefix}.linear_q`);
        this.linearK.loadWeights(weights, `${prefix}.linear_k`);
        this.linearV.loadWeights(weights, `${prefix}.linear_v`);
        this.linearOut.loadWeights(weights, `${prefix}.linear_out`);
        this.linearPos.loadWeights(weights, `${prefix}.linear_pos`);
        const entry = weights.get(`${prefix}.pos_bias_u`);
        if (entry) {
            this.posBiasU = MxArray.fromFloat32(entry.data, BigInt64Array.from(entry.shape.map(BigInt)));
        }
        else {
            this.posBiasU = MxArray.zeros(s(this.nHead, this.headDim), null);
        }
        const entryV = weights.get(`${prefix}.pos_bias_v`);
        if (entryV) {
            this.posBiasV = MxArray.fromFloat32(entryV.data, BigInt64Array.from(entryV.shape.map(BigInt)));
        }
        else {
            this.posBiasV = MxArray.zeros(s(this.nHead, this.headDim), null);
        }
    }
    /** Copy the loaded parameters from another rel-pos module of the same shape. */
    copyWeightsFrom(other) {
        this.linearQ.weight = other.linearQ.weight;
        this.linearQ.bias = other.linearQ.bias;
        this.linearK.weight = other.linearK.weight;
        this.linearK.bias = other.linearK.bias;
        this.linearV.weight = other.linearV.weight;
        this.linearV.bias = other.linearV.bias;
        this.linearOut.weight = other.linearOut.weight;
        this.linearOut.bias = other.linearOut.bias;
        this.linearPos.weight = other.linearPos.weight;
        this.linearPos.bias = other.linearPos.bias;
        this.posBiasU = other.posBiasU;
        this.posBiasV = other.posBiasV;
    }
}
// ---------------------------------------------------------------------------
// Local relative-position attention
// Uses standard attention with a local window mask (no custom Metal kernel)
// ---------------------------------------------------------------------------
export class RelPositionMultiHeadLocalAttention extends RelPositionMultiHeadAttention {
    contextSize;
    constructor(nHead, nFeat, bias = true, contextSize = [256, 256]) {
        super(nHead, nFeat, bias);
        this.contextSize = contextSize;
    }
    forward(q, k, v, posEmb, mask, cache) {
        if (posEmb === null)
            throw new Error('posEmb is required');
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
        const kOffset = kLen - qSeq; // queries align to the right of the (cached) keys
        // Content scores: [batch, heads, qSeq, kLen]
        const Qu = Qr.add(this.posBiasU.expandDims(1));
        const Qv = Qr.add(this.posBiasV.expandDims(1));
        const matrixAC = Qu.matmul(Kr.transpose(new Int32Array([0, 1, 3, 2])));
        // Raw position scores against the relative-position buffer:
        //   [batch, heads, qSeq, posLen], posLen = leftCtx + rightCtx + 1.
        // pe row r encodes relative distance (leftCtx - r).
        const matrixBDraw = Qv.matmul(Pr.transpose(new Int32Array([0, 1, 3, 2])));
        // For each (query qi, key ki), select the pe row for distance
        //   d = (kOffset + qi) - ki   ->   r = leftCtx - d
        // and mask keys outside the [qi-leftCtx, qi+rightCtx] window. r in
        // [0, posLen) is exactly the in-window condition, so it drives both.
        const block = new Int32Array(qSeq * kLen);
        const maskData = new Float32Array(qSeq * kLen);
        for (let qi = 0; qi < qSeq; qi++) {
            for (let ki = 0; ki < kLen; ki++) {
                const r = leftCtx - kOffset - qi + ki;
                const flat = qi * kLen + ki;
                if (r >= 0 && r < posLen) {
                    block[flat] = r;
                    maskData[flat] = 0.0;
                }
                else {
                    block[flat] = 0; // clamped; masked out below
                    maskData[flat] = -Infinity;
                }
            }
        }
        // Gather the aligned position scores: [batch, heads, qSeq, kLen].
        // takeAlongAxis requires indices to match the source rank and non-gathered
        // dims, so replicate the per-(qi,ki) block across batch and heads.
        const idxData = new Int32Array(batch * this.nHead * qSeq * kLen);
        for (let i = 0; i < batch * this.nHead; i++) {
            idxData.set(block, i * qSeq * kLen);
        }
        const idx = MxArray.fromInt32(idxData, s(batch, this.nHead, qSeq, kLen));
        const matrixBD = matrixBDraw.takeAlongAxis(idx, 3);
        let scores = matrixAC.add(matrixBD).mulScalar(this.scale);
        const localMask = MxArray.fromFloat32(maskData, s(1, 1, qSeq, kLen));
        scores = scores.add(localMask);
        if (mask !== null) {
            scores = scores.add(mask);
        }
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
    dModel;
    maxLen;
    scaleInput;
    scale;
    pe; // [1, 2*maxLen-1, dModel]
    constructor(dModel, maxLen = 5000, scaleInput = true) {
        super();
        this.dModel = dModel;
        this.maxLen = maxLen;
        this.scaleInput = scaleInput;
        this.scale = scaleInput ? Math.sqrt(dModel) : 1.0;
        this.calculatePE();
    }
    calculatePE() {
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
    forward(x, offset = 0) {
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
    loadWeights(_weights, _prefix) {
        // PE is computed, not learned (though pos_bias_u/v in attention are learned)
    }
}
export class LocalRelPositionalEncoding extends Module {
    dModel;
    maxLen;
    scaleInput;
    scale;
    leftContext;
    rightContext;
    pe; // [1, leftCtx+rightCtx+1, dModel]
    constructor(dModel, maxLen = 5000, scaleInput = true, contextSize = [256, 256]) {
        super();
        this.dModel = dModel;
        this.maxLen = maxLen;
        this.scaleInput = scaleInput;
        this.scale = scaleInput ? Math.sqrt(dModel) : 1.0;
        [this.leftContext, this.rightContext] = contextSize;
        this.calculatePE();
    }
    calculatePE() {
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
    forward(x, _offset = 0) {
        const scaledX = x.mulScalar(this.scale);
        return [scaledX, this.pe];
    }
    loadWeights(_weights, _prefix) { }
}
//# sourceMappingURL=attention.js.map