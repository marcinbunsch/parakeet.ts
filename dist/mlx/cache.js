import { MxArray } from '@mlx-node/core';
function s(...dims) {
    return BigInt64Array.from(dims.map(BigInt));
}
/**
 * Cache for a single Conformer layer — stores attention K/V and conv state.
 */
export class ConformerCache {
    keys = null;
    values = null;
    conv = null;
    offset = 0;
    step = 256;
    updateAndFetchKV(k, v) {
        if (this.keys === null || this.values === null) {
            // Allocate initial cache
            const kShape = k.shape();
            const batch = Number(kShape[0]);
            const heads = Number(kShape[1]);
            const headDim = Number(kShape[3]);
            const initLen = Math.ceil(Number(kShape[2]) / this.step) * this.step;
            this.keys = MxArray.zeros(s(batch, heads, initLen, headDim), null);
            this.values = MxArray.zeros(s(batch, heads, initLen, headDim), null);
        }
        const newSeq = Number(k.shape()[2]);
        // Grow if needed
        while (this.offset + newSeq > Number(this.keys.shape()[2])) {
            const batch = Number(this.keys.shape()[0]);
            const heads = Number(this.keys.shape()[1]);
            const headDim = Number(this.keys.shape()[3]);
            const extra = MxArray.zeros(s(batch, heads, this.step, headDim), null);
            this.keys = MxArray.concatenate(this.keys, extra, 2);
            this.values = MxArray.concatenate(this.values, extra, 2);
        }
        // Write new K and V into cache at [offset : offset+newSeq]
        // MLX doesn't have scatter_nd; we rebuild by concatenation
        const before = this.offset;
        const total = Number(this.keys.shape()[2]);
        const keysBefore = this.keys.slice(BigInt64Array.from([0n, 0n, 0n, 0n]), BigInt64Array.from([this.keys.shape()[0], this.keys.shape()[1], BigInt(before), this.keys.shape()[3]]));
        const keysAfter = this.keys.slice(BigInt64Array.from([0n, 0n, BigInt(before + newSeq), 0n]), BigInt64Array.from([this.keys.shape()[0], this.keys.shape()[1], BigInt(total), this.keys.shape()[3]]));
        this.keys = MxArray.concatenateMany([keysBefore, k, keysAfter], 2);
        const valsBefore = this.values.slice(BigInt64Array.from([0n, 0n, 0n, 0n]), BigInt64Array.from([this.values.shape()[0], this.values.shape()[1], BigInt(before), this.values.shape()[3]]));
        const valsAfter = this.values.slice(BigInt64Array.from([0n, 0n, BigInt(before + newSeq), 0n]), BigInt64Array.from([this.values.shape()[0], this.values.shape()[1], BigInt(total), this.values.shape()[3]]));
        this.values = MxArray.concatenateMany([valsBefore, v, valsAfter], 2);
        this.offset += newSeq;
        const cachedK = this.keys.slice(BigInt64Array.from([0n, 0n, 0n, 0n]), BigInt64Array.from([this.keys.shape()[0], this.keys.shape()[1], BigInt(this.offset), this.keys.shape()[3]]));
        const cachedV = this.values.slice(BigInt64Array.from([0n, 0n, 0n, 0n]), BigInt64Array.from([this.values.shape()[0], this.values.shape()[1], BigInt(this.offset), this.values.shape()[3]]));
        return [cachedK, cachedV];
    }
    updateAndFetchConv(x, padding) {
        // x: [batch, seq, channels]
        if (this.conv === null) {
            // Pad with zeros on the left
            const xShape = x.shape();
            const batch = Number(xShape[0]);
            const ch = Number(xShape[2]);
            const pad = MxArray.zeros(s(batch, padding, ch), null);
            this.conv = MxArray.concatenate(pad, x, 1);
        }
        else {
            // Keep only the last `padding` frames plus new input
            const convLen = Number(this.conv.shape()[1]);
            const keep = Math.min(padding, convLen);
            const tail = this.conv.slice(BigInt64Array.from([0n, BigInt(convLen - keep), 0n]), this.conv.shape());
            this.conv = MxArray.concatenate(tail, x, 1);
        }
        return this.conv;
    }
}
/**
 * Rotating cache for streaming Conformer: drops old frames beyond keep_size.
 */
export class RotatingConformerCache extends ConformerCache {
    keepSize;
    dropSize;
    constructor(keepSize, cacheDrop) {
        super();
        this.keepSize = keepSize;
        this.dropSize = cacheDrop;
    }
    updateAndFetchKV(k, v) {
        const [cachedK, cachedV] = super.updateAndFetchKV(k, v);
        // Trim to keepSize if we've accumulated too many frames
        const currentLen = Number(cachedK.shape()[2]);
        if (currentLen > this.keepSize) {
            const start = currentLen - this.keepSize;
            const trimmedK = cachedK.slice(BigInt64Array.from([0n, 0n, BigInt(start), 0n]), cachedK.shape());
            const trimmedV = cachedV.slice(BigInt64Array.from([0n, 0n, BigInt(start), 0n]), cachedV.shape());
            // Update internal state
            this.keys = trimmedK;
            this.values = trimmedV;
            this.offset = this.keepSize;
            return [trimmedK, trimmedV];
        }
        return [cachedK, cachedV];
    }
}
//# sourceMappingURL=cache.js.map