import { MxArray } from '@mlx-node/core';

function s(...dims: number[]): BigInt64Array {
  return BigInt64Array.from(dims.map(BigInt));
}

/**
 * Cache for a single Conformer layer — stores attention K/V and conv state.
 */
export class ConformerCache {
  keys: MxArray | null = null;
  values: MxArray | null = null;
  conv: MxArray | null = null;
  offset = 0;
  private readonly step = 256;

  updateAndFetchKV(k: MxArray, v: MxArray): [MxArray, MxArray] {
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

    const keysBefore = this.keys.slice(
      BigInt64Array.from([0n, 0n, 0n, 0n]),
      BigInt64Array.from([this.keys.shape()[0], this.keys.shape()[1], BigInt(before), this.keys.shape()[3]]),
    );
    const keysAfter = this.keys.slice(
      BigInt64Array.from([0n, 0n, BigInt(before + newSeq), 0n]),
      BigInt64Array.from([this.keys.shape()[0], this.keys.shape()[1], BigInt(total), this.keys.shape()[3]]),
    );
    this.keys = MxArray.concatenateMany([keysBefore, k, keysAfter], 2);

    const valsBefore = this.values.slice(
      BigInt64Array.from([0n, 0n, 0n, 0n]),
      BigInt64Array.from([this.values.shape()[0], this.values.shape()[1], BigInt(before), this.values.shape()[3]]),
    );
    const valsAfter = this.values.slice(
      BigInt64Array.from([0n, 0n, BigInt(before + newSeq), 0n]),
      BigInt64Array.from([this.values.shape()[0], this.values.shape()[1], BigInt(total), this.values.shape()[3]]),
    );
    this.values = MxArray.concatenateMany([valsBefore, v, valsAfter], 2);

    this.offset += newSeq;

    const cachedK = this.keys.slice(
      BigInt64Array.from([0n, 0n, 0n, 0n]),
      BigInt64Array.from([this.keys.shape()[0], this.keys.shape()[1], BigInt(this.offset), this.keys.shape()[3]]),
    );
    const cachedV = this.values.slice(
      BigInt64Array.from([0n, 0n, 0n, 0n]),
      BigInt64Array.from([this.values.shape()[0], this.values.shape()[1], BigInt(this.offset), this.values.shape()[3]]),
    );

    return [cachedK, cachedV];
  }

  updateAndFetchConv(x: MxArray, padding: number): MxArray {
    // x: [batch, seq, channels]. Prepend `padding` cached frames of left
    // context and append `padding` zeros so the depthwise conv (kernel
    // 2*padding+1, no internal padding) returns a sequence of the same length.
    if (padding === 0) return x;

    const xShape = x.shape();
    const B = Number(xShape[0]);
    const S = Number(xShape[1]);
    const D = Number(xShape[2]);

    if (this.conv === null) {
      this.conv = MxArray.zeros(s(B, padding, D), null);
    }

    const tokensToCache = Math.min(padding, S);
    const cacheUpdate = x.slice(s(0, S - tokensToCache, 0), s(B, S, D));

    if (tokensToCache < padding) {
      const kept = this.conv.slice(s(0, tokensToCache, 0), s(B, padding, D));
      this.conv = MxArray.concatenate(kept, cacheUpdate, 1);
    } else {
      this.conv = cacheUpdate;
    }

    let result = MxArray.concatenate(this.conv, x, 1);
    result = result.pad(new Int32Array([0, 0, 0, padding, 0, 0]), 0.0);
    return result;
  }
}

/**
 * Rotating cache for streaming Conformer: drops old frames beyond keep_size.
 */
export class RotatingConformerCache extends ConformerCache {
  private readonly keepSize: number;
  private readonly dropSize: number;

  constructor(keepSize: number, cacheDrop: number) {
    super();
    this.keepSize = keepSize;
    this.dropSize = cacheDrop;
  }

  updateAndFetchKV(k: MxArray, v: MxArray): [MxArray, MxArray] {
    // Return the cached history (up to keepSize frames) concatenated with ALL
    // of the new keys/values. Only the frames that will be finalized — i.e.
    // everything except the last `dropSize` frames — are committed to the
    // history; the drop tail is recomputed on the next call. Frames re-fed via
    // overlapping mel are therefore never double-counted (to_cache is 0 while
    // the sequence is shorter than dropSize).
    const kShape = k.shape();
    const B = Number(kShape[0]);
    const H = Number(kShape[1]);
    const S = Number(kShape[2]);
    const D = Number(kShape[3]);

    const kOut = this.keys === null ? k : MxArray.concatenate(this.keys, k, 2);
    const vOut = this.values === null ? v : MxArray.concatenate(this.values, v, 2);

    const toCache = Math.min(Math.max(0, S - this.dropSize), this.keepSize);
    if (toCache > 0) {
      const startIdx = S - this.dropSize - toCache;
      const endIdx = S - this.dropSize;
      const kChunk = k.slice(s(0, 0, startIdx, 0), s(B, H, endIdx, D));
      const vChunk = v.slice(s(0, 0, startIdx, 0), s(B, H, endIdx, D));

      let newK = this.keys === null ? kChunk : MxArray.concatenate(this.keys, kChunk, 2);
      let newV = this.values === null ? vChunk : MxArray.concatenate(this.values, vChunk, 2);

      // Keep only the most recent keepSize frames of history.
      const curLen = Number(newK.shape()[2]);
      if (curLen > this.keepSize) {
        const st = curLen - this.keepSize;
        newK = newK.slice(s(0, 0, st, 0), s(B, H, curLen, D));
        newV = newV.slice(s(0, 0, st, 0), s(B, H, curLen, D));
      }

      this.keys = newK;
      this.values = newV;
      this.offset += toCache;
    }

    return [kOut, vOut];
  }

  updateAndFetchConv(x: MxArray, padding: number): MxArray {
    // As ConformerCache.updateAndFetchConv, but only the frames that will be
    // finalized (beyond cache_drop_size) are kept as left context for the next
    // call — the drop_size tail is recomputed each step, not carried over.
    if (padding === 0) return x;

    const xShape = x.shape();
    const B = Number(xShape[0]);
    const S = Number(xShape[1]);
    const D = Number(xShape[2]);

    if (this.conv === null) {
      this.conv = MxArray.zeros(s(B, padding, D), null);
    }

    if (S > this.dropSize) {
      const tokensToCache = Math.min(padding, S - this.dropSize);
      const cacheUpdate = x.slice(s(0, S - tokensToCache, 0), s(B, S, D));

      if (tokensToCache < padding) {
        const kept = this.conv.slice(s(0, tokensToCache, 0), s(B, padding, D));
        this.conv = MxArray.concatenate(kept, cacheUpdate, 1);
      } else {
        this.conv = cacheUpdate;
      }
    }

    let result = MxArray.concatenate(this.conv, x, 1);
    result = result.pad(new Int32Array([0, 0, 0, padding, 0, 0]), 0.0);
    return result;
  }
}
