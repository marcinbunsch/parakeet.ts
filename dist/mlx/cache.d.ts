import { MxArray } from '@mlx-node/core';
/**
 * Cache for a single Conformer layer — stores attention K/V and conv state.
 */
export declare class ConformerCache {
    keys: MxArray | null;
    values: MxArray | null;
    conv: MxArray | null;
    offset: number;
    private readonly step;
    updateAndFetchKV(k: MxArray, v: MxArray): [MxArray, MxArray];
    updateAndFetchConv(x: MxArray, padding: number): MxArray;
}
/**
 * Rotating cache for streaming Conformer: drops old frames beyond keep_size.
 */
export declare class RotatingConformerCache extends ConformerCache {
    private readonly keepSize;
    private readonly dropSize;
    constructor(keepSize: number, cacheDrop: number);
    updateAndFetchKV(k: MxArray, v: MxArray): [MxArray, MxArray];
}
//# sourceMappingURL=cache.d.ts.map