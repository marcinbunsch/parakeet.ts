import { MxArray } from '@mlx-node/core';
import { Module, Linear, WeightMap } from './nn.js';
import { ConformerCache } from './cache.js';
export declare class MultiHeadAttention extends Module {
    linearQ: Linear;
    linearK: Linear;
    linearV: Linear;
    linearOut: Linear;
    readonly nHead: number;
    readonly headDim: number;
    readonly scale: number;
    constructor(nHead: number, nFeat: number, bias?: boolean);
    forward(q: MxArray, k: MxArray, v: MxArray, posEmb: MxArray | null, mask: MxArray | null, cache: ConformerCache | null): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class RelPositionMultiHeadAttention extends Module {
    linearQ: Linear;
    linearK: Linear;
    linearV: Linear;
    linearOut: Linear;
    linearPos: Linear;
    posBiasU: MxArray;
    posBiasV: MxArray;
    readonly nHead: number;
    readonly headDim: number;
    readonly scale: number;
    constructor(nHead: number, nFeat: number, bias?: boolean);
    private relShift;
    forward(q: MxArray, k: MxArray, v: MxArray, posEmb: MxArray | null, mask: MxArray | null, cache: ConformerCache | null): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class RelPositionMultiHeadLocalAttention extends RelPositionMultiHeadAttention {
    readonly contextSize: [number, number];
    constructor(nHead: number, nFeat: number, bias?: boolean, contextSize?: [number, number]);
    forward(q: MxArray, k: MxArray, v: MxArray, posEmb: MxArray | null, mask: MxArray | null, cache: ConformerCache | null): MxArray;
}
export declare class RelPositionalEncoding extends Module {
    readonly dModel: number;
    maxLen: number;
    readonly scaleInput: boolean;
    readonly scale: number;
    pe: MxArray;
    constructor(dModel: number, maxLen?: number, scaleInput?: boolean);
    calculatePE(): void;
    forward(x: MxArray, offset?: number): [MxArray, MxArray];
    loadWeights(_weights: WeightMap, _prefix: string): void;
}
export declare class LocalRelPositionalEncoding extends Module {
    readonly dModel: number;
    readonly maxLen: number;
    readonly scaleInput: boolean;
    readonly scale: number;
    readonly leftContext: number;
    readonly rightContext: number;
    pe: MxArray;
    constructor(dModel: number, maxLen?: number, scaleInput?: boolean, contextSize?: [number, number]);
    calculatePE(): void;
    forward(x: MxArray, _offset?: number): [MxArray, MxArray];
    loadWeights(_weights: WeightMap, _prefix: string): void;
}
//# sourceMappingURL=attention.d.ts.map