import { MxArray } from '@mlx-node/core';
import { Module, WeightMap, Linear, LayerNorm, BatchNorm, Conv1d, Conv2d } from './nn.js';
import { MultiHeadAttention, RelPositionMultiHeadAttention, RelPositionMultiHeadLocalAttention, RelPositionalEncoding, LocalRelPositionalEncoding } from './attention.js';
import { ConformerCache } from './cache.js';
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
declare class FeedForward extends Module {
    linear1: Linear;
    linear2: Linear;
    constructor(dModel: number, dFf: number, useBias: boolean);
    forward(x: MxArray): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
declare class Convolution extends Module {
    readonly padding: number;
    pointwiseConv1: Conv1d;
    depthwiseConv: Conv1d;
    batchNorm: BatchNorm;
    pointwiseConv2: Conv1d;
    constructor(args: ConformerArgs);
    forward(x: MxArray, cache: ConformerCache | null): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
type AttentionModel = 'rel_pos' | 'rel_pos_local_attn' | 'normal';
export declare class ConformerBlock extends Module {
    normFF1: LayerNorm;
    ff1: FeedForward;
    normSelfAtt: LayerNorm;
    selfAttn: MultiHeadAttention | RelPositionMultiHeadAttention | RelPositionMultiHeadLocalAttention;
    normConv: LayerNorm;
    conv: Convolution;
    normFF2: LayerNorm;
    ff2: FeedForward;
    normOut: LayerNorm;
    private readonly args;
    constructor(args: ConformerArgs);
    private buildAttention;
    setAttentionModel(name: AttentionModel, contextSize?: [number, number]): void;
    forward(x: MxArray, posEmb: MxArray | null, mask: MxArray | null, cache: ConformerCache | null): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
declare class DwStridingSubsampling extends Module {
    private readonly samplingNum;
    private readonly stride;
    private readonly kernelSize;
    private readonly padding;
    convLayers: Array<Conv2d | null>;
    out: Linear;
    constructor(args: ConformerArgs);
    forward(x: MxArray, lengths: MxArray): [MxArray, MxArray];
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class Conformer extends Module {
    readonly args: ConformerArgs;
    posEnc: RelPositionalEncoding | LocalRelPositionalEncoding | null;
    preEncode: DwStridingSubsampling | Linear;
    layers: ConformerBlock[];
    constructor(args: ConformerArgs);
    setAttentionModel(name: AttentionModel, contextSize?: [number, number]): void;
    forward(x: MxArray, // [batch, seq, mel]
    lengths: MxArray | null, cache: Array<ConformerCache | null> | null): [MxArray, MxArray];
    loadWeights(weights: WeightMap, prefix: string): void;
}
export {};
//# sourceMappingURL=conformer.d.ts.map