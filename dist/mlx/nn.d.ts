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
export interface TensorEntry {
    data: Float32Array;
    shape: number[];
}
export type WeightMap = Map<string, TensorEntry>;
export declare abstract class Module {
    abstract loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class Linear extends Module {
    inFeatures: number;
    outFeatures: number;
    useBias: boolean;
    weight: MxArray;
    bias: MxArray | null;
    constructor(inFeatures: number, outFeatures: number, useBias?: boolean);
    forward(x: MxArray): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class Embedding extends Module {
    numEmbeddings: number;
    embeddingDim: number;
    weight: MxArray;
    constructor(numEmbeddings: number, embeddingDim: number);
    forward(indices: MxArray): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class LayerNorm extends Module {
    features: number;
    eps: number;
    weight: MxArray;
    bias: MxArray;
    constructor(features: number, eps?: number);
    forward(x: MxArray): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class BatchNorm extends Module {
    features: number;
    eps: number;
    weight: MxArray;
    bias: MxArray;
    runningMean: MxArray;
    runningVar: MxArray;
    constructor(features: number, eps?: number);
    forward(x: MxArray): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class Conv1d extends Module {
    inChannels: number;
    outChannels: number;
    kernelSize: number;
    stride: number;
    padding: number;
    groups: number;
    useBias: boolean;
    weight: MxArray;
    bias: MxArray | null;
    constructor(inChannels: number, outChannels: number, kernelSize: number, stride?: number, padding?: number, groups?: number, useBias?: boolean);
    forward(x: MxArray): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
/** Im2col-based Conv1d. weight: [out_ch, kernel_size, in_ch] */
export declare function conv1d(x: MxArray, weight: MxArray, bias: MxArray | null, stride: number, padding: number, groups: number): MxArray;
export declare class Conv2d extends Module {
    inChannels: number;
    outChannels: number;
    kernelSize: number;
    stride: number;
    padding: number;
    groups: number;
    useBias: boolean;
    weight: MxArray;
    bias: MxArray | null;
    constructor(inChannels: number, outChannels: number, kernelSize: number, stride?: number, padding?: number, groups?: number, useBias?: boolean);
    forward(x: MxArray): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
/** Im2col-based Conv2d. weight: [out_ch, kH, kW, in_ch], input: [batch, H, W, in_ch] */
export declare function conv2d(x: MxArray, weight: MxArray, bias: MxArray | null, stride: number, padding: number, groups: number): MxArray;
export declare function sigmoid(x: MxArray): MxArray;
export declare function silu(x: MxArray): MxArray;
export declare function relu(x: MxArray): MxArray;
export declare function tanh(x: MxArray): MxArray;
export declare function glu(x: MxArray, axis: number): MxArray;
export declare function softmax(x: MxArray, axis: number): MxArray;
export declare function logSoftmax(x: MxArray, axis: number): MxArray;
export declare class LSTMLayer extends Module {
    inputSize: number;
    hiddenSize: number;
    useBias: boolean;
    Wx: MxArray;
    Wh: MxArray;
    b: MxArray | null;
    constructor(inputSize: number, hiddenSize: number, useBias?: boolean);
    /** x: [seq, batch, input], h: [batch, hidden] | null, c: [batch, hidden] | null */
    forward(x: MxArray, h: MxArray | null, c: MxArray | null): [MxArray, MxArray];
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class LSTM extends Module {
    layers: LSTMLayer[];
    constructor(inputSize: number, hiddenSize: number, numLayers: number, useBias?: boolean);
    /**
     * x: [batch, seq, input] (batch_first=true as in Python code)
     * Returns [output: [batch, seq, hidden], [h: [numLayers, batch, H], c: [numLayers, batch, H]]]
     */
    forward(x: MxArray, hc?: [MxArray, MxArray] | null): [MxArray, [MxArray, MxArray]];
    loadWeights(weights: WeightMap, prefix: string): void;
}
//# sourceMappingURL=nn.d.ts.map