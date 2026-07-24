import { MxArray } from '@mlx-node/core';
import { Module, WeightMap, Linear, Embedding, LSTM } from './nn.js';
import { AlignedToken } from '../alignment.js';
export interface PredictNetworkArgs {
    predHidden: number;
    predRnnLayers: number;
    rnnHiddenSize?: number;
}
export interface JointNetworkArgs {
    jointHidden: number;
    activation: string;
    encoderHidden: number;
    predHidden: number;
}
export interface PredictArgs {
    blankAsPad: boolean;
    vocabSize: number;
    prednet: PredictNetworkArgs;
}
export interface JointArgs {
    numClasses: number;
    vocabulary: string[];
    jointnet: JointNetworkArgs;
    numExtraOutputs?: number;
}
export declare class PredictNetwork extends Module {
    embed: Embedding;
    decRnn: LSTM;
    readonly predHidden: number;
    constructor(args: PredictArgs);
    /**
     * y: [batch, 1] token ids or null for blank start
     * hc: LSTM hidden/cell state or null
     * Returns [output: [batch, 1, predHidden], [h, c]]
     */
    forward(y: MxArray | null, hc: [MxArray, MxArray] | null): [MxArray, [MxArray, MxArray]];
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare class JointNetwork extends Module {
    pred: Linear;
    enc: Linear;
    activation: (x: MxArray) => MxArray;
    jointOut: Linear;
    readonly numClasses: number;
    constructor(args: JointArgs);
    forward(enc: MxArray, pred: MxArray): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
export interface DecoderState {
    lastToken: number | null;
    hiddenState: [MxArray, MxArray] | null;
}
/** Greedy TDT decoder. Returns decoded token list per batch element. */
export declare function decodeTDTGreedy(features: MxArray, // [batch, seqLen, encoderDim]
lengths: MxArray, // [batch]
decoder: PredictNetwork, joint: JointNetwork, vocabulary: string[], durations: number[], maxSymbols: number | null, states: DecoderState[], timeRatio: number): [Array<AlignedToken[]>, DecoderState[]];
/** Greedy RNNT decoder */
export declare function decodeRNNTGreedy(features: MxArray, lengths: MxArray, decoder: PredictNetwork, joint: JointNetwork, vocabulary: string[], maxSymbols: number | null, states: DecoderState[], timeRatio: number): [Array<Array<import('../alignment.js').AlignedToken>>, DecoderState[]];
//# sourceMappingURL=rnnt.d.ts.map