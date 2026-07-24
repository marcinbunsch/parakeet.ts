import { MxArray } from '@mlx-node/core';
import { Module, WeightMap, Conv1d } from './nn.js';
import { AlignedToken } from '../alignment.js';
export interface ConvASRDecoderArgs {
    featIn: number;
    numClasses: number;
    vocabulary: string[];
}
export interface AuxCTCArgs {
    decoder: ConvASRDecoderArgs;
}
export declare class ConvASRDecoder extends Module {
    decoderLayer: Conv1d;
    temperature: number;
    constructor(args: ConvASRDecoderArgs);
    forward(x: MxArray): MxArray;
    loadWeights(weights: WeightMap, prefix: string): void;
}
export declare function decodeCTCGreedy(features: MxArray, // [batch, seq, encoderDim]
lengths: MxArray, // [batch]
decoder: ConvASRDecoder, vocabulary: string[], timeRatio: number): Array<AlignedToken[]>;
//# sourceMappingURL=ctc.d.ts.map