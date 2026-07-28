import type { ParakeetBackend, EncoderOutput, DecodeStepResult, DecoderStateHandle } from '../backend.js';
import { Conformer } from './conformer.js';
import { PredictNetwork, JointNetwork } from './rnnt.js';
export declare class MlxBackend implements ParakeetBackend {
    readonly name = "mlx";
    readonly encoderDim: number;
    private readonly encoder;
    private readonly predict;
    private readonly joint;
    constructor(args: {
        encoder: Conformer;
        predict: PredictNetwork;
        joint: JointNetwork;
        encoderDim: number;
    });
    /**
     * @param mel log-mel laid out [nMels, numFrames] (the shared front-end's order)
     */
    encode(mel: Float32Array, nMels: number, numFrames: number): Promise<EncoderOutput>;
    decodeStep(encFrame: Float32Array, token: number | null, state: DecoderStateHandle | null): Promise<DecodeStepResult>;
}
//# sourceMappingURL=backend.d.ts.map