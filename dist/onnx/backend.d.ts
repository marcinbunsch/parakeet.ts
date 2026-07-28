/**
 * ONNX Runtime backend — CUDA execution provider on Linux/Nvidia.
 *
 * Two sessions: the conformer encoder, and the fused prediction+joint
 * `decoder_joint` graph. `onnxruntime-node` is loaded lazily so that a
 * Mac-only install never has to resolve it.
 *
 * Graph contract (parakeet-tdt-0.6b-v3 exports):
 *   encoder       : audio_signal [1,nMels,T_mel], length [1]
 *                -> outputs [1,D,T], encoded_lengths [1]
 *   decoder_joint : encoder_outputs [1,D,1], targets [1,1], target_length [1],
 *                   input_states_1 [L,1,H], input_states_2 [L,1,H]
 *                -> outputs [1,1,1,C], output_states_1, output_states_2
 */
import type { ParakeetBackend, EncoderOutput, DecodeStepResult, DecoderStateHandle } from '../backend.js';
export type ExecutionProvider = 'cuda' | 'cpu' | 'tensorrt' | 'coreml' | 'dml';
export interface OnnxBackendOptions {
    encoderPath: string;
    decoderJointPath: string;
    /** Execution provider for the encoder. Default: 'cuda' on linux/win x64, else 'cpu'. */
    executionProvider?: ExecutionProvider;
    /**
     * Execution provider for the autoregressive decode step. Defaults to the same
     * as `executionProvider`. CPU is a reasonable choice — the step is small
     * (~0.5 ms) and CPU avoids per-step device round trips.
     */
    decoderExecutionProvider?: ExecutionProvider;
    /** LSTM layers in the prediction network (state dim 0). Default 2. */
    predRnnLayers?: number;
    /** LSTM hidden size (state dim 2). Default 640. */
    predHidden?: number;
    /** ORT log severity: 0 verbose .. 4 fatal. Default 3 (error). */
    logSeverityLevel?: number;
}
export declare function defaultExecutionProvider(): ExecutionProvider;
export declare class OnnxBackend implements ParakeetBackend {
    readonly name: string;
    readonly encoderDim: number;
    private readonly ort;
    private readonly encoder;
    private readonly decoderJoint;
    private readonly predRnnLayers;
    private readonly predHidden;
    private readonly zeroState;
    /** Reused scratch for the single-element int32 inputs. */
    private readonly targetLength;
    private constructor();
    static create(opts: OnnxBackendOptions): Promise<OnnxBackend>;
    encode(mel: Float32Array, nMels: number, numFrames: number): Promise<EncoderOutput>;
    decodeStep(encFrame: Float32Array, token: number | null, state: DecoderStateHandle | null): Promise<DecodeStepResult>;
    /** Blank/padding token id, set by the model wrapper (vocabulary length). */
    blankId: number;
    dispose(): Promise<void>;
}
//# sourceMappingURL=backend.d.ts.map