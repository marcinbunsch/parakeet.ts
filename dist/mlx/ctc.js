import { Module, Conv1d, logSoftmax } from './nn.js';
import { makeAlignedToken } from '../alignment.js';
import { decode } from '../tokenizer.js';
function s(...dims) {
    return BigInt64Array.from(dims.map(BigInt));
}
// ---------------------------------------------------------------------------
// ConvASR Decoder — 1×1 Conv1d acting as a linear projection
// ---------------------------------------------------------------------------
export class ConvASRDecoder extends Module {
    decoderLayer;
    temperature;
    constructor(args) {
        super();
        const numClasses = (args.numClasses <= 0 ? args.vocabulary.length : args.numClasses) + 1;
        this.decoderLayer = new Conv1d(args.featIn, numClasses, 1, 1, 0, 1, true);
        this.temperature = 1.0;
    }
    forward(x) {
        // x: [batch, seq, feat_in]
        // → log_softmax over last dim
        const logits = this.decoderLayer.forward(x); // [batch, seq, numClasses]
        return logSoftmax(logits.divScalar(this.temperature), 2);
    }
    loadWeights(weights, prefix) {
        this.decoderLayer.loadWeights(weights, `${prefix}.decoder_layers.0`);
    }
}
// ---------------------------------------------------------------------------
// CTC greedy decoding
// ---------------------------------------------------------------------------
export function decodeCTCGreedy(features, // [batch, seq, encoderDim]
lengths, // [batch]
decoder, vocabulary, timeRatio) {
    const logits = decoder.forward(features); // [batch, seq, numClasses]
    logits.eval();
    lengths.eval();
    const fShape = features.shape();
    const B = Number(fShape[0]);
    const S = Number(fShape[1]);
    const numClasses = Number(logits.shape()[2]);
    // Move to CPU
    const logitsData = logits.toFloat32(); // [B * S * numClasses]
    const lengthsData = lengths.toInt32();
    const results = [];
    for (let b = 0; b < B; b++) {
        const length = lengthsData[b];
        const hypothesis = [];
        const tokenBoundaries = [];
        let prevToken = -1;
        // Build predictions and probs for this batch
        const predictions = new Int32Array(length);
        for (let t = 0; t < length; t++) {
            let maxLogit = -Infinity;
            let maxIdx = 0;
            for (let c = 0; c < numClasses; c++) {
                const val = logitsData[(b * S + t) * numClasses + c];
                if (val > maxLogit) {
                    maxLogit = val;
                    maxIdx = c;
                }
            }
            predictions[t] = maxIdx;
        }
        // Convert log-probs to probs for confidence
        const probs = new Float32Array(length * numClasses);
        for (let t = 0; t < length; t++) {
            for (let c = 0; c < numClasses; c++) {
                probs[t * numClasses + c] = Math.exp(logitsData[(b * S + t) * numClasses + c]);
            }
        }
        const vocabSize = vocabulary.length; // blank token index = vocabSize
        for (let t = 0; t < length; t++) {
            const tokenIdx = predictions[t];
            if (tokenIdx === vocabSize)
                continue; // skip blank
            if (tokenIdx === prevToken)
                continue; // skip repeat
            if (prevToken !== -1) {
                const startFrame = tokenBoundaries[tokenBoundaries.length - 1];
                const startTime = startFrame * timeRatio;
                const endTime = t * timeRatio;
                const duration = endTime - startTime;
                // Compute entropy-based confidence over the token frames
                let entropy = 0;
                let count = 0;
                for (let ft = startFrame; ft < t; ft++) {
                    for (let c = 0; c < numClasses; c++) {
                        const p = probs[ft * numClasses + c] + 1e-10;
                        entropy -= p * Math.log(p);
                    }
                    count++;
                }
                if (count > 0)
                    entropy /= count;
                const maxEntropy = Math.log(numClasses);
                const confidence = 1.0 - entropy / maxEntropy;
                hypothesis.push(makeAlignedToken(prevToken, decode([prevToken], vocabulary), startTime, duration, confidence));
            }
            tokenBoundaries.push(t);
            prevToken = tokenIdx;
        }
        // Handle last token
        if (prevToken !== -1) {
            const startFrame = tokenBoundaries[tokenBoundaries.length - 1];
            // Find last non-blank frame
            let lastNonBlank = length - 1;
            for (let t = length - 1; t >= startFrame; t--) {
                if (predictions[t] !== vocabSize) {
                    lastNonBlank = t;
                    break;
                }
            }
            const startTime = startFrame * timeRatio;
            const endTime = (lastNonBlank + 1) * timeRatio;
            const duration = endTime - startTime;
            let entropy = 0;
            let count = 0;
            for (let ft = startFrame; ft <= lastNonBlank; ft++) {
                for (let c = 0; c < numClasses; c++) {
                    const p = probs[ft * numClasses + c] + 1e-10;
                    entropy -= p * Math.log(p);
                }
                count++;
            }
            if (count > 0)
                entropy /= count;
            const confidence = 1.0 - entropy / Math.log(numClasses);
            hypothesis.push(makeAlignedToken(prevToken, decode([prevToken], vocabulary), startTime, duration, confidence));
        }
        results.push(hypothesis);
    }
    return results;
}
//# sourceMappingURL=ctc.js.map