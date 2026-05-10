import { describe, it } from 'vitest';
import path from 'node:path';
import { MxArray } from '@mlx-node/core';
import { fromLocal } from '../../src/utils.js';
import { loadAudio, getLogMel } from '../../src/audio.js';
import { ParakeetTDT } from '../../src/parakeet.js';
import { softmax } from '../../src/nn.js';

const MODEL_PATH = path.join(
  process.env['HOME'] ?? '/tmp',
  '.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3/snapshots/ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15',
);
const INPUTS = path.join(import.meta.dirname, 'inputs');

function s(...dims: number[]): BigInt64Array {
  return BigInt64Array.from(dims.map(BigInt));
}

describe('diagnose3', () => {
  it('manual TDT decode trace', async () => {
    const model = fromLocal(MODEL_PATH) as ParakeetTDT;
    const audio = loadAudio(path.join(INPUTS, 'sample-1.wav'), model.preprocessorConfig.sampleRate);
    let mel = getLogMel(audio, model.preprocessorConfig);
    if (mel.ndim() === 2) mel = mel.expandDims(0);

    const [features, lengths] = model.encoder.forward(mel, null, null);
    features.eval();
    lengths.eval();
    const length = Number(lengths.toInt32()[0]);
    console.log(`features shape=${features.shape()}, length=${length}`);

    const featData = features.toFloat32();
    let mn = Infinity, mx = -Infinity;
    for (const v of featData) { if (v < mn) mn = v; if (v > mx) mx = v; }
    console.log(`features stats: min=${mn.toFixed(4)}, max=${mx.toFixed(4)}`);

    const vocabSize = model.vocabulary.length; // 8192
    const numDurations = model.durations.length; // 5

    // Just run a few steps and log
    let lastToken: number | null = null;
    let hiddenState: [MxArray, MxArray] | null = null;

    for (let step = 0; step < Math.min(length, 10); step++) {
      const yInput = lastToken !== null
        ? MxArray.fromInt32(new Int32Array([lastToken]), s(1, 1))
        : null;

      const [decOut, [newH, newC]] = model.decoder.forward(yInput, hiddenState);
      decOut.eval();
      const decExpand = decOut.expandDims(1);

      const encStep = features
        .slice(s(0, step, 0), s(1, step + 1, Number(features.shape()[2])))
        .expandDims(2);

      const jointOut = model.joint.forward(encStep, decExpand);
      jointOut.eval();
      const jShape = jointOut.shape();
      const numClasses = Number(jShape[3]);

      const tokenLogits = jointOut.slice(s(0, 0, 0, 0), s(1, 1, 1, vocabSize + 1)).squeeze(new Int32Array([0, 1, 2]));
      const durationLogits = jointOut.slice(s(0, 0, 0, vocabSize + 1), s(1, 1, 1, numClasses)).squeeze(new Int32Array([0, 1, 2]));

      const tokenProbs = softmax(tokenLogits, 0).toFloat32();
      const durationProbs = softmax(durationLogits, 0).toFloat32();

      let predTok = 0, maxP = tokenProbs[0];
      for (let i = 1; i < tokenProbs.length; i++) if (tokenProbs[i] > maxP) { maxP = tokenProbs[i]; predTok = i; }
      let predDur = 0, maxD = durationProbs[0];
      for (let i = 1; i < durationProbs.length; i++) if (durationProbs[i] > maxD) { maxD = durationProbs[i]; predDur = i; }

      // Top 5 tokens
      const idx = Array.from({ length: tokenProbs.length }, (_, i) => i);
      idx.sort((a, b) => tokenProbs[b] - tokenProbs[a]);
      const top5 = idx.slice(0, 5).map(i => `[${i}:${i === vocabSize ? 'BLANK' : model.vocabulary[i]}]=${tokenProbs[i].toFixed(4)}`).join(' ');

      console.log(`step=${step} predTok=${predTok} (${predTok === vocabSize ? 'BLANK' : model.vocabulary[predTok]}) p=${maxP.toFixed(4)}, dur=${predDur} p=${maxD.toFixed(4)} | top5: ${top5}`);

      // Update state only if not blank
      if (predTok !== vocabSize) {
        lastToken = predTok;
        hiddenState = [newH, newC];
      }
    }
  }, 600_000);
});
