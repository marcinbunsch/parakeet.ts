import { describe, it } from 'vitest';
import path from 'node:path';
import { fromLocal } from '../../src/utils.js';
import { loadAudio, getLogMel } from '../../src/audio.js';
import { ParakeetTDT } from '../../src/parakeet.js';

const MODEL_PATH = path.join(
  process.env['HOME'] ?? '/tmp',
  '.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3/snapshots/ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15',
);
const INPUTS = path.join(import.meta.dirname, 'inputs');

describe('diagnose2', () => {
  it('checks encoder output', async () => {
    const model = fromLocal(MODEL_PATH) as ParakeetTDT;
    console.log('vocabulary length:', model.vocabulary.length);
    console.log('vocabulary first 30:', model.vocabulary.slice(0, 30));
    console.log('durations:', model.durations);

    const wav = path.join(INPUTS, 'sample-1.wav');
    const audio = loadAudio(wav, model.preprocessorConfig.sampleRate);
    let mel = getLogMel(audio, model.preprocessorConfig);
    if (mel.ndim() === 2) mel = mel.expandDims(0);
    console.log('mel shape:', mel.shape().toString());

    const t0 = Date.now();
    const [features, lengths] = model.encoder.forward(mel, null, null);
    features.eval();
    lengths.eval();
    console.log(`encoder forward: ${Date.now() - t0}ms`);
    console.log('features shape:', features.shape().toString());
    console.log('lengths:', Array.from(lengths.toInt32()));

    const featData = features.toFloat32();
    console.log('features stats: len=' + featData.length);
    let mn = Infinity, mx = -Infinity, sum = 0;
    let nan = 0, inf = 0;
    for (let i = 0; i < featData.length; i++) {
      const v = featData[i];
      if (Number.isNaN(v)) nan++;
      else if (!Number.isFinite(v)) inf++;
      else {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
      }
    }
    console.log(`features: min=${mn}, max=${mx}, mean=${sum / featData.length}, nan=${nan}, inf=${inf}`);
    console.log('features[0:10]:', Array.from(featData.slice(0, 10)));
  }, 600_000);
});
