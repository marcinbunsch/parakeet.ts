import { describe, it } from 'vitest';
import path from 'node:path';
import { fromLocal } from '../../src/utils.js';
import { loadAudio, getLogMel } from '../../src/audio.js';

const MODEL_PATH = path.join(
  process.env['HOME'] ?? '/tmp',
  '.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3/snapshots/ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15',
);
const INPUTS = path.join(import.meta.dirname, 'inputs');

describe('diagnose', () => {
  it('checks pipeline stages', async () => {
    console.log('[1] loading model...');
    const t0 = Date.now();
    const model = fromLocal(MODEL_PATH);
    console.log(`[1] model loaded in ${Date.now() - t0}ms, kind=${model.constructor.name}`);
    console.log(`[1] preprocessor sampleRate=${(model as any).preprocessorConfig.sampleRate}`);

    const wav = path.join(INPUTS, 'sample-1.wav');
    console.log('[2] loading audio:', wav);
    const t1 = Date.now();
    const audio = loadAudio(wav, (model as any).preprocessorConfig.sampleRate);
    console.log(`[2] audio loaded in ${Date.now() - t1}ms, shape=${audio.shape()}, ndim=${audio.ndim()}`);
    const af32 = audio.toFloat32();
    console.log(`[2] audio length=${af32.length}, first=${af32.slice(0, 5)}, last=${af32.slice(af32.length - 5)}`);

    console.log('[3] computing mel...');
    const t2 = Date.now();
    const mel = getLogMel(audio, (model as any).preprocessorConfig);
    console.log(`[3] mel computed in ${Date.now() - t2}ms, shape=${mel.shape()}`);
    const melData = mel.toFloat32();
    console.log(`[3] mel data: len=${melData.length}, first 5=${Array.from(melData.slice(0, 5))}, last 5=${Array.from(melData.slice(melData.length - 5))}`);

    console.log('[4] running generate...');
    const t3 = Date.now();
    const result = (model as any).generate(mel);
    console.log(`[4] generate done in ${Date.now() - t3}ms`);
    console.log('[4] result:', JSON.stringify(result, null, 2).slice(0, 500));
  }, 600_000);
});
