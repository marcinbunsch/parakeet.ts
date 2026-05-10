import { describe, it } from 'vitest';
import path from 'node:path';
import { fromLocal } from '../../src/utils.js';

const MODEL_PATH = path.join(
  process.env['HOME'] ?? '/tmp',
  '.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3/snapshots/ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15',
);

const INPUTS = path.join(import.meta.dirname, 'inputs');

const SAMPLES = [
  { file: 'sample-1.wav', expected: 'All right, let\'s give this a go then' },
  { file: 'sample-2.wav', expected: 'I absolutely hate smalltalk' },
  { file: 'sample-3.wav', expected: 'The best thing you can do is give them a card' },
];

describe('transcription', () => {
  const model = fromLocal(MODEL_PATH);

  for (const { file, expected } of SAMPLES) {
    it(`transcribes ${file}`, async () => {
      const result = await model.transcribe(path.join(INPUTS, file));
      console.log(`[${file}] expected: "${expected}"`);
      console.log(`[${file}] got:      "${result.text}"`);
    }, 120_000);
  }
});
