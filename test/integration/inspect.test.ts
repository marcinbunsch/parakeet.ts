import { describe, it } from 'vitest';
import path from 'node:path';
import fs from 'node:fs';

const MODEL_PATH = path.join(
  process.env['HOME'] ?? '/tmp',
  '.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3/snapshots/ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15',
);

describe('inspect', () => {
  it('safetensors header', () => {
    const fp = path.join(MODEL_PATH, 'model.safetensors');
    const fd = fs.openSync(fp, 'r');
    const lenBuf = Buffer.allocUnsafe(8);
    fs.readSync(fd, lenBuf, 0, 8, 0);
    const headerLen = Number(new DataView(lenBuf.buffer, lenBuf.byteOffset, 8).getBigUint64(0, true));
    const headerBuf = Buffer.allocUnsafe(headerLen);
    fs.readSync(fd, headerBuf, 0, headerLen, 8);
    fs.closeSync(fd);
    const header = JSON.parse(headerBuf.toString('utf8'));

    const dtypes = new Map<string, number>();
    const sampleKeys: string[] = [];
    let total = 0;
    for (const [k, v] of Object.entries(header)) {
      if (k === '__metadata__') continue;
      const dt = (v as any).dtype;
      dtypes.set(dt, (dtypes.get(dt) ?? 0) + 1);
      if (sampleKeys.length < 30) sampleKeys.push(`${k} (${dt} ${(v as any).shape})`);
      total++;
    }
    console.log(`total tensors: ${total}`);
    console.log(`dtypes:`, [...dtypes.entries()]);
    console.log(`sample keys:`);
    for (const k of sampleKeys) console.log(`  ${k}`);

    // Sample some specific keys
    const allKeys = Object.keys(header).filter(k => k !== '__metadata__');
    console.log('first 10:', allKeys.slice(0, 10));
    console.log('keys containing "decoder":', allKeys.filter(k => k.includes('decoder')).slice(0, 5));
    console.log('keys containing "joint":', allKeys.filter(k => k.includes('joint')).slice(0, 5));
    console.log('keys containing "encoder.layers.0":', allKeys.filter(k => k.includes('encoder.layers.0')).slice(0, 10));
  }, 60_000);

  it('config.json', () => {
    const cfg = JSON.parse(fs.readFileSync(path.join(MODEL_PATH, 'config.json'), 'utf8'));
    console.log(JSON.stringify(cfg, null, 2));
  });
});
