import { describe, it } from 'vitest';
import path from 'node:path';
import fs from 'node:fs';
import { fromLocal } from '../../src/utils.js';
import { ParakeetTDT } from '../../src/parakeet.js';

const MODEL_PATH = path.join(
  process.env['HOME'] ?? '/tmp',
  '.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3/snapshots/ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15',
);

describe('diagnose4', () => {
  it('check weight loading coverage', async () => {
    // Read safetensors header
    const fp = path.join(MODEL_PATH, 'model.safetensors');
    const fd = fs.openSync(fp, 'r');
    const lenBuf = Buffer.allocUnsafe(8);
    fs.readSync(fd, lenBuf, 0, 8, 0);
    const headerLen = Number(new DataView(lenBuf.buffer, lenBuf.byteOffset, 8).getBigUint64(0, true));
    const headerBuf = Buffer.allocUnsafe(headerLen);
    fs.readSync(fd, headerBuf, 0, headerLen, 8);
    fs.closeSync(fd);
    const header = JSON.parse(headerBuf.toString('utf8'));

    const allKeys = Object.keys(header).filter(k => k !== '__metadata__');
    console.log(`Total tensor keys in safetensors: ${allKeys.length}`);

    // Group keys by top-level prefix
    const byPrefix = new Map<string, number>();
    for (const k of allKeys) {
      const top = k.split('.')[0];
      byPrefix.set(top, (byPrefix.get(top) ?? 0) + 1);
    }
    console.log('Keys by top-level prefix:');
    for (const [p, c] of byPrefix.entries()) console.log(`  ${p}: ${c}`);

    // Print sample keys for encoder layers.0
    console.log('\nencoder.layers.0.* keys:');
    for (const k of allKeys.filter(x => x.startsWith('encoder.layers.0.'))) {
      const m = header[k];
      console.log(`  ${k} (${m.dtype} ${m.shape})`);
    }

    console.log('\nencoder.pre_encode.* keys:');
    for (const k of allKeys.filter(x => x.startsWith('encoder.pre_encode.'))) {
      const m = header[k];
      console.log(`  ${k} (${m.dtype} ${m.shape})`);
    }

    console.log('\ndecoder.* keys:');
    for (const k of allKeys.filter(x => x.startsWith('decoder.'))) {
      const m = header[k];
      console.log(`  ${k} (${m.dtype} ${m.shape})`);
    }

    console.log('\njoint.* keys:');
    for (const k of allKeys.filter(x => x.startsWith('joint.'))) {
      const m = header[k];
      console.log(`  ${k} (${m.dtype} ${m.shape})`);
    }

    // Now load model and instrument weight loading to track which were used
    const model = fromLocal(MODEL_PATH) as ParakeetTDT;
    console.log('\nModel loaded:', model.constructor.name);
  }, 60_000);
});
