#!/usr/bin/env node
/**
 * `pnpm test` entry point: pick the platform-appropriate integration suite.
 *
 *   darwin  -> MLX / Apple Silicon path (transcribe + streaming)
 *   other   -> ONNX Runtime path + MLX-vs-ONNX parity
 *
 * Every test also self-skips when its checkpoint or backend is unavailable, so
 * this only decides which files to load — it never forces a native library that
 * isn't there. Vitest is run directly (not via a nested `pnpm run`) to avoid
 * pnpm's pre-run dependency check tripping over the intentionally-skipped
 * onnxruntime-node build on a Mac.
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const MAC = ['test/integration/transcribe.test.ts', 'test/integration/streaming.test.ts'];
const LINUX = ['test/integration/backend-parity.test.ts'];

const files = process.platform === 'darwin' ? MAC : LINUX;
console.log(`[test] platform ${process.platform} -> vitest run ${files.join(' ')}`);

const vitest = path.join(process.cwd(), 'node_modules', '.bin', 'vitest');
const res = spawnSync(vitest, ['run', ...files], { stdio: 'inherit' });
process.exit(res.status ?? 1);
