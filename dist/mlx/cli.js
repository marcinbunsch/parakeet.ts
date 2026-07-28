#!/usr/bin/env node
/**
 * parakeet.ts CLI
 *
 * Usage:
 *   parakeet file.wav
 *   parakeet file.wav --json
 *   parakeet file.wav --model mlx-community/parakeet-tdt-0.6b-v3
 *   parakeet --stream < pcm_f32le_16k.raw
 */
import { parseArgs } from 'node:util';
import { fromPretrained } from './load.js';
import { consumePcmStream } from '../model.js';
const DEFAULT_MODEL = 'mlx-community/parakeet-tdt-0.6b-v3';
// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------
const { values, positionals } = parseArgs({
    options: {
        model: { type: 'string', short: 'm', default: DEFAULT_MODEL },
        json: { type: 'boolean', short: 'j', default: false },
        stream: { type: 'boolean', short: 's', default: false },
        help: { type: 'boolean', short: 'h', default: false },
    },
    allowPositionals: true,
    args: process.argv.slice(2),
});
if (values.help) {
    process.stdout.write(`
parakeet — Nvidia Parakeet ASR (parakeet.ts)

Usage:
  parakeet <file.wav> [options]
  parakeet --stream [options] < pcm_f32le_16k.raw

Options:
  --model, -m <id>   HuggingFace repo or local dir (default: ${DEFAULT_MODEL})
  --json, -j         Output full AlignedResult JSON instead of plain text
  --stream, -s       Read raw f32le mono 16kHz PCM from stdin
  --help, -h         Show this help

Exit codes: 0 success  1 file/IO error  2 model error
`.trimStart());
    process.exit(0);
}
// ---------------------------------------------------------------------------
// Progress bar helpers
// ---------------------------------------------------------------------------
function progressBar(label, downloaded, total) {
    if (!process.stderr.isTTY) {
        return; // non-TTY: only emit the one-shot line (handled at call site)
    }
    if (total <= 0) {
        process.stderr.write(`\r${label}: ${formatBytes(downloaded)}`);
        return;
    }
    const pct = Math.min(100, Math.round((downloaded / total) * 100));
    const barWidth = 30;
    const filled = Math.round((pct / 100) * barWidth);
    const bar = '█'.repeat(filled) + '░'.repeat(barWidth - filled);
    process.stderr.write(`\r${label}: [${bar}] ${pct}% (${formatBytes(downloaded)}/${formatBytes(total)})`);
}
function formatBytes(n) {
    if (n < 1024)
        return `${n} B`;
    if (n < 1024 * 1024)
        return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
    const modelId = values.model ?? DEFAULT_MODEL;
    // Load model
    process.stderr.write(`Loading model: ${modelId}\n`);
    let lastFile = '';
    const model = await fromPretrained(modelId, {
        onProgress(file, downloaded, total) {
            if (file !== lastFile) {
                if (lastFile !== '')
                    process.stderr.write('\n'); // end previous file's line
                if (!process.stderr.isTTY) {
                    process.stderr.write(`Downloading ${file}...\n`);
                }
                lastFile = file;
            }
            progressBar(file, downloaded, total);
            if (downloaded >= total && total > 0) {
                if (process.stderr.isTTY)
                    process.stderr.write('\n');
            }
        },
    });
    if (lastFile !== '' && process.stderr.isTTY) {
        // Ensure cursor is on a new line after progress bars
        process.stderr.write('\n');
    }
    process.stderr.write('Model ready.\n');
    let result;
    if (values.stream) {
        // Streaming mode: read raw f32le PCM from stdin
        process.stderr.write('Streaming PCM from stdin (16kHz mono f32le)...\n');
        const stream = model.transcribeStream();
        async function* stdinPcm() {
            for await (const chunk of process.stdin) {
                // Reinterpret raw bytes as Float32 (little-endian f32)
                const buf = chunk instanceof Buffer ? chunk : Buffer.from(chunk);
                // Ensure alignment: process only whole float32 values
                const floatCount = Math.floor(buf.byteLength / 4);
                if (floatCount > 0) {
                    yield new Float32Array(buf.buffer, buf.byteOffset, floatCount);
                }
            }
        }
        result = await consumePcmStream(stream, stdinPcm());
    }
    else {
        // File mode
        const filePath = positionals[0];
        if (!filePath) {
            process.stderr.write('Error: provide a WAV file path or use --stream\n');
            process.exit(1);
        }
        process.stderr.write(`Transcribing: ${filePath}\n`);
        result = await model.transcribe(filePath);
    }
    // Output
    if (values.json) {
        process.stdout.write(JSON.stringify(result, null, 2) + '\n');
    }
    else {
        process.stdout.write(result.text + '\n');
    }
}
main().catch(err => {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes('ENOENT') || msg.includes('EACCES') || msg.includes('Failed to download')) {
        process.stderr.write(`Error: ${msg}\n`);
        process.exit(1);
    }
    process.stderr.write(`Model error: ${msg}\n`);
    process.exit(2);
});
//# sourceMappingURL=cli.js.map