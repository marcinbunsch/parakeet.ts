/**
 * Low-level model-loading helpers for the MLX backend.
 *
 *  - `loadSafetensors` parses a SafeTensors weight file into a WeightMap.
 *  - `downloadFromHub` is re-exported from `../hub.ts` for backwards
 *    compatibility; the implementation is backend-agnostic.
 *
 * The public loaders (`fromLocal` / `fromPretrained`) live in `./load.ts`, which
 * builds a backend-agnostic `ParakeetModel` from these pieces.
 */
import fs from 'node:fs';
/**
 * Parse a safetensors file and return a WeightMap.
 * The safetensors format: 8 bytes (header length LE uint64) + JSON header + data.
 * Uses fd-based random access to support files larger than the 2 GiB Buffer limit.
 */
export function loadSafetensors(filePath) {
    const fd = fs.openSync(filePath, 'r');
    try {
        // Read the 8-byte header length
        const lenBuf = Buffer.allocUnsafe(8);
        fs.readSync(fd, lenBuf, 0, 8, 0);
        const headerLen = Number(new DataView(lenBuf.buffer, lenBuf.byteOffset, 8).getBigUint64(0, true));
        // Read the JSON header
        const headerBuf = Buffer.allocUnsafe(headerLen);
        fs.readSync(fd, headerBuf, 0, headerLen, 8);
        const header = JSON.parse(headerBuf.toString('utf8'));
        const dataStart = 8 + headerLen;
        const map = new Map();
        for (const [name, meta] of Object.entries(header)) {
            if (name === '__metadata__')
                continue;
            const [start, end] = meta.data_offsets;
            const byteLen = end - start;
            const rawSlice = Buffer.allocUnsafe(byteLen);
            fs.readSync(fd, rawSlice, 0, byteLen, dataStart + start);
            const shape = meta.shape;
            let data;
            switch (meta.dtype) {
                case 'F32':
                    data = new Float32Array(rawSlice.buffer, rawSlice.byteOffset, byteLen / 4);
                    break;
                case 'BF16':
                    data = bf16ToF32(rawSlice);
                    break;
                case 'F16':
                    data = f16ToF32(rawSlice);
                    break;
                case 'I32':
                    data = new Float32Array(new Int32Array(rawSlice.buffer, rawSlice.byteOffset, byteLen / 4));
                    break;
                default:
                    continue;
            }
            map.set(name, { data: new Float32Array(data), shape });
        }
        return map;
    }
    finally {
        fs.closeSync(fd);
    }
}
function bf16ToF32(buf) {
    const n = buf.byteLength / 2;
    const out = new Float32Array(n);
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    for (let i = 0; i < n; i++) {
        const bits = view.getUint16(i * 2, true);
        // BF16: sign[1] + exponent[8] + mantissa[7] → F32: shift mantissa left by 16
        const f32bits = bits << 16;
        const f32view = new DataView(new ArrayBuffer(4));
        f32view.setInt32(0, f32bits, true);
        out[i] = f32view.getFloat32(0, true);
    }
    return out;
}
function f16ToF32(buf) {
    const n = buf.byteLength / 2;
    const out = new Float32Array(n);
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    for (let i = 0; i < n; i++) {
        const h = view.getUint16(i * 2, true);
        const sign = (h >> 15) & 1;
        const exp = (h >> 10) & 0x1f;
        const mant = h & 0x3ff;
        if (exp === 0) {
            out[i] = sign ? -0 : 0;
        }
        else if (exp === 0x1f) {
            out[i] = mant ? NaN : sign ? -Infinity : Infinity;
        }
        else {
            const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
            out[i] = sign ? -f : f;
        }
    }
    return out;
}
// ---------------------------------------------------------------------------
// HuggingFace Hub download — implementation lives in ../hub.ts (backend-agnostic)
// ---------------------------------------------------------------------------
export { downloadFromHub } from '../hub.js';
//# sourceMappingURL=utils.js.map