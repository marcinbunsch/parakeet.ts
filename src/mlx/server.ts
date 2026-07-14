/**
 * parakeet.ts/mlx/server — Hono route factory for the streaming ASR endpoint.
 *
 * Usage:
 *   import { Hono } from 'hono';
 *   import { serve } from '@hono/node-server';
 *   import { fromPretrained } from 'parakeet.ts/mlx';
 *   import { createParakeetRoutes } from 'parakeet.ts/mlx/server';
 *
 *   const model = await fromPretrained('mlx-community/parakeet-tdt-0.6b-v3');
 *   const app = new Hono();
 *   app.route('/asr', createParakeetRoutes({ model }));
 *   serve({ fetch: app.fetch, port: 8080 });
 *
 * Requires hono as a peer dependency: npm i hono @hono/node-server
 */

import { Hono } from 'hono';
import type { Context } from 'hono';
import { consumePcmStream } from './parakeet.js';
import type { BaseParakeet } from './parakeet.js';

const REQUIRED_CONTENT_TYPE = 'audio/pcm; rate=16000; channels=1; format=f32le';

export interface ParakeetRouteOptions {
  model: BaseParakeet;
  /** Maximum duration of audio accepted per request (default: 300 s). */
  maxDurationSeconds?: number;
  /** Close request if no bytes arrive within this window (default: 30 000 ms). */
  idleTimeoutMs?: number;
}

/**
 * Create a Hono sub-application with the Parakeet ASR routes.
 * Mount it with `app.route('/asr', createParakeetRoutes({ model }))`.
 *
 * Routes:
 *   POST /transcribe   — stream raw PCM, get AlignedResult JSON
 */
export function createParakeetRoutes(options: ParakeetRouteOptions): Hono {
  const { model, maxDurationSeconds = 300, idleTimeoutMs = 30_000 } = options;

  const app = new Hono();

  // Maximum bytes of audio we'll accept: maxDurationSeconds * 16000 Hz * 4 bytes/sample
  const maxBytes = maxDurationSeconds * 16_000 * 4;

  app.post('/transcribe', async (c: Context) => {
    // Validate Content-Type
    const contentType = c.req.header('content-type') ?? '';
    if (contentType.trim().toLowerCase() !== REQUIRED_CONTENT_TYPE) {
      return c.json(
        { error: 'unsupported_media_type', message: `Content-Type must be "${REQUIRED_CONTENT_TYPE}"` },
        415,
      );
    }

    const streamingSession = model.transcribeStream();
    let totalBytes = 0;
    let timedOut = false;
    let tooLarge = false;

    async function* bodyToPcm(): AsyncIterable<Float32Array> {
      const body: ReadableStream<Uint8Array> | null = c.req.raw.body;
      if (!body) return;

      const reader = body.getReader();
      let leftover = new Uint8Array(0);

      try {
        for (;;) {
          // Idle timeout: race reader.read() against a timeout promise
          let timeoutHandle: ReturnType<typeof setTimeout>;
          const timeoutPromise = new Promise<null>(resolve => {
            timeoutHandle = setTimeout(() => resolve(null), idleTimeoutMs);
          });
          const readPromise = reader.read();
          const raceResult = await Promise.race([readPromise, timeoutPromise]);

          clearTimeout(timeoutHandle!);

          if (raceResult === null) {
            timedOut = true;
            reader.cancel().catch(() => {});
            return;
          }

          const chunk = raceResult as { done: boolean; value?: Uint8Array };
          if (chunk.done) break;

          const incoming = chunk.value!;
          totalBytes += incoming.byteLength;

          if (totalBytes > maxBytes) {
            tooLarge = true;
            reader.cancel().catch(() => {});
            return;
          }

          // Combine leftover bytes with new chunk
          const combined = new Uint8Array(leftover.byteLength + incoming.byteLength);
          combined.set(leftover);
          combined.set(incoming, leftover.byteLength);

          const floatCount = Math.floor(combined.byteLength / 4);
          if (floatCount > 0) {
            const usedBytes = floatCount * 4;
            yield new Float32Array(combined.buffer, combined.byteOffset, floatCount);
            leftover = combined.slice(usedBytes);
          } else {
            leftover = combined;
          }
        }
      } finally {
        reader.releaseLock();
      }
    }

    try {
      const result = await consumePcmStream(streamingSession, bodyToPcm());

      if (tooLarge) {
        return c.json(
          { error: 'payload_too_large', message: `Audio exceeds maximum of ${maxDurationSeconds} seconds` },
          413,
        );
      }

      if (timedOut) {
        return c.json(
          { error: 'idle_timeout', message: `No audio received for ${idleTimeoutMs} ms` },
          408,
        );
      }

      return c.json(result, 200);

    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);

      // Client disconnect mid-stream: silent drop
      if (message.includes('aborted') || message.includes('disconnected')) {
        console.error('[parakeet-mlx] client disconnected mid-stream');
        return new Response(null, { status: 499 });
      }

      console.error('[parakeet-mlx] internal error:', err);
      return c.json(
        { error: 'internal_error', message: 'Unexpected error during transcription' },
        500,
      );
    }
  });

  return app;
}
