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
import type { ParakeetModel } from '../model.js';
export interface ParakeetRouteOptions {
    model: ParakeetModel;
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
export declare function createParakeetRoutes(options: ParakeetRouteOptions): Hono;
//# sourceMappingURL=server.d.ts.map