/**
 * HuggingFace Hub downloads — backend-agnostic.
 *
 * Files are cached under the HF hub layout so both backends share one cache
 * directory. Downloads are atomic (write to `.tmp`, then rename) so an
 * interrupted fetch never leaves a truncated file that later looks cached.
 */
import fs from 'node:fs';
import path from 'node:path';

export type ProgressFn = (downloaded: number, total: number) => void;

/** Default HF cache root, honouring HF_HOME / HF_HUB_CACHE like huggingface_hub. */
export function defaultCacheDir(): string {
  if (process.env['HF_HUB_CACHE']) return process.env['HF_HUB_CACHE'];
  if (process.env['HF_HOME']) return path.join(process.env['HF_HOME'], 'hub');
  return path.join(
    process.env['HOME'] ?? process.env['USERPROFILE'] ?? '/tmp',
    '.cache', 'huggingface', 'hub',
  );
}

/** Local directory a repo's files are cached in. */
export function repoDir(repoId: string, cacheDir?: string): string {
  return path.join(cacheDir ?? defaultCacheDir(), repoId.replace(/\//g, '--'));
}

/**
 * Fetch one file from a repo, returning its local path. No-op if already cached.
 */
export async function downloadFromHub(
  repoId: string,
  filename: string,
  cacheDir?: string,
  onProgress?: ProgressFn,
): Promise<string> {
  const modelDir = repoDir(repoId, cacheDir);
  fs.mkdirSync(modelDir, { recursive: true });

  const localPath = path.join(modelDir, filename);
  if (fs.existsSync(localPath)) return localPath;

  fs.mkdirSync(path.dirname(localPath), { recursive: true });

  const url = `https://huggingface.co/${repoId}/resolve/main/${filename}`;
  const response = await fetch(url, { headers: { 'User-Agent': 'parakeet.ts/1.0.0' } });

  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }

  const total = parseInt(response.headers.get('content-length') ?? '0', 10);
  const tmpPath = `${localPath}.tmp`;

  if (response.body) {
    const writeStream = fs.createWriteStream(tmpPath);
    let downloaded = 0;
    const reader = response.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!writeStream.write(value)) {
          await new Promise<void>(resolve => writeStream.once('drain', () => resolve()));
        }
        downloaded += value.byteLength;
        if (onProgress) onProgress(downloaded, total);
      }
    } finally {
      reader.releaseLock();
    }
    await new Promise<void>((resolve, reject) => {
      writeStream.end((err?: Error | null) => (err ? reject(err) : resolve()));
    });
  } else {
    const arrayBuffer = await response.arrayBuffer();
    fs.writeFileSync(tmpPath, Buffer.from(arrayBuffer));
  }

  fs.renameSync(tmpPath, localPath);
  return localPath;
}

/**
 * Fetch several files, skipping any marked optional that 404.
 * Returns the repo's local directory.
 */
export async function downloadRepoFiles(
  repoId: string,
  files: Array<{ name: string; optional?: boolean }>,
  cacheDir?: string,
  onProgress?: (file: string, downloaded: number, total: number) => void,
): Promise<string> {
  for (const { name, optional } of files) {
    try {
      await downloadFromHub(repoId, name, cacheDir,
        onProgress ? (d, t) => onProgress(name, d, t) : undefined);
    } catch (err) {
      if (!optional) throw err;
    }
  }
  return repoDir(repoId, cacheDir);
}
