export type ProgressFn = (downloaded: number, total: number) => void;
/** Default HF cache root, honouring HF_HOME / HF_HUB_CACHE like huggingface_hub. */
export declare function defaultCacheDir(): string;
/** Local directory a repo's files are cached in. */
export declare function repoDir(repoId: string, cacheDir?: string): string;
/**
 * Fetch one file from a repo, returning its local path. No-op if already cached.
 */
export declare function downloadFromHub(repoId: string, filename: string, cacheDir?: string, onProgress?: ProgressFn): Promise<string>;
/**
 * Fetch several files, skipping any marked optional that 404.
 * Returns the repo's local directory.
 */
export declare function downloadRepoFiles(repoId: string, files: Array<{
    name: string;
    optional?: boolean;
}>, cacheDir?: string, onProgress?: (file: string, downloaded: number, total: number) => void): Promise<string>;
//# sourceMappingURL=hub.d.ts.map