export interface AlignedToken {
    id: number;
    text: string;
    start: float;
    duration: float;
    confidence: float;
    end: float;
}
type float = number;
export declare function makeAlignedToken(id: number, text: string, start: number, duration: number, confidence?: number): AlignedToken;
export interface AlignedSentence {
    text: string;
    tokens: AlignedToken[];
    start: number;
    end: number;
    duration: number;
    confidence: number;
}
export declare function makeAlignedSentence(text: string, tokens: AlignedToken[]): AlignedSentence;
export interface AlignedResult {
    text: string;
    sentences: AlignedSentence[];
    tokens: AlignedToken[];
}
export declare function makeAlignedResult(text: string, sentences: AlignedSentence[]): AlignedResult;
export interface SentenceConfig {
    maxWords?: number;
    silenceGap?: number;
    maxDuration?: number;
}
export declare function tokensToSentences(tokens: AlignedToken[], config?: SentenceConfig): AlignedSentence[];
export declare function sentencesToResult(sentences: AlignedSentence[]): AlignedResult;
export declare function mergeLongestContiguous(a: AlignedToken[], b: AlignedToken[], overlapDuration: number): AlignedToken[];
export declare function mergeLongestCommonSubsequence(a: AlignedToken[], b: AlignedToken[], overlapDuration: number): AlignedToken[];
export {};
//# sourceMappingURL=alignment.d.ts.map