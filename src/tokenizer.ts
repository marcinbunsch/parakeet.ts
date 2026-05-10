/**
 * Decode a list of token IDs to a string using the vocabulary.
 * The vocabulary uses '▁' as a word-boundary prefix (SentencePiece convention).
 */
export function decode(tokens: number[], vocabulary: string[]): string {
  return tokens.map(id => vocabulary[id].replace(/▁/g, ' ')).join('');
}
