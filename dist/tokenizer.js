/**
 * Decode a list of token IDs to a string using the vocabulary.
 * The vocabulary uses '▁' as a word-boundary prefix (SentencePiece convention).
 */
export function decode(tokens, vocabulary) {
    return tokens.map(id => vocabulary[id].replace(/▁/g, ' ')).join('');
}
//# sourceMappingURL=tokenizer.js.map