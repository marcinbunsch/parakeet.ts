export interface AlignedToken {
  id: number;
  text: string;
  start: float;
  duration: float;
  confidence: float;
  end: float;
}

// TypeScript doesn't have a float type, use number
type float = number;

export function makeAlignedToken(
  id: number,
  text: string,
  start: number,
  duration: number,
  confidence: number = 1.0,
): AlignedToken {
  return { id, text, start, duration, confidence, end: start + duration };
}

export interface AlignedSentence {
  text: string;
  tokens: AlignedToken[];
  start: number;
  end: number;
  duration: number;
  confidence: number;
}

export function makeAlignedSentence(
  text: string,
  tokens: AlignedToken[],
): AlignedSentence {
  const sorted = [...tokens].sort((a, b) => a.start - b.start);
  const start = sorted[0].start;
  const end = sorted[sorted.length - 1].end;
  const duration = end - start;
  // Geometric mean of token confidences
  const logSum = sorted.reduce((acc, t) => acc + Math.log(t.confidence + 1e-10), 0);
  const confidence = Math.exp(logSum / sorted.length);
  return { text, tokens: sorted, start, end, duration, confidence };
}

export interface AlignedResult {
  text: string;
  sentences: AlignedSentence[];
  tokens: AlignedToken[];
}

export function makeAlignedResult(
  text: string,
  sentences: AlignedSentence[],
): AlignedResult {
  return {
    text: text.trim(),
    sentences,
    get tokens() {
      return sentences.flatMap(s => s.tokens);
    },
  };
}

export interface SentenceConfig {
  maxWords?: number;
  silenceGap?: number;
  maxDuration?: number;
}

export function tokensToSentences(
  tokens: AlignedToken[],
  config: SentenceConfig = {},
): AlignedSentence[] {
  if (tokens.length === 0) return [];

  const sentences: AlignedSentence[] = [];
  let current: AlignedToken[] = [];

  for (let idx = 0; idx < tokens.length; idx++) {
    const token = tokens[idx];
    current.push(token);

    const isPunctuation =
      token.text.includes('!') ||
      token.text.includes('?') ||
      token.text.includes('。') ||
      token.text.includes('？') ||
      token.text.includes('！') ||
      (token.text.includes('.') &&
        (idx === tokens.length - 1 || tokens[idx + 1].text.includes(' ')));

    const isWordLimit =
      config.maxWords !== undefined &&
      idx !== tokens.length - 1 &&
      current.filter(t => t.text.includes(' ')).length +
        (tokens[idx + 1].text.includes(' ') ? 1 : 0) >
        config.maxWords;

    const isLongSilence =
      config.silenceGap !== undefined &&
      idx !== tokens.length - 1 &&
      tokens[idx + 1].start - token.end >= config.silenceGap;

    const isOverDuration =
      config.maxDuration !== undefined &&
      token.end - current[0].start >= config.maxDuration;

    if (isPunctuation || isWordLimit || isLongSilence || isOverDuration) {
      const text = current.map(t => t.text).join('');
      sentences.push(makeAlignedSentence(text, current));
      current = [];
    }
  }

  if (current.length > 0) {
    const text = current.map(t => t.text).join('');
    sentences.push(makeAlignedSentence(text, current));
  }

  return sentences;
}

export function sentencesToResult(sentences: AlignedSentence[]): AlignedResult {
  const text = sentences.map(s => s.text).join('');
  return makeAlignedResult(text, sentences);
}

export function mergeLongestContiguous(
  a: AlignedToken[],
  b: AlignedToken[],
  overlapDuration: number,
): AlignedToken[] {
  if (a.length === 0) return b;
  if (b.length === 0) return a;

  const aEndTime = a[a.length - 1].end;
  const bStartTime = b[0].start;

  if (aEndTime <= bStartTime) return [...a, ...b];

  const overlapA = a.filter(t => t.end > bStartTime - overlapDuration);
  const overlapB = b.filter(t => t.start < aEndTime + overlapDuration);

  const enoughPairs = Math.floor(overlapA.length / 2);

  if (overlapA.length < 2 || overlapB.length < 2) {
    const cutoffTime = (aEndTime + bStartTime) / 2;
    return [
      ...a.filter(t => t.end <= cutoffTime),
      ...b.filter(t => t.start >= cutoffTime),
    ];
  }

  // Find longest contiguous matching subsequence
  let bestContiguous: Array<[number, number]> = [];

  for (let i = 0; i < overlapA.length; i++) {
    for (let j = 0; j < overlapB.length; j++) {
      if (
        overlapA[i].id === overlapB[j].id &&
        Math.abs(overlapA[i].start - overlapB[j].start) < overlapDuration / 2
      ) {
        const current: Array<[number, number]> = [];
        let k = i, l = j;
        while (
          k < overlapA.length &&
          l < overlapB.length &&
          overlapA[k].id === overlapB[l].id &&
          Math.abs(overlapA[k].start - overlapB[l].start) < overlapDuration / 2
        ) {
          current.push([k, l]);
          k++;
          l++;
        }
        if (current.length > bestContiguous.length) {
          bestContiguous = current;
        }
      }
    }
  }

  if (bestContiguous.length >= enoughPairs) {
    const aStartIdx = a.length - overlapA.length;
    const lcsIndicesA = bestContiguous.map(p => aStartIdx + p[0]);
    const lcsIndicesB = bestContiguous.map(p => p[1]);

    const result: AlignedToken[] = [];
    result.push(...a.slice(0, lcsIndicesA[0]));

    for (let i = 0; i < bestContiguous.length; i++) {
      result.push(a[lcsIndicesA[i]]);

      if (i < bestContiguous.length - 1) {
        const nextA = lcsIndicesA[i + 1];
        const nextB = lcsIndicesB[i + 1];
        const gapA = a.slice(lcsIndicesA[i] + 1, nextA);
        const gapB = b.slice(lcsIndicesB[i] + 1, nextB);
        result.push(...(gapB.length > gapA.length ? gapB : gapA));
      }
    }

    result.push(...b.slice(lcsIndicesB[lcsIndicesB.length - 1] + 1));
    return result;
  }

  throw new Error(`No pairs exceeding ${enoughPairs}`);
}

export function mergeLongestCommonSubsequence(
  a: AlignedToken[],
  b: AlignedToken[],
  overlapDuration: number,
): AlignedToken[] {
  if (a.length === 0) return b;
  if (b.length === 0) return a;

  const aEndTime = a[a.length - 1].end;
  const bStartTime = b[0].start;

  if (aEndTime <= bStartTime) return [...a, ...b];

  const overlapA = a.filter(t => t.end > bStartTime - overlapDuration);
  const overlapB = b.filter(t => t.start < aEndTime + overlapDuration);

  if (overlapA.length < 2 || overlapB.length < 2) {
    const cutoffTime = (aEndTime + bStartTime) / 2;
    return [
      ...a.filter(t => t.end <= cutoffTime),
      ...b.filter(t => t.start >= cutoffTime),
    ];
  }

  // LCS dynamic programming
  const m = overlapA.length;
  const n = overlapB.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (
        overlapA[i - 1].id === overlapB[j - 1].id &&
        Math.abs(overlapA[i - 1].start - overlapB[j - 1].start) < overlapDuration / 2
      ) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  // Backtrack
  const lcsPairs: Array<[number, number]> = [];
  let i = m, j = n;
  while (i > 0 && j > 0) {
    if (
      overlapA[i - 1].id === overlapB[j - 1].id &&
      Math.abs(overlapA[i - 1].start - overlapB[j - 1].start) < overlapDuration / 2
    ) {
      lcsPairs.push([i - 1, j - 1]);
      i--;
      j--;
    } else if (dp[i - 1][j] > dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }
  lcsPairs.reverse();

  if (lcsPairs.length === 0) {
    const cutoffTime = (aEndTime + bStartTime) / 2;
    return [
      ...a.filter(t => t.end <= cutoffTime),
      ...b.filter(t => t.start >= cutoffTime),
    ];
  }

  const aStartIdx = a.length - overlapA.length;
  const lcsIndicesA = lcsPairs.map(p => aStartIdx + p[0]);
  const lcsIndicesB = lcsPairs.map(p => p[1]);

  const result: AlignedToken[] = [];
  result.push(...a.slice(0, lcsIndicesA[0]));

  for (let k = 0; k < lcsPairs.length; k++) {
    result.push(a[lcsIndicesA[k]]);

    if (k < lcsPairs.length - 1) {
      const nextA = lcsIndicesA[k + 1];
      const nextB = lcsIndicesB[k + 1];
      const gapA = a.slice(lcsIndicesA[k] + 1, nextA);
      const gapB = b.slice(lcsIndicesB[k] + 1, nextB);
      result.push(...(gapB.length > gapA.length ? gapB : gapA));
    }
  }

  result.push(...b.slice(lcsIndicesB[lcsIndicesB.length - 1] + 1));
  return result;
}
