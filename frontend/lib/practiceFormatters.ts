import { colors } from '../components/AppUI';
import type { PracticeFeedback, ProblemPhoneme } from '../types';

const ARPABET_TO_IPA: Record<string, string> = {
  // vowels
  AA: 'ɑ', AA0: 'ɑ', AA1: 'ɑ', AA2: 'ɑ',
  AE: 'æ', AE0: 'æ', AE1: 'æ', AE2: 'æ',
  AH: 'ʌ', AH0: 'ə', AH1: 'ʌ', AH2: 'ʌ',
  AO: 'ɔ', AO0: 'ɔ', AO1: 'ɔ', AO2: 'ɔ',
  AW: 'aʊ', AW0: 'aʊ', AW1: 'aʊ', AW2: 'aʊ',
  AY: 'aɪ', AY0: 'aɪ', AY1: 'aɪ', AY2: 'aɪ',
  EH: 'ɛ', EH0: 'ɛ', EH1: 'ɛ', EH2: 'ɛ',
  ER: 'ɜr', ER0: 'ər', ER1: 'ɜr', ER2: 'ɜr',
  EY: 'eɪ', EY0: 'eɪ', EY1: 'eɪ', EY2: 'eɪ',
  IH: 'ɪ', IH0: 'ɪ', IH1: 'ɪ', IH2: 'ɪ',
  IY: 'iː', IY0: 'iː', IY1: 'iː', IY2: 'iː',
  OW: 'oʊ', OW0: 'oʊ', OW1: 'oʊ', OW2: 'oʊ',
  OY: 'ɔɪ', OY0: 'ɔɪ', OY1: 'ɔɪ', OY2: 'ɔɪ',
  UH: 'ʊ', UH0: 'ʊ', UH1: 'ʊ', UH2: 'ʊ',
  UW: 'uː', UW0: 'uː', UW1: 'uː', UW2: 'uː',
  // consonants
  B: 'b', CH: 'tʃ', D: 'd', DH: 'ð', F: 'f',
  G: 'ɡ', HH: 'h', JH: 'dʒ', K: 'k', L: 'l',
  M: 'm', N: 'n', NG: 'ŋ', P: 'p', R: 'r',
  S: 's', SH: 'ʃ', T: 't', TH: 'θ', V: 'v',
  W: 'w', WH: 'ʍ', Y: 'j', Z: 'z', ZH: 'ʒ',
};

function arpabetToIpa(code: string): string {
  const ipa = ARPABET_TO_IPA[code.toUpperCase()] ?? ARPABET_TO_IPA[code] ?? null;
  return ipa ? `/${ipa}/` : `/${code}/`;
}

export type ScoreTone = {
  color: string;
  softColor: string;
  label: string;
};

export function getScoreTone(score: number | null | undefined): ScoreTone {
  const value = clampScore(score);

  if (value <= 39) {
    return {
      color: colors.error,
      softColor: colors.softRed,
      label: 'Cần luyện lại',
    };
  }
  if (value <= 59) {
    return {
      color: colors.accent,
      softColor: colors.softOrange,
      label: 'Cần cải thiện',
    };
  }
  if (value <= 74) {
    return {
      color: colors.warning,
      softColor: '#FFFBEB',
      label: 'Khá ổn',
    };
  }
  if (value <= 89) {
    return {
      color: colors.primary,
      softColor: colors.softBlue,
      label: 'Tốt',
    };
  }

  return {
    color: colors.success,
    softColor: '#F0FDF4',
    label: 'Rất tốt',
  };
}

export function clampScore(score: number | null | undefined): number {
  if (typeof score !== 'number' || Number.isNaN(score)) {
    return 0;
  }

  return Math.max(0, Math.min(100, score));
}

export function formatProblemPhonemes(
  problemPhonemes: ProblemPhoneme[] | null | undefined,
): string[] {
  if (!problemPhonemes?.length) {
    return ['Chưa phát hiện lỗi nổi bật.'];
  }

  const lines = problemPhonemes
    .map(formatProblemPhoneme)
    .filter((line): line is string => Boolean(line));

  return lines.length ? lines : ['Chưa phát hiện lỗi nổi bật.'];
}

export function formatFeedbackLines(
  feedback: PracticeFeedback | null | undefined,
): string[] {
  if (!feedback || Object.keys(feedback).length === 0) {
    return ['Chưa có nhận xét chi tiết.'];
  }

  const lines: string[] = [];
  const summary = toDisplayText(feedback.summary ?? feedback.message ?? feedback.text);
  if (summary) {
    lines.push(summary);
  }

  if (Array.isArray(feedback.tips)) {
    feedback.tips.forEach((tip) => {
      const tipText = toDisplayText(tip);
      if (tipText) {
        lines.push(`• ${tipText}`);
      }
    });
  }

  return lines.length ? lines : ['Chưa có nhận xét chi tiết.'];
}

function formatProblemPhoneme(item: ProblemPhoneme): string | null {
  if (typeof item === 'string') {
    const code = item.trim();
    return code ? arpabetToIpa(code) : null;
  }

  if (!item || typeof item !== 'object') {
    return null;
  }

  const rawPhoneme = toDisplayText(item.phoneme);
  const phoneme = rawPhoneme ? arpabetToIpa(rawPhoneme) : null;
  const word = toDisplayText(item.word);
  const type = toDisplayText(item.type);
  const severity = toDisplayText(item.severity);
  const tip = toDisplayText(item.tip ?? item.message ?? item.description);
  const subject = [word, phoneme].filter(Boolean).join(' ');
  const context = [type, severity].filter(Boolean).join(', ');
  const leading = subject || context;

  if (leading && tip) {
    return `${leading} — ${tip}`;
  }
  if (leading) {
    return leading;
  }
  if (tip) {
    return tip;
  }

  return null;
}

function toDisplayText(value: unknown): string | null {
  if (typeof value === 'string') {
    return value.trim() || null;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  return null;
}
