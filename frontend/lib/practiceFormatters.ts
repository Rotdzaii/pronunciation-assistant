import { colors } from '../components/AppUI';
import type { PracticeFeedback, ProblemPhoneme } from '../types';

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
    return item.trim() || null;
  }

  if (!item || typeof item !== 'object') {
    return null;
  }

  const phoneme = toDisplayText(item.phoneme);
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
