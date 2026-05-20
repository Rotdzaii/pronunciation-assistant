import { colors } from '../components/AppUI';
import type { PracticeFeedback, ProblemPhoneme } from '../types';

export type ScoreTone = {
  color: string;
  softColor: string;
  label: string;
};

export type FeedbackMetadataLine = {
  label: string;
  value: string;
  tone?: 'warning' | 'muted';
};

export type ModelConfidenceDisplay = {
  percent: number;
  level: string;
  isReliable: boolean;
  label: string;
};

export type ResultLabelDisplay = {
  value: string;
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
  const seen = new Set<string>();
  const summary = toDisplayText(feedback.summary ?? feedback.message ?? feedback.text);
  if (summary) {
    lines.push(summary);
    seen.add(normalizeForCompare(summary));
  }

  let visibleTipCount = 0;
  if (Array.isArray(feedback.tips)) {
    feedback.tips.forEach((tip) => {
      const tipText = toDisplayText(tip);
      const normalized = normalizeForCompare(tipText);
      if (tipText && normalized && !seen.has(normalized) && visibleTipCount < 3) {
        seen.add(normalized);
        visibleTipCount += 1;
        lines.push(`• ${tipText}`);
      }
    });
  }

  return lines.length ? lines : ['Chưa có nhận xét chi tiết.'];
}

export function getFeedbackSummary(
  feedback: PracticeFeedback | null | undefined,
): string | null {
  return toDisplayText(feedback?.summary ?? feedback?.message ?? feedback?.text);
}

export function getVisibleFeedbackTips(
  feedback: PracticeFeedback | null | undefined,
  limit = 3,
): string[] {
  if (!feedback || !Array.isArray(feedback.tips)) {
    return [];
  }

  const summary = normalizeForCompare(getFeedbackSummary(feedback));
  const seen = new Set<string>();
  const tips: string[] = [];

  feedback.tips.forEach((tip) => {
    const tipText = toDisplayText(tip);
    const normalized = normalizeForCompare(tipText);
    if (!tipText || !normalized || normalized === summary || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    tips.push(tipText);
  });

  return tips.slice(0, limit);
}

export function getModelConfidenceDisplay(
  feedback: PracticeFeedback | null | undefined,
): ModelConfidenceDisplay | null {
  const confidence = parseModelConfidence(feedback?.model_confidence);
  if (!confidence) {
    return null;
  }

  const percent = Math.round(confidence.value * 100);
  return {
    percent,
    level: confidence.level,
    isReliable: confidence.isReliable,
    label: `${percent}% (${confidence.level})`,
  };
}

export function isWav2Vec2Feedback(feedback: PracticeFeedback | null | undefined): boolean {
  return toDisplayText(feedback?.scorer)?.toLowerCase() === 'wav2vec2';
}

export function getFeedbackScorerBadge(
  feedback: PracticeFeedback | null | undefined,
): string | null {
  const scorer = toDisplayText(feedback?.scorer)?.toLowerCase();
  if (scorer === 'mock') {
    return 'Mock AI';
  }
  if (scorer === 'wav2vec2') {
    return 'AI chấm tự động';
  }
  return null;
}

export function formatBaselineMetadata(
  feedback: PracticeFeedback | null | undefined,
): FeedbackMetadataLine[] {
  if (!feedback || !isWav2Vec2Feedback(feedback)) {
    return [];
  }

  const lines: FeedbackMetadataLine[] = [];
  const resultLabel = getResultLabelDisplay(feedback);
  const recognizedText = toDisplayText(feedback.recognized_text);
  const targetWord = toDisplayText(feedback.target_word);
  const confidence = parseModelConfidence(feedback.model_confidence);

  if (resultLabel) {
    lines.push({ label: 'Kết quả', value: resultLabel.label });
  }
  if (recognizedText) {
    lines.push({ label: 'AI nghe được', value: recognizedText });
  }
  if (targetWord) {
    lines.push({ label: 'Từ mục tiêu', value: targetWord });
  }
  if (confidence) {
    const percent = Math.round(confidence.value * 100);
    lines.push({
      label: 'Độ tin cậy',
      value: `${percent}% (${confidence.level})`,
      tone: confidence.isReliable ? undefined : 'warning',
    });
  }

  return lines;
}

export function getResultLabelDisplay(
  feedback: PracticeFeedback | null | undefined,
): ResultLabelDisplay | null {
  const value = toDisplayText(feedback?.result_label);
  if (!value) {
    return null;
  }

  const labels: Record<string, string> = {
    correct: 'Đúng',
    near_correct: 'Gần đúng',
    partial_match: 'Khớp một phần',
    mismatch: 'Chưa khớp',
    failed: 'Không chấm được',
  };

  return {
    value,
    label: labels[value] ?? value,
  };
}

export function hasLowConfidenceWarning(
  feedback: PracticeFeedback | null | undefined,
): boolean {
  const confidence = parseModelConfidence(feedback?.model_confidence);
  return Boolean(confidence && !confidence.isReliable);
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

function normalizeForCompare(value: string | null): string {
  return (value ?? '').trim().replace(/^â€¢\s*/, '').replace(/^•\s*/, '').toLowerCase();
}

function parseModelConfidence(value: unknown): {
  value: number;
  level: string;
  isReliable: boolean;
} | null {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const confidence = value as {
    value?: unknown;
    level?: unknown;
    is_reliable?: unknown;
  };
  if (typeof confidence.value !== 'number' || Number.isNaN(confidence.value)) {
    return null;
  }

  return {
    value: Math.max(0, Math.min(1, confidence.value)),
    level: toDisplayText(confidence.level) ?? 'unknown',
    isReliable: confidence.is_reliable === true,
  };
}
