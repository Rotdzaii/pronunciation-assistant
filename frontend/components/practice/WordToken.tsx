import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { colors, radius } from '../../constants/theme';

export type WordStatus = 'correct' | 'incorrect' | 'warning' | 'neutral';

type Props = {
  word: string;
  status: WordStatus;
};

// Tailwind colors used verbatim in result.html for word tokens
// (not in M3 palette — standard Tailwind emerald/rose/amber scale)
const EMERALD_200 = '#a7f3d0';
const ROSE_200    = '#fecdd3';
const AMBER_200   = '#fde68a';

type TokenConfig = {
  textColor: string;
  bgColor?: string;
  underlineColor?: string;
  brackets?: boolean;
};

const CONFIG: Record<WordStatus, TokenConfig> = {
  /** text-emerald-500 border-b-2 border-emerald-200 */
  correct: {
    textColor:      colors.success,
    underlineColor: EMERALD_200,
  },
  /** text-rose-500 border-b-2 border-rose-200 (decoration-wavy → straight in RN) */
  incorrect: {
    textColor:      colors.danger,
    underlineColor: ROSE_200,
  },
  /** text-amber-500 bg-amber-50 border-b-2 border-amber-200 → displayed as [word] */
  warning: {
    textColor:      colors.warning,
    bgColor:        colors.warningLight,
    underlineColor: AMBER_200,
    brackets:       true,
  },
  /** plain text-on-surface */
  neutral: {
    textColor: colors.onSurface,
  },
};

/**
 * A single word chip in the pronunciation analysis sentence.
 * Based on the Interactive Sentence Card in result.html.
 *
 * Renders the word with colour-coded status:
 *  - correct  → green text + green underline
 *  - incorrect → red text + rose underline
 *  - warning  → amber text + amber bg + brackets + amber underline
 *  - neutral  → default dark text, no underline
 */
export default function WordToken({ word, status }: Props) {
  const cfg = CONFIG[status];
  const displayWord = cfg.brackets ? `[${word}]` : word;

  return (
    <View
      style={[
        styles.wrapper,
        cfg.bgColor && {
          backgroundColor: cfg.bgColor,
          paddingHorizontal: 8,
          paddingVertical: 4,
          borderRadius: radius.sm,   // 8 dp — rounded-lg
        },
      ]}
    >
      <Text style={[styles.word, { color: cfg.textColor }]}>
        {displayWord}
      </Text>

      {cfg.underlineColor ? (
        <View
          style={[styles.underline, { backgroundColor: cfg.underlineColor }]}
        />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    alignItems: 'center',
  },
  word: {
    fontSize: 26,           // web text-3xl=30px, scaled ~13% for mobile
    fontWeight: '700',
    lineHeight: 34,
  },
  underline: {
    height: 2,
    alignSelf: 'stretch',
    borderRadius: 1,
    marginTop: 3,
  },
});
